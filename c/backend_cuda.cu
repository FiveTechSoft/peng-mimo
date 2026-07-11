#include "backend_cuda.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cstring>

/* Profile: cumulative host-side time spent in PCIe H2D/D2H copies. */
static double g_copy_sec = 0.0;
extern "C" double coli_cuda_copy_seconds(void) { return g_copy_sec; }
static double now_s(void) {
    using namespace std::chrono;
    return duration_cast<duration<double>>(steady_clock::now().time_since_epoch()).count();
}

struct ColiCudaTensor {
    void *weights;
    float *scales;
    size_t weight_bytes;
    int fmt, I, O, device;
    int tracked;
};

typedef struct {
    int device;
    float *x, *y;
    size_t x_cap, y_cap;
    float *ma, *mb;
    size_t ma_cap, mb_cap;
    float *yacc;                                 /* MoE accumulate: sum w*swiglu on device */
    size_t yacc_cap;
    size_t tensor_count, tensor_bytes;
    int sticky_S, sticky_D;
    int sticky_valid;
    int moe_active, moe_S, moe_D;
    cudaStream_t stream;
} DeviceContext;

static DeviceContext g_ctx[COLI_CUDA_MAX_DEVICES];
static int g_nctx;

static int cuda_ok(cudaError_t err, const char *what) {
    if (err == cudaSuccess) return 1;
    std::fprintf(stderr, "[CUDA] %s: %s\n", what, cudaGetErrorString(err));
    return 0;
}

static DeviceContext *find_ctx(int device) {
    for (int i = 0; i < g_nctx; i++) if (g_ctx[i].device == device) return &g_ctx[i];
    return nullptr;
}

static int select_ctx(DeviceContext *ctx) {
    return ctx && cuda_ok(cudaSetDevice(ctx->device), "select device");
}

static size_t row_bytes(int fmt, int I) {
    if (fmt == 0) return (size_t)I * sizeof(float);
    if (fmt == 1) return (size_t)I;
    if (fmt == 2) return (size_t)(I + 1) / 2;
    if (fmt == 3) return (size_t)(I + 3) / 4;
    return 0;
}

/* ---- kernels ---- */

/* Generic path (f32 / uncommon). One block per (o,s). */
__global__ static void quant_matmul(float *y, const float *x, const void *weights,
                                    const float *scales, int fmt, int S, int I, int O,
                                    size_t rb) {
    int o = blockIdx.x;
    int s = blockIdx.y;
    float sum = 0.0f;
    size_t row = (size_t)o * rb;
    const float *xs = x + (size_t)s * I;
    const uint8_t *base = static_cast<const uint8_t *>(weights) + row;
    if (fmt == 0) {
        const float *w = reinterpret_cast<const float *>(base);
        for (int i = threadIdx.x; i < I; i += blockDim.x) sum += xs[i] * w[i];
    } else if (fmt == 1) {
        const int8_t *w = reinterpret_cast<const int8_t *>(base);
        for (int i = threadIdx.x; i < I; i += blockDim.x) sum += xs[i] * (float)w[i];
    } else if (fmt == 2) {
        for (int i = threadIdx.x; i < I; i += blockDim.x) {
            uint8_t v = base[i >> 1];
            float w = (float)(((i & 1) ? (v >> 4) : (v & 15)) - 8);
            sum += xs[i] * w;
        }
    } else {
        for (int i = threadIdx.x; i < I; i += blockDim.x) {
            uint8_t v = base[i >> 2];
            float w = (float)(((v >> ((i & 3) * 2)) & 3) - 2);
            sum += xs[i] * w;
        }
    }

    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int n = blockDim.x >> 1; n; n >>= 1) {
        if (threadIdx.x < n) partial[threadIdx.x] += partial[threadIdx.x + n];
        __syncthreads();
    }
    if (!threadIdx.x)
        y[(size_t)s * O + o] = partial[0] * (fmt ? scales[o] : 1.0f);
}

/*
 * Fast int4 GEMV for decode-heavy shapes (S small, I up to 8k):
 * - Stage x[S,I] into shared memory once per block (huge win vs re-reading global x
 *   for every output row independently from L2 only).
 * - 8 warps/block → 8 output rows per block; warp shuffle reduction.
 * MiMo experts: I=4096 (gate/up cols) or I=2048 (down cols), O=2048 or 4096.
 */
__global__ static void quant_gemv_i4(float *y, const float *x, const uint8_t *weights,
                                    const float *scales, int S, int I, int O) {
    const int warps = blockDim.x >> 5;           /* typically 8 */
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int o0 = (int)blockIdx.x * warps + warp;
    extern __shared__ float shx[];               /* [S * I] */

    /* Cooperative load of x into shared (all threads). */
    const int nxi = S * I;
    for (int i = threadIdx.x; i < nxi; i += blockDim.x) shx[i] = x[i];
    __syncthreads();

    if (o0 >= O) return;
    const int rb = (I + 1) / 2;
    const uint8_t *wrow = weights + (size_t)o0 * (size_t)rb;
    for (int s = 0; s < S; s++) {
        const float *xs = shx + s * I;
        float sum = 0.0f;
        /* 32 lanes × 2 nibble-pairs via 1 byte each step of 32 bytes of weights
         * covering 64 activations when unrolled — simple lane-strided over i. */
        for (int i = lane; i < I; i += 32) {
            uint8_t v = wrow[i >> 1];
            float w = (float)(((i & 1) ? (v >> 4) : (v & 15)) - 8);
            sum += xs[i] * w;
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, off);
        if (lane == 0) y[(size_t)s * O + o0] = sum * scales[o0];
    }
}

__global__ static void quant_gemv_i8(float *y, const float *x, const int8_t *weights,
                                    const float *scales, int S, int I, int O) {
    const int warps = blockDim.x >> 5;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int o0 = (int)blockIdx.x * warps + warp;
    extern __shared__ float shx[];

    const int nxi = S * I;
    for (int i = threadIdx.x; i < nxi; i += blockDim.x) shx[i] = x[i];
    __syncthreads();

    if (o0 >= O) return;
    const int8_t *wrow = weights + (size_t)o0 * (size_t)I;
    for (int s = 0; s < S; s++) {
        const float *xs = shx + s * I;
        float sum = 0.0f;
        for (int i = lane; i < I; i += 32) sum += xs[i] * (float)wrow[i];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, off);
        if (lane == 0) y[(size_t)s * O + o0] = sum * scales[o0];
    }
}

/*
 * Fused gate+up for int4: same x, two weight matrices → silu(g)*u in one pass.
 * Output mid[S,I_ffn] where I_ffn == O of gate/up. Cuts x traffic in half vs 2 GEMVs.
 */
__global__ static void quant_gate_up_silu_i4(float *mid, const float *x,
                                            const uint8_t *wg, const float *sg,
                                            const uint8_t *wu, const float *su,
                                            int S, int D, int Iffn) {
    const int warps = blockDim.x >> 5;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int o0 = (int)blockIdx.x * warps + warp;
    extern __shared__ float shx[];

    const int nxi = S * D;
    for (int i = threadIdx.x; i < nxi; i += blockDim.x) shx[i] = x[i];
    __syncthreads();
    if (o0 >= Iffn) return;

    const int rb = (D + 1) / 2;
    const uint8_t *rg = wg + (size_t)o0 * (size_t)rb;
    const uint8_t *ru = wu + (size_t)o0 * (size_t)rb;
    for (int s = 0; s < S; s++) {
        const float *xs = shx + s * D;
        float sg_acc = 0.0f, su_acc = 0.0f;
        for (int i = lane; i < D; i += 32) {
            uint8_t vg = rg[i >> 1], vu = ru[i >> 1];
            float wg_ = (float)(((i & 1) ? (vg >> 4) : (vg & 15)) - 8);
            float wu_ = (float)(((i & 1) ? (vu >> 4) : (vu & 15)) - 8);
            float xv = xs[i];
            sg_acc += xv * wg_;
            su_acc += xv * wu_;
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            sg_acc += __shfl_down_sync(0xffffffffu, sg_acc, off);
            su_acc += __shfl_down_sync(0xffffffffu, su_acc, off);
        }
        if (lane == 0) {
            float g = sg_acc * sg[o0];
            float u = su_acc * su[o0];
            mid[(size_t)s * Iffn + o0] = (g / (1.0f + expf(-g))) * u;
        }
    }
}

__global__ static void quant_gate_up_silu_i8(float *mid, const float *x,
                                            const int8_t *wg, const float *sg,
                                            const int8_t *wu, const float *su,
                                            int S, int D, int Iffn) {
    const int warps = blockDim.x >> 5;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int o0 = (int)blockIdx.x * warps + warp;
    extern __shared__ float shx[];

    const int nxi = S * D;
    for (int i = threadIdx.x; i < nxi; i += blockDim.x) shx[i] = x[i];
    __syncthreads();
    if (o0 >= Iffn) return;

    const int8_t *rg = wg + (size_t)o0 * (size_t)D;
    const int8_t *ru = wu + (size_t)o0 * (size_t)D;
    for (int s = 0; s < S; s++) {
        const float *xs = shx + s * D;
        float sg_acc = 0.0f, su_acc = 0.0f;
        for (int i = lane; i < D; i += 32) {
            float xv = xs[i];
            sg_acc += xv * (float)rg[i];
            su_acc += xv * (float)ru[i];
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            sg_acc += __shfl_down_sync(0xffffffffu, sg_acc, off);
            su_acc += __shfl_down_sync(0xffffffffu, su_acc, off);
        }
        if (lane == 0) {
            float g = sg_acc * sg[o0];
            float u = su_acc * su[o0];
            mid[(size_t)s * Iffn + o0] = (g / (1.0f + expf(-g))) * u;
        }
    }
}

__global__ static void silu_mul_kernel(float *g, const float *u, int n) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        float v = g[i];
        g[i] = (v / (1.0f + expf(-v))) * u[i];
    }
}

__global__ static void axpy_kernel(float *y, const float *x, float a, int n) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) y[i] += a * x[i];
}

__global__ static void zero_kernel(float *y, int n) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) y[i] = 0.0f;
}

static int reserve(float **ptr, size_t *cap, size_t bytes) {
    if (*cap >= bytes) return 1;
    if (*ptr) cudaFree(*ptr);
    *ptr = nullptr;
    *cap = 0;
    if (!cuda_ok(cudaMalloc(ptr, bytes), "scratch allocation")) return 0;
    *cap = bytes;
    return 1;
}

/* Prefer fast GEMV when shared mem fits. sm_86 can do ~100 KB dynamic smem;
 * dense down uses I=16384 → 64 KB for S=1. Cap 96 KB. */
static int launch_fast_gemv(float *dy, const float *dx, ColiCudaTensor *t,
                            int S, int I, int O, cudaStream_t stream) {
    size_t shmem = (size_t)S * (size_t)I * sizeof(float);
    if (shmem == 0 || shmem > 96 * 1024) return 0;
    const int threads = 256;                     /* 8 warps */
    const int warps = threads / 32;
    const int blocks = (O + warps - 1) / warps;
    if (t->fmt == 2) {
        if (shmem > 48 * 1024)
            cudaFuncSetAttribute(quant_gemv_i4, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);
        quant_gemv_i4<<<blocks, threads, shmem, stream>>>(
            dy, dx, static_cast<const uint8_t *>(t->weights), t->scales, S, I, O);
        return cuda_ok(cudaGetLastError(), "gemv_i4 launch");
    }
    if (t->fmt == 1) {
        if (shmem > 48 * 1024)
            cudaFuncSetAttribute(quant_gemv_i8, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);
        quant_gemv_i8<<<blocks, threads, shmem, stream>>>(
            dy, dx, static_cast<const int8_t *>(t->weights), t->scales, S, I, O);
        return cuda_ok(cudaGetLastError(), "gemv_i8 launch");
    }
    return 0;
}

static int launch_quant_matmul(float *dy, const float *dx, ColiCudaTensor *t,
                               int S, int I, int O, cudaStream_t stream) {
    if (t->I != I || t->O != O) return 0;
    if (launch_fast_gemv(dy, dx, t, S, I, O, stream)) return 1;
    size_t rb = row_bytes(t->fmt, I);
    if (!rb) return 0;
    dim3 grid((unsigned)O, (unsigned)S);
    quant_matmul<<<grid, 256, 0, stream>>>(dy, dx, t->weights, t->scales, t->fmt, S, I, O, rb);
    return cuda_ok(cudaGetLastError(), "matmul launch");
}

static int launch_gate_up_silu(float *mid, const float *dx,
                               ColiCudaTensor *tg, ColiCudaTensor *tu,
                               int S, int D, int Iffn, cudaStream_t stream) {
    if (tg->fmt != tu->fmt) return 0;
    if (tg->I != D || tg->O != Iffn || tu->I != D || tu->O != Iffn) return 0;
    size_t shmem = (size_t)S * (size_t)D * sizeof(float);
    if (shmem == 0 || shmem > 96 * 1024) return 0;
    const int threads = 256;
    const int warps = threads / 32;
    const int blocks = (Iffn + warps - 1) / warps;
    if (tg->fmt == 2) {
        if (shmem > 48 * 1024)
            cudaFuncSetAttribute(quant_gate_up_silu_i4, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);
        quant_gate_up_silu_i4<<<blocks, threads, shmem, stream>>>(
            mid, dx,
            static_cast<const uint8_t *>(tg->weights), tg->scales,
            static_cast<const uint8_t *>(tu->weights), tu->scales,
            S, D, Iffn);
        return cuda_ok(cudaGetLastError(), "gate_up_i4 launch");
    }
    if (tg->fmt == 1) {
        if (shmem > 48 * 1024)
            cudaFuncSetAttribute(quant_gate_up_silu_i8, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);
        quant_gate_up_silu_i8<<<blocks, threads, shmem, stream>>>(
            mid, dx,
            static_cast<const int8_t *>(tg->weights), tg->scales,
            static_cast<const int8_t *>(tu->weights), tu->scales,
            S, D, Iffn);
        return cuda_ok(cudaGetLastError(), "gate_up_i8 launch");
    }
    return 0;
}

extern "C" int coli_cuda_init(const int *devices, int count) {
    int available = 0;
    if (!devices || count < 1 || count > COLI_CUDA_MAX_DEVICES) return 0;
    if (!cuda_ok(cudaGetDeviceCount(&available), "device discovery")) return 0;
    g_nctx = 0;
    for (int i = 0; i < count; i++) {
        int device = devices[i];
        if (device < 0 || device >= available) {
            std::fprintf(stderr, "[CUDA] invalid device %d (available: 0..%d)\n", device, available - 1);
            g_nctx = 0;
            return 0;
        }
        if (find_ctx(device)) {
            std::fprintf(stderr, "[CUDA] duplicate device %d\n", device);
            g_nctx = 0;
            return 0;
        }
        DeviceContext *ctx = &g_ctx[g_nctx];
        *ctx = {};
        ctx->device = device;
        if (!select_ctx(ctx)) { g_nctx = 0; return 0; }
        if (!cuda_ok(cudaStreamCreateWithFlags(&ctx->stream, cudaStreamNonBlocking), "stream")) {
            g_nctx = 0; return 0;
        }
        cudaDeviceProp prop{};
        if (!cuda_ok(cudaGetDeviceProperties(&prop, device), "device properties")) { g_nctx = 0; return 0; }
        g_nctx++;
        std::fprintf(stderr, "[CUDA] device %d: %s, %.1f GB VRAM, sm_%d%d (fast GEMV+SwiGLU)\n",
                     device, prop.name, prop.totalGlobalMem / 1e9, prop.major, prop.minor);
    }
    return 1;
}

extern "C" void coli_cuda_shutdown(void) {
    for (int i = 0; i < g_nctx; i++) {
        DeviceContext *ctx = &g_ctx[i];
        if (!select_ctx(ctx)) continue;
        if (ctx->stream) cudaStreamDestroy(ctx->stream);
        if (ctx->x) cudaFree(ctx->x);
        if (ctx->y) cudaFree(ctx->y);
        if (ctx->ma) cudaFree(ctx->ma);
        if (ctx->mb) cudaFree(ctx->mb);
        if (ctx->yacc) cudaFree(ctx->yacc);
        ctx->x = ctx->y = ctx->ma = ctx->mb = ctx->yacc = nullptr;
        ctx->x_cap = ctx->y_cap = ctx->ma_cap = ctx->mb_cap = ctx->yacc_cap = 0;
        ctx->stream = nullptr;
        ctx->sticky_valid = 0;
        ctx->moe_active = 0;
    }
    g_nctx = 0;
}

extern "C" int coli_cuda_device_count(void) { return g_nctx; }

extern "C" int coli_cuda_device_at(int index) {
    return index >= 0 && index < g_nctx ? g_ctx[index].device : -1;
}

extern "C" int coli_cuda_mem_info(int device, size_t *free_bytes, size_t *total_bytes) {
    DeviceContext *ctx = find_ctx(device);
    if (!free_bytes || !total_bytes || !select_ctx(ctx)) return 0;
    return cuda_ok(cudaMemGetInfo(free_bytes, total_bytes), "memory info");
}

extern "C" void coli_cuda_stats(int device, size_t *tensor_count, size_t *tensor_bytes) {
    size_t count = 0, bytes = 0;
    for (int i = 0; i < g_nctx; i++) if (device < 0 || g_ctx[i].device == device) {
        count += g_ctx[i].tensor_count;
        bytes += g_ctx[i].tensor_bytes;
    }
    if (tensor_count) *tensor_count = count;
    if (tensor_bytes) *tensor_bytes = bytes;
}

extern "C" int coli_cuda_tensor_upload(ColiCudaTensor **tensor,
                                        const void *weights, const float *scales,
                                        int fmt, int I, int O, int device) {
    DeviceContext *ctx = find_ctx(device);
    if (!tensor || I < 1 || O < 1 || !select_ctx(ctx)) return 0;
    if (*tensor) {
        ColiCudaTensor *t = *tensor;
        return t->fmt == fmt && t->I == I && t->O == O && t->device == device;
    }
    if (!weights) return 0;
    size_t rb = row_bytes(fmt, I);
    if (!rb || (fmt && !scales)) return 0;
    ColiCudaTensor *t = static_cast<ColiCudaTensor *>(std::calloc(1, sizeof(*t)));
    if (!t) return 0;
    t->fmt = fmt; t->I = I; t->O = O; t->device = device; t->weight_bytes = rb * (size_t)O;
    if (!cuda_ok(cudaMalloc(&t->weights, t->weight_bytes), "tensor allocation") ||
        !cuda_ok(cudaMemcpy(t->weights, weights, t->weight_bytes, cudaMemcpyHostToDevice), "tensor upload")) {
        coli_cuda_tensor_free(t);
        return 0;
    }
    if (fmt) {
        if (!cuda_ok(cudaMalloc(&t->scales, (size_t)O * sizeof(float)), "scale allocation") ||
            !cuda_ok(cudaMemcpy(t->scales, scales, (size_t)O * sizeof(float), cudaMemcpyHostToDevice), "scale upload")) {
            coli_cuda_tensor_free(t);
            return 0;
        }
    }
    t->tracked = 1;
    ctx->tensor_count++;
    ctx->tensor_bytes += t->weight_bytes + (fmt ? (size_t)O * sizeof(float) : 0);
    *tensor = t;
    return 1;
}

extern "C" int coli_cuda_matmul(ColiCudaTensor **tensor,
                                 float *y, const float *x,
                                 const void *weights, const float *scales,
                                 int fmt, int S, int I, int O, int device) {
    if (S < 1 || !coli_cuda_tensor_upload(tensor, weights, scales, fmt, I, O, device)) return 0;
    ColiCudaTensor *t = *tensor;
    DeviceContext *ctx = find_ctx(t->device);
    if (!select_ctx(ctx)) return 0;
    size_t xb = (size_t)S * I * sizeof(float), yb = (size_t)S * O * sizeof(float);
    if (!reserve(&ctx->x, &ctx->x_cap, xb) || !reserve(&ctx->y, &ctx->y_cap, yb)) return 0;
    double tc0 = now_s();
    if (!cuda_ok(cudaMemcpyAsync(ctx->x, x, xb, cudaMemcpyHostToDevice, ctx->stream), "input upload")) return 0;
    if (!launch_quant_matmul(ctx->y, ctx->x, t, S, I, O, ctx->stream)) return 0;
    if (!cuda_ok(cudaMemcpyAsync(y, ctx->y, yb, cudaMemcpyDeviceToHost, ctx->stream), "output download")) return 0;
    if (!cuda_ok(cudaStreamSynchronize(ctx->stream), "matmul sync")) return 0;
    double tc1 = now_s();
    /* Full API call wall includes compute; copy timer stays best-effort (async). */
    g_copy_sec += 0; /* H2D/D2H overlapped; not isolated under async */
    (void)tc0; (void)tc1;
    ctx->sticky_valid = 0;   /* x buffer repurposed */
    return 1;
}

extern "C" void coli_cuda_x_invalidate(int device) {
    DeviceContext *ctx = find_ctx(device);
    if (ctx) { ctx->sticky_valid = 0; ctx->moe_active = 0; }
}

/* ---- MoE layer accumulate: 1× H2D x, N× (swiglu + axpy) no sync, 1× D2H ---- */

extern "C" int coli_cuda_moe_begin(int device, const float *x, int S, int D) {
    if (S < 1 || D < 1 || !x) return 0;
    DeviceContext *ctx = find_ctx(device);
    if (!select_ctx(ctx)) return 0;
    size_t xb = (size_t)S * D * sizeof(float);
    size_t yb = xb;
    size_t mb = (size_t)S * 2048 * sizeof(float);
    if (!reserve(&ctx->x, &ctx->x_cap, xb) ||
        !reserve(&ctx->y, &ctx->y_cap, yb) ||
        !reserve(&ctx->yacc, &ctx->yacc_cap, yb) ||
        !reserve(&ctx->ma, &ctx->ma_cap, mb) ||
        !reserve(&ctx->mb, &ctx->mb_cap, mb)) return 0;
    if (!cuda_ok(cudaMemcpyAsync(ctx->x, x, xb, cudaMemcpyHostToDevice, ctx->stream), "moe x H2D"))
        return 0;
    {
        int n = S * D;
        int block = 256;
        int grid = (n + block - 1) / block;
        zero_kernel<<<grid, block, 0, ctx->stream>>>(ctx->yacc, n);
        if (!cuda_ok(cudaGetLastError(), "moe zero")) return 0;
    }
    ctx->sticky_S = S; ctx->sticky_D = D; ctx->sticky_valid = 1;
    ctx->moe_active = 1; ctx->moe_S = S; ctx->moe_D = D;
    return 1;
}

extern "C" int coli_cuda_moe_acc(ColiCudaTensor **gate, ColiCudaTensor **up, ColiCudaTensor **down,
                                  const void *gw, const float *gs, int gfmt,
                                  const void *uw, const float *us, int ufmt,
                                  const void *dw, const float *ds, int dfmt,
                                  float weight, int S, int D, int I, int device) {
    /* Single-stream pipeline: multi-stream+atomic was measured slower on 3060
     * (atomic contention + launch overhead). Keep one stream, batch sync at end. */
    DeviceContext *ctx = find_ctx(device);
    if (!ctx || !ctx->moe_active || ctx->moe_S != S || ctx->moe_D != D) return 0;
    if (!select_ctx(ctx)) return 0;
    if (!coli_cuda_tensor_upload(gate, gw, gs, gfmt, D, I, device)) return 0;
    if (!coli_cuda_tensor_upload(up,   uw, us, ufmt, D, I, device)) return 0;
    if (!coli_cuda_tensor_upload(down, dw, ds, dfmt, I, D, device)) return 0;
    ColiCudaTensor *tg = *gate, *tu = *up, *td = *down;
    if (tg->device != device || tu->device != device || td->device != device) return 0;
    size_t mb = (size_t)S * I * sizeof(float);
    if (!reserve(&ctx->ma, &ctx->ma_cap, mb) || !reserve(&ctx->mb, &ctx->mb_cap, mb)) return 0;

    cudaStream_t st = ctx->stream;
    int fused_gu = launch_gate_up_silu(ctx->ma, ctx->x, tg, tu, S, D, I, st);
    if (!fused_gu) {
        if (!launch_quant_matmul(ctx->ma, ctx->x, tg, S, D, I, st)) return 0;
        if (!launch_quant_matmul(ctx->mb, ctx->x, tu, S, D, I, st)) return 0;
        int n = S * I;
        int block = 256;
        int grid = (n + block - 1) / block;
        silu_mul_kernel<<<grid, block, 0, st>>>(ctx->ma, ctx->mb, n);
        if (!cuda_ok(cudaGetLastError(), "moe silu")) return 0;
    }
    if (!launch_quant_matmul(ctx->y, ctx->ma, td, S, I, D, st)) return 0;
    {
        int n = S * D;
        int block = 256;
        int grid = (n + block - 1) / block;
        axpy_kernel<<<grid, block, 0, st>>>(ctx->yacc, ctx->y, weight, n);
        if (!cuda_ok(cudaGetLastError(), "moe axpy")) return 0;
    }
    return 1;
}

extern "C" int coli_cuda_moe_end(float *y_host, int S, int D, int device) {
    DeviceContext *ctx = find_ctx(device);
    if (!ctx || !ctx->moe_active) return 0;
    if (!select_ctx(ctx)) return 0;
    if (ctx->moe_S != S || ctx->moe_D != D) { ctx->moe_active = 0; return 0; }
    size_t yb = (size_t)S * D * sizeof(float);
    if (!cuda_ok(cudaMemcpyAsync(y_host, ctx->yacc, yb, cudaMemcpyDeviceToHost, ctx->stream),
                 "moe y D2H")) {
        ctx->moe_active = 0;
        return 0;
    }
    if (!cuda_ok(cudaStreamSynchronize(ctx->stream), "moe sync")) {
        ctx->moe_active = 0;
        return 0;
    }
    ctx->moe_active = 0;
    return 1;
}

extern "C" int coli_cuda_swiglu(ColiCudaTensor **gate, ColiCudaTensor **up, ColiCudaTensor **down,
                                 float *y, const float *x,
                                 const void *gw, const float *gs, int gfmt,
                                 const void *uw, const float *us, int ufmt,
                                 const void *dw, const float *ds, int dfmt,
                                 int S, int D, int I, int device, int flags) {
    if (S < 1 || D < 1 || I < 1) return 0;
    if (!coli_cuda_tensor_upload(gate, gw, gs, gfmt, D, I, device)) return 0;
    if (!coli_cuda_tensor_upload(up,   uw, us, ufmt, D, I, device)) return 0;
    if (!coli_cuda_tensor_upload(down, dw, ds, dfmt, I, D, device)) return 0;
    ColiCudaTensor *tg = *gate, *tu = *up, *td = *down;
    if (tg->device != device || tu->device != device || td->device != device) return 0;
    DeviceContext *ctx = find_ctx(device);
    if (!select_ctx(ctx)) return 0;

    size_t xb = (size_t)S * D * sizeof(float);
    size_t yb = (size_t)S * D * sizeof(float);
    size_t mb = (size_t)S * I * sizeof(float);
    if (!reserve(&ctx->x,  &ctx->x_cap,  xb) ||
        !reserve(&ctx->y,  &ctx->y_cap,  yb) ||
        !reserve(&ctx->ma, &ctx->ma_cap, mb) ||
        !reserve(&ctx->mb, &ctx->mb_cap, mb)) return 0;

    int reuse = (flags & COLI_CUDA_SWIGLU_REUSE_X) &&
                ctx->sticky_valid && ctx->sticky_S == S && ctx->sticky_D == D;

    double tc0 = now_s();
    if (!reuse) {
        if (!cuda_ok(cudaMemcpyAsync(ctx->x, x, xb, cudaMemcpyHostToDevice, ctx->stream),
                     "swiglu x H2D")) return 0;
        ctx->sticky_S = S; ctx->sticky_D = D; ctx->sticky_valid = 1;
    }
    (void)tc0;

    /* Prefer fused gate+up+silu (int4/int8); else 2 GEMV + silu_mul. */
    int fused_gu = launch_gate_up_silu(ctx->ma, ctx->x, tg, tu, S, D, I, ctx->stream);
    if (!fused_gu) {
        if (!launch_quant_matmul(ctx->ma, ctx->x, tg, S, D, I, ctx->stream)) return 0;
        if (!launch_quant_matmul(ctx->mb, ctx->x, tu, S, D, I, ctx->stream)) return 0;
        int n = S * I;
        int block = 256;
        int grid = (n + block - 1) / block;
        silu_mul_kernel<<<grid, block, 0, ctx->stream>>>(ctx->ma, ctx->mb, n);
        if (!cuda_ok(cudaGetLastError(), "silu_mul launch")) return 0;
    }
    if (!launch_quant_matmul(ctx->y, ctx->ma, td, S, I, D, ctx->stream)) return 0;
    if (!cuda_ok(cudaMemcpyAsync(y, ctx->y, yb, cudaMemcpyDeviceToHost, ctx->stream), "swiglu y D2H"))
        return 0;
    if (!cuda_ok(cudaStreamSynchronize(ctx->stream), "swiglu sync")) return 0;
    return 1;
}

extern "C" void coli_cuda_tensor_free(ColiCudaTensor *tensor) {
    if (!tensor) return;
    DeviceContext *ctx = find_ctx(tensor->device);
    if (ctx) select_ctx(ctx);
    if (tensor->tracked && ctx) {
        size_t bytes = tensor->weight_bytes + (tensor->fmt ? (size_t)tensor->O * sizeof(float) : 0);
        if (ctx->tensor_count) ctx->tensor_count--;
        if (ctx->tensor_bytes >= bytes) ctx->tensor_bytes -= bytes;
    }
    if (tensor->weights) cudaFree(tensor->weights);
    if (tensor->scales) cudaFree(tensor->scales);
    std::free(tensor);
}

extern "C" size_t coli_cuda_tensor_bytes(const ColiCudaTensor *tensor) {
    return tensor ? tensor->weight_bytes + (tensor->fmt ? (size_t)tensor->O * sizeof(float) : 0) : 0;
}

extern "C" int coli_cuda_tensor_device(const ColiCudaTensor *tensor) {
    return tensor ? tensor->device : -1;
}
