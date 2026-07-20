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
        /* Byte-strided: one weight load → two nibble MACs (half global weight traffic). */
        for (int b = lane; b < rb; b += 32) {
            uint8_t v = wrow[b];
            int i0 = b << 1;
            sum += xs[i0] * (float)((int)(v & 15) - 8);
            if (i0 + 1 < I)
                sum += xs[i0 + 1] * (float)((int)(v >> 4) - 8);
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
        /* Byte-strided dual nibble (same as quant_gemv_i4). */
        for (int b = lane; b < rb; b += 32) {
            uint8_t vg = rg[b], vu = ru[b];
            int i0 = b << 1;
            float x0 = xs[i0];
            sg_acc += x0 * (float)((int)(vg & 15) - 8);
            su_acc += x0 * (float)((int)(vu & 15) - 8);
            if (i0 + 1 < D) {
                float x1 = xs[i0 + 1];
                sg_acc += x1 * (float)((int)(vg >> 4) - 8);
                su_acc += x1 * (float)((int)(vu >> 4) - 8);
            }
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
    /* Sync policy: SPIN by default. Measured 2026-07-19 on the 311B run: under
     * the expert-pipeline CPU load a yielding cudaStreamSynchronize deschedules
     * the forward thread for ~5-8 ms per decode layer, eating the whole
     * CUDA_ATTN gain (8.6 s instead of ~1 s per 24 tok). Spinning keeps the
     * sync at ~40 us. COLI_CUDA_SYNC=yield|block restores the old behavior. */
    {
        const char *sync = std::getenv("COLI_CUDA_SYNC");
        unsigned flags = cudaDeviceScheduleSpin;
        if (sync && !std::strcmp(sync, "yield")) flags = cudaDeviceScheduleYield;
        else if (sync && !std::strcmp(sync, "block")) flags = cudaDeviceScheduleBlockingSync;
        cudaSetDeviceFlags(flags);   /* best-effort: before context creation */
    }
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

extern "C" int coli_cuda_matmul_ex(ColiCudaTensor **tensor,
                                    float *y, float **y_dev, const float *x,
                                    const void *weights, const float *scales,
                                    int fmt, int S, int I, int O, int device, int flags) {
    if (S < 1 || !coli_cuda_tensor_upload(tensor, weights, scales, fmt, I, O, device)) return 0;
    ColiCudaTensor *t = *tensor;
    DeviceContext *ctx = find_ctx(t->device);
    if (!select_ctx(ctx)) return 0;
    size_t xb = (size_t)S * I * sizeof(float), yb = (size_t)S * O * sizeof(float);
    const float *dx = x;
    if (!(flags & COLI_CUDA_MM_X_DEV)) {
        if (!reserve(&ctx->x, &ctx->x_cap, xb)) return 0;
        if (!cuda_ok(cudaMemcpyAsync(ctx->x, x, xb, cudaMemcpyHostToDevice, ctx->stream), "input upload")) return 0;
        dx = ctx->x;
        ctx->sticky_valid = 0;   /* x buffer repurposed */
    }
    if (!reserve(&ctx->y, &ctx->y_cap, yb)) return 0;
    if (!launch_quant_matmul(ctx->y, dx, t, S, I, O, ctx->stream)) return 0;
    if (flags & COLI_CUDA_MM_Y_DEV) {
        if (y_dev) *y_dev = ctx->y;
        return 1;   /* no D2H, no sync: caller chains more device work */
    }
    if (!cuda_ok(cudaMemcpyAsync(y, ctx->y, yb, cudaMemcpyDeviceToHost, ctx->stream), "output download")) return 0;
    if (!cuda_ok(cudaStreamSynchronize(ctx->stream), "matmul sync")) return 0;
    return 1;
}

extern "C" int coli_cuda_matmul(ColiCudaTensor **tensor,
                                 float *y, const float *x,
                                 const void *weights, const float *scales,
                                 int fmt, int S, int I, int O, int device) {
    return coli_cuda_matmul_ex(tensor, y, nullptr, x, weights, scales, fmt, S, I, O, device, 0);
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

/* ================= fused decode attention (S=1), KV resident on device ====
 * One block per query head. Inside a block:
 *   1) load q/k, apply partial RoPE (pairs j, j+rd/2), scale v by v_scale
 *   2) block with h%group==0 appends the new K/V row to the device cache
 *   3) per-warp online softmax over the causal window [lo,pos] (t strided by
 *      warp; t==pos uses the smem copy, no cross-block dependency)
 *   4) cross-warp merge -> ctx[h*vd]
 * FP order differs from the CPU 2-pass reference (online softmax + different
 * dot reduction): the whole path is opt-in (CUDA_ATTN=1), same contract as
 * the existing CUDA dense/expert tiers (chat-speed, not oracle-exact). */

#define COLI_ATTN_MAX_LAYERS 130
#define ATTN_THREADS 128                 /* 4 warps */
#define ATTN_HD_MAX 192
#define ATTN_VD_MAX 128

typedef struct {
    int allocated;
    int n_layer, n_heads, window, max_t;
    int rows[COLI_ATTN_MAX_LAYERS];      /* physical KV rows per layer */
    int kvh[COLI_ATTN_MAX_LAYERS];
    int hd[COLI_ATTN_MAX_LAYERS];
    int vd[COLI_ATTN_MAX_LAYERS];
    int rd[COLI_ATTN_MAX_LAYERS];        /* rope dims (even) */
    int swa[COLI_ATTN_MAX_LAYERS];
    int ring[COLI_ATTN_MAX_LAYERS];      /* swa && window>0 && rows<max_t */
    int has_sink[COLI_ATTN_MAX_LAYERS];
    float *K[COLI_ATTN_MAX_LAYERS];      /* device */
    float *V[COLI_ATTN_MAX_LAYERS];      /* device */
    float *sink;                         /* device [n_layer][n_heads] packed */
    float *rope_cos[2], *rope_sin[2];    /* device, per type 0=full 1=swa */
    int rope_half[2], rope_rows[2];
    float *Kh[COLI_ATTN_MAX_LAYERS];     /* host mirror bases (nullable) */
    float *Vh[COLI_ATTN_MAX_LAYERS];
    float *ctx;                          /* device [n_heads*max_vd] */
    size_t ctx_cap;
} AttnState;

static AttnState g_attn[COLI_CUDA_MAX_DEVICES];

static AttnState *attn_for(int device) {
    for (int i = 0; i < g_nctx; i++) if (g_ctx[i].device == device) return &g_attn[i];
    return nullptr;
}

__global__ static void attn_decode_kernel(
    const float *__restrict__ qkv,       /* [H*hd + kvh*hd + kvh*vd] */
    float *__restrict__ K, float *__restrict__ V,
    const float *__restrict__ cosr, const float *__restrict__ sinr,   /* [half] */
    const float *__restrict__ sink,      /* [H] or nullptr */
    float *__restrict__ ctx,             /* [H*vd] */
    int pos, int lo, int nt,
    int kvh, int group, int hd, int vd, int rd,
    int rows, int ring,
    float scale, float v_scale)
{
    const int h = blockIdx.x;
    const int g = h / group;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int nwarps = blockDim.x >> 5;
    __shared__ float sq[ATTN_HD_MAX];
    __shared__ float sk[ATTN_HD_MAX];
    __shared__ float sv[ATTN_VD_MAX];
    __shared__ float sacc[ATTN_THREADS / 32][ATTN_VD_MAX];
    __shared__ float sm[ATTN_THREADS / 32], sl[ATTN_THREADS / 32];

    const int qs = (int)gridDim.x * hd;  /* H*hd */
    const int half = rd >> 1;
    const float *qsrc = qkv + (size_t)h * hd;
    const float *ksrc = qkv + (size_t)qs + (size_t)g * hd;
    const float *vsrc = qkv + (size_t)qs + (size_t)kvh * hd + (size_t)g * vd;

    for (int d = threadIdx.x; d < hd; d += blockDim.x) { sq[d] = qsrc[d]; sk[d] = ksrc[d]; }
    for (int d = threadIdx.x; d < vd; d += blockDim.x) sv[d] = vsrc[d] * v_scale;
    __syncthreads();
    if (threadIdx.x < (unsigned)half) {
        const int j = threadIdx.x;
        const float cs = cosr[j], sn = sinr[j];
        float a = sq[j], b = sq[half + j];
        sq[j] = a * cs - b * sn; sq[half + j] = b * cs + a * sn;
        a = sk[j]; b = sk[half + j];
        sk[j] = a * cs - b * sn; sk[half + j] = b * cs + a * sn;
    }
    __syncthreads();

    /* append the new row (exactly one block per kv group) */
    const int pphy = ring ? (pos % rows) : pos;
    if (h % group == 0) {
        float *kd = K + (size_t)pphy * ((size_t)kvh * hd) + (size_t)g * hd;
        float *vdst = V + (size_t)pphy * ((size_t)kvh * vd) + (size_t)g * vd;
        for (int d = threadIdx.x; d < hd; d += blockDim.x) kd[d] = sk[d];
        for (int d = threadIdx.x; d < vd; d += blockDim.x) vdst[d] = sv[d];
    }

    /* per-warp online softmax over [lo,pos], t strided by warp */
    float m = -1e30f, l = 0.f;
    if (warp == 0 && sink) { m = sink[h]; l = 1.0f; }
    for (int d = lane; d < vd; d += 32) sacc[warp][d] = 0.f;
    __syncwarp();
    const float *Kb = K + (size_t)g * hd;
    const float *Vb = V + (size_t)g * vd;
    const size_t kstride = (size_t)kvh * hd, vstride = (size_t)kvh * vd;
    for (int j = warp; j < nt; j += nwarps) {
        const int t = lo + j;
        const float *kr, *vr;
        if (t == pos) { kr = sk; vr = sv; }
        else {
            const int tp = ring ? (t % rows) : t;
            kr = Kb + (size_t)tp * kstride;
            vr = Vb + (size_t)tp * vstride;
        }
        float sum = 0.f;
        for (int d = lane; d < hd; d += 32) sum += sq[d] * kr[d];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, off);
        const float s = sum * scale;
        const float mn = fmaxf(m, s);
        const float corr = expf(m - mn);
        const float wgt = expf(s - mn);
        l = l * corr + wgt;
        m = mn;
        for (int d = lane; d < vd; d += 32)
            sacc[warp][d] = sacc[warp][d] * corr + wgt * vr[d];
    }
    if (lane == 0) { sm[warp] = m; sl[warp] = l; }
    __syncthreads();

    /* cross-warp merge */
    float M = sm[0];
    for (int w = 1; w < nwarps; w++) M = fmaxf(M, sm[w]);
    float L = 0.f;
    for (int w = 0; w < nwarps; w++) L += sl[w] * expf(sm[w] - M);
    const float inv = 1.f / L;
    for (int d = threadIdx.x; d < vd; d += blockDim.x) {
        float a = 0.f;
        for (int w = 0; w < nwarps; w++) a += sacc[w][d] * expf(sm[w] - M);
        ctx[(size_t)h * vd + d] = a * inv;
    }
}

static void attn_free(AttnState *A) {
    if (!A) return;
    for (int i = 0; i < A->n_layer; i++) {
        if (A->K[i]) cudaFree(A->K[i]);
        if (A->V[i]) cudaFree(A->V[i]);
        A->K[i] = A->V[i] = nullptr;
    }
    for (int t = 0; t < 2; t++) {
        if (A->rope_cos[t]) cudaFree(A->rope_cos[t]);
        if (A->rope_sin[t]) cudaFree(A->rope_sin[t]);
        A->rope_cos[t] = A->rope_sin[t] = nullptr;
        A->rope_half[t] = A->rope_rows[t] = 0;
    }
    if (A->sink) { cudaFree(A->sink); A->sink = nullptr; }
    if (A->ctx)  { cudaFree(A->ctx);  A->ctx = nullptr; }
    A->ctx_cap = 0;
    A->allocated = 0;
}

extern "C" int coli_cuda_attn_kv_alloc(int device, int n_layer,
                                        const int *rows, const int *kvh,
                                        const int *hd, const int *vd, const int *swa,
                                        int window, int max_t, int n_heads) {
    DeviceContext *ctx = find_ctx(device);
    AttnState *A = attn_for(device);
    if (!ctx || !A || !select_ctx(ctx)) return 0;
    if (n_layer < 1 || n_layer > COLI_ATTN_MAX_LAYERS || n_heads < 1) return 0;
    attn_free(A);
    A->n_layer = n_layer; A->n_heads = n_heads;
    A->window = window; A->max_t = max_t;
    int max_vd = 0;
    size_t total = 0;
    for (int i = 0; i < n_layer; i++) {
        A->rows[i] = rows[i]; A->kvh[i] = kvh[i];
        A->hd[i] = hd[i]; A->vd[i] = vd[i]; A->swa[i] = swa[i];
        A->ring[i] = (swa[i] && window > 0 && rows[i] < max_t) ? 1 : 0;
        A->has_sink[i] = 0; A->Kh[i] = nullptr; A->Vh[i] = nullptr;
        if (hd[i] > ATTN_HD_MAX || vd[i] > ATTN_VD_MAX) { attn_free(A); return 0; }
        if (vd[i] > max_vd) max_vd = vd[i];
        size_t kb = (size_t)rows[i] * kvh[i] * hd[i] * sizeof(float);
        size_t vb = (size_t)rows[i] * kvh[i] * vd[i] * sizeof(float);
        if (!cuda_ok(cudaMalloc(&A->K[i], kb), "attn K alloc") ||
            !cuda_ok(cudaMalloc(&A->V[i], vb), "attn V alloc")) { attn_free(A); return 0; }
        total += kb + vb;
    }
    if (!cuda_ok(cudaMalloc(&A->sink, (size_t)n_layer * n_heads * sizeof(float)), "attn sink alloc")) {
        attn_free(A); return 0;
    }
    A->ctx_cap = (size_t)n_heads * max_vd * sizeof(float);
    if (!cuda_ok(cudaMalloc(&A->ctx, A->ctx_cap), "attn ctx alloc")) { attn_free(A); return 0; }
    A->allocated = 1;
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    std::fprintf(stderr, "[CUDA] attn KV: %d layers, %.2f GB device (free %.2f GB)\n",
                 n_layer, total / 1e9, free_b / 1e9);
    return 1;
}

extern "C" int coli_cuda_attn_rope(int device, int type, const float *cos_t, const float *sin_t,
                                    int rows, int half) {
    DeviceContext *ctx = find_ctx(device);
    AttnState *A = attn_for(device);
    if (!ctx || !A || !A->allocated || type < 0 || type > 1 || !select_ctx(ctx)) return 0;
    if (!cos_t || !sin_t || rows < 1 || half < 1) return 0;
    size_t b = (size_t)rows * half * sizeof(float);
    if (A->rope_rows[type] < rows || A->rope_half[type] != half) {
        if (A->rope_cos[type]) cudaFree(A->rope_cos[type]);
        if (A->rope_sin[type]) cudaFree(A->rope_sin[type]);
        A->rope_cos[type] = A->rope_sin[type] = nullptr;
        A->rope_rows[type] = 0; A->rope_half[type] = half;
        if (!cuda_ok(cudaMalloc(&A->rope_cos[type], b), "rope cos alloc") ||
            !cuda_ok(cudaMalloc(&A->rope_sin[type], b), "rope sin alloc")) return 0;
        A->rope_rows[type] = rows;
    }
    if (!cuda_ok(cudaMemcpy(A->rope_cos[type], cos_t, b, cudaMemcpyHostToDevice), "rope cos upload")) return 0;
    if (!cuda_ok(cudaMemcpy(A->rope_sin[type], sin_t, b, cudaMemcpyHostToDevice), "rope sin upload")) return 0;
    return 1;
}

extern "C" int coli_cuda_attn_layer_cfg(int device, int layer, int rd, const float *sink, int n_heads) {
    DeviceContext *ctx = find_ctx(device);
    AttnState *A = attn_for(device);
    if (!ctx || !A || !A->allocated || !select_ctx(ctx)) return 0;
    if (layer < 0 || layer >= A->n_layer || rd < 0 || rd > ATTN_HD_MAX || n_heads != A->n_heads) return 0;
    A->rd[layer] = rd;
    A->has_sink[layer] = sink ? 1 : 0;
    if (sink && !cuda_ok(cudaMemcpy(A->sink + (size_t)layer * n_heads, sink,
                                    (size_t)n_heads * sizeof(float), cudaMemcpyHostToDevice), "sink upload"))
        return 0;
    return 1;
}

extern "C" int coli_cuda_attn_mirror(int device, int layer, float *K_host, float *V_host) {
    AttnState *A = attn_for(device);
    if (!A || !A->allocated || layer < 0 || layer >= A->n_layer) return 0;
    A->Kh[layer] = K_host; A->Vh[layer] = V_host;
    return 1;
}

extern "C" int coli_cuda_attn_upload_rows(int device, int layer, int r0, int r1,
                                           const float *K, const float *V) {
    DeviceContext *ctx = find_ctx(device);
    AttnState *A = attn_for(device);
    if (!ctx || !A || !A->allocated || !select_ctx(ctx)) return 0;
    if (layer < 0 || layer >= A->n_layer || r0 < 0 || r1 > A->rows[layer] || r1 <= r0) return 0;
    size_t koff = (size_t)r0 * A->kvh[layer] * A->hd[layer];
    size_t voff = (size_t)r0 * A->kvh[layer] * A->vd[layer];
    size_t kb = (size_t)(r1 - r0) * A->kvh[layer] * A->hd[layer] * sizeof(float);
    size_t vb = (size_t)(r1 - r0) * A->kvh[layer] * A->vd[layer] * sizeof(float);
    if (!cuda_ok(cudaMemcpyAsync(A->K[layer] + koff, K, kb, cudaMemcpyHostToDevice, ctx->stream),
                 "KV rows K upload")) return 0;
    if (!cuda_ok(cudaMemcpyAsync(A->V[layer] + voff, V, vb, cudaMemcpyHostToDevice, ctx->stream),
                 "KV rows V upload")) return 0;
    if (!cuda_ok(cudaStreamSynchronize(ctx->stream), "KV rows sync")) return 0;
    return 1;
}

extern "C" float *coli_cuda_attn_decode(int device, int layer, const float *qkv_dev,
                                         int pos, int kv_start, float v_scale) {
    DeviceContext *ctx = find_ctx(device);
    AttnState *A = attn_for(device);
    if (!ctx || !A || !A->allocated || !qkv_dev || !select_ctx(ctx)) return nullptr;
    if (layer < 0 || layer >= A->n_layer) return nullptr;
    const int swa = A->swa[layer], rows = A->rows[layer], ring = A->ring[layer];
    int lo = (swa && A->window > 0) ? pos - A->window + 1 : 0;
    if (lo < 0) lo = 0;
    if (lo < kv_start) lo = kv_start;
    if (pos < lo) return nullptr;
    const int nt = pos + 1 - lo;
    const int half = A->rd[layer] >> 1;
    const float *cosr = A->rope_cos[swa] ? A->rope_cos[swa] + (size_t)pos * half : nullptr;
    const float *sinr = A->rope_sin[swa] ? A->rope_sin[swa] + (size_t)pos * half : nullptr;
    if (half > 0 && (!cosr || !sinr)) return nullptr;
    const int hd = A->hd[layer], vd = A->vd[layer], kvh = A->kvh[layer];
    const int H = A->n_heads, group = H / kvh;
    const float scale = 1.f / sqrtf((float)hd);
    const float *sink = A->has_sink[layer] ? A->sink + (size_t)layer * H : nullptr;
    attn_decode_kernel<<<H, ATTN_THREADS, 0, ctx->stream>>>(
        qkv_dev, A->K[layer], A->V[layer], cosr, sinr, sink, A->ctx,
        pos, lo, nt, kvh, group, hd, vd, A->rd[layer], rows, ring, scale, v_scale);
    if (!cuda_ok(cudaGetLastError(), "attn decode launch")) return nullptr;
    /* mirror the appended row back to host K/V (async; flushed by the caller's
     * end-of-layer sync) so S>1 CPU paths keep reading warm host KV. */
    if (A->Kh[layer]) {
        const int pphy = ring ? (pos % rows) : pos;
        const size_t ko = (size_t)pphy * kvh * hd, vo = (size_t)pphy * kvh * vd;
        cudaMemcpyAsync(A->Kh[layer] + ko, A->K[layer] + ko, kvh * hd * sizeof(float),
                        cudaMemcpyDeviceToHost, ctx->stream);
        cudaMemcpyAsync(A->Vh[layer] + vo, A->V[layer] + vo, kvh * vd * sizeof(float),
                        cudaMemcpyDeviceToHost, ctx->stream);
    }
    return A->ctx;
}
