#ifndef COLIBRI_BACKEND_CUDA_H
#define COLIBRI_BACKEND_CUDA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define COLI_CUDA_MAX_DEVICES 16

/* Opaque, persistent device copy of one resident quantized tensor. */
typedef struct ColiCudaTensor ColiCudaTensor;

/* Devices are CUDA ordinals, not positions in the input list. */
int coli_cuda_init(const int *devices, int count);
void coli_cuda_shutdown(void);
int coli_cuda_device_count(void);
int coli_cuda_device_at(int index);
int coli_cuda_mem_info(int device, size_t *free_bytes, size_t *total_bytes);
/* device < 0 returns aggregate statistics for all configured devices. */
void coli_cuda_stats(int device, size_t *tensor_count, size_t *tensor_bytes);

/* Upload without executing, so capacity failures happen during model startup. */
int coli_cuda_tensor_upload(ColiCudaTensor **tensor,
                            const void *weights, const float *scales,
                            int fmt, int I, int O, int device);

/*
 * y[S,O] = x[S,I] @ W[O,I]^T.
 * fmt matches QT in glm.c: 0=f32, 1=int8, 2=int4, 3=int2.
 * The first successful call uploads W and its row scales; later calls reuse it.
 * Returns 1 on success and 0 when CUDA is not initialized or the format is invalid.
 */
int coli_cuda_matmul(ColiCudaTensor **tensor,
                     float *y, const float *x,
                     const void *weights, const float *scales,
                     int fmt, int S, int I, int O, int device);

/* flags for coli_cuda_swiglu */
#define COLI_CUDA_SWIGLU_REUSE_X  1  /* skip H2D if device-x sticky matches S,D */

/*
 * Fused expert / dense SwiGLU MLP on one device:
 *   y[S,D] = down( silu(gate(x)) * up(x) )
 * gate/up: W[I,D] (fmt row-major O=I, cols=D); down: W[D,I] (O=D, cols=I).
 * Fast path: int4/int8 GEMV with shared-x + fused gate+up+silu kernel.
 * REUSE_X: consecutive experts in decode share the same packed x (skip H2D).
 */
int coli_cuda_swiglu(ColiCudaTensor **gate, ColiCudaTensor **up, ColiCudaTensor **down,
                     float *y, const float *x,
                     const void *gw, const float *gs, int gfmt,
                     const void *uw, const float *us, int ufmt,
                     const void *dw, const float *ds, int dfmt,
                     int S, int D, int I, int device, int flags);

/*
 * Decode MoE accumulate path (S=1 typically): one H2D of x, N experts with
 * y_acc += w * swiglu(x) on device (no per-expert D2H/sync), one D2H of y_acc.
 * begin → acc* → end. end() is a no-op if begin never succeeded.
 */
int coli_cuda_moe_begin(int device, const float *x, int S, int D);
int coli_cuda_moe_acc(ColiCudaTensor **gate, ColiCudaTensor **up, ColiCudaTensor **down,
                      const void *gw, const float *gs, int gfmt,
                      const void *uw, const float *us, int ufmt,
                      const void *dw, const float *ds, int dfmt,
                      float weight, int S, int D, int I, int device);
int coli_cuda_moe_end(float *y_host, int S, int D, int device);

/* Drop sticky device-x (call when host x packing may change). */
void coli_cuda_x_invalidate(int device);

/* ---- fused decode attention (S=1), KV cache resident on device ----------
 * Opt-in (CUDA_ATTN=1). Per decode layer:
 *   qkv GEMV (y stays on device) -> attn kernel (RoPE + v_scale + KV append +
 *   scores + online softmax + weighted-V) -> o GEMV (x on device) -> D2H out.
 * One sync per layer instead of two, no qkv/ctx PCIe round-trips.
 * Host K/V stay authoritative for S>1 (prefill/verify): those run on CPU and
 * are mirrored to the device with coli_cuda_attn_upload_rows(); the S=1
 * device path mirrors its appended row back to host (async, same stream). */

/* flags for coli_cuda_matmul_ex */
#define COLI_CUDA_MM_X_DEV  1   /* x is a device pointer (skip H2D)          */
#define COLI_CUDA_MM_Y_DEV  2   /* leave y on device (*y_dev), skip D2H+sync */

int coli_cuda_matmul_ex(ColiCudaTensor **tensor,
                        float *y, float **y_dev, const float *x,
                        const void *weights, const float *scales,
                        int fmt, int S, int I, int O, int device, int flags);

/* (Re)allocate the device KV store: per-layer physical rows (ring W or max_t),
 * geometry and type. Frees previous allocations. swa[i]: 1=SWA layer.
 * Returns 1 on success (all layers), 0 on any failure (store disabled). */
int coli_cuda_attn_kv_alloc(int device, int n_layer,
                            const int *rows, const int *kvh,
                            const int *hd, const int *vd, const int *swa,
                            int window, int max_t, int n_heads);

/* Upload RoPE tables for one layer type: cos/sin [max_t][half]. type: 0=full 1=swa. */
int coli_cuda_attn_rope(int device, int type, const float *cos_t, const float *sin_t,
                        int rows, int half);

/* Per-layer rope dim + optional sink bias [n_heads] (NULL = no sink). */
int coli_cuda_attn_layer_cfg(int device, int layer, int rd, const float *sink, int n_heads);

/* Register host K/V base pointers of one layer for the row mirror. */
int coli_cuda_attn_mirror(int device, int layer, float *K_host, float *V_host);

/* Mirror host rows [r0,r1) of one layer to the device (after S>1 CPU paths). */
int coli_cuda_attn_upload_rows(int device, int layer, int r0, int r1,
                               const float *K, const float *V);

/* Fused decode attention for one token at pos (S=1).
 * qkv_dev: device row [H*hd + kvh*hd + kvh*vd] (no RoPE/v_scale applied yet).
 * Returns the device ctx pointer [H*vd] (valid until next decode), NULL on error. */
float *coli_cuda_attn_decode(int device, int layer, const float *qkv_dev,
                             int pos, int kv_start, float v_scale);

void coli_cuda_tensor_free(ColiCudaTensor *tensor);
size_t coli_cuda_tensor_bytes(const ColiCudaTensor *tensor);
int coli_cuda_tensor_device(const ColiCudaTensor *tensor);

/* Cumulative host time spent in PCIe H2D/D2H copies inside coli_cuda_matmul
 * (kernel time excluded). Caller subtracts a baseline taken before the
 * timed region to isolate copy overhead from compute. */
double coli_cuda_copy_seconds(void);

/* ---- event-based profiler (opt-in via env COLI_CUDA_PROF=1) ---------------
 * Cumulative per-device counters, reset with coli_cuda_prof_reset(). GPU times
 * are measured with CUDA events on the device stream (drained at the existing
 * sync points — no extra syncs are introduced); ms_sync_wait is host wall time
 * inside the stream syncs. Synchronous pin-time weight uploads are accounted
 * in ms_h2d/h2d_bytes with host timing. When COLI_CUDA_PROF is unset/0 all
 * hooks are inert (one branch per call). */
typedef struct ColiCudaProf {
    uint64_t n_launch;              /* kernel launches (copies excluded) */
    uint64_t n_h2d, n_d2h;          /* copy operations */
    uint64_t h2d_bytes, d2h_bytes;
    double ms_gate_up;              /* fused gate+up+silu kernels */
    double ms_gemv;                 /* down / projection GEMV kernels */
    double ms_axpy;                 /* weighted-accumulate kernels */
    double ms_attn;                 /* fused decode attention kernels */
    double ms_kernel_other;         /* zero/silu_mul and misc kernels */
    double ms_h2d, ms_d2h;          /* event-measured copy time */
    double ms_sync_wait;            /* host wall time inside syncs */
} ColiCudaProf;

int  coli_cuda_prof_enabled(void);
void coli_cuda_prof_reset(int device);
/* Drains only completed segments (never blocks); snapshot of counters. */
int  coli_cuda_prof_snapshot(int device, ColiCudaProf *out);
/* Syncs the stream and drains everything pending: call once before the final
 * report so the tail of the last layer is not lost. */
void coli_cuda_prof_flush(int device);

#ifdef __cplusplus
}
#endif

#endif
