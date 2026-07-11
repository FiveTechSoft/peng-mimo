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

void coli_cuda_tensor_free(ColiCudaTensor *tensor);
size_t coli_cuda_tensor_bytes(const ColiCudaTensor *tensor);
int coli_cuda_tensor_device(const ColiCudaTensor *tensor);

/* Cumulative host time spent in PCIe H2D/D2H copies inside coli_cuda_matmul
 * (kernel time excluded). Caller subtracts a baseline taken before the
 * timed region to isolate copy overhead from compute. */
double coli_cuda_copy_seconds(void);

#ifdef __cplusplus
}
#endif

#endif
