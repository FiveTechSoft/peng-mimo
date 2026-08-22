/*
 * Header for kv_cache PoC
 */

#ifndef KV_CACHE_H
#define KV_CACHE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Initialize cache with a maximum resident bytes on host (0 = unlimited) */
int kv_cache_init(size_t max_bytes);

/* Store per-layer K/V arrays. rows[] and kvh[] are arrays of length n_layer.
 * K_src and V_src are arrays of pointers (one pointer per layer) to float arrays
 * of length rows[i]*kvh[i]. The function makes its own copies. */
int kv_cache_put(const char *model_hash, const char *prompt_id,
                 int n_layer, const int *rows, const int *kvh,
                 const float **K_src, const float **V_src);

/* Retrieve stored K/V. On success, returns 1 and fills outputs. The returned
 * arrays (rows_out, kvh_out) must be free()'d by the caller. The returned
 * per-layer K/V pointers point into the cache internal buffers (do not free
 * them). */
int kv_cache_get(const char *model_hash, const char *prompt_id,
                 int *n_layer_out, int **rows_out, int **kvh_out,
                 float ***K_out, float ***V_out);

/* Free all cache entries */
void kv_cache_free_all(void);

/* Introspect */
size_t kv_cache_cur_bytes(void);
size_t kv_cache_max_bytes(void);

#ifdef __cplusplus
}
#endif

#endif /* KV_CACHE_H */
