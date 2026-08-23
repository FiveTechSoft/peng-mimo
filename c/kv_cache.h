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

/* Store per-layer K/V arrays. rows[], kvh_k[] and kvh_v[] are arrays of length
 * n_layer (K and V may have different per-row widths, e.g. MLA latent vs rope).
 * K_src and V_src are arrays of pointers (one pointer per layer) to float arrays
 * of length rows[i]*kvh_k[i] / rows[i]*kvh_v[i]. The function makes its own copies. */
int kv_cache_put(const char *model_hash, const char *prompt_id,
                 int n_layer, const int *rows, const int *kvh_k, const int *kvh_v,
                 const float **K_src, const float **V_src);

/* Retrieve stored K/V. On success, returns 1 and fills outputs. The returned
 * arrays (rows_out, kvh_k_out, kvh_v_out) must be free()'d by the caller. The
 * returned per-layer K/V pointers point into the cache internal buffers (do not
 * free them). */
int kv_cache_get(const char *model_hash, const char *prompt_id,
                 int *n_layer_out, int **rows_out, int **kvh_k_out, int **kvh_v_out,
                 float ***K_out, float ***V_out);

/* Free all cache entries (RAM only; disk files survive) */
void kv_cache_free_all(void);

/* Optional disk backing (PoC: same-machine format, raw f32, no endianness
 * handling). kv_cache_set_dir(dir): entries written by kv_cache_put are also
 * stored as one file per (model_hash, prompt_id) under dir, and a kv_cache_get
 * that misses in RAM tries to load the file back. Files survive the process:
 * this is the warm tier across restarts. kv_cache_set_disk_max caps the total
 * disk usage; when exceeded, the oldest files (by mtime) are deleted.
 * RAM evictions never delete the disk copy. */
void kv_cache_set_dir(const char *dir);
void kv_cache_set_disk_max(size_t max_bytes);

/* Introspect */
size_t kv_cache_cur_bytes(void);
size_t kv_cache_max_bytes(void);

/* FreeToken prefix-tree (§52.2): attach token ids to an entry and find the
 * cached entry with the longest common prefix against ids[0..n). The returned
 * prompt_id_out can be fed to kv_cache_get to restore that prefix length. */
int kv_cache_set_ids(const char *model_hash, const char *prompt_id,
                     const int *ids, int n);
int kv_cache_best_prefix(const char *model_hash, const int *ids, int n,
                         char *prompt_id_out, size_t outlen, int *lcp_out);

#ifdef __cplusplus
}
#endif

#endif /* KV_CACHE_H */
