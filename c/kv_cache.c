/*
 * Simple KV cache PoC for FreeToken feature.
 * Lightweight, single-process in-memory cache with conservative semantics.
 * Not optimized — intended as a minimal PoC to integrate with the C runtime.
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "kv_cache.h"

typedef struct KVLayer {
    int rows;
    int kvh_k;
    int kvh_v;
    float *K; /* rows * kvh_k */
    float *V; /* rows * kvh_v */
} KVLayer;

typedef struct KVEntry {
    char *model_hash;
    char *prompt_id;
    int n_layer;
    KVLayer *layers;
    struct KVEntry *next;
    size_t bytes;
} KVEntry;

static KVEntry *g_head = NULL;
static size_t g_max_bytes = 0;
static size_t g_cur_bytes = 0;

int kv_cache_init(size_t max_bytes) {
    g_max_bytes = max_bytes;
    g_cur_bytes = 0;
    g_head = NULL;
    return 1;
}

static void free_entry(KVEntry *e) {
    if (!e) return;
    free(e->model_hash);
    free(e->prompt_id);
    if (e->layers) {
        for (int i = 0; i < e->n_layer; ++i) {
            free(e->layers[i].K);
            free(e->layers[i].V);
        }
        free(e->layers);
    }
    free(e);
}

int kv_cache_put(const char *model_hash, const char *prompt_id,
                 int n_layer, const int *rows, const int *kvh_k, const int *kvh_v,
                 const float **K_src, const float **V_src) {
    if (!model_hash || !prompt_id) return 0;

    KVEntry *e = (KVEntry*)calloc(1, sizeof(KVEntry));
    e->model_hash = strdup(model_hash);
    e->prompt_id = strdup(prompt_id);
    e->n_layer = n_layer;
    e->layers = (KVLayer*)calloc(n_layer, sizeof(KVLayer));
    size_t bytes = 0;
    for (int i = 0; i < n_layer; ++i) {
        int r = rows[i];
        int hk = kvh_k[i];
        int hv = kvh_v[i];
        e->layers[i].rows = r;
        e->layers[i].kvh_k = hk;
        e->layers[i].kvh_v = hv;
        size_t nk = (size_t)r * (size_t)hk;
        size_t nv = (size_t)r * (size_t)hv;
        e->layers[i].K = (float*)malloc(sizeof(float) * nk);
        e->layers[i].V = (float*)malloc(sizeof(float) * nv);
        if (K_src && K_src[i]) memcpy(e->layers[i].K, K_src[i], sizeof(float) * nk);
        if (V_src && V_src[i]) memcpy(e->layers[i].V, V_src[i], sizeof(float) * nv);
        bytes += sizeof(float) * (nk + nv);
    }
    e->bytes = bytes;

    /* conservative eviction: if not enough space, drop oldest entries until fit */
    while (g_max_bytes > 0 && g_cur_bytes + e->bytes > g_max_bytes) {
        /* evict tail */
        KVEntry *prev = NULL;
        KVEntry *cur = g_head;
        if (!cur) break;
        while (cur->next) { prev = cur; cur = cur->next; }
        if (prev) prev->next = NULL; else g_head = NULL;
        g_cur_bytes -= cur->bytes;
        free_entry(cur);
    }

    /* insert at head (MRU) */
    e->next = g_head;
    g_head = e;
    g_cur_bytes += e->bytes;
    return 1;
}

int kv_cache_get(const char *model_hash, const char *prompt_id,
                 int *n_layer_out, int **rows_out, int **kvh_k_out, int **kvh_v_out,
                 float ***K_out, float ***V_out) {
    KVEntry *prev = NULL;
    KVEntry *cur = g_head;
    while (cur) {
        if (strcmp(cur->model_hash, model_hash) == 0 && strcmp(cur->prompt_id, prompt_id) == 0) {
            /* move to head (MRU) */
            if (prev) {
                prev->next = cur->next;
                cur->next = g_head;
                g_head = cur;
            }
            if (n_layer_out) *n_layer_out = cur->n_layer;
            if (rows_out) {
                int *rows = (int*)malloc(sizeof(int) * cur->n_layer);
                for (int i = 0; i < cur->n_layer; ++i) rows[i] = cur->layers[i].rows;
                *rows_out = rows;
            }
            if (kvh_k_out) {
                int *kvh = (int*)malloc(sizeof(int) * cur->n_layer);
                for (int i = 0; i < cur->n_layer; ++i) kvh[i] = cur->layers[i].kvh_k;
                *kvh_k_out = kvh;
            }
            if (kvh_v_out) {
                int *kvh = (int*)malloc(sizeof(int) * cur->n_layer);
                for (int i = 0; i < cur->n_layer; ++i) kvh[i] = cur->layers[i].kvh_v;
                *kvh_v_out = kvh;
            }
            if (K_out) {
                float **Ks = (float**)malloc(sizeof(float*) * cur->n_layer);
                for (int i = 0; i < cur->n_layer; ++i) Ks[i] = cur->layers[i].K;
                *K_out = Ks;
            }
            if (V_out) {
                float **Vs = (float**)malloc(sizeof(float*) * cur->n_layer);
                for (int i = 0; i < cur->n_layer; ++i) Vs[i] = cur->layers[i].V;
                *V_out = Vs;
            }
            return 1;
        }
        prev = cur;
        cur = cur->next;
    }
    return 0; /* miss */
}

void kv_cache_free_all(void) {
    KVEntry *cur = g_head;
    while (cur) {
        KVEntry *n = cur->next;
        free_entry(cur);
        cur = n;
    }
    g_head = NULL;
    g_cur_bytes = 0;
}

size_t kv_cache_cur_bytes(void) { return g_cur_bytes; }
size_t kv_cache_max_bytes(void) { return g_max_bytes; }
