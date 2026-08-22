/*
 * Simple KV cache PoC for FreeToken feature.
 * Lightweight, single-process in-memory cache with conservative semantics.
 * Not optimized — intended as a minimal PoC to integrate with the C runtime.
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <limits.h>
#include <dirent.h>
#include <sys/stat.h>
#ifdef _WIN32
#include <direct.h>
#endif
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

static void free_entry(KVEntry *e);

/* insert at head (MRU), evicting oldest RAM entries until the cap fits */
static void insert_head(KVEntry *e) {
    while (g_max_bytes > 0 && g_cur_bytes + e->bytes > g_max_bytes) {
        KVEntry *prev = NULL, *cur = g_head;
        if (!cur) break;
        while (cur->next) { prev = cur; cur = cur->next; }
        if (prev) prev->next = NULL; else g_head = NULL;
        g_cur_bytes -= cur->bytes;
        free_entry(cur);
    }
    e->next = g_head;
    g_head = e;
    g_cur_bytes += e->bytes;
}

int kv_cache_init(size_t max_bytes) {
    g_max_bytes = max_bytes;
    g_cur_bytes = 0;
    g_head = NULL;
    return 1;
}

/* ---- disk backing (warm tier across process restarts) ------------------ */

#define KVC_MAGIC "KVCC0001"
static char g_dir[2048] = "";
static size_t g_disk_max = 0;

void kv_cache_set_dir(const char *dir) {
    if (!dir || !*dir) { g_dir[0] = 0; return; }
    snprintf(g_dir, sizeof(g_dir), "%s", dir);
#ifdef _WIN32
    _mkdir(g_dir);
#else
    mkdir(g_dir, 0755);
#endif
}

void kv_cache_set_disk_max(size_t max_bytes) { g_disk_max = max_bytes; }

static uint64_t fnv_str(const char *s) {
    uint64_t h = 14695981039346656037ULL;
    while (*s) { h ^= (uint8_t)*s++; h *= 1099511628211ULL; }
    return h;
}

static void entry_path(char *out, size_t outlen,
                       const char *model_hash, const char *prompt_id) {
    snprintf(out, outlen, "%s/%016llx_%s.kvc", g_dir,
             (unsigned long long)fnv_str(model_hash), prompt_id);
}

static int write_entry_file(const KVEntry *e) {
    if (!g_dir[0]) return 0;
    char path[2200]; entry_path(path, sizeof(path), e->model_hash, e->prompt_id);
    FILE *f = fopen(path, "wb");
    if (!f) return 0;
    uint32_t ml = (uint32_t)strlen(e->model_hash), pl = (uint32_t)strlen(e->prompt_id);
    uint32_t nl = (uint32_t)e->n_layer;
    int ok = fwrite(KVC_MAGIC, 1, 8, f) == 8
          && fwrite(&ml, 4, 1, f) == 1 && fwrite(e->model_hash, 1, ml, f) == ml
          && fwrite(&pl, 4, 1, f) == 1 && fwrite(e->prompt_id, 1, pl, f) == pl
          && fwrite(&nl, 4, 1, f) == 1;
    for (uint32_t i = 0; ok && i < nl; ++i) {
        uint32_t r = (uint32_t)e->layers[i].rows, hk = (uint32_t)e->layers[i].kvh_k,
                 hv = (uint32_t)e->layers[i].kvh_v;
        size_t nk = (size_t)r * hk, nv = (size_t)r * hv;
        ok = fwrite(&r, 4, 1, f) == 1 && fwrite(&hk, 4, 1, f) == 1 && fwrite(&hv, 4, 1, f) == 1
          && fwrite(e->layers[i].K, 4, nk, f) == nk
          && fwrite(e->layers[i].V, 4, nv, f) == nv;
    }
    if (fclose(f) != 0) ok = 0;
    if (!ok) remove(path);
    return ok;
}

/* delete oldest .kvc files (by mtime) until total size fits the disk cap */
static void disk_evict_if_needed(void) {
    if (!g_dir[0] || !g_disk_max) return;
    DIR *d = opendir(g_dir);
    if (!d) return;
    char (*names)[2304] = NULL; long *mt = NULL; size_t *sz = NULL;
    size_t n = 0, cap = 0, total = 0;
    struct dirent *de;
    while ((de = readdir(d))) {
        size_t l = strlen(de->d_name);
        if (l < 4 || strcmp(de->d_name + l - 4, ".kvc")) continue;
        char p[2304]; snprintf(p, sizeof(p), "%s/%s", g_dir, de->d_name);
        struct stat st; if (stat(p, &st)) continue;
        if (n == cap) {
            cap = cap ? cap * 2 : 16;
            names = realloc(names, cap * sizeof(*names));
            mt = realloc(mt, cap * sizeof(*mt));
            sz = realloc(sz, cap * sizeof(*sz));
            if (!names || !mt || !sz) goto out;
        }
        snprintf(names[n], sizeof(*names), "%s", p);
        mt[n] = (long)st.st_mtime; sz[n] = (size_t)st.st_size;
        total += sz[n]; n++;
    }
    while (total > g_disk_max) {
        size_t old = SIZE_MAX;
        for (size_t i = 0; i < n; ++i)
            if (mt[i] != LONG_MAX && (old == SIZE_MAX || mt[i] < mt[old])) old = i;
        if (old == SIZE_MAX) break;
        if (!remove(names[old])) total -= sz[old];
        mt[old] = LONG_MAX;                         /* borrado (o imposible): no reintentar */
    }
out:
    closedir(d);
    free(names); free(mt); free(sz);
}

static KVEntry *read_entry_file(const char *model_hash, const char *prompt_id) {
    if (!g_dir[0]) return NULL;
    char path[2200]; entry_path(path, sizeof(path), model_hash, prompt_id);
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    char magic[8]; uint32_t ml, pl, nl;
    KVEntry *e = NULL;
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, KVC_MAGIC, 8)) goto fail;
    if (fread(&ml, 4, 1, f) != 1 || ml > 4096) goto fail;
    e = (KVEntry*)calloc(1, sizeof(KVEntry));
    e->model_hash = (char*)malloc(ml + 1);
    if (fread(e->model_hash, 1, ml, f) != ml) goto fail;
    e->model_hash[ml] = 0;
    if (strcmp(e->model_hash, model_hash)) goto fail;   /* colision de nombre de fichero */
    if (fread(&pl, 4, 1, f) != 1 || pl > 4096) goto fail;
    e->prompt_id = (char*)malloc(pl + 1);
    if (fread(e->prompt_id, 1, pl, f) != pl) goto fail;
    e->prompt_id[pl] = 0;
    if (strcmp(e->prompt_id, prompt_id)) goto fail;
    if (fread(&nl, 4, 1, f) != 1 || nl > 4096) goto fail;
    e->n_layer = (int)nl;
    e->layers = (KVLayer*)calloc(nl, sizeof(KVLayer));
    for (uint32_t i = 0; i < nl; ++i) {
        uint32_t r, hk, hv;
        if (fread(&r, 4, 1, f) != 1 || fread(&hk, 4, 1, f) != 1 || fread(&hv, 4, 1, f) != 1)
            goto fail;
        size_t nk = (size_t)r * hk, nv = (size_t)r * hv;
        if (nk > (1u << 30) || nv > (1u << 30)) goto fail;
        e->layers[i].rows = (int)r; e->layers[i].kvh_k = (int)hk; e->layers[i].kvh_v = (int)hv;
        e->layers[i].K = (float*)malloc(4 * nk);
        e->layers[i].V = (float*)malloc(4 * nv);
        if (!e->layers[i].K || !e->layers[i].V) goto fail;
        if (fread(e->layers[i].K, 4, nk, f) != nk || fread(e->layers[i].V, 4, nv, f) != nv)
            goto fail;
        e->bytes += 4 * (nk + nv);
    }
    fclose(f);
    return e;
fail:
    if (e) { e->n_layer = e->layers ? e->n_layer : 0; free_entry(e); }
    fclose(f);
    remove(path);                       /* fichero corrupto: fuera */
    return NULL;
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

    insert_head(e);
    if (write_entry_file(e)) disk_evict_if_needed();
    return 1;
}

int kv_cache_get(const char *model_hash, const char *prompt_id,
                 int *n_layer_out, int **rows_out, int **kvh_k_out, int **kvh_v_out,
                 float ***K_out, float ***V_out) {
    KVEntry *prev = NULL;
    KVEntry *cur = g_head;
    for (;;) {
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
        /* RAM miss: intenta el tier de disco (sobrevive al proceso) */
        if (!g_dir[0]) return 0;
        KVEntry *e = read_entry_file(model_hash, prompt_id);
        if (!e) return 0;
        insert_head(e);
        prev = NULL; cur = g_head;              /* reintenta el lookup en RAM */
    }
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
