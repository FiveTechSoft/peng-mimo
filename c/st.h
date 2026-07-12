/* Indicizzazione e lettura on-demand di tensori da piu' file safetensors.
 * Equivale a Shards in engine.py, ma:
 *   - legge con pread (niente mmap) + posix_fadvise(DONTNEED) -> le pagine NON
 *     restano residenti nel processo. E' la correzione del bug di RSS: cosi' la
 *     RAM di picco resta densa+cache, non l'intero modello. (vedi memoria mmap-rss-bug)
 *   - converte sempre in float32 in uscita (BF16/F16/F32 supportati). */
#ifndef ST_H
#define ST_H
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>
#include "json.h"
#include "compat.h"

typedef struct {
    char   *name;
    int     fd;
    int64_t off;       /* offset assoluto del dato dentro al file */
    int64_t nbytes;
    int     dtype;     /* 0=BF16 1=F16 2=F32 */
    int64_t numel;
} st_tensor;

typedef struct {
    st_tensor *t;
    int        n, cap;
    int        fds[512];
    int        dfds[512];  /* gemelli O_DIRECT (aperti pigramente): -2 = non ancora provato */
    char      *paths[512];
    int        nfd;
    int       *hidx;      /* hash map nome->indice (open addressing): con ~120k tensori
                           * (GLM: 256 expert x 78 layer x 3 x 2) la scansione lineare
                           * costava decine di secondi/token (misurato sul primo run reale) */
    int        hcap;
} shards;
#define ST_MAX_SHARDS 512

static uint64_t st_hash(const char *s){
    uint64_t h=1469598103934665603ULL;
    while(*s){ h^=(unsigned char)*s++; h*=1099511628211ULL; }
    return h;
}

static int st_dtype_code(const char *s) {
    if (!strcmp(s, "BF16")) return 0;
    if (!strcmp(s, "F16"))  return 1;
    if (!strcmp(s, "F32"))  return 2;
    if (!strcmp(s, "U8"))   return 3;   /* dati quantizzati (int4 packed / int8) */
    if (!strcmp(s, "I8"))   return 3;
    fprintf(stderr, "dtype non gestito: %s\n", s); exit(1);
}

static inline float bf16_to_f32(uint16_t h) {
    uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f;
}
static inline float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t man  = h & 0x3FF;
    uint32_t u;
    if (exp == 0) {            /* subnormale o zero */
        if (man == 0) u = sign;
        else { exp = 127 - 15 + 1; while (!(man & 0x400)) { man <<= 1; exp--; } man &= 0x3FF; u = sign | (exp << 23) | (man << 13); }
    } else if (exp == 0x1F) {  /* inf/nan */
        u = sign | 0x7F800000 | (man << 13);
    } else {
        u = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float f; memcpy(&f, &u, 4); return f;
}

static int st_open_fd(shards *S, const char *path) {
    for (int i = 0; i < S->nfd; i++) if (!strcmp(S->paths[i], path)) return S->fds[i];
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror(path); exit(1); }
    S->paths[S->nfd] = strdup(path); S->fds[S->nfd] = fd;
#ifdef O_DIRECT
    S->dfds[S->nfd] = open(path, O_RDONLY | O_DIRECT);   /* eager: lookup poi thread-safe */
#elif defined(__APPLE__)
    S->dfds[S->nfd] = compat_open_direct(path);          /* macOS: F_NOCACHE ~ O_DIRECT */
#else
    S->dfds[S->nfd] = -1;                                /* niente equivalente: solo buffered */
#endif
    S->nfd++;
    return fd;
}

/* fd gemello O_DIRECT dello stesso file (bypassa la page cache: il buffered read su
 * ext4-in-VHDX si strozza a ~0.8 GB/s, O_DIRECT arriva a 2.3+; misurato). -1 se non disponibile. */
static int st_direct_fd(shards *S, int fd) {
    for (int i = 0; i < S->nfd; i++) if (S->fds[i] == fd) return S->dfds[i];
    return -1;
}

/* F-01: dtype payload size (bytes/element). U8/I8 = 1 (packed int4/int2 may use
 * nbytes < numel*1 if shape is logical — still require nbytes > 0 and in-file). */
static int st_dtype_esz(int dtype) {
    if (dtype == 2) return 4;   /* F32 */
    if (dtype == 0 || dtype == 1) return 2; /* BF16 / F16 */
    if (dtype == 3) return 1;   /* U8 / I8 */
    return -1;
}
/* product of shape dims with overflow detection; empty shape [] => scalar numel 1. */
static int64_t st_shape_numel(jval *shp) {
    if (!shp || shp->t != J_ARR) return -1;
    int64_t numel = 1;
    for (int k = 0; k < shp->len; k++) {
        if (!shp->kids[k] || shp->kids[k]->t != J_NUM) return -1;
        double d = shp->kids[k]->num;
        if (d < 0 || d != (double)(int64_t)d) return -1;
        int64_t dim = (int64_t)d;
        if (dim < 0) return -1;
        if (dim > 0 && numel > INT64_MAX / dim) return -1;
        numel *= dim;
    }
    return numel;
}
/* Linear name scan (used during st_init before the hash index exists). */
static int st_name_seen(shards *S, const char *name) {
    for (int i = 0; i < S->n; i++) if (!strcmp(S->t[i].name, name)) return 1;
    return 0;
}
/* pread completo: una singola pread() Linux tronca a ~2 GB (0x7ffff000). */
static ssize_t pread_full(int fd, void *buf, int64_t count, int64_t off) {
    int64_t got = 0;
    while (got < count) {
        ssize_t r = pread(fd, (char *)buf + got, (size_t)(count - got), off + got);
        if (r < 0) return -1;
        if (r == 0) { errno = 0; return got; }   /* EOF prematuro */
        got += r;
    }
    return got;
}

/* Max JSON header we will accept (safetensors header is rarely > few MB). */
#ifndef ST_MAX_HEADER
#define ST_MAX_HEADER (64ULL << 20)   /* 64 MiB */
#endif

/* indicizza tutti i model-*.safetensors in snap_dir */
static void st_init(shards *S, const char *snap_dir) {
    memset(S, 0, sizeof(*S));
    S->cap = 4096; S->t = calloc(S->cap, sizeof(st_tensor));
    /* raccoglie ordinatamente i nomi dei file shard */
    static char files[ST_MAX_SHARDS][1024]; int nf = 0;
    DIR *d = opendir(snap_dir); struct dirent *e;
    if (!d) { perror(snap_dir); exit(1); }
    while ((e = readdir(d))) {
        const char *dot = strrchr(e->d_name, '.');
        if (dot && !strcmp(dot, ".safetensors")) {  /* model.safetensors o model-0000N-of-... */
            if (nf >= ST_MAX_SHARDS) { fprintf(stderr, "troppi shard (>%d): alza ST_MAX_SHARDS\n", ST_MAX_SHARDS); exit(1); }
            snprintf(files[nf++], 1024, "%s/%s", snap_dir, e->d_name);
        }
    }
    closedir(d);
    for (int a = 0; a < nf; a++) for (int b = a+1; b < nf; b++)
        if (strcmp(files[a], files[b]) > 0) { char tmp[1024]; strcpy(tmp, files[a]); strcpy(files[a], files[b]); strcpy(files[b], tmp); }

    for (int fi = 0; fi < nf; fi++) {
        int fd = st_open_fd(S, files[fi]);
        struct stat sb;
        if (fstat(fd, &sb) != 0) { perror(files[fi]); exit(1); }
        int64_t fsz = (int64_t)sb.st_size;
        if (fsz < 8) { fprintf(stderr, "safetensors too small: %s (%lld bytes)\n", files[fi], (long long)fsz); exit(1); }
        uint64_t hlen = 0;
        if (pread(fd, &hlen, 8, 0) != 8) { perror("pread hlen"); exit(1); }
        /* F-01: header size must fit in file and stay under ST_MAX_HEADER */
        if (hlen == 0 || hlen > ST_MAX_HEADER || (int64_t)hlen > fsz - 8) {
            fprintf(stderr, "safetensors bad header length: %s hlen=%llu file=%lld\n",
                    files[fi], (unsigned long long)hlen, (long long)fsz);
            exit(1);
        }
        char *hdr = malloc((size_t)hlen + 1);
        if (!hdr) { fprintf(stderr, "OOM safetensors header %s\n", files[fi]); exit(1); }
        if (pread_full(fd, hdr, (int64_t)hlen, 8) != (ssize_t)hlen) {
            fprintf(stderr, "pread hdr short: %s\n", files[fi]); exit(1);
        }
        hdr[hlen] = 0;
        int64_t data_start = 8 + (int64_t)hlen;
        char *arena = NULL;
        jval *root = json_parse(hdr, &arena);
        if (!root || root->t != J_OBJ) {
            fprintf(stderr, "safetensors header JSON invalid: %s\n", files[fi]);
            free(arena); free(hdr); exit(1);
        }
        for (int i = 0; i < root->len; i++) {
            const char *name = root->keys[i];
            if (!name || !strcmp(name, "__metadata__")) continue;
            jval *m = root->kids[i];
            if (!m || m->t != J_OBJ) {
                fprintf(stderr, "safetensors tensor entry not object: %s in %s\n", name, files[fi]);
                exit(1);
            }
            jval *dt = json_get(m, "dtype");
            jval *off = json_get(m, "data_offsets");
            jval *shp = json_get(m, "shape");
            if (!dt || dt->t != J_STR || !dt->str ||
                !off || off->t != J_ARR || off->len != 2 ||
                !off->kids[0] || !off->kids[1] ||
                off->kids[0]->t != J_NUM || off->kids[1]->t != J_NUM) {
                fprintf(stderr, "safetensors missing dtype/data_offsets: %s in %s\n", name, files[fi]);
                exit(1);
            }
            int64_t a0 = (int64_t)off->kids[0]->num, b0 = (int64_t)off->kids[1]->num;
            if (a0 < 0 || b0 < a0) {
                fprintf(stderr, "safetensors bad offsets %lld..%lld for %s in %s\n",
                        (long long)a0, (long long)b0, name, files[fi]);
                exit(1);
            }
            int64_t nbytes = b0 - a0;
            int64_t abs_end = data_start + b0;
            if (abs_end < data_start || abs_end > fsz) {
                fprintf(stderr, "safetensors range out of file: %s off=%lld..%lld file=%lld (%s)\n",
                        name, (long long)(data_start + a0), (long long)abs_end,
                        (long long)fsz, files[fi]);
                exit(1);
            }
            int64_t numel = st_shape_numel(shp);
            if (numel < 0) {
                fprintf(stderr, "safetensors bad shape: %s in %s\n", name, files[fi]);
                exit(1);
            }
            int dtype = st_dtype_code(dt->str);
            int esz = st_dtype_esz(dtype);
            if (esz < 0) {
                fprintf(stderr, "safetensors bad dtype code for %s\n", name);
                exit(1);
            }
            /* F32/BF16/F16: payload size must match shape. U8/I8 may be packed (int4). */
            if (dtype != 3) {
                if (numel > 0 && esz > 0 && numel > INT64_MAX / esz) {
                    fprintf(stderr, "safetensors numel*esz overflow: %s\n", name);
                    exit(1);
                }
                int64_t expect = numel * (int64_t)esz;
                if (nbytes != expect) {
                    fprintf(stderr, "safetensors nbytes mismatch: %s nbytes=%lld expect=%lld (numel=%lld esz=%d) in %s\n",
                            name, (long long)nbytes, (long long)expect,
                            (long long)numel, esz, files[fi]);
                    exit(1);
                }
            } else if (nbytes < 1 && numel > 0) {
                fprintf(stderr, "safetensors empty U8 payload: %s in %s\n", name, files[fi]);
                exit(1);
            }
            if (st_name_seen(S, name)) {
                fprintf(stderr, "safetensors duplicate tensor name: %s (also in %s)\n", name, files[fi]);
                exit(1);
            }
            if (S->n == S->cap) { S->cap *= 2; S->t = realloc(S->t, S->cap*sizeof(st_tensor)); }
            st_tensor *t = &S->t[S->n++];
            t->name = strdup(name); t->fd = fd; t->off = data_start + a0;
            t->nbytes = nbytes; t->dtype = dtype; t->numel = numel;
        }
        free(arena); /* i jval restano leakati: ok, una tantum all'avvio */
        free(hdr);
    }
    /* indice hash costruito a fine indicizzazione (gli indici restano validi dopo i realloc) */
    S->hcap = 1; while (S->hcap < S->n * 2) S->hcap <<= 1;
    S->hidx = malloc(S->hcap * sizeof(int));
    for (int i = 0; i < S->hcap; i++) S->hidx[i] = -1;
    for (int i = 0; i < S->n; i++) {
        uint64_t h = st_hash(S->t[i].name) & (S->hcap - 1);
        while (S->hidx[h] >= 0) h = (h + 1) & (S->hcap - 1);
        S->hidx[h] = i;
    }
}

static st_tensor *st_find(shards *S, const char *name) {
    if (S->hidx) {
        uint64_t h = st_hash(name) & (S->hcap - 1);
        while (S->hidx[h] >= 0) {
            st_tensor *t = &S->t[S->hidx[h]];
            if (!strcmp(t->name, name)) return t;
            h = (h + 1) & (S->hcap - 1);
        }
        return NULL;
    }
    for (int i = 0; i < S->n; i++) if (!strcmp(S->t[i].name, name)) return &S->t[i];
    return NULL;
}
static int st_has(shards *S, const char *name) { return st_find(S, name) != NULL; }

/* prefetch ASINCRONO: dice al kernel di iniziare a leggere le pagine del tensore in
 * background (readahead). Serve a sovrapporre l'I/O degli expert col calcolo: si
 * prefetcha tutto il set di expert di un layer, poi le pread sincrone trovano la cache
 * gia' calda. No-op se il tensore non esiste (es. il primo .qs prima della lettura). */
static void st_prefetch(shards *S, const char *name) {
    st_tensor *t = st_find(S, name);
    if (t) posix_fadvise(t->fd, t->off, t->nbytes, POSIX_FADV_WILLNEED);
}

/* Physical readahead: sort tensors by (fd, off), merge adjacent/nearby ranges, one
 * fadvise per merged span. Beats N random WILLNEEDs when experts share a shard. */
static void st_prefetch_phys(shards *S, st_tensor **ts, int n) {
    if (!S || !ts || n < 1) return;
    if (n == 1) {
        if (ts[0]) posix_fadvise(ts[0]->fd, ts[0]->off, ts[0]->nbytes, POSIX_FADV_WILLNEED);
        return;
    }
    /* insertion sort by fd then off (n is small: pathpack radius * 6 tensors) */
    for (int i = 1; i < n; i++) {
        st_tensor *v = ts[i]; int j = i - 1;
        while (j >= 0 && ts[j] && v &&
               (ts[j]->fd > v->fd || (ts[j]->fd == v->fd && ts[j]->off > v->off))) {
            ts[j + 1] = ts[j]; j--;
        }
        /* also bubble nulls to end */
        if (!v) { /* keep nulls at end via simple skip in merge */ }
        ts[j + 1] = v;
    }
    const int64_t gap = 2 * 1024 * 1024; /* merge if within 2 MB on same fd */
    int i = 0;
    while (i < n) {
        while (i < n && !ts[i]) i++;
        if (i >= n) break;
        int fd = ts[i]->fd;
        int64_t lo = ts[i]->off, hi = ts[i]->off + ts[i]->nbytes;
        int k = i + 1;
        while (k < n && ts[k] && ts[k]->fd == fd) {
            int64_t a = ts[k]->off, b = ts[k]->off + ts[k]->nbytes;
            if (a > hi + gap) break;
            if (a < lo) lo = a;
            if (b > hi) hi = b;
            k++;
        }
        if (hi > lo) posix_fadvise(fd, lo, hi - lo, POSIX_FADV_WILLNEED);
        i = k;
    }
    (void)S;
}

/* legge un tensore in un buffer float32 fornito dal chiamante.
 * out_cap = numero di float disponibili in `out` (F-01: no write past end).
 * drop=1 -> fadvise DONTNEED (streaming expert). */
static int64_t st_read_f32(shards *S, const char *name, float *out, int64_t out_cap, int drop) {
    st_tensor *t = st_find(S, name);
    if (!t) { fprintf(stderr, "tensore mancante: %s\n", name); exit(1); }
    if (!out || out_cap < t->numel) {
        fprintf(stderr, "st_read_f32 buffer too small: %s need %lld floats, cap %lld\n",
                name, (long long)t->numel, (long long)out_cap);
        exit(1);
    }
    if (t->dtype == 3) {
        fprintf(stderr, "st_read_f32: U8 tensor %s — use st_read_raw\n", name);
        exit(1);
    }
    void *raw = malloc((size_t)t->nbytes);
    if (!raw) { fprintf(stderr, "OOM st_read_f32 %s (%lld bytes)\n", name, (long long)t->nbytes); exit(1); }
    if (pread_full(t->fd, raw, t->nbytes, t->off) != t->nbytes) {
        struct stat sb; fstat(t->fd, &sb);
        fprintf(stderr, "pread data corto: %s off=%lld nbytes=%lld file=%lld byte\n",
                t->name, (long long)t->off, (long long)t->nbytes, (long long)sb.st_size);
        perror("pread data"); exit(1);
    }
    if (t->dtype == 2) {
        memcpy(out, raw, (size_t)t->nbytes);
    } else if (t->dtype == 0) {
        uint16_t *p = (uint16_t *)raw; for (int64_t i = 0; i < t->numel; i++) out[i] = bf16_to_f32(p[i]);
    } else if (t->dtype == 1) {
        uint16_t *p = (uint16_t *)raw; for (int64_t i = 0; i < t->numel; i++) out[i] = f16_to_f32(p[i]);
    } else {
        fprintf(stderr, "st_read_f32: dtype %d non gestito per %s\n", t->dtype, name);
        exit(1);
    }
    free(raw);
    if (drop) posix_fadvise(t->fd, t->off, t->nbytes, POSIX_FADV_DONTNEED);
    return t->numel;
}

static int64_t st_numel(shards *S, const char *name) {
    st_tensor *t = st_find(S, name); return t ? t->numel : -1;
}
static int64_t st_nbytes(shards *S, const char *name) {
    st_tensor *t = st_find(S, name); return t ? t->nbytes : -1;
}

/* legge i byte GREZZI di un tensore (U8 quant). out_cap_bytes = capacità del buffer (F-01). */
static void st_read_raw(shards *S, const char *name, void *out, int64_t out_cap_bytes, int drop) {
    st_tensor *t = st_find(S, name);
    if (!t) { fprintf(stderr, "tensore mancante: %s\n", name); exit(1); }
    if (!out || out_cap_bytes < t->nbytes) {
        fprintf(stderr, "st_read_raw buffer too small: %s need %lld bytes, cap %lld\n",
                name, (long long)t->nbytes, (long long)out_cap_bytes);
        exit(1);
    }
    if (pread_full(t->fd, out, t->nbytes, t->off) != t->nbytes) { perror("pread raw"); exit(1); }
    if (drop) posix_fadvise(t->fd, t->off, t->nbytes, POSIX_FADV_DONTNEED);
}

/* legge una FETTA di un tensore: n_elems a partire dall'elemento elem_off.
 * out_cap = float disponibili in out. Serve per expert fusi GLM [E,...]. */
static void st_read_slice_f32(shards *S, const char *name, int64_t elem_off, int64_t n_elems,
                              float *out, int64_t out_cap, int drop) {
    st_tensor *t = st_find(S, name);
    if (!t) { fprintf(stderr, "tensore mancante: %s\n", name); exit(1); }
    if (!out || out_cap < n_elems) {
        fprintf(stderr, "st_read_slice_f32 buffer too small: %s need %lld floats, cap %lld\n",
                name, (long long)n_elems, (long long)out_cap);
        exit(1);
    }
    if (t->dtype == 3) {
        fprintf(stderr, "st_read_slice_f32: U8 tensor %s\n", name); exit(1);
    }
    if (elem_off < 0 || n_elems < 0 || elem_off > t->numel || n_elems > t->numel - elem_off) {
        fprintf(stderr, "st_read_slice_f32 range: %s off=%lld n=%lld numel=%lld\n",
                name, (long long)elem_off, (long long)n_elems, (long long)t->numel);
        exit(1);
    }
    int esz = st_dtype_esz(t->dtype);
    if (esz < 2) { fprintf(stderr, "st_read_slice_f32 bad esz for %s\n", name); exit(1); }
    int64_t boff = t->off + elem_off * esz, nb = n_elems * esz;
    void *raw = malloc((size_t)nb);
    if (!raw) { fprintf(stderr, "OOM st_read_slice_f32\n"); exit(1); }
    if (pread_full(t->fd, raw, nb, boff) != nb) { perror("pread slice"); exit(1); }
    if (t->dtype == 2) memcpy(out, raw, (size_t)nb);
    else if (t->dtype == 0) { uint16_t *p = raw; for (int64_t i = 0; i < n_elems; i++) out[i] = bf16_to_f32(p[i]); }
    else { uint16_t *p = raw; for (int64_t i = 0; i < n_elems; i++) out[i] = f16_to_f32(p[i]); }
    free(raw);
    if (drop) posix_fadvise(t->fd, boff, nb, POSIX_FADV_DONTNEED);
}

#endif
