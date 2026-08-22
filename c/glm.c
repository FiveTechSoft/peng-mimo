/* Motore GLM-5.2 (architettura glm_moe_dsa) in C puro.
 * Stadio B: replica fedele del forward di transformers (modeling_glm_moe_dsa.py):
 *   - attenzione MLA (q/kv-LoRA, RoPE interleaved parziale)
 *   - router sigmoid + noaux_tc (n_group=1) con routed_scaling_factor
 *   - shared expert + expert routed in streaming dal disco (per-expert)
 *   - primi first_k_dense_replace layer densi
 * Il DSA indexer e' un NO-OP per seq <= index_topk (seleziona tutte le key): qui si usa
 * attenzione causale densa -> output identico all'oracolo su prompt corti.
 *
 * QUANTIZZAZIONE: gli expert (streaming) e la parte DENSA residente (attenzione, lm_head,
 * embed, mlp densa, shared expert) sono tenuti in int8 per-riga + scala per riga (dequant-on-use).
 * E' cio' che fa entrare GLM-5.2 nei 15 GB: ~17B param residenti a int4 ~= 8.7 GB.
 * Norme/router/bias restano f32 (piccoli e sensibili).
 *
 * Validazione: stessi token id di ref_glm.json (oracolo transformers, c/tools/make_glm_oracle.py).
 *   build: make glm   run: SNAP=./glm_tiny ./glm <cap> <expert_bits> <dense_bits>
 *   TF=1 -> teacher-forcing (valida il prefill su tutta la sequenza)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <limits.h>
#include <pthread.h>                              /* thread I/O del PILOTA */
#include <unistd.h>
#include <sys/resource.h>
#if defined(__APPLE__) || defined(__linux__)
#include <sys/mman.h>                             /* mlock: inchioda le pagine in RAM / wire pages into RAM */
#endif
#include "st.h"
#include "tok.h"
#include "tier.h"
#ifdef COLI_CUDA
#include <omp.h>
#include "backend_cuda.h"
#endif
#include "kv_cache.h" /* FreeToken PoC cache API */
#ifdef __AVX2__
#include <immintrin.h>
static inline float hsum256(__m256 v){            /* somma orizzontale di 8 float */
    __m128 lo=_mm256_castps256_ps128(v), hi=_mm256_extractf128_ps(v,1);
    lo=_mm_add_ps(lo,hi); __m128 sh=_mm_movehl_ps(lo,lo); lo=_mm_add_ss(lo,sh); return _mm_cvtss_f32(lo);
}
#elif defined(__ARM_NEON)
#include <arm_neon.h>                             /* Apple Silicon / aarch64: kernel NEON */
#endif
#ifdef __APPLE__
#include <mach/mach.h>                            /* host_statistics64: MemAvailable di macOS */
#endif

/* Simple FNV-1a 64-bit over an int array, hex output for prompt id */
static void prompt_hash(const int *ids, int n, char *out, size_t outlen){
    uint64_t h = 14695981039346656037ULL;
    for(int i=0;i<n;i++){
        uint32_t v = (uint32_t)ids[i];
        /* mix 4 bytes of v */
        for(int b=0;b<4;b++){ uint8_t c = (v >> (b*8)) & 0xFF; h ^= c; h *= 1099511628211ULL; }
    }
    snprintf(out, outlen, "%016llx", (unsigned long long)h);
}

typedef struct {
    int hidden, n_layers, n_heads, n_experts, topk, moe_inter, dense_inter;
    int first_dense, q_lora, kv_lora, qk_nope, qk_rope, qk_head, v_head, n_shared, vocab;
    int n_group, topk_group, norm_topk;
    int stop_ids[8], n_stop;                     /* eos_token_id dal config (GLM-5.2 ne ha 3!) */
    int index_topk, index_nh, index_hd;          /* DSA lightning indexer */
    int8_t idx_type[128];                        /* per layer: 1=full (calcola), 0=shared (riusa) */
    float eps, theta, attn_scale, routed_scale;
} Cfg;

/* tensore [O,I] in uno di tre formati:
 *   fmt=0 F32   -> qf
 *   fmt=1 INT8  -> q8 (1 byte/param) + scala per riga
 *   fmt=2 INT4  -> q4 (2 valori per byte, impacchettati) + scala per riga
 * INT4 e' cio' che fa stare la densa residente nei 15 GB (0.5 byte/param). */
/* fmt: 0 F32, 1 INT8, 2 INT4 (2/byte), 3 INT2 (4/byte). q4 ospita sia int4 che int2 packed. */
typedef struct {
    int fmt; float *qf; int8_t *q8; uint8_t *q4; float *s; int O, I;
#ifdef COLI_CUDA
    ColiCudaTensor *cuda;
#endif
    int cuda_eligible, cuda_failed, cuda_device;  /* resident tensor, never a reused expert slot */
} QT;
static int64_t qt_bytes(const QT *t){    /* byte residenti del tensore */
    int64_t n=(int64_t)t->O*t->I;
    if(t->fmt==0) return n*4;
    if(t->fmt==1) return n + (int64_t)t->O*4;
    if(t->fmt==3) return (int64_t)t->O*((t->I+3)/4) + (int64_t)t->O*4;
    return (int64_t)t->O*((t->I+1)/2) + (int64_t)t->O*4;
}

/* --- FreeToken runtime helpers (PoC) ---
 * try_restore_kv_from_cache: attempt to restore per-layer Lc/Rc for the prefix `ids[0..n-1]`.
 * try_save_kv_to_cache: save current m->Lc/m->Rc prefix for ids.
 * Model identification: use SNAP env value (string) as model id in the cache.
 */
static int try_restore_kv_from_cache(Model *m, const int *ids, int n){
    if(!ids || n<=0) return 0;
    char prompt_id[32]; prompt_hash(ids,n,prompt_id,sizeof(prompt_id));
    const char *snap = getenv("SNAP"); const char *model_id = snap?snap:"unknown";
    int n_layer=0; int *rows=NULL; int *kvh=NULL; float **Ks=NULL; float **Vs=NULL;
    if(!kv_cache_get(model_id, prompt_id, &n_layer, &rows, &kvh, &Ks, &Vs)){
        free(rows); free(kvh); free(Ks); free(Vs); return 0; }
    Cfg *c=&m->c; int NR=c->n_layers+1;
    int upto = n; if(upto > m->max_t) upto = m->max_t;
    for(int i=0;i<NR && i<n_layer;i++){
        int rowcnt = rows[i]; if(rowcnt>upto) rowcnt=upto;
        /* copy into m->Lc/m->Rc storage (which is allocated by kv_alloc) */
        if(rowcnt>0 && m->Lc && m->Rc){
            memcpy(m->Lc[i], Ks[i], (size_t)rowcnt * (size_t)c->kv_lora * sizeof(float));
            memcpy(m->Rc[i], Vs[i], (size_t)rowcnt * (size_t)c->qk_rope * sizeof(float));
        }
    }
    /* set kv_start conservatively to 0 for layers (PoC) */
    for(int i=0;i<NR;i++) m->kv_start[i]=0;
    free(rows); free(kvh); free(Ks); free(Vs);
    return 1;
}

static int try_save_kv_to_cache(Model *m, const int *ids, int n){
    if(!ids || n<=0) return 0;
    char prompt_id[32]; prompt_hash(ids,n,prompt_id,sizeof(prompt_id));
    const char *snap = getenv("SNAP"); const char *model_id = snap?snap:"unknown";
    Cfg *c=&m->c; int NR=c->n_layers+1;
    int *rows = malloc(NR*sizeof(int)); int *kvh = malloc(NR*sizeof(int));
    const float **Ks = malloc(NR*sizeof(float*)); const float **Vs = malloc(NR*sizeof(float*));
    if(!rows || !kvh || !Ks || !Vs){ free(rows); free(kvh); free(Ks); free(Vs); return 0; }
    int upto = n; if(upto > m->max_t) upto = m->max_t;
    for(int i=0;i<NR;i++){
        rows[i]=upto; kvh[i]=c->kv_lora; Ks[i]= (const float*)(m->Lc[i]); Vs[i]=(const float*)(m->Rc[i]);
    }
    int ok = kv_cache_put(model_id, prompt_id, NR, rows, kvh, Ks, Vs);
    free(rows); free(kvh); free(Ks); free(Vs);
    return ok;
}

/* Rest of original file follows unchanged... */

