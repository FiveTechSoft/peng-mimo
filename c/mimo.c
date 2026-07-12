/* Motore MiMo-V2.5 (architettura mimo_v2) in C puro. Derivato da glm.c (donatore).
 * Attenzione ibrida full/SWA con pattern per-layer, GQA con qkv FUSO, RoPE parziale
 * NON-interleaved a doppia theta, sink bias per-testa sui layer SWA,
 * V*attention_value_scale prima della cache. Validato token-exact contro l'oracolo
 * transformers: tiny TF 32/32 + greedy 20/20, fixture 396M TF 20/20 + greedy 8/8.
 *   - router sigmoid + noaux_tc (n_group=1) con routed_scaling_factor: IDENTICO a GLM
 *   - expert routed in streaming dal disco (per-expert); NIENTE shared expert
 *   - niente MLA/DSA: MiMo-V2.5 non li ha
 *   - MTP NATIVA (model.mtp.layers.0, auto-rilevata): decodifica speculativa LOSSLESS
 *     con la testa multi-token del checkpoint (layer DENSO in geometria SWA, vedi vLLM
 *     mimo_v2_mtp.py); fallback n-gram quando la testa non c'e' (tiny/fixture)
 *
 * QUANTIZZAZIONE: gli expert (streaming) e la parte DENSA residente (attenzione, lm_head,
 * embed, mlp densa) sono tenuti in int8/int4 per-riga + scala (dequant-on-use).
 * Norme/router/bias restano f32 (piccoli e sensibili).
 *
 * Validazione: stessi token id di ref_mimo.json (oracolo transformers).
 *   build: make mimo   run: SNAP=./mimo_tiny ./mimo <cap> <expert_bits> <dense_bits>
 *   TF=1 -> teacher-forcing (valida il prefill su tutta la sequenza)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>
#include <time.h>
#include <limits.h>
#include <unistd.h>
#include <sys/resource.h>
#if defined(__APPLE__) || defined(__linux__)
#include <sys/mman.h>                             /* mlock: inchioda le pagine in RAM / wire pages into RAM */
#endif
#include <pthread.h>                              /* PILOT: async I/O thread for router-lookahead */
#ifdef __linux__
#include <sys/syscall.h>                          /* OVERLAP: gettid per il nice dei thread loader */
#include <execinfo.h>                             /* SEGV backtrace (no debugger available) */
#include <signal.h>
/* SEGV backtrace: no gdb/cuda-gdb in the dev environment; on a host
 * segfault print a symbolized backtrace to stderr before aborting. */
static void segv_handler(int sig){
    void *bt[64]; int n=backtrace(bt,64);
    fprintf(stderr,"\n[SEGV] signal %d, backtrace (%d frames):\n",sig,n);
    backtrace_symbols_fd(bt,n,2);
    fflush(stderr);
    _exit(2);
}
#endif

/* PILOT/LOOKA router-lookahead prefetch (ported from colibri/glm.c). Globals are
 * declared here (before moe()) so moe() can record the LOOKA accuracy counters;
 * the helper functions live further down, just before layer_forward. */
static int g_looka=0, g_pilot=0, g_pilot_k=8, g_pilot_depth=1;
static struct { int l,e; } pilot_q[4096];
static volatile unsigned pilot_w=0, pilot_r=0;
static void *pilot_m=NULL;                        /* Model* (defined later); cast at use */
static int  pilot_pred[256][64];                  /* LOOKA: predicted top-K per layer */
static int  pilot_pred_kt[256];
static long looka_hit=0, looka_tot=0;
/* Trajectory bulk WILLNEED (TRAJ=): knobs only; impl after Model. */
static int g_traj=-1, g_traj_k=8, g_traj_depth=2;
#include "st.h"
#include "tok.h"
#include "tier.h"
#ifdef COLI_CUDA
#include <omp.h>
#include "backend_cuda.h"
#endif
#ifdef __AVX2__
#include <immintrin.h>
static inline float hsum256(__m256 v){            /* somma orizzontale di 8 float */
    __m128 lo=_mm256_castps256_ps128(v), hi=_mm256_extractf128_ps(v,1);
    lo=_mm_add_ps(lo,hi); __m128 sh=_mm_movehl_ps(lo,lo); lo=_mm_add_ps(lo,sh);
    sh=_mm_shuffle_ps(lo,lo,1); lo=_mm_add_ss(lo,sh); return _mm_cvtss_f32(lo);
}
#elif defined(__ARM_NEON)
#include <arm_neon.h>                             /* Apple Silicon / aarch64: kernel NEON */
#endif
#ifdef __APPLE__
#include <mach/mach.h>                            /* host_statistics64: MemAvailable di macOS */
#endif

typedef struct {
    int hidden, n_layers, n_heads, n_experts, topk, moe_inter, dense_inter;
    int kv_heads_full, kv_heads_swa;             /* 4 / 8 nel modello reale */
    int head_dim, v_head_dim, rope_dim;          /* 192 / 128 / 64 */
    int swa_head_dim, swa_v_head_dim;            /* i layer SWA possono avere teste diverse
                                                  * (oracolo tiny: 24/24 vs 24/16 dei full) */
    int sliding_window, vocab;
    int n_group, topk_group, norm_topk;
    int qkv_grouped;                             /* layout del qkv fuso NEL CHECKPOINT: 1 = righe
                                                  * raggruppate per rank TP [Q|K|V] x NR
                                                  * (export FP8 di Xiaomi), 0 = piatto [Q|K|V]
                                                  * (modelli salvati da modeling_mimo_v2.py) */
    int stop_ids[8], n_stop;
    int8_t is_swa[128];                          /* hybrid_layer_pattern: 1=SWA */
    int8_t is_moe[128];                          /* moe_layer_freq: 1=MoE */
    int8_t has_sink_full, has_sink_swa;
    float eps, theta_full, theta_swa, v_scale, routed_scale;
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
    int gpu_only;                       /* VRAM-only: host slab freed, no CPU fallback */
} QT;
static int64_t qt_bytes(const QT *t){    /* byte residenti del tensore */
    int64_t n=(int64_t)t->O*t->I;
    if(t->fmt==0) return n*4;
    if(t->fmt==1) return n + (int64_t)t->O*4;
    if(t->fmt==3) return (int64_t)t->O*((t->I+3)/4) + (int64_t)t->O*4;
    return (int64_t)t->O*((t->I+1)/2) + (int64_t)t->O*4;
}

typedef struct {
    float *in_ln, *post_ln;
    QT qkv, o;                                   /* qkv fuso [qs+ks+vs, hidden] + o_proj */
    float *sink;                                 /* attention_sink_bias [n_heads] o NULL */
    int sparse;                                  /* is_moe */
    /* dense mlp (sparse==0) */
    QT gate_proj, up_proj, down_proj;
    /* moe (sparse==1) */
    float *router, *router_bias;                 /* router f32 (sensibile) */
} Layer;

/* slot di un expert: pesi quantizzati + scale. Nel container pre-quantizzato g/u/d sono
 * VISTE dentro `slab` (una sola pread coalescente); nel fallback hanno buffer propri.
 * slab_cap/fslab_cap: capienza allocata — gli slot ws[] sono riusati TRA layer
 * (in MiMo tutti gli expert hanno la stessa taglia; il buffer resta dimensionato al max). */
typedef struct { int eid; QT g,u,d; uint8_t *slab; float *fslab;
                 int64_t slab_cap, fslab_cap; uint64_t used; } ESlot;

typedef struct {
    Cfg c; shards S;
    int ebits, dbits;                            /* bit expert / bit densa */
    QT embed, lm_head; float *final_norm;
    Layer *L;
    /* KV-cache GQA per layer: K [max_t, kvh*head_dim], V [max_t, kvh*v_head_dim].
     * kvh e le dimensioni di testa dipendono dal TIPO di layer (full o SWA).
     * Lineare a tutta lunghezza anche per i layer SWA (il ring buffer arriva dopo). */
    float **K, **V; int max_t;
    int *kv_start;                               /* prima pos valida nella KV del layer (MTP: parziale) */
    /* testa MTP nativa (model.mtp.layers.0, convertita con --mtp): layer DENSO con
     * attenzione in geometria SWA + eh_proj/enorm/hnorm; final_layernorm = norma della
     * testa condivisa (lm_head del modello). KV alla riga n_layers. */
    int has_mtp; Layer mtpL; QT eh_proj;
    float *enorm, *hnorm, *mtp_norm;
    float *hlast, *h_all;                        /* hidden PRE-norm: ultima pos / batch spec (<=64) */
    uint64_t mtp_prop, mtp_acc;                  /* statistica acceptance */
    ESlot **ecache; int *ecn; int ecap;          /* LRU expert per-layer */
    ESlot ws[64];                                /* working set del layer corrente (load paralleli) */
    ESlot **pin; int *npin;                      /* HOT-STORE: expert pinnati in RAM (mai evicted) */
    ESlot **gpu_pin; int *ngpin;                 /* VRAM tier: complementary experts (not in RAM pin) */
    /* Bitmaps (the other way): 256 experts → 4×u64 per layer.
     * res_bits  = resident (VRAM|RAM pin|LRU) — O(1) hit test / skip WILLNEED
     * pref_bits = already WILLNEED'd this epoch — dedupe PILOT∪TRAJ∪sticky∪block */
    uint64_t *res_bits;                          /* [n_layers * 4] */
    uint64_t *pref_bits;                         /* [n_layers * 4] */
    uint32_t **eusage;                           /* contatori persistenti (per STATS/PIN) */
    uint32_t **eheat;                            /* calore recente per promotion/demotion live */
    int **eroute; int *enr;                      /* metodo C: routing dell'ULTIMO token per layer */
    uint64_t eclock, hits, miss, ereq;
    uint64_t gpu_expert_calls; int gpu_expert_count; int64_t gpu_expert_bytes;
    uint64_t n_fw, n_emit;                       /* metodo E: forward de decode / token emessi */
    double t_edisk, t_emm, t_attn, t_head;       /* profiling: dove va il tempo (sempre attivo) */
    double t_router, t_cuda_copy;                 /* fine-grained profile: router + PCIe copy (CUDA) */
    int64_t resident_bytes;
    /* tabella RoPE precalcolata: gli angoli dipendono SOLO da (pos, j, tipo-layer),
     * non dalla testa. La ricomputazione con powf() costava ~100k chiamate/token; qui
     * si calcolano una volta per max_t e si consultano via lookup (bit-identico). */
    float *rope_full_cos, *rope_full_sin;        /* [max_t][rope_dim/2]   theta_full */
    float *rope_swa_cos,  *rope_swa_sin;         /* [max_t][rd_swa/2]     theta_swa  */
    int rope_cap, rope_rd_full, rope_rd_swa;
} Model;

static void traj_observe_layer(Model *m, int layer, const int *idx, int Ke);
static void traj_commit_prev(Model *m);
static void traj_warm(Model *m, int k_per_layer);
static void usage_save(Model *m);        /* cache che impara: definita accanto a stats_dump */
static double now_s(void);               /* monotonic seconds; defined below CUDA helpers */
/* DRAFT must be visible before cuda_upload_dense_all (MTP densas only when >0). */
static int g_draft=0;    /* Method E: DRAFT=n speculative tokens per forward (0=off). With the
                          * native MTP head in the container, draft comes from that head
                          * (37-64% acceptance on the real model); without it, n-gram lookup
                          * (~5%). LOSSLESS: verified output is byte-identical to pure greedy
                          * (DRAFT=0 vs 2 vs 4). Default OFF: on disk-bound hosts each draft
                          * position pays its own experts from disk; enable DRAFT=2 when
                          * experts are pinned / NVMe is fast. */
#ifdef COLI_CUDA
static int g_cuda_enabled;
static double g_cuda_expert_gb;          /* 0=off, >0=GB cap, <0=auto (free-headroom) */
static int g_cuda_dense;
static int g_cuda_devices[COLI_CUDA_MAX_DEVICES], g_cuda_ndev, g_cuda_rr;
static int64_t g_cuda_dense_projected[COLI_CUDA_MAX_DEVICES];
static void qt_cuda_reset(QT *t){
    if(t->cuda){ coli_cuda_tensor_free(t->cuda); t->cuda=NULL; }
    t->cuda_failed=0;
}
static int qt_cuda_upload(QT *t){
    const void *weights = t->fmt==0 ? (const void*)t->qf
                        : t->fmt==1 ? (const void*)t->q8 : (const void*)t->q4;
    return coli_cuda_tensor_upload(&t->cuda,weights,t->s,t->fmt,t->I,t->O,t->cuda_device);
}
/* Drop a tensor from the dense VRAM plan (before upload). Used for embed/lm_head
 * (never on GPU) and for the MTP block when DRAFT=0. */
static void qt_cuda_unmark(QT *t){
    if(!t->cuda_eligible) return;
    if(!t->cuda){
        for(int i=0;i<g_cuda_ndev;i++) if(g_cuda_devices[i]==t->cuda_device){
            g_cuda_dense_projected[i]-=qt_bytes(t);
            if(g_cuda_dense_projected[i]<0) g_cuda_dense_projected[i]=0;
            break;
        }
    }
    t->cuda_eligible=0;
}
/* Free host weight buffers after a successful GPU upload. Sets gpu_only so
 * matmul_qt never falls through to a CPU path on NULL/freed pointers. */
static void qt_host_release(QT *t){
    if(t->fmt==0){ free(t->qf); t->qf=NULL; }
    else if(t->fmt==1){ free(t->q8); t->q8=NULL; }
    else { free(t->q4); t->q4=NULL; }
    free(t->s); t->s=NULL;
    t->gpu_only=1;
}
static int qt_cuda_upload_and_free(QT *t){
    if(!t->cuda_eligible || t->cuda_failed) return 0;
    if(t->cuda){ if(!t->gpu_only) qt_host_release(t); return 1; }
    if(!qt_cuda_upload(t)){ t->cuda_failed=1; t->cuda_eligible=0; return 0; }
    qt_host_release(t);
    return 1;
}
/* Eager-upload all CUDA-eligible dense tensors so pin_load sees the REAL free
 * VRAM (lazy first-use was racing experts for the same memory). Frees host
 * copies on success → more RAM for the expert pin/LRU. embed/lm_head stay on
 * CPU (gather path / better spent as expert budget). MTP densas only when
 * DRAFT>0. */
static void cuda_upload_dense_all(Model *m){
    if(!g_cuda_enabled || !g_cuda_dense) return;
    Cfg *c=&m->c;
    qt_cuda_unmark(&m->embed);
    qt_cuda_unmark(&m->lm_head);
    if(m->has_mtp && g_draft<=0){
        Layer *l=&m->mtpL;
        qt_cuda_unmark(&l->qkv); qt_cuda_unmark(&l->o);
        qt_cuda_unmark(&l->gate_proj); qt_cuda_unmark(&l->up_proj); qt_cuda_unmark(&l->down_proj);
        qt_cuda_unmark(&m->eh_proj);
    }
    double t0=now_s(); int ok=0, fail=0; int64_t freed=0;
    #define UP(t) do{ int64_t hb=qt_bytes(&(t)); \
        if((t).cuda_eligible){ if(qt_cuda_upload_and_free(&(t))){ ok++; freed+=hb; m->resident_bytes-=hb; } \
                               else fail++; } }while(0)
    for(int i=0;i<c->n_layers;i++){
        Layer *l=&m->L[i];
        UP(l->qkv); UP(l->o);
        if(!l->sparse){ UP(l->gate_proj); UP(l->up_proj); UP(l->down_proj); }
    }
    if(m->has_mtp && g_draft>0){
        Layer *l=&m->mtpL;
        UP(l->qkv); UP(l->o);
        UP(l->gate_proj); UP(l->up_proj); UP(l->down_proj);
        UP(m->eh_proj);
    }
    #undef UP
    if(m->resident_bytes<0) m->resident_bytes=0;
    size_t free_b=0,total_b=0;
    if(g_cuda_ndev>0) coli_cuda_mem_info(g_cuda_devices[0],&free_b,&total_b);
    fprintf(stderr,"[CUDA] dense eager upload: %d ok, %d fail, freed %.2f GB host in %.1fs | VRAM free %.2f GB\n",
        ok,fail,freed/1e9,now_s()-t0,free_b/1e9);
}
static void cuda_stats_print(void){
    size_t n=0,b=0; coli_cuda_stats(-1,&n,&b);
    fprintf(stderr,"[CUDA] resident set: %zu tensor, %.2f GB VRAM\n",n,b/1e9);
    if(g_cuda_ndev>1) for(int i=0;i<g_cuda_ndev;i++){
        coli_cuda_stats(g_cuda_devices[i],&n,&b);
        fprintf(stderr,"[CUDA]   device %d: %zu tensor, %.2f GB\n",g_cuda_devices[i],n,b/1e9);
    }
}
static int parse_cuda_devices(const char *list, int *out){
    if(!list||!*list) return 0;
    int n=0; const char *p=list;
    while(*p){
        char *end=NULL; long v=strtol(p,&end,10);
        if(end==p||v<0||v>INT_MAX||n>=COLI_CUDA_MAX_DEVICES) return 0;
        for(int i=0;i<n;i++) if(out[i]==(int)v) return 0;
        out[n++]=(int)v; p=end;
        while(*p==' '||*p=='\t') p++;
        if(!*p) break;
        if(*p++!=',') return 0;
        while(*p==' '||*p=='\t') p++;
        if(!*p) return 0;
    }
    return n;
}
#endif
static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static double rss_gb(void){ struct rusage r; getrusage(RUSAGE_SELF,&r);
#ifdef __APPLE__
    return r.ru_maxrss/(1024.0*1024.0*1024.0);   /* macOS: ru_maxrss in BYTE */
#else
    return r.ru_maxrss/(1024.0*1024.0);          /* Linux: in KB */
#endif
}
static float *falloc(int64_t n){
    /* guardia anti-wrap (report PR #25): n assurdo da file modello ostili non deve
     * diventare una malloc piccola. Niente calloc: il memset nel percorso caldo costa. */
    if(n<0 || (uint64_t)n > SIZE_MAX/sizeof(float)){ fprintf(stderr,"falloc: n=%lld fuori range\n",(long long)n); exit(1); }
    float *p=malloc((size_t)n*sizeof(float)); if(!p){fprintf(stderr,"OOM\n");exit(1);} return p; }

/* y[S,O] = x[S,I] @ W^T, W[O,I] f32 */
static void matmul(float *y, const float *x, const float *W, int S, int I, int O){
    #pragma omp parallel for schedule(static)
    for (int o=0;o<O;o++){ const float *w=W+(int64_t)o*I;
        for (int s=0;s<S;s++){ const float *xs=x+(int64_t)s*I; float a=0; for(int i=0;i<I;i++) a+=xs[i]*w[i]; y[(int64_t)s*O+o]=a; } }
}
/* y[S,O] = x[S,I] @ W^T con W quantizzato int8 per-riga + scala[O] (dequant-on-use) */
static void matmul_q(float *y, const float *x, const int8_t *q, const float *scale, int S, int I, int O){
    #pragma omp parallel for schedule(static)
    for (int o=0;o<O;o++){ const int8_t *w=q+(int64_t)o*I; float sc=scale[o];
        for (int s=0;s<S;s++){ const float *xs=x+(int64_t)s*I; float a=0; int i=0;
#ifdef __AVX2__
            __m256 acc=_mm256_setzero_ps();
            for(;i+8<=I;i+=8){ __m256i wi=_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(w+i)));
                acc=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i), _mm256_cvtepi32_ps(wi), acc); }
            a=hsum256(acc);
#elif defined(__ARM_NEON)
            float32x4_t ac0=vdupq_n_f32(0), ac1=vdupq_n_f32(0);
            for(;i+8<=I;i+=8){ int16x8_t w16=vmovl_s8(vld1_s8(w+i));
                ac0=vfmaq_f32(ac0, vld1q_f32(xs+i),   vcvtq_f32_s32(vmovl_s16(vget_low_s16(w16))));
                ac1=vfmaq_f32(ac1, vld1q_f32(xs+i+4), vcvtq_f32_s32(vmovl_s16(vget_high_s16(w16)))); }
            a=vaddvq_f32(vaddq_f32(ac0,ac1));
#endif
            for(;i<I;i++) a+=xs[i]*(float)w[i]; y[(int64_t)s*O+o]=a*sc; } }
}
/* y[S,O] = x[S,I] @ W^T con W int4 impacchettato (2 valori/byte) + scala[O]. */
static void matmul_i4(float *y, const float *x, const uint8_t *q4, const float *scale, int S, int I, int O){
    int rb=(I+1)/2;
    #pragma omp parallel for schedule(static)
    for (int o=0;o<O;o++){ const uint8_t *w=q4+(int64_t)o*rb; float sc=scale[o];
        for (int s=0;s<S;s++){ const float *xs=x+(int64_t)s*I; float a=0; int i=0;
#ifdef __AVX2__
            const __m128i m4=_mm_set1_epi8(0x0F); const __m256i b8=_mm256_set1_epi32(8);
            __m256 acc=_mm256_setzero_ps();
            for(;i+16<=I;i+=16){ __m128i by=_mm_loadl_epi64((const __m128i*)(w+(i>>1)));   /* 8 byte=16 nibble */
                __m128i lo=_mm_and_si128(by,m4), hi=_mm_and_si128(_mm_srli_epi16(by,4),m4);
                __m128i nib=_mm_unpacklo_epi8(lo,hi);                                       /* nibble in ordine */
                __m256 w0=_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(nib),b8));
                __m256 w1=_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(nib,8)),b8));
                acc=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i),   w0, acc);
                acc=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i+8), w1, acc); }
            a=hsum256(acc);
#elif defined(__ARM_NEON)
            const uint8x8_t m4=vdup_n_u8(0x0F); const int8x8_t b8=vdup_n_s8(8);
            float32x4_t ac0=vdupq_n_f32(0), ac1=vdupq_n_f32(0);
            for(;i+16<=I;i+=16){ uint8x8_t by=vld1_u8(w+(i>>1));               /* 8 byte=16 nibble */
                uint8x8x2_t z=vzip_u8(vand_u8(by,m4), vshr_n_u8(by,4));        /* nibble in ordine */
                int16x8_t w0=vmovl_s8(vsub_s8(vreinterpret_s8_u8(z.val[0]),b8));
                int16x8_t w1=vmovl_s8(vsub_s8(vreinterpret_s8_u8(z.val[1]),b8));
                ac0=vfmaq_f32(ac0, vld1q_f32(xs+i),    vcvtq_f32_s32(vmovl_s16(vget_low_s16(w0))));
                ac1=vfmaq_f32(ac1, vld1q_f32(xs+i+4),  vcvtq_f32_s32(vmovl_s16(vget_high_s16(w0))));
                ac0=vfmaq_f32(ac0, vld1q_f32(xs+i+8),  vcvtq_f32_s32(vmovl_s16(vget_low_s16(w1))));
                ac1=vfmaq_f32(ac1, vld1q_f32(xs+i+12), vcvtq_f32_s32(vmovl_s16(vget_high_s16(w1)))); }
            a=vaddvq_f32(vaddq_f32(ac0,ac1));
#endif
            for(;i+1<I;i+=2){ uint8_t byte=w[i>>1]; int lo=(int)(byte&0xF)-8, hi=(int)(byte>>4)-8;
                a += xs[i]*(float)lo + xs[i+1]*(float)hi; }
            if(i<I){ uint8_t byte=w[i>>1]; int lo=(int)(byte&0xF)-8; a += xs[i]*(float)lo; }
            y[(int64_t)s*O+o]=a*sc; } }
}
/* y[S,O] = x[S,I] @ W^T con W int2 impacchettato (4 valori/byte) + scala[O]. nibble 2-bit -> [-2,1]. */
static void matmul_i2(float *y, const float *x, const uint8_t *q2, const float *scale, int S, int I, int O){
    int rb=(I+3)/4;
    #pragma omp parallel for schedule(static)
    for (int o=0;o<O;o++){ const uint8_t *w=q2+(int64_t)o*rb; float sc=scale[o];
        for (int s=0;s<S;s++){ const float *xs=x+(int64_t)s*I; float a=0; int i=0;
#ifdef __AVX2__
            const __m128i m2=_mm_set1_epi8(0x03); const __m256i b2=_mm256_set1_epi32(2);
            __m256 acc=_mm256_setzero_ps();
            for(;i+16<=I;i+=16){ __m128i by=_mm_cvtsi32_si128(*(const int*)(w+(i>>2)));    /* 4 byte=16 valori */
                __m128i p0=_mm_and_si128(by,m2), p1=_mm_and_si128(_mm_srli_epi16(by,2),m2);
                __m128i p2=_mm_and_si128(_mm_srli_epi16(by,4),m2), p3=_mm_and_si128(_mm_srli_epi16(by,6),m2);
                __m128i lo=_mm_unpacklo_epi8(p0,p1), hi=_mm_unpacklo_epi8(p2,p3);
                __m128i nib=_mm_unpacklo_epi16(lo,hi);                                      /* 16 valori in ordine */
                __m256 w0=_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(nib),b2));
                __m256 w1=_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(nib,8)),b2));
                acc=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i),   w0, acc);
                acc=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i+8), w1, acc); }
            a=hsum256(acc);
#elif defined(__ARM_NEON)
            const uint8x8_t m2v=vdup_n_u8(3); const int8x8_t b2v=vdup_n_s8(2);
            float32x4_t ac0=vdupq_n_f32(0), ac1=vdupq_n_f32(0);
            for(;i+16<=I;i+=16){ uint32_t wd; memcpy(&wd, w+(i>>2), 4);        /* 4 byte=16 valori */
                uint8x8_t by=vreinterpret_u8_u32(vdup_n_u32(wd));
                uint8x8x2_t z01=vzip_u8(vand_u8(by,m2v),              vand_u8(vshr_n_u8(by,2),m2v));
                uint8x8x2_t z23=vzip_u8(vand_u8(vshr_n_u8(by,4),m2v), vshr_n_u8(by,6));
                uint16x4x2_t zz=vzip_u16(vreinterpret_u16_u8(z01.val[0]), vreinterpret_u16_u8(z23.val[0]));
                int16x8_t w0=vmovl_s8(vsub_s8(vreinterpret_s8_u16(zz.val[0]),b2v));  /* 16 valori in ordine */
                int16x8_t w1=vmovl_s8(vsub_s8(vreinterpret_s8_u16(zz.val[1]),b2v));
                ac0=vfmaq_f32(ac0, vld1q_f32(xs+i),    vcvtq_f32_s32(vmovl_s16(vget_low_s16(w0))));
                ac1=vfmaq_f32(ac1, vld1q_f32(xs+i+4),  vcvtq_f32_s32(vmovl_s16(vget_high_s16(w0))));
                ac0=vfmaq_f32(ac0, vld1q_f32(xs+i+8),  vcvtq_f32_s32(vmovl_s16(vget_low_s16(w1))));
                ac1=vfmaq_f32(ac1, vld1q_f32(xs+i+12), vcvtq_f32_s32(vmovl_s16(vget_high_s16(w1)))); }
            a=vaddvq_f32(vaddq_f32(ac0,ac1));
#endif
            for(;i<I;i++){ uint8_t byte=w[i>>2]; int sh=(i&3)*2; a += xs[i]*(float)((int)((byte>>sh)&3)-2); }
            y[(int64_t)s*O+o]=a*sc; } }
}
/* ---- KERNEL INTERI (IDOT): attivazioni quantizzate a int8 per riga (absmax/127,
 * stile Q8_0), prodotto scalare INTERO via maddubs/madd AVX2 — niente conversione
 * f32 dei pesi nel ciclo caldo. ~2-3x sui matmul quantizzati; errore aggiunto ~0.3%
 * RMS per matmul (attivazione int8), IDOT=0 torna al percorso f32 esatto. */
#if defined(__AVX512VNNI__) && defined(__AVX512BW__)
#define IDOT_KERNEL "avx512-vnni"
#elif defined(__AVX2__)
#define IDOT_KERNEL "avx2"
#elif defined(__ARM_NEON)
#define IDOT_KERNEL "neon"
#else
#define IDOT_KERNEL "scalar"
#endif
static int g_idot=1;
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
static int g_i4s=1;   /* SDOT presente: int4 IDOT conviene anche a S=1 (decode). Misurato
                       * su Apple M-series: +14%%, expert-matmul -16%%. EN: with SDOT, int4
                       * IDOT pays even at S=1 (decode); measured on Apple M-series. */
#else
static int g_i4s=2;   /* senza SDOT / altrove: soglia originale (misura AVX2 dell'autore).
                       * EN: without SDOT / elsewhere: original threshold (author's AVX2). */
#endif
static inline float qrow_i8(const float *x, int8_t *q, int I){
    float amax=0; for(int i=0;i<I;i++){ float a=fabsf(x[i]); if(a>amax)amax=a; }
    float s=amax/127.f; if(s<1e-12f) s=1e-12f; float inv=1.f/s;
    for(int i=0;i<I;i++) q[i]=(int8_t)lrintf(x[i]*inv);
    return s;
}
#ifdef __AVX2__
static inline int hsum256_i32(__m256i v){
    __m128i lo=_mm256_castsi256_si128(v), hi=_mm256_extracti128_si256(v,1);
    lo=_mm_add_epi32(lo,hi); lo=_mm_hadd_epi32(lo,lo); lo=_mm_hadd_epi32(lo,lo);
    return _mm_cvtsi128_si32(lo);
}
#endif
/* dot int8·int8: trucco del segno (|w| unsigned × x·sign(w) signed). Sicuro:
 * coppie <= 128*127*2 = 32512 < 32767, accumulo s32 fino a I=16384. */
static inline int32_t dot_i8i8(const int8_t *w, const int8_t *x, int I){
    int32_t sum=0; int i=0;
#if defined(__AVX512VNNI__) && defined(__AVX512BW__)
    /* VNNI: vpdpbusd u8*s8 -> s32 directly, 64 bytes/iter, no 16-bit intermediate.
     * AVX-512 has no vpsignb: |w| via abs, sign folded into x with a mask-negate
     * (w==0 -> product 0 either way). |x|<=127 (qrow_i8), |w|<=128 as u8: each
     * s32 lane adds <= 4*128*127, safe up to I=16384 like the AVX2 bound. */
    __m512i acc=_mm512_setzero_si512();
    for(;i+64<=I;i+=64){
        __m512i wv=_mm512_loadu_si512((const void*)(w+i));
        __m512i xv=_mm512_loadu_si512((const void*)(x+i));
        __mmask64 neg=_mm512_movepi8_mask(wv);
        __m512i xs=_mm512_mask_sub_epi8(xv,neg,_mm512_setzero_si512(),xv);
        acc=_mm512_dpbusd_epi32(acc,_mm512_abs_epi8(wv),xs);
    }
    sum=_mm512_reduce_add_epi32(acc);
#elif defined(__AVX2__)
    __m256i acc=_mm256_setzero_si256(); const __m256i ones=_mm256_set1_epi16(1);
    for(;i+32<=I;i+=32){
        __m256i wv=_mm256_loadu_si256((const __m256i*)(w+i));
        __m256i xv=_mm256_loadu_si256((const __m256i*)(x+i));
        __m256i p=_mm256_maddubs_epi16(_mm256_sign_epi8(wv,wv),_mm256_sign_epi8(xv,wv));
        acc=_mm256_add_epi32(acc,_mm256_madd_epi16(p,ones));
    }
    sum=hsum256_i32(acc);
#elif defined(__ARM_NEON)
    /* ARM: SDOT nativo se disponibile (Apple Silicon: sempre); altrimenti vmull/vpadal.
     * Stesso bound anti-overflow del trucco AVX2: coppie <= 128*127*2 = 32512 < 32767. */
    int32x4_t acc=vdupq_n_s32(0);
    for(;i+16<=I;i+=16){
        int8x16_t wv=vld1q_s8(w+i), xv=vld1q_s8(x+i);
#if defined(__ARM_FEATURE_DOTPROD)
        acc=vdotq_s32(acc,wv,xv);
#else
        int16x8_t p=vmull_s8(vget_low_s8(wv),vget_low_s8(xv));
        p=vmlal_s8(p,vget_high_s8(wv),vget_high_s8(xv));
        acc=vpadalq_s16(acc,p);
#endif
    }
    sum=vaddvq_s32(acc);
#endif
    for(;i<I;i++) sum+=(int32_t)w[i]*x[i];
    return sum;
}
/* dot int4(packed)·int8: nibble -> int8 [-8,7] al volo, poi stesso trucco */
static inline int32_t dot_i4i8(const uint8_t *w4, const int8_t *x, int I){
    int32_t sum=0; int i=0;
#if defined(__AVX512VNNI__) && defined(__AVX512BW__)
    /* 32 bytes = 64 nibbles -> int8 in [-8,7], one vpdpbusd per 64 values.
     * 256-bit unpack leaves values in per-128-lane order [0-15][32-47]/[16-31][48-63];
     * dot pairing is order-invariant, so permute x's 128-bit blocks to match
     * instead of re-ordering w (one vpermq per iter, off the critical unpack path). */
    const __m256i m4v=_mm256_set1_epi8(0x0F);
    const __m512i b8v=_mm512_set1_epi8(8);
    const __m512i xidx=_mm512_setr_epi64(0,1,4,5,2,3,6,7);
    __m512i acc=_mm512_setzero_si512();
    for(;i+64<=I;i+=64){
        __m256i by=_mm256_loadu_si256((const __m256i*)(w4+(i>>1)));
        __m256i lo=_mm256_and_si256(by,m4v), hi=_mm256_and_si256(_mm256_srli_epi16(by,4),m4v);
        __m256i z0=_mm256_unpacklo_epi8(lo,hi), z1=_mm256_unpackhi_epi8(lo,hi);
        __m512i wv=_mm512_sub_epi8(_mm512_inserti64x4(_mm512_castsi256_si512(z0),z1,1),b8v);
        __m512i xv=_mm512_permutexvar_epi64(xidx,_mm512_loadu_si512((const void*)(x+i)));
        __mmask64 neg=_mm512_movepi8_mask(wv);
        __m512i xs=_mm512_mask_sub_epi8(xv,neg,_mm512_setzero_si512(),xv);
        acc=_mm512_dpbusd_epi32(acc,_mm512_abs_epi8(wv),xs);
    }
    sum=_mm512_reduce_add_epi32(acc);
#elif defined(__AVX2__)
    const __m128i m4=_mm_set1_epi8(0x0F); const __m256i b8=_mm256_set1_epi8(8);
    const __m256i ones=_mm256_set1_epi16(1);
    __m256i acc=_mm256_setzero_si256();
    for(;i+32<=I;i+=32){
        __m128i by=_mm_loadu_si128((const __m128i*)(w4+(i>>1)));   /* 16 byte = 32 nibble */
        __m128i lo=_mm_and_si128(by,m4), hi=_mm_and_si128(_mm_srli_epi16(by,4),m4);
        __m128i n0=_mm_unpacklo_epi8(lo,hi), n1=_mm_unpackhi_epi8(lo,hi);   /* in ordine */
        __m256i wv=_mm256_sub_epi8(_mm256_set_m128i(n1,n0),b8);
        __m256i xv=_mm256_loadu_si256((const __m256i*)(x+i));
        __m256i p=_mm256_maddubs_epi16(_mm256_sign_epi8(wv,wv),_mm256_sign_epi8(xv,wv));
        acc=_mm256_add_epi32(acc,_mm256_madd_epi16(p,ones));
    }
    sum=hsum256_i32(acc);
#elif defined(__ARM_NEON)
    const uint8x16_t m4q=vdupq_n_u8(0x0F); const int8x16_t b8q=vdupq_n_s8(8);
    int32x4_t acc=vdupq_n_s32(0);
    for(;i+32<=I;i+=32){
        uint8x16_t by=vld1q_u8(w4+(i>>1));                          /* 16 byte = 32 nibble */
        uint8x16x2_t z=vzipq_u8(vandq_u8(by,m4q), vshrq_n_u8(by,4)); /* nibble in ordine */
        int8x16_t w0=vsubq_s8(vreinterpretq_s8_u8(z.val[0]),b8q);
        int8x16_t w1=vsubq_s8(vreinterpretq_s8_u8(z.val[1]),b8q);
        int8x16_t x0=vld1q_s8(x+i), x1=vld1q_s8(x+i+16);
#if defined(__ARM_FEATURE_DOTPROD)
        acc=vdotq_s32(acc,w0,x0); acc=vdotq_s32(acc,w1,x1);
#else
        int16x8_t p=vmull_s8(vget_low_s8(w0),vget_low_s8(x0));      /* |w|<=8: nessun overflow */
        p=vmlal_s8(p,vget_high_s8(w0),vget_high_s8(x0));
        acc=vpadalq_s16(acc,p);
        p=vmull_s8(vget_low_s8(w1),vget_low_s8(x1));
        p=vmlal_s8(p,vget_high_s8(w1),vget_high_s8(x1));
        acc=vpadalq_s16(acc,p);
#endif
    }
    sum=vaddvq_s32(acc);
#endif
    for(;i+1<I;i+=2){ uint8_t b=w4[i>>1]; sum+=((int)(b&0xF)-8)*x[i]+((int)(b>>4)-8)*x[i+1]; }
    if(i<I){ uint8_t b=w4[i>>1]; sum+=((int)(b&0xF)-8)*x[i]; }
    return sum;
}
static void matmul_q_idot(float *y, const int8_t *xq, const float *sx, const int8_t *q,
                          const float *scale, int S, int I, int O){
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){ const int8_t *w=q+(int64_t)o*I; float sc=scale[o];
        for(int s=0;s<S;s++) y[(int64_t)s*O+o]=(float)dot_i8i8(w,xq+(int64_t)s*I,I)*sc*sx[s]; }
}
static void matmul_i4_idot(float *y, const int8_t *xq, const float *sx, const uint8_t *q4,
                           const float *scale, int S, int I, int O){
    int rb=(I+1)/2;
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){ const uint8_t *w=q4+(int64_t)o*rb; float sc=scale[o];
        for(int s=0;s<S;s++) y[(int64_t)s*O+o]=(float)dot_i4i8(w,xq+(int64_t)s*I,I)*sc*sx[s]; }
}

/* LOSSLESS SPEC (bugfix col gate MTP "DRAFT=0 vs 2 byte-identici"): la scelta del
 * kernel int4 dipende da S (S>=g_i4s -> IDOT intero, S=1 -> dequant f32 esatto). Il
 * forward di VERIFICA speculativa e' un batch S=1+g: senza pin le sue logits differiscono
 * da quelle del decode sequenziale S=1 e un argmax al limite puo' flippare -> il greedy
 * con DRAFT>0 non era piu' byte-identico. g_fw1=1 (attivo solo durante la verifica)
 * valuta la soglia come se S=1: ogni riga del batch riproduce ESATTAMENTE il forward
 * singolo (tutti i kernel sono per-riga indipendenti, quindi basta pareggiare la scelta).
 * EN: the int4 kernel choice is S-dependent; the speculative verify batch must reproduce
 * S=1 numerics bit-exactly or greedy with DRAFT>0 diverges. g_fw1 pins the choice to S=1. */
static int g_fw1=0;

/* Thread-local quant scratch (ported from colibri #43): IDOT path used to
 * malloc/free S*I bytes on every matmul_qt. On decode that is thousands of
 * allocs/token on the CPU expert path. Grow-only per-thread buffers. */
typedef struct { int8_t *xq; size_t xq_cap; float *sx; size_t sx_cap; } QScratch;
static _Thread_local QScratch g_qscratch;
static void quant_scratch(size_t xn, size_t sn, int8_t **xq, float **sx){
    if(xn>g_qscratch.xq_cap){
        int8_t *p=realloc(g_qscratch.xq,xn);
        if(!p){ fprintf(stderr,"OOM quant scratch xq\n"); exit(1); }
        g_qscratch.xq=p; g_qscratch.xq_cap=xn;
    }
    if(sn>g_qscratch.sx_cap){
        float *p=realloc(g_qscratch.sx,sn*sizeof(float));
        if(!p){ fprintf(stderr,"OOM quant scratch sx\n"); exit(1); }
        g_qscratch.sx=p; g_qscratch.sx_cap=sn;
    }
    *xq=g_qscratch.xq; *sx=g_qscratch.sx;
}

static const void *qt_wptr(const QT *w){
    if(w->fmt==0) return (const void*)w->qf;
    if(w->fmt==1) return (const void*)w->q8;
    return (const void*)w->q4;   /* int4 and int2 share q4 packing */
}
#ifdef COLI_CUDA
/* Fused SwiGLU on GPU. flags: COLI_CUDA_SWIGLU_REUSE_X for sticky device-x (decode). */
static int swiglu_qt(float *y, const float *x, QT *g, QT *u, QT *d, int S, int flags){
    if(!g_cuda_enabled || S<1) return 0;
    if(!g->cuda_eligible||!u->cuda_eligible||!d->cuda_eligible) return 0;
    if(g->cuda_failed||u->cuda_failed||d->cuda_failed) return 0;
    if(g->cuda_device!=u->cuda_device||g->cuda_device!=d->cuda_device) return 0;
    if(omp_in_parallel()) return 0;
    int D=g->I, I=g->O;
    if(u->I!=D||u->O!=I||d->I!=I||d->O!=D) return 0;
    if(coli_cuda_swiglu(&g->cuda,&u->cuda,&d->cuda,y,x,
                        qt_wptr(g),g->s,g->fmt,
                        qt_wptr(u),u->s,u->fmt,
                        qt_wptr(d),d->s,d->fmt,
                        S,D,I,g->cuda_device,flags))
        return 1;
    if(g->gpu_only||u->gpu_only||d->gpu_only){
        fprintf(stderr,"[CUDA] fatal: fused SwiGLU failed on gpu_only expert/dense device %d\n",
                g->cuda_device);
        exit(1);
    }
    g->cuda_failed=u->cuda_failed=d->cuda_failed=1;
    coli_cuda_x_invalidate(g->cuda_device);
    fprintf(stderr,"[CUDA] fused SwiGLU failed device %d; falling back to 3x matmul\n",
            g->cuda_device);
    return 0;
}
/* True if all three expert tensors live on one CUDA device and can use moe_acc. */
static int expert_gpu_ready(const ESlot *e){
    if(!g_cuda_enabled) return 0;
    if(!e->g.cuda_eligible||!e->u.cuda_eligible||!e->d.cuda_eligible) return 0;
    if(e->g.cuda_failed||e->u.cuda_failed||e->d.cuda_failed) return 0;
    if(e->g.cuda_device!=e->u.cuda_device||e->g.cuda_device!=e->d.cuda_device) return 0;
    if(!(e->g.cuda && e->u.cuda && e->d.cuda) && !(qt_wptr(&e->g)||qt_wptr(&e->u))) return 0;
    return 1;
}
static int moe_acc_qt(ESlot *e, float w, int S){
    QT *g=&e->g,*u=&e->u,*d=&e->d;
    int D=g->I, I=g->O;
    if(u->I!=D||u->O!=I||d->I!=I||d->O!=D) return 0;
    return coli_cuda_moe_acc(&g->cuda,&u->cuda,&d->cuda,
                             qt_wptr(g),g->s,g->fmt,
                             qt_wptr(u),u->s,u->fmt,
                             qt_wptr(d),d->s,d->fmt,
                             w,S,D,I,g->cuda_device);
}
#endif
static void matmul_qt(float *y, const float *x, QT *w, int S){
#ifdef COLI_CUDA
    /* VRAM-only tensor (dense freed after eager upload, or complementary expert
     * tier): host qf/q8/q4/s are NULL. Weights live ONLY on the GPU. Never fall
     * through to the CPU path on a freed slab — that was a silent-zero / SEGV
     * footgun. On failure with no host copy: fail loud (exit), never return
     * untouched y. Guard is independent of omp_in_parallel(). */
    if(g_cuda_enabled && w->cuda_eligible && !w->cuda_failed && w->gpu_only){
        if(omp_in_parallel()){
            fprintf(stderr,"[CUDA] fatal: gpu_only matmul inside OpenMP [%d,%d] device %d\n",
                w->O,w->I,w->cuda_device);
            exit(1);
        }
        const void *weights = w->fmt==0 ? (const void*)w->qf
                            : w->fmt==1 ? (const void*)w->q8 : (const void*)w->q4;
        if(coli_cuda_matmul(&w->cuda,y,x,weights,w->s,w->fmt,S,w->I,w->O,w->cuda_device)) return;
        w->cuda_failed=1;
        if(!(w->qf||w->q8||w->q4)){
            fprintf(stderr,"[CUDA] fatal: gpu_only matmul failed and host slab freed [%d,%d] device %d\n",
                w->O,w->I,w->cuda_device);
            exit(1);
        }
        /* host still present (upload-and-free skipped): fall through to CPU */
    } else if(g_cuda_enabled && w->cuda_eligible && !w->cuda_failed && !omp_in_parallel()){
        /* Resident tensors that still have a host copy (lazy path / not freed).
         * Nested OpenMP stays on CPU: one synchronous scratch stream per device. */
        const void *weights = w->fmt==0 ? (const void*)w->qf
                            : w->fmt==1 ? (const void*)w->q8 : (const void*)w->q4;
        if(coli_cuda_matmul(&w->cuda,y,x,weights,w->s,w->fmt,S,w->I,w->O,w->cuda_device)) return;
        w->cuda_failed=1;
        fprintf(stderr,"[CUDA] tensor [%d,%d] on device %d disabled after error; CPU fallback\n",
            w->O,w->I,w->cuda_device);
    }
#endif
    if(w->fmt==0){ matmul(y,x,w->qf,S,w->I,w->O); return; }
    /* int8 IDOT vince sempre (1.4-2.5x). int4 IDOT: l'autore su AVX2 trovo' che a S=1
     * non ripaga (soglia S>=2); ma su ARM/SDOT il singolo token CONVIENE (vedi g_i4s /
     * PR #9 per il gemello VNNI). Soglia configurabile con I4S.
     * EN: int8 IDOT always wins (1.4-2.5x). int4 IDOT: on AVX2 the author found S=1 didn't
     * pay (S>=2 gate); on ARM/SDOT single-token DOES pay (see g_i4s / PR #9 for the VNNI
     * twin). Threshold configurable via I4S. */
    if(g_idot && (w->fmt==1 || (w->fmt==2 && (g_fw1?1:S)>=g_i4s))){
        int I=w->I; int8_t *xq; float *sx;
        if(S<0 || I<0 || (size_t)S>SIZE_MAX/(size_t)(I?I:1)){
            fprintf(stderr,"matmul_qt: shape overflow S=%d I=%d\n",S,I); exit(1);
        }
        quant_scratch((size_t)S*(size_t)I,(size_t)S,&xq,&sx);
        for(int s=0;s<S;s++) sx[s]=qrow_i8(x+(int64_t)s*I, xq+(int64_t)s*I, I);
        if(w->fmt==1) matmul_q_idot(y,xq,sx,w->q8,w->s,S,I,w->O);
        else matmul_i4_idot(y,xq,sx,w->q4,w->s,S,I,w->O);
        return;
    }
    if(w->fmt==1) matmul_q(y,x,w->q8,w->s,S,w->I,w->O);
    else if(w->fmt==3) matmul_i2(y,x,w->q4,w->s,S,w->I,w->O);
    else matmul_i4(y,x,w->q4,w->s,S,w->I,w->O);
}

/* quantizza w[O,I] f32 -> int8 q[O,I] + scala[O] simmetrica per riga */
static void quantize_rows(const float *w, int8_t *q, float *scale, int O, int I, int bits){
    int qmax=(1<<(bits-1))-1;
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){ const float *wr=w+(int64_t)o*I; float amax=0;
        for(int i=0;i<I;i++){ float a=fabsf(wr[i]); if(a>amax)amax=a; }
        float s=amax/qmax; if(s<1e-8f)s=1e-8f; scale[o]=s;
        int8_t *qr=q+(int64_t)o*I;
        for(int i=0;i<I;i++){ int v=(int)lrintf(wr[i]/s); if(v>qmax)v=qmax; if(v<-qmax-1)v=-qmax-1; qr[i]=(int8_t)v; }
    }
}
/* quantizza w[O,I] f32 -> int4 impacchettato (2/byte) + scala[O].
 * bits<=4: valori in [-qmax-1,qmax] stanno in un nibble [-8,7]; memorizzati come v+8 (0..15). */
static void pack_int4(const float *w, uint8_t *q4, float *scale, int O, int I, int bits){
    int qmax=(1<<(bits-1))-1, rb=(I+1)/2;
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){ const float *wr=w+(int64_t)o*I; float amax=0;
        for(int i=0;i<I;i++){ float a=fabsf(wr[i]); if(a>amax)amax=a; }
        float s=amax/qmax; if(s<1e-8f)s=1e-8f; scale[o]=s;
        uint8_t *qr=q4+(int64_t)o*rb;
        for(int i=0;i<I;i+=2){
            int v0=(int)lrintf(wr[i]/s); if(v0>qmax)v0=qmax; if(v0<-8)v0=-8;
            int v1=0; if(i+1<I){ v1=(int)lrintf(wr[i+1]/s); if(v1>qmax)v1=qmax; if(v1<-8)v1=-8; }
            qr[i>>1] = (uint8_t)((v0+8) | ((v1+8)<<4));
        }
    }
}

/* quantizza w[O,I] f32 -> int2 impacchettato (4/byte) + scala[O]. valori nibble 2-bit in [-2,1]. */
static void pack_int2(const float *w, uint8_t *q2, float *scale, int O, int I, int bits){
    int qmax=(1<<(bits-1))-1, rb=(I+3)/4;
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){ const float *wr=w+(int64_t)o*I; float amax=0;
        for(int i=0;i<I;i++){ float a=fabsf(wr[i]); if(a>amax)amax=a; }
        float s=amax/qmax; if(s<1e-8f)s=1e-8f; scale[o]=s;
        uint8_t *qr=q2+(int64_t)o*rb;
        for(int i=0;i<I;i+=4){ uint8_t byte=0;
            for(int k=0;k<4 && i+k<I;k++){ int v=(int)lrintf(wr[i+k]/s); if(v>qmax)v=qmax; if(v<-2)v=-2; byte|=(uint8_t)((v+2)<<(k*2)); }
            qr[i>>2]=byte;
        }
    }
}

static int g_nopack=0;   /* NOPACK=1 -> tiene i valori <=4bit in contenitore int8 (per validare il packing) */
static int g_drop=0;     /* DROP=1 -> scarta le pagine expart dopo l'uso. Default 0: le lascia in
                          * page-cache (buff/cache, NON RSS) come L2 gratuito -> sfrutta lo
                          * sbilanciamento del routing MoE (pochi expert "caldi" riusati). */
static int g_prefetch=-1; /* PREFETCH sticky same-layer WILLNEED for next token (enr/eroute).
                           * -1=auto ON with SERVE/PILOT; 0=off; 1=force on. */
static int g_direct=0;   /* DIRECT=1 -> O_DIRECT sugli slab expert. Default OFF: su questo host
                          * (VHDX su NVMe DRAM-less, latenza serializzata ~60ms/req) il buffered
                          * liscio e' risultato il migliore; su NVMe veri DIRECT=1 rende di piu'. */
static float g_temp=-1;  /* TEMP: temperatura di sampling sui TOKEN. <0 = auto (1.0 in chat/testo,
                          * 0=greedy in validazione). 0 = greedy puro. */
static float g_nuc=0.95f;/* NUCLEUS: top-p sul vocabolario (0.95 = generation_config di
                          * MiMo-V2.5, verificato: temperature=1.0, top_p=0.95) */
static int g_topk=0;     /* TOPK=n -> usa n expert/token invece di config (ricerca: meno disco) */
static float g_topp=0;   /* TOPP=p (0..1) -> top-p adattivo: tieni gli expert fino a peso cumulato p */
static int g_spec=1;     /* metodo C: SPEC=0 disabilita il prefetch speculativo cross-layer */
static int g_overlap=1;  /* OVERLAP=0 -> disattiva la pipeline load/compute dentro il layer
                          * (torna alle due fasi serializzate: prima tutti i load, poi i matmul) */
static int g_overlap_t=4;/* OVERLAP_T: thread del pool di load asincroni. 4 misurato migliore
                          * di 8 su WSL/VHDX (meno contesa col team OpenMP dei matmul) */
/* g_draft is declared above (near Model) — needed by cuda_upload_dense_all. */
/* Choose format from `bits`: >=16 f32, 5..8 int8, <=4 int4-packed */
static void qt_alloc(QT *t, int O, int I, int bits){
    t->O=O; t->I=I; t->qf=NULL; t->q8=NULL; t->q4=NULL; t->s=NULL;
    if(bits>=16){ t->fmt=0; t->qf=falloc((int64_t)O*I); }
    else if(bits>=5 || g_nopack){ t->fmt=1; t->q8=malloc((int64_t)O*I); t->s=falloc(O); }
    else if(bits>=3){ t->fmt=2; t->q4=malloc((int64_t)O*((I+1)/2)); t->s=falloc(O); }
    else { t->fmt=3; t->q4=malloc((int64_t)O*((I+3)/4)); t->s=falloc(O); }
}
static void qt_fill(QT *t, const float *w, int bits){
    if(t->fmt==0) memcpy(t->qf, w, (int64_t)t->O*t->I*sizeof(float));
    else if(t->fmt==1) quantize_rows(w, t->q8, t->s, t->O, t->I, bits);
    else if(t->fmt==3) pack_int2(w, t->q4, t->s, t->O, t->I, bits);
    else pack_int4(w, t->q4, t->s, t->O, t->I, bits);
}

static void rmsnorm(float *out, const float *x, const float *w, int D, float eps){
    double ms=0; for(int i=0;i<D;i++) ms+=(double)x[i]*x[i];
    float r=1.f/sqrtf((float)(ms/D)+eps); for(int i=0;i<D;i++) out[i]=x[i]*r*w[i];
}
static void softmax(float *x,int n){ float m=-1e30f; for(int i=0;i<n;i++) if(x[i]>m)m=x[i];
    float s=0; for(int i=0;i<n;i++){x[i]=expf(x[i]-m);s+=x[i];} for(int i=0;i<n;i++) x[i]/=s; }
static inline float sigmoidf(float x){ return 1.f/(1.f+expf(-x)); }
static inline float siluf(float x){ return x/(1.f+expf(-x)); }

/* ---------- config ---------- */
static jval* cfg_root(const char *snap, char **arena){
    char p[2048]; snprintf(p,sizeof(p),"%s/config.json",snap);
    FILE *f=fopen(p,"rb"); if(!f){perror(p);exit(1);}
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char *b=malloc(n+1); if(fread(b,1,n,f)!=(size_t)n){} b[n]=0; fclose(f);
    return json_parse(b,arena);
}
static int gi(jval*r,const char*k){ jval*v=json_get(r,k); return v?(int)v->num:0; }
/* legge un array 0/1 per-layer (hybrid_layer_pattern / moe_layer_freq) in dst[]; assente -> tutti 0 */
static void load_pattern(jval *r, const char *key, int n_layers, int8_t *dst){
    memset(dst,0,128);
    jval *a=json_get(r,key); if(!a || a->t!=J_ARR) return;
    if(a->len!=n_layers){ fprintf(stderr,"config: %s ha %d elementi, num_hidden_layers=%d\n",key,a->len,n_layers); exit(1); }
    for(int i=0;i<n_layers && i<128;i++) dst[i]=(int8_t)(a->kids[i]->num!=0);
}
static void load_cfg(Cfg *c, const char *snap){
    char *ar=NULL; jval *r=cfg_root(snap,&ar); jval *v;
    c->hidden=gi(r,"hidden_size"); c->n_layers=gi(r,"num_hidden_layers");
    c->n_heads=gi(r,"num_attention_heads");
    c->kv_heads_full=gi(r,"num_key_value_heads");
    v=json_get(r,"swa_num_key_value_heads"); c->kv_heads_swa = v&&v->t==J_NUM ? (int)v->num : c->kv_heads_full;
    v=json_get(r,"head_dim");     c->head_dim   = v&&v->t==J_NUM ? (int)v->num : (c->n_heads>0 ? c->hidden/c->n_heads : 0);
    v=json_get(r,"v_head_dim");   c->v_head_dim = v&&v->t==J_NUM ? (int)v->num : c->head_dim;
    /* l'oracolo lo conferma: i layer SWA hanno teste TUTTE loro (swa_head_dim/swa_v_head_dim) */
    v=json_get(r,"swa_head_dim");   c->swa_head_dim   = v&&v->t==J_NUM ? (int)v->num : c->head_dim;
    v=json_get(r,"swa_v_head_dim"); c->swa_v_head_dim = v&&v->t==J_NUM ? (int)v->num : c->v_head_dim;
    /* i campi rope: transformers li serializza SIA al top level SIA dentro rope_parameters */
    jval *rp=json_get(r,"rope_parameters");
    v=json_get(r,"partial_rotary_factor"); if((!v||v->t!=J_NUM)&&rp) v=json_get(rp,"partial_rotary_factor");
    float prf = v&&v->t==J_NUM ? (float)v->num : 1.0f;
    c->rope_dim=(int)(c->head_dim*prf);
    if(c->rope_dim&1){ fprintf(stderr,"config: rope_dim=%d dispari (head_dim=%d x partial_rotary_factor=%g)\n",
        c->rope_dim,c->head_dim,(double)prf); exit(1); }
    v=json_get(r,"sliding_window"); c->sliding_window = v&&v->t==J_NUM ? (int)v->num : 0;   /* 0 = niente SWA */
    v=json_get(r,"rope_theta"); if((!v||v->t!=J_NUM)&&rp) v=json_get(rp,"rope_theta");
    c->theta_full = v&&v->t==J_NUM ? (float)v->num : 10000.f;
    v=json_get(r,"swa_rope_theta"); c->theta_swa = v&&v->t==J_NUM ? (float)v->num : c->theta_full;
    v=json_get(r,"attention_value_scale"); c->v_scale = v&&v->t==J_NUM ? (float)v->num : 1.0f;
    v=json_get(r,"add_full_attention_sink_bias"); c->has_sink_full = (v&&v->t==J_BOOL)?(int8_t)v->boolean:0;
    v=json_get(r,"add_swa_attention_sink_bias");  c->has_sink_swa  = (v&&v->t==J_BOOL)?(int8_t)v->boolean:0;
    v=json_get(r,"layernorm_epsilon"); c->eps = v&&v->t==J_NUM ? (float)v->num : 1e-5f;   /* NB: non rms_norm_eps */
    load_pattern(r,"hybrid_layer_pattern",c->n_layers,c->is_swa);
    load_pattern(r,"moe_layer_freq",c->n_layers,c->is_moe);
    c->n_experts=gi(r,"n_routed_experts"); c->topk=gi(r,"num_experts_per_tok");
    c->moe_inter=gi(r,"moe_intermediate_size"); c->dense_inter=gi(r,"intermediate_size");
    c->n_group=gi(r,"n_group"); c->topk_group=gi(r,"topk_group");
    jval *nt=json_get(r,"norm_topk_prob"); c->norm_topk=(nt&&nt->t==J_BOOL)?nt->boolean:0;
    jval *rs=json_get(r,"routed_scaling_factor"); c->routed_scale=(rs&&rs->t==J_NUM)?(float)rs->num:1.f;   /* null -> 1 */
    c->vocab=gi(r,"vocab_size");
    /* LAYOUT DEL QKV FUSO (il bug " Paris"->"00.0" del 2026-07-11): il checkpoint FP8
     * ufficiale di MiMo-V2.5 memorizza qkv_proj.weight INTERLEAVED per rank TP
     * ([Q|K|V] ripetuto NR=4 volte, Q = (H/NR)*hd righe per rank), mentre
     * modeling_mimo_v2.py (e i nostri modelli tiny/fixture, salvati DA quel codice)
     * usano lo split piatto [Q|K|V]. vLLM lo documenta e de-interleava al load
     * (_shard_fp8_qkv_proj in vllm/model_executor/models/mimo_v2.py). Il marcatore
     * affidabile e' la provenienza: solo l'export fp8 di Xiaomi scrive il layout
     * raggruppato, e solo lui mette quantization_config.quant_method=="fp8" nel
     * config.json (che il convertitore copia nel container). QKV_GROUPED=0/1 forza.
     * EN: MiMo-V2.5's official FP8 checkpoint stores fused qkv interleaved per TP
     * rank; the HF modeling (and our oracles, saved by it) use the flat layout. vLLM
     * de-interleaves at load. Provenance marker: quantization_config.quant_method
     * =="fp8". Env QKV_GROUPED overrides. */
    { jval *qc=json_get(r,"quantization_config");
      jval *qm=qc?json_get(qc,"quant_method"):NULL;
      c->qkv_grouped = (qm && qm->t==J_STR && qm->str && !strcmp(qm->str,"fp8"));
      if(getenv("QKV_GROUPED")) c->qkv_grouped=atoi(getenv("QKV_GROUPED")); }
    /* token di stop: numero singolo o array */
    c->n_stop=0;
    jval *eo=json_get(r,"eos_token_id");
    if(eo){ if(eo->t==J_NUM) c->stop_ids[c->n_stop++]=(int)eo->num;
            else if(eo->t==J_ARR) for(int i=0;i<eo->len && c->n_stop<8;i++)
                c->stop_ids[c->n_stop++]=(int)eo->kids[i]->num; }
    if(c->n_group!=1){ fprintf(stderr,"questo motore assume n_group=1 (MiMo-V2.5)\n"); exit(1); }
    /* VALIDAZIONE (report PR #25): il config.json arriva da mirror non fidati — dimensioni
     * ostili non devono superare questo punto. Un solo choke point protegge ogni alloc a valle. */
    #define CKR(name,v,lo,hi) if((v)<(lo)||(v)>(hi)){ \
        fprintf(stderr,"config: %s=%d fuori range [%d,%d]\n",name,(int)(v),(int)(lo),(int)(hi)); exit(1); }
    CKR("hidden_size",c->hidden,1,1<<20)         CKR("num_hidden_layers",c->n_layers,1,128)
    CKR("num_attention_heads",c->n_heads,1,1024)
    CKR("num_key_value_heads",c->kv_heads_full,1,1024)
    CKR("swa_num_key_value_heads",c->kv_heads_swa,1,1024)
    CKR("head_dim",c->head_dim,1,1<<16)          CKR("v_head_dim",c->v_head_dim,1,1<<16)
    CKR("swa_head_dim",c->swa_head_dim,1,1<<16)  CKR("swa_v_head_dim",c->swa_v_head_dim,1,1<<16)
    CKR("rope_dim",c->rope_dim,2,c->head_dim)    CKR("sliding_window",c->sliding_window,0,1<<20)
    CKR("n_routed_experts",c->n_experts,1,4096)  CKR("num_experts_per_tok",c->topk,1,64)
    CKR("moe_intermediate_size",c->moe_inter,1,1<<24) CKR("intermediate_size",c->dense_inter,1,1<<24)
    CKR("vocab_size",c->vocab,1,1<<24)
    #undef CKR
    if(c->n_heads % c->kv_heads_full){ fprintf(stderr,"config: num_key_value_heads=%d non divide num_attention_heads=%d\n",c->kv_heads_full,c->n_heads); exit(1); }
    if(c->n_heads % c->kv_heads_swa){ fprintf(stderr,"config: swa_num_key_value_heads=%d non divide num_attention_heads=%d\n",c->kv_heads_swa,c->n_heads); exit(1); }
    /* un layer marcato SWA con finestra 0 degraderebbe SILENZIOSAMENTE a full attention:
     * meglio un errore di config che risultati sbagliati senza avviso */
    for(int i=0;i<c->n_layers;i++) if(c->is_swa[i] && c->sliding_window<=0){
        fprintf(stderr,"config: hybrid_layer_pattern marca il layer %d SWA ma sliding_window=%d\n",i,c->sliding_window); exit(1); }
    /* rope_dim derivato per tipo di layer (riscala intera da head_dim): rope_neox assume coppie */
    { int rd_full=c->rope_dim, rd_swa=(int)((int64_t)c->swa_head_dim*c->rope_dim/c->head_dim);
      if(rd_full%2 || rd_swa%2){ fprintf(stderr,"config: rope_dim derivato dispari (full=%d swa=%d)\n",rd_full,rd_swa); exit(1); } }
    free(ar);
}

/* costruisce un QT [O,I] dal disco in `t` (buffer riusabili tra chiamate).
 *  - se esiste `name.qs`: pesi GIA' quantizzati nel container (U8 qdata + F32 scala) -> letti diretti
 *  - altrimenti: tensore pieno (f32/bf16) -> quantizzato a runtime a `bits` (oracolo tiny / pesi pieni)
 * drop=1 -> fadvise DONTNEED (streaming expert). */
static void qt_from_disk(Model *m, const char *name, int O, int I, int bits, int drop, QT *t){
    char sn[300]; snprintf(sn,sizeof(sn),"%s.qs",name);
    if(st_has(&m->S,sn)){
        int64_t nb=st_nbytes(&m->S,name);
        int64_t expect8=(int64_t)O*I, expect4=(int64_t)O*((I+1)/2), expect2=(int64_t)O*((I+3)/4);
        int fmt = (nb==expect8)?1 : (nb==expect4)?2 : (nb==expect2)?3 : -1;
        if(fmt<0){
            fprintf(stderr,"qt_from_disk: %s nbytes=%lld not int8/4/2 for O=%d I=%d\n",
                    name,(long long)nb,O,I); exit(1);
        }
        int64_t nscale=st_numel(&m->S,sn);
        if(nscale!=O){ fprintf(stderr,"qt_from_disk: %s.qs numel=%lld expect O=%d\n",
                               name,(long long)nscale,O); exit(1); }
        if(fmt==1){ if(t->fmt!=1||!t->q8){ t->fmt=1; t->O=O; t->I=I; t->q8=malloc((size_t)nb); t->s=falloc(O); }
                    st_read_raw(&m->S,name,t->q8,nb,drop); }
        else      { if(t->fmt!=fmt||!t->q4){ t->fmt=fmt; t->O=O; t->I=I; t->q4=malloc((size_t)nb); t->s=falloc(O); }
                    st_read_raw(&m->S,name,t->q4,nb,drop); }
        st_read_f32(&m->S,sn,t->s,O,drop);
    } else {
        int64_t n=st_numel(&m->S,name);
        if(n!=(int64_t)O*I){ fprintf(stderr,"qt_from_disk: %s numel=%lld expect %lld (O=%d I=%d)\n",
                                     name,(long long)n,(long long)O*I,O,I); exit(1); }
        if(!t->qf && !t->q8 && !t->q4) qt_alloc(t,O,I,bits);
        if(t->fmt==0) st_read_f32(&m->S,name,t->qf,(int64_t)O*I,drop);
        else { float *tmp=falloc((int64_t)O*I); st_read_f32(&m->S,name,tmp,(int64_t)O*I,drop); qt_fill(t,tmp,bits); free(tmp); }
    }
}
static QT qt_load(Model *m, const char *name, int O, int I, int bits){
    QT t; memset(&t,0,sizeof(t)); qt_from_disk(m,name,O,I,bits,0,&t);
#ifdef COLI_CUDA
    if(g_cuda_enabled&&g_cuda_dense){
        t.cuda_eligible=1;
        int slot=g_cuda_rr++%g_cuda_ndev; t.cuda_device=g_cuda_devices[slot];
        g_cuda_dense_projected[slot]+=qt_bytes(&t);
    }
#endif
    return t;
}
static float *ld(Model *m, const char *name){   /* tensore 1D f32 residente (norme/bias) */
    int64_t n=st_numel(&m->S,name); if(n<0){fprintf(stderr,"manca %s\n",name);exit(1);}
    float *p=falloc(n); st_read_f32(&m->S,name,p,n,0); return p;
}
/* F-01: reject token ids outside [0, vocab) before embedding/lm_head indexing. */
static void tokens_check_vocab(const int *ids, int n, int vocab, const char *ctx){
    if(!ids || n<0 || vocab<1) return;
    for(int i=0;i<n;i++){
        if(ids[i]<0 || ids[i]>=vocab){
            fprintf(stderr,"token id out of range: %s[%d]=%d vocab=%d\n",ctx,i,ids[i],vocab);
            exit(1);
        }
    }
}

/* De-interleava il qkv fuso dal layout del checkpoint FP8 al layout piatto [Q|K|V]
 * atteso da attention(). Il checkpoint di MiMo-V2.5 concatena NR=4 blocchi di rank
 * tensor-parallel, ognuno [Q (H/NR teste) | K (kvh/NR teste) | V (kvh/NR teste)]:
 * verificato sui dati (le scale fp8 a blocchi 128 del checkpoint sono emesse PER RANK:
 * i layer full hanno 108 = 4 x ceil(3392/128) righe di scala, non ceil(13568/128)=106).
 * E' una PERMUTAZIONE DI RIGHE: con quantizzazione per-riga (dati + scala per riga) e'
 * esatta, nessuna requantizzazione (vLLM invece deve dequant/requant per le sue scale
 * a blocchi: vedi _shard_fp8_qkv_proj in vllm/model_executor/models/mimo_v2.py).
 * EN: de-interleave fused qkv from the FP8 checkpoint's per-TP-rank layout (NR=4 blocks
 * of [Q|K|V]) to the flat [Q|K|V] attention() expects. Pure row permutation: exact
 * under per-row quantization. NR verified from the checkpoint's per-rank scale grids. */
static void qkv_degroup(QT *t, int H, int kvh, int hd, int vd, int nr){
    if(nr<1 || H%nr || kvh%nr){ fprintf(stderr,"qkv_degroup: nr=%d incompatibile con H=%d kvh=%d\n",nr,H,kvh); exit(1); }
    int qpr=(H/nr)*hd, kpr=(kvh/nr)*hd, vpr=(kvh/nr)*vd, rpr=qpr+kpr+vpr;
    int qs=H*hd, ks=kvh*hd;
    if((int64_t)nr*rpr != t->O){ fprintf(stderr,"qkv_degroup: O=%d != nr %d x rpr %d\n",t->O,nr,rpr); exit(1); }
    int64_t rb = t->fmt==0 ? (int64_t)t->I*4 : t->fmt==1 ? t->I
               : t->fmt==2 ? ((int64_t)t->I+1)/2 : ((int64_t)t->I+3)/4;   /* byte per riga */
    uint8_t *w = t->fmt==0 ? (uint8_t*)t->qf : t->fmt==1 ? (uint8_t*)t->q8 : t->q4;
    uint8_t *tmp=malloc((size_t)((int64_t)t->O*rb)); if(!tmp){fprintf(stderr,"OOM degroup\n");exit(1);}
    float *stmp = t->s ? falloc(t->O) : NULL;
    for(int g=0;g<nr;g++){
        int64_t sq=(int64_t)g*rpr;                                  /* inizio del blocco di rank g */
        int64_t dq=(int64_t)g*qpr, dk=(int64_t)qs+(int64_t)g*kpr, dv=(int64_t)qs+ks+(int64_t)g*vpr;
        memcpy(tmp+dq*rb, w+sq*rb,          (size_t)((int64_t)qpr*rb));  /* Q del rank g */
        memcpy(tmp+dk*rb, w+(sq+qpr)*rb,    (size_t)((int64_t)kpr*rb));  /* K del rank g */
        memcpy(tmp+dv*rb, w+(sq+qpr+kpr)*rb,(size_t)((int64_t)vpr*rb));  /* V del rank g */
        if(stmp){
            memcpy(stmp+dq, t->s+sq,        (size_t)qpr*sizeof(float));
            memcpy(stmp+dk, t->s+sq+qpr,    (size_t)kpr*sizeof(float));
            memcpy(stmp+dv, t->s+sq+qpr+kpr,(size_t)vpr*sizeof(float));
        }
    }
    memcpy(w, tmp, (size_t)((int64_t)t->O*rb)); free(tmp);
    if(stmp){ memcpy(t->s, stmp, (size_t)t->O*sizeof(float)); free(stmp); }
}

/* dimensioni di attenzione del layer li (dipendono dal tipo: full o SWA) */
static inline int lyr_kvh(const Cfg *c, int li){ return c->is_swa[li] ? c->kv_heads_swa   : c->kv_heads_full; }
static inline int lyr_hd (const Cfg *c, int li){ return c->is_swa[li] ? c->swa_head_dim   : c->head_dim; }
static inline int lyr_vd (const Cfg *c, int li){ return c->is_swa[li] ? c->swa_v_head_dim : c->v_head_dim; }
/* F-11 / speed: SWA (and MTP-as-SWA) KV is a ring of sliding_window rows, not max_t.
 * Full-attention layers stay linear. RoPE still uses the logical position. */
static inline int lyr_kv_rows(const Cfg *c, int li, int max_t){
    if(c->is_swa[li] && c->sliding_window>0){
        int w=c->sliding_window;
        return (w<max_t)?w:max_t;
    }
    return max_t;
}
static inline int kv_phys(const Cfg *c, int li, int pos, int max_t){
    int rows=lyr_kv_rows(c,li,max_t);
    if(c->is_swa[li] && c->sliding_window>0 && rows<max_t){
        /* pos >= 0 always on the decode path */
        return pos % rows;
    }
    return pos;
}

static void model_init(Model *m, const char *snap, int cap, int ebits, int dbits){
    memset(m,0,sizeof(*m)); m->ebits=ebits; m->dbits=dbits;
    load_cfg(&m->c,snap); st_init(&m->S,snap);
    Cfg *c=&m->c; char nm[256]; int D=c->hidden;
    /* rd del RoPE per tipo di layer: full usa head_dim (=rope_dim), SWA usa swa_head_dim.
     * Deve combaciare ESATTAMENTE con la rd calcolata in attention() (hd*c->rope_dim/c->head_dim). */
    m->rope_rd_full = c->rope_dim;
    m->rope_rd_swa  = (int)((int64_t)c->swa_head_dim*c->rope_dim/c->head_dim);
    m->rope_cap = 0; m->rope_full_cos=m->rope_full_sin=m->rope_swa_cos=m->rope_swa_sin=NULL;
    /* embed e lm_head sono il confine I/O: tenerli ad alta precisione (come i quant dynamic
     * reali). dbits>=8 -> qui f32; piu' basso -> dbits. */
    int io_bits = dbits>=8 ? 16 : dbits;
    m->embed   = qt_load(m,"model.embed_tokens.weight", c->vocab, D, io_bits);
    m->lm_head = qt_load(m,"lm_head.weight", c->vocab, D, io_bits);
    m->final_norm = ld(m,"model.norm.weight");
    m->L=calloc(c->n_layers,sizeof(Layer));
    int NR=c->n_layers;
    m->ecap=cap; m->ecache=calloc(NR,sizeof(ESlot*)); m->ecn=calloc(NR,sizeof(int));
    m->eroute=calloc(NR,sizeof(int*)); m->enr=calloc(NR,sizeof(int));
    m->pin=calloc(NR,sizeof(ESlot*)); m->npin=calloc(NR,sizeof(int));
    m->gpu_pin=calloc(NR,sizeof(ESlot*)); m->ngpin=calloc(NR,sizeof(int));
    m->res_bits=calloc((size_t)NR*4,sizeof(uint64_t));
    m->pref_bits=calloc((size_t)NR*4,sizeof(uint64_t));
    m->eusage=calloc(NR,sizeof(uint32_t*)); m->eheat=calloc(NR,sizeof(uint32_t*));
    m->kv_start=calloc(NR+1,sizeof(int));        /* +1: riga KV del layer MTP */
    for(int i=0;i<c->n_layers;i++){
        Layer *l=&m->L[i];
        int kvh=lyr_kvh(c,i), hd=lyr_hd(c,i), vd=lyr_vd(c,i);
        int qs=c->n_heads*hd, ks=kvh*hd, vs=kvh*vd;   /* layout del qkv fuso [q|k|v] */
        #define P(s) (snprintf(nm,sizeof(nm),"model.layers.%d." s,i),nm)
        l->in_ln=ld(m,P("input_layernorm.weight"));
        l->post_ln=ld(m,P("post_attention_layernorm.weight"));
        { int64_t want=(int64_t)(qs+ks+vs)*D, got=st_numel(&m->S,P("self_attn.qkv_proj.weight"));
          if(got!=want){ fprintf(stderr,"layer %d (%s): qkv_proj ha %lld elementi, attesi %lld = (%d+%d+%d)x%d\n",
              i, c->is_swa[i]?"SWA":"full", (long long)got,(long long)want, qs,ks,vs,D); exit(1); } }
        l->qkv = qt_load(m,P("self_attn.qkv_proj.weight"), qs+ks+vs, D, dbits);
        if(c->qkv_grouped){
            /* NR di default: il numero minimo di teste KV tra i tipi di layer (=4 per
             * MiMo-V2.5, coerente con le griglie di scala per-rank del checkpoint).
             * QKV_RANKS lo forza per checkpoint futuri con altro grado di TP. */
            int nr = getenv("QKV_RANKS") ? atoi(getenv("QKV_RANKS"))
                   : (c->kv_heads_full < c->kv_heads_swa ? c->kv_heads_full : c->kv_heads_swa);
            qkv_degroup(&l->qkv, c->n_heads, kvh, hd, vd, nr);
        }
        l->o   = qt_load(m,P("self_attn.o_proj.weight"), D, c->n_heads*vd, dbits);
        if(c->is_swa[i] ? c->has_sink_swa : c->has_sink_full){
            int64_t sn=st_numel(&m->S,P("self_attn.attention_sink_bias"));
            if(sn!=c->n_heads){ fprintf(stderr,"layer %d: attention_sink_bias ha %lld elementi, attesi %d\n",
                i,(long long)sn,c->n_heads); exit(1); }
            l->sink = ld(m,P("self_attn.attention_sink_bias"));
        } else l->sink = NULL;
        l->sparse = c->is_moe[i];
        if(!l->sparse){
            l->gate_proj = qt_load(m,P("mlp.gate_proj.weight"), c->dense_inter, D, dbits);
            l->up_proj   = qt_load(m,P("mlp.up_proj.weight"),   c->dense_inter, D, dbits);
            l->down_proj = qt_load(m,P("mlp.down_proj.weight"), D, c->dense_inter, dbits);
        } else {
            l->router=ld(m,P("mlp.gate.weight"));
            l->router_bias=ld(m,P("mlp.gate.e_score_correction_bias"));
            m->ecache[i]=calloc(cap,sizeof(ESlot));
            m->eroute[i]=calloc(c->topk,sizeof(int));      /* metodo C: ultimo routing del layer */
            m->eusage[i]=calloc(c->n_experts,sizeof(uint32_t));
            m->eheat[i]=calloc(c->n_experts,sizeof(uint32_t));
        }
        #undef P
    }
    /* ---- testa MTP nativa (MiMo-V2.5: model.mtp.layers.0.*, dal container --mtp) ----
     * Struttura (vLLM mimo_v2_mtp.py, il riferimento autorevole per questo checkpoint):
     * h = eh_proj([enorm(emb(tok)) ; hnorm(h_prev)]) -> blocco decoder DENSO con attenzione
     * in GEOMETRIA SWA (swa_num_attention_heads/kv/dim, sliding window, sink bias,
     * swa_rope_theta, v_scale) -> final_layernorm -> lm_head condiviso del modello.
     * Attiva SOLO se il set di tensori e' completo; MTP=0 la spegne comunque. */
    {
        const char *req[]={"eh_proj.weight","enorm.weight","hnorm.weight","final_layernorm.weight",
            "input_layernorm.weight","pre_mlp_layernorm.weight","self_attn.qkv_proj.weight",
            "self_attn.o_proj.weight","mlp.gate_proj.weight","mlp.up_proj.weight","mlp.down_proj.weight"};
        m->has_mtp = c->n_layers<128;            /* is_swa[n_layers] deve esistere */
        for(unsigned q=0;q<sizeof(req)/sizeof(req[0]) && m->has_mtp;q++){
            snprintf(nm,sizeof(nm),"model.mtp.layers.0.%s",req[q]);
            if(!st_has(&m->S,nm)) m->has_mtp=0;
        }
        if(getenv("MTP") && atoi(getenv("MTP"))==0) m->has_mtp=0;
        if(m->has_mtp){
            int li=c->n_layers; Layer *l=&m->mtpL;
            c->is_swa[li]=1;                     /* geometria/rope/finestra SWA per la riga MTP */
            m->kv_start[li]=-1;                  /* KV MTP: parte dalla prima posizione assorbita */
            int kvh=lyr_kvh(c,li), hd=lyr_hd(c,li), vd=lyr_vd(c,li);
            int qs=c->n_heads*hd, ks=kvh*hd, vs=kvh*vd;
            #define PM(s) (snprintf(nm,sizeof(nm),"model.mtp.layers.0." s),nm)
            l->in_ln  =ld(m,PM("input_layernorm.weight"));
            l->post_ln=ld(m,PM("pre_mlp_layernorm.weight"));
            { int64_t want=(int64_t)(qs+ks+vs)*D, got=st_numel(&m->S,PM("self_attn.qkv_proj.weight"));
              if(got!=want){ fprintf(stderr,"MTP: qkv_proj ha %lld elementi, attesi %lld = (%d+%d+%d)x%d\n",
                  (long long)got,(long long)want,qs,ks,vs,D); exit(1); } }
            l->qkv = qt_load(m,PM("self_attn.qkv_proj.weight"), qs+ks+vs, D, dbits);
            if(c->qkv_grouped){
                /* stesso layout per-rank dei layer del modello: la griglia di scale fp8
                 * del qkv MTP nel checkpoint ([116,32]) e' IDENTICA a quella dei layer
                 * SWA principali, gia' validati con questo de-interleave (NR=4). */
                int nr = getenv("QKV_RANKS") ? atoi(getenv("QKV_RANKS"))
                       : (c->kv_heads_full < c->kv_heads_swa ? c->kv_heads_full : c->kv_heads_swa);
                qkv_degroup(&l->qkv, c->n_heads, kvh, hd, vd, nr);
            }
            l->o = qt_load(m,PM("self_attn.o_proj.weight"), D, c->n_heads*vd, dbits);
            if(c->has_sink_swa){
                int64_t sn=st_numel(&m->S,PM("self_attn.attention_sink_bias"));
                if(sn!=c->n_heads){ fprintf(stderr,"MTP: attention_sink_bias ha %lld elementi, attesi %d\n",
                    (long long)sn,c->n_heads); exit(1); }
                l->sink=ld(m,PM("self_attn.attention_sink_bias"));
            } else l->sink=NULL;
            l->sparse=0;                         /* mlp DENSO (intermediate_size): niente expert */
            l->gate_proj = qt_load(m,PM("mlp.gate_proj.weight"), c->dense_inter, D, dbits);
            l->up_proj   = qt_load(m,PM("mlp.up_proj.weight"),   c->dense_inter, D, dbits);
            l->down_proj = qt_load(m,PM("mlp.down_proj.weight"), D, c->dense_inter, dbits);
            m->eh_proj = qt_load(m,PM("eh_proj.weight"), D, 2*D, dbits);
            m->enorm=ld(m,PM("enorm.weight")); m->hnorm=ld(m,PM("hnorm.weight"));
            m->mtp_norm=ld(m,PM("final_layernorm.weight"));
            #undef PM
        }
    }
    m->hlast=falloc(D); m->h_all=falloc((int64_t)64*D);
    /* byte della parte DENSA residente (embed+lm_head+attn+mlp densa+norme) */
    int64_t rb=qt_bytes(&m->embed)+qt_bytes(&m->lm_head);
    for(int i=0;i<c->n_layers;i++){ Layer *l=&m->L[i];
        rb+=qt_bytes(&l->qkv)+qt_bytes(&l->o);
        if(!l->sparse) rb+=qt_bytes(&l->gate_proj)+qt_bytes(&l->up_proj)+qt_bytes(&l->down_proj);
    }
    if(m->has_mtp){ Layer *l=&m->mtpL;
        rb+=qt_bytes(&l->qkv)+qt_bytes(&l->o)+qt_bytes(&m->eh_proj);
        rb+=qt_bytes(&l->gate_proj)+qt_bytes(&l->up_proj)+qt_bytes(&l->down_proj);
    }
    m->resident_bytes=rb;
}

/* embed: dequantizza la riga del token (scala per-riga) in x[hidden] */
static void embed_row(Model *m, int tok, float *x){
    int D=m->c.hidden; QT *e=&m->embed;
    if(e->fmt==0){ memcpy(x, e->qf+(int64_t)tok*D, D*sizeof(float)); return; }
    if(e->fmt==1){ const int8_t *q=e->q8+(int64_t)tok*D; float s=e->s[tok];
        for(int i=0;i<D;i++) x[i]=(float)q[i]*s; return; }
    if(e->fmt==2){ const uint8_t *q=e->q4+(int64_t)tok*((D+1)/2); float s=e->s[tok];   /* int4 */
        for(int i=0;i<D;i+=2){ uint8_t byte=q[i>>1]; x[i]=(float)((int)(byte&0xF)-8)*s;
            if(i+1<D) x[i+1]=(float)((int)(byte>>4)-8)*s; }
        return; }
    const uint8_t *q=e->q4+(int64_t)tok*((D+3)/4); float s=e->s[tok];   /* int2 */
    for(int i=0;i<D;i++){ uint8_t byte=q[i>>2]; int sh=(i&3)*2; x[i]=(float)((int)((byte>>sh)&3)-2)*s; }
}

/* carica un expert nello slot. Container pre-quantizzato: le 3 matrici sono contigue nel
 * file -> UNA pread coalescente da ~19 MB dentro `slab` (+ le scale in fslab); i QT sono
 * viste dentro lo slab (zero copie). Fallback per modelli non quantizzati (oracolo tiny).
 * THREAD-SAFE su slot distinti (pread posizionale, st_find read-only). */
static void expert_load(Model *m, int layer, int eid, ESlot *s){
#ifdef COLI_CUDA
    /* A live REPIN may reuse a GPU-enabled pinned slot for a different expert.
     * Keep its tier assignment, but invalidate the old device weights. */
    if(s->eid!=eid){ qt_cuda_reset(&s->g); qt_cuda_reset(&s->u); qt_cuda_reset(&s->d); }
#endif
    Cfg *c=&m->c; int I=c->moe_inter, D=c->hidden, b=m->ebits;
    char nm[3][288]; const char *suf[3]={"gate_proj","up_proj","down_proj"};
    for(int k=0;k<3;k++) snprintf(nm[k],sizeof(nm[k]),"model.layers.%d.mlp.experts.%d.%s.weight",layer,eid,suf[k]);
    char qn[300]; snprintf(qn,sizeof(qn),"%s.qs",nm[0]);
    if(!st_has(&m->S,qn)){                       /* fallback: tensori pieni, quantizza a runtime */
        qt_from_disk(m,nm[0],I,D,b,g_drop,&s->g);
        qt_from_disk(m,nm[1],I,D,b,g_drop,&s->u);
        qt_from_disk(m,nm[2],D,I,b,g_drop,&s->d);
        s->eid=eid; return;
    }
    st_tensor *tw[3], *tq[3];
    for(int k=0;k<3;k++){
        tw[k]=st_find(&m->S,nm[k]);
        snprintf(qn,sizeof(qn),"%s.qs",nm[k]); tq[k]=st_find(&m->S,qn);
        if(!tw[k]||!tq[k]){ fprintf(stderr,"manca %s\n",nm[k]); exit(1); }
    }
    int64_t wtot=tw[0]->nbytes+tw[1]->nbytes+tw[2]->nbytes;
    int64_t ftot=(tq[0]->nbytes+tq[1]->nbytes+tq[2]->nbytes)/4;
    /* rialloca se lo slot (riusato tra layer) e' troppo piccolo per QUESTO expert:
     * pread oltre la mappatura = short-read o CORRUZIONE silenziosa dei vicini */
    if(!s->slab || wtot+8192 > s->slab_cap){
        free(s->slab);
        if(posix_memalign((void**)&s->slab,4096,wtot+8192)){fprintf(stderr,"OOM slab\n");exit(1);}
        s->slab_cap=wtot+8192;
    }
    if(!s->fslab || ftot > s->fslab_cap){ free(s->fslab); s->fslab=falloc(ftot); s->fslab_cap=ftot; }
    int ord[3]={0,1,2};                          /* ordina per offset nel file */
    for(int a=0;a<3;a++) for(int bb=a+1;bb<3;bb++) if(tw[ord[bb]]->off<tw[ord[a]]->off){ int t=ord[a]; ord[a]=ord[bb]; ord[bb]=t; }
    int contig = tw[ord[0]]->fd==tw[ord[1]]->fd && tw[ord[1]]->fd==tw[ord[2]]->fd
              && tw[ord[0]]->off+tw[ord[0]]->nbytes==tw[ord[1]]->off
              && tw[ord[1]]->off+tw[ord[1]]->nbytes==tw[ord[2]]->off;
    int64_t pos[3]; int done=0;
    if(contig){
        int64_t off0=tw[ord[0]]->off;
        int dfd = g_direct ? st_direct_fd(&m->S, tw[ord[0]]->fd) : -1;
        if(dfd>=0){                              /* O_DIRECT: offset/len allineati a 4K */
            int64_t base=off0 & ~4095LL, need=(off0-base)+wtot;
            int64_t len=(need+4095)&~4095LL;
            ssize_t r=pread(dfd, s->slab, len, base);
            if(r>=need){
                pos[ord[0]]=off0-base; pos[ord[1]]=pos[ord[0]]+tw[ord[0]]->nbytes;
                pos[ord[2]]=pos[ord[1]]+tw[ord[1]]->nbytes; done=1;
            }
        }
        if(!done){                               /* fallback bufferizzato */
            if(pread(tw[ord[0]]->fd, s->slab, wtot, off0)!=wtot){ perror("pread expert"); exit(1); }
            pos[ord[0]]=0; pos[ord[1]]=tw[ord[0]]->nbytes; pos[ord[2]]=tw[ord[0]]->nbytes+tw[ord[1]]->nbytes; done=1;
        }
    }
    if(!done){                                   /* non contigui: 3 pread bufferizzate */
        int64_t o=0;
        for(int a=0;a<3;a++){ int k=ord[a];
            if(pread(tw[k]->fd, s->slab+o, tw[k]->nbytes, tw[k]->off)!=tw[k]->nbytes){ perror("pread expert"); exit(1); }
            pos[k]=o; o+=tw[k]->nbytes; }
    }
    float *fp[3]; int64_t fo=0;                  /* scale (piccole) */
    for(int k=0;k<3;k++){
        if(pread(tq[k]->fd, (char*)(s->fslab+fo), tq[k]->nbytes, tq[k]->off)!=tq[k]->nbytes){ perror("pread qs"); exit(1); }
        fp[k]=s->fslab+fo; fo+=tq[k]->nbytes/4; }
    if(g_drop){                                  /* scarta subito le pagine: evita che la page
                                                  * cache in pressione strangoli il throughput */
        posix_fadvise(tw[ord[0]]->fd, tw[ord[0]]->off, wtot, POSIX_FADV_DONTNEED);
        for(int k=0;k<3;k++) posix_fadvise(tq[k]->fd, tq[k]->off, tq[k]->nbytes, POSIX_FADV_DONTNEED);
    }
    QT *qt[3]={&s->g,&s->u,&s->d}; int OO[3]={I,I,D}, II[3]={D,D,I};
    for(int k=0;k<3;k++){
        int64_t nb=tw[k]->nbytes;
        int fmt = (nb==(int64_t)OO[k]*II[k])?1 : (nb==(int64_t)OO[k]*((II[k]+1)/2))?2 : 3;
        qt[k]->fmt=fmt; qt[k]->O=OO[k]; qt[k]->I=II[k]; qt[k]->qf=NULL;
        qt[k]->q8=(int8_t*)(s->slab+pos[k]); qt[k]->q4=s->slab+pos[k]; qt[k]->s=fp[k];
    }
    s->eid=eid;
}

/* Free host slabs after weights live on the GPU (complementary VRAM tier).
 * NULLs every host pointer so matmul_qt cannot touch freed memory. */
static void expert_cpu_free(Model *m, ESlot *s){
    (void)m;
    free(s->slab);  s->slab=NULL;  s->slab_cap=0;
    free(s->fslab); s->fslab=NULL; s->fslab_cap=0;
    QT *qt[3]={&s->g,&s->u,&s->d};
    for(int k=0;k<3;k++){ qt[k]->qf=NULL; qt[k]->q8=NULL; qt[k]->q4=NULL; qt[k]->s=NULL; }
}

/* ---- Expert bitmaps (256 experts → 4×uint64_t words per layer) -------------------- */
#define EMAP_W 4
static inline int emap_ok(const Model *m, int layer, int eid){
    return m && layer>=0 && layer<m->c.n_layers && eid>=0 && eid<m->c.n_experts
        && eid<256 && m->res_bits && m->pref_bits;
}
static inline uint64_t *emap_row(uint64_t *base, int layer){ return base+(size_t)layer*EMAP_W; }
static inline int emap_test(const uint64_t *row, int eid){
    return (row[eid>>6] >> (eid&63)) & 1ull;
}
static inline void emap_set(uint64_t *row, int eid){
    row[eid>>6] |= 1ull << (eid&63);
}
static inline void emap_clr(uint64_t *row, int eid){
    row[eid>>6] &= ~(1ull << (eid&63));
}
/* Rebuild residency mask for one layer (pin ∪ gpu ∪ LRU). */
static void res_bits_rebuild_layer(Model *m, int l){
    if(!m->res_bits || l<0 || l>=m->c.n_layers) return;
    uint64_t *row=emap_row(m->res_bits,l);
    memset(row,0,EMAP_W*sizeof(uint64_t));
    if(m->pin && m->pin[l])
        for(int z=0;z<m->npin[l];z++){ int e=m->pin[l][z].eid; if(e>=0&&e<256) emap_set(row,e); }
#ifdef COLI_CUDA
    if(m->gpu_pin && m->gpu_pin[l])
        for(int z=0;z<m->ngpin[l];z++){ int e=m->gpu_pin[l][z].eid; if(e>=0&&e<256) emap_set(row,e); }
#endif
    if(m->ecache && m->ecache[l])
        for(int z=0;z<m->ecn[l];z++){ int e=m->ecache[l][z].eid; if(e>=0&&e<256) emap_set(row,e); }
}
static void res_bits_rebuild_all(Model *m){
    if(!m) return;
    for(int l=0;l<m->c.n_layers;l++) res_bits_rebuild_layer(m,l);
}
/* Clear WILLNEED-epoch mask (call once per SERVE turn / PROMPT). Within a turn,
 * bits stick: first hint wins, duplicates are free no-ops. */
static void pref_bits_clear(Model *m){
    if(m && m->pref_bits)
        memset(m->pref_bits,0,(size_t)m->c.n_layers*EMAP_W*sizeof(uint64_t));
}
/* Is eid resident in RAM pin, VRAM tier, or LRU for this layer? O(1) via bitmap. */
static int expert_resident(Model *m, int l, int eid){
    if(!emap_ok(m,l,eid)){
        /* Fallback scan if bitmaps not ready (early boot). */
        if(m->pin && m->pin[l]) for(int z=0;z<m->npin[l];z++) if(m->pin[l][z].eid==eid) return 1;
#ifdef COLI_CUDA
        if(m->gpu_pin && m->gpu_pin[l]) for(int z=0;z<m->ngpin[l];z++) if(m->gpu_pin[l][z].eid==eid) return 1;
#endif
        if(m->ecache && m->ecache[l]) for(int z=0;z<m->ecn[l];z++) if(m->ecache[l][z].eid==eid) return 1;
        return 0;
    }
    return emap_test(emap_row(m->res_bits,l), eid);
}
/* prefetch asincrono dei pesi di un expert (e delle sue scale .qs): avvia il readahead
 * cosi' le letture sincrone successive trovano la page-cache calda.
 * Bitmap path: skip if already resident OR already hinted this epoch (no fadvise storm). */
static void expert_prefetch(Model *m, int layer, int eid){
    if(eid<0 || layer<0) return;
    if(emap_ok(m,layer,eid)){
        uint64_t *rb=emap_row(m->res_bits,layer);
        uint64_t *pb=emap_row(m->pref_bits,layer);
        if(emap_test(rb,eid) || emap_test(pb,eid)) return;
        emap_set(pb,eid);
    }
    char nm[300];
    const char *suf[3]={"gate_proj.weight","up_proj.weight","down_proj.weight"};
    for(int k=0;k<3;k++){
        snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.%d.%s",layer,eid,suf[k]); st_prefetch(&m->S,nm);
        char qs[320]; snprintf(qs,sizeof(qs),"%s.qs",nm); st_prefetch(&m->S,qs);
    }
}

/* === OVERLAP: pipeline load/compute DENTRO il layer ==================================
 * Prima: FASE load (omp parallel sui miss) POI FASE compute — il profilo sul modello
 * reale mostrava expert-disk ed expert-matmul ADDITIVI (74.5s + 46.4s su 139s totali:
 * serializzati). Qui i load dei miss partono su un piccolo pool di pthread e il loop di
 * calcolo CONSUMA gli slot ws[] nello STESSO ordine di prima, aspettando il flag ready
 * del singolo slot: mentre l'expert j viene moltiplicato, j+1.. sono in lettura.
 * LOSSLESS per costruzione: stessi kernel, stessi slot, stesso ordine di accumulo
 * (findings §17: la somma float non commuta e l'ordine top-k a S=1 deve restare
 * bit-esatto) — cambia solo QUANDO parte la pread. Compone con PILOT (hint WILLNEED
 * per la capa L+1 su un altro thread) e col WILLNEED del blocco-64 successivo.
 * t_edisk diventa il tempo di STALLO (attesa slot), non piu' il tempo di lettura:
 * e' la metrica onesta dell'overlap. OVERLAP=0 ripristina il percorso a fasi.
 * MISURATO (WSL2/VHDX, 32 token TOPP=0.7): lo stallo scende davvero (43s -> 26s,
 * profilo non piu' additivo) ma expert-matmul si gonfia quasi 1:1 (17s -> 30s):
 * su questo host il "disco" e' in gran parte CPU dell'I/O-stack host (VHDX) sugli
 * STESSI core dei matmul, quindi il wall-clock resta ~pari (DIRECT=1) o ~-7%%
 * (buffered, rumoroso). Su NVMe nativo con vero DMA il tempo nascosto e' latenza
 * reale e l'overlap deve pagare: si lascia ON di default con pool piccolo (T=4).
 * EN: async expert loads on a tiny pthread pool; the compute loop consumes slots in
 * the ORIGINAL order via per-slot ready flags, so accumulation order (and thus every
 * output bit) is unchanged. The waiter helps run unclaimed jobs, so forward progress
 * never depends on the pool. Single producer (main); ring of 64 = max miss per block.
 * Measured on the WSL2/VHDX reference host: stall drops 43s->26s but expert-matmul
 * inflates 17s->30s (host-side I/O CPU shares the cores) -> wall-clock break-even
 * with T=4; the win materializes where I/O is DMA/latency-bound, not CPU-bound. */
typedef struct { Model *m; int layer, eid; ESlot *slot; } OvJob;
static OvJob ov_job[64];
static volatile int ov_done[64];
static volatile unsigned ov_next=0, ov_total=0;   /* indici monotoni sul ring (1P/NC) */
static int ov_started=0;
/* prende ed esegue il job non reclamato piu' vecchio, se il suo indice e' <=idx_max.
 * Ritorna 0 se non c'era niente da fare (il chiamante puo' dormire). */
static int ov_claim(unsigned idx_max){
    unsigned n=__atomic_load_n(&ov_next,__ATOMIC_RELAXED);
    unsigned t=__atomic_load_n(&ov_total,__ATOMIC_ACQUIRE);
    if(n>=t || n>idx_max) return 0;
    if(!__atomic_compare_exchange_n(&ov_next,&n,n+1,0,__ATOMIC_ACQ_REL,__ATOMIC_RELAXED)) return 1;
    OvJob *j=&ov_job[n&63];
    expert_load(j->m,j->layer,j->eid,j->slot);
    __atomic_store_n(&ov_done[n&63],1,__ATOMIC_RELEASE);
    return 1;
}
static void *ov_worker(void *a){ (void)a;
#ifdef __linux__
    /* i loader NON devono rubare cicli al team OpenMP dei matmul: senza nice l'expert-
     * matmul raddoppiava (17s -> 32s su 32 token, misurato) e mangiava il guadagno
     * dell'overlap. I loader stanno quasi sempre bloccati in pread; il nice conta solo
     * nei tratti CPU (copia dalla page-cache), dove devono cedere il passo.
     * EN: deprioritize loaders so the OMP matmul team keeps the cores; loaders mostly
     * block in pread anyway (per-thread nice is Linux-specific and exactly what we want). */
    setpriority(PRIO_PROCESS,(id_t)syscall(SYS_gettid),10);
#endif
    for(;;) if(!ov_claim(UINT_MAX)) usleep(200);
    return NULL; }
/* accoda i load dei miss del blocco corrente; ritorna l'indice ring del primo.
 * Chiamato SOLO dal thread principale, e solo dopo che il blocco precedente e' stato
 * interamente consumato (ov_wait su tutti i suoi job) -> il ring non si sovrascrive. */
static unsigned ov_submit(Model *m,int layer,const int *eids,const int *missk,int nmiss){
    if(!ov_started){
        for(int i=0;i<g_overlap_t;i++){ pthread_t t;
            if(pthread_create(&t,NULL,ov_worker,NULL)==0) pthread_detach(t); }
        ov_started=1;   /* se qualche create fallisse: ov_wait sa eseguire i job da solo */
    }
    unsigned b=ov_total;
    for(int q=0;q<nmiss;q++){
        OvJob j={m,layer,eids[missk[q]],&m->ws[q]};
        ov_job[(b+(unsigned)q)&63]=j;
        __atomic_store_n(&ov_done[(b+(unsigned)q)&63],0,__ATOMIC_RELAXED);
    }
    __atomic_store_n(&ov_total,b+(unsigned)nmiss,__ATOMIC_RELEASE);
    return b;
}
static void ov_wait(unsigned idx){
    while(!__atomic_load_n(&ov_done[idx&63],__ATOMIC_ACQUIRE))
        if(!ov_claim(idx)) usleep(50);            /* job orfano o in coda: aiuta il chiamante */
}

/* RoPE parziale NON-interleaved (NeoX/rotate_half) su UNA testa: ruota i primi rd dim,
 * il resto passa intatto. Coppie (j, j+half) con half=rd/2. cos/sin arrivano GIA'
 * precalcolati in cos[j]/sin[j] (riga della posizione corrente nella tabella RoPE),
 * cosi' il caldo non rifa mai powf()/cosf()/sinf() per token. Bit-identico a rope_neox. */
static inline void rope_apply(float *v, int rd, const float *cos, const float *sin){
    int half=rd/2;
    for(int j=0;j<half;j++){
        float cs=cos[j], sn=sin[j];
        float a=v[j], b=v[half+j];
        v[j]=a*cs-b*sn; v[half+j]=b*cs+a*sn;
    }
}

/* Precalcola le tabelle RoPE fino a max_t per I DUE tipi di layer (full/SWA, theta e rd
 * diversi). Chiamata da kv_alloc(): gli angoli dipendono solo da (pos, j) per tipo, quindi
 * si calcolano una volta sola e si riusano per tutti i token/capi. Ricostruisce solo se
 * serve piu' spazio (max_t crescente). */
static void rope_build(Model *m, int max_t){
    Cfg *c=&m->c;
    if(max_t<=m->rope_cap) return;
    if(m->rope_full_cos){ free(m->rope_full_cos); free(m->rope_full_sin);
                          free(m->rope_swa_cos);  free(m->rope_swa_sin); }
    m->rope_cap = max_t;
    int nf=c->rope_dim/2, ns=m->rope_rd_swa/2;
    size_t szf=(size_t)max_t*nf, szs=(size_t)max_t*ns;
    m->rope_full_cos=falloc(szf); m->rope_full_sin=falloc(szf);
    m->rope_swa_cos =falloc(szs); m->rope_swa_sin =falloc(szs);
    for(int pos=0;pos<max_t;pos++){
        for(int j=0;j<nf;j++){
            float ang=(float)pos*powf(c->theta_full,-2.f*(float)j/(float)c->rope_dim);
            m->rope_full_cos[(size_t)pos*nf+j]=cosf(ang);
            m->rope_full_sin[(size_t)pos*nf+j]=sinf(ang);
        }
        for(int j=0;j<ns;j++){
            float ang=(float)pos*powf(c->theta_swa,-2.f*(float)j/(float)m->rope_rd_swa);
            m->rope_swa_cos[(size_t)pos*ns+j]=cosf(ang);
            m->rope_swa_sin[(size_t)pos*ns+j]=sinf(ang);
        }
    }
}

/* attenzione ibrida MiMo (full/SWA, GQA su qkv fuso, RoPE parziale non-interleaved a doppia
 * theta, sink bias, V*v_scale) su token nuovi x[S,hidden], pos_base = pos del primo.
 * Ordine del reference (_forward_attention + eager_attention_forward):
 *   qkv fuso -> v*v_scale (PRIMA della cache) -> RoPE su q/k -> append KV -> score/softmax
 *   (con eventuale colonna sink: entra nel denominatore, la sua massa viene scartata)
 *   -> ctx -> o_proj. Layer SWA: finestra causale [pos-window+1, pos] (include se stesso). */
static void attention(Model *m, Layer *l, int layer, float *x, int S, int pos_base, float *out){
    Cfg *c=&m->c; int H=c->n_heads;
    int kvh=lyr_kvh(c,layer), hd=lyr_hd(c,layer), vd=lyr_vd(c,layer);
    int group=H/kvh, qs=H*hd, ks=kvh*hd, vs=kvh*vd, rowsz=qs+ks+vs;
    int swa=c->is_swa[layer];
    /* rope_dim e scala dipendono dal TIPO di layer: derivati da hd, non dai campi full-only
     * di Cfg (rope_dim fu calcolato da head_dim*prf: qui lo riscaliamo in interi, esatto) */
    int rd=(int)((int64_t)hd*c->rope_dim/c->head_dim);
    float scale=1.f/sqrtf((float)hd);
    double ta0=now_s();
    /* 1) proiezione fusa qkv [q|k|v] per tutti i token nuovi */
    float *qkv=falloc((int64_t)S*rowsz);
    matmul_qt(qkv, x, &l->qkv, S);
    /* 2) per token: v*v_scale prima della cache, RoPE su q (in place) e k, append in KV.
     * cos/sin arrivano dalla tabella precalcolata per (pos, tipo-layer): niente powf() sul
     * percorso caldo. */
    const float *base_cos = swa ? m->rope_swa_cos : m->rope_full_cos;
    const float *base_sin = swa ? m->rope_swa_sin : m->rope_full_sin;
    int half=rd/2;
    for(int s=0;s<S;s++){
        int pos=pos_base+s; float *r=qkv+(int64_t)s*rowsz;
        int pphy=kv_phys(c,layer,pos,m->max_t);
        float *Kd=m->K[layer]+(int64_t)pphy*ks, *Vd=m->V[layer]+(int64_t)pphy*vs;
        memcpy(Kd, r+qs, ks*sizeof(float));
        for(int i=0;i<vs;i++) Vd[i]=r[qs+ks+i]*c->v_scale;
        const float *ct=base_cos+(size_t)pos*half;
        const float *st=base_sin+(size_t)pos*half;
        for(int h=0;h<H;h++)   rope_apply(r+(int64_t)h*hd, rd, ct, st);
        for(int g=0;g<kvh;g++) rope_apply(Kd+(int64_t)g*hd, rd, ct, st);
    }
    /* 3) attenzione causale per (s,h): GQA, testa kv g=h/group */
    float *ctx=falloc((int64_t)S*H*vd);
    const float *Kc=m->K[layer], *Vc=m->V[layer];
    #pragma omp parallel for collapse(2) schedule(static)
    for(int s=0;s<S;s++) for(int h=0;h<H;h++){
        int pos=pos_base+s, g=h/group;
        int lo = (swa && c->sliding_window>0) ? pos-c->sliding_window+1 : 0;
        if(lo<0) lo=0;
        if(lo<m->kv_start[layer]) lo=m->kv_start[layer];
        int nt=pos+1-lo;
        float scb[4096]; float *sc = nt<=4096 ? scb : falloc(nt);
        const float *q=qkv+(int64_t)s*rowsz+(int64_t)h*hd;
        float mx=-1e30f;
        for(int j=0;j<nt;j++){
            int tphy=kv_phys(c,layer,lo+j,m->max_t);
            const float *kt=Kc+(int64_t)tphy*ks+(int64_t)g*hd;
            float a=0; int d=0;
#if defined(__AVX2__)
            {
                __m256 acc=_mm256_setzero_ps();
                for(; d+8<=hd; d+=8){
                    __m256 qv=_mm256_loadu_ps(q+d), kv=_mm256_loadu_ps(kt+d);
#if defined(__FMA__)
                    acc=_mm256_fmadd_ps(qv,kv,acc);
#else
                    acc=_mm256_add_ps(acc,_mm256_mul_ps(qv,kv));
#endif
                }
                __m128 t=_mm_add_ps(_mm256_castps256_ps128(acc),_mm256_extractf128_ps(acc,1));
                t=_mm_add_ps(t,_mm_movehl_ps(t,t));
                t=_mm_add_ss(t,_mm_shuffle_ps(t,t,1));
                a=_mm_cvtss_f32(t);
            }
#endif
            for(;d<hd;d++) a+=q[d]*kt[d];
            a*=scale; sc[j]=a; if(a>mx) mx=a;
        }
        float den=0;
        if(l->sink){ float sk=l->sink[h]; if(sk>mx) mx=sk; den=expf(sk-mx); }
        for(int j=0;j<nt;j++){ sc[j]=expf(sc[j]-mx); den+=sc[j]; }
        float inv=1.f/den;
        float *cx=ctx+((int64_t)s*H+h)*vd;
        for(int d=0;d<vd;d++) cx[d]=0;
        for(int j=0;j<nt;j++){
            int tphy=kv_phys(c,layer,lo+j,m->max_t);
            const float *vt=Vc+(int64_t)tphy*vs+(int64_t)g*vd;
            float a=sc[j]*inv; int d=0;
#if defined(__AVX2__)
            {
                __m256 av=_mm256_set1_ps(a);
                for(; d+8<=vd; d+=8){
                    __m256 c=_mm256_loadu_ps(cx+d), vv=_mm256_loadu_ps(vt+d);
#if defined(__FMA__)
                    c=_mm256_fmadd_ps(av,vv,c);
#else
                    c=_mm256_add_ps(c,_mm256_mul_ps(av,vv));
#endif
                    _mm256_storeu_ps(cx+d,c);
                }
            }
#endif
            for(;d<vd;d++) cx[d]+=a*vt[d];
        }
        if(sc!=scb) free(sc);
    }
    /* 4) concat teste [S, H*vd] -> o_proj */
    matmul_qt(out, ctx, &l->o, S);
    free(ctx); free(qkv);
    m->t_attn += now_s()-ta0;
}

/* MoE MiMo su x[S,hidden] -> out (router sigmoid/noaux_tc, n_group=1, NIENTE shared expert).
 * BATCH-UNION: per S>1 (prefill, verifica speculativa) ogni expert UNICO del batch viene
 * caricato una volta sola e moltiplicato per tutte le posizioni che lo usano (pesi letti
 * 1 volta). Per posizione l'accumulo resta nell'ordine di union dei routed.
 * LOSSLESS SPEC, parte 2 (secondo bug trovato col gate "DRAFT=0 vs 2 byte-identici" a 32
 * token): l'ordine di union NON e' l'ordine top-k della singola posizione — a S=1 una
 * posizione accumula i suoi expert nel SUO ordine, nel batch li accumula nell'ordine di
 * prima-apparizione tra le righe. La somma float non e' associativa: dopo 48 layer la
 * deriva sugli ultimi bit flippa argmax al limite e il greedy con DRAFT>0 divergeva.
 * Con g_fw1 (solo verifica speculativa) i contributi vengono bufferizzati per (pos,k) e
 * accumulati alla fine nell'ordine top-k della posizione: ogni riga del batch riproduce
 * bit-esatto il forward S=1. I percorsi validati (prefill, decode) restano invariati.
 * EN: union order != per-position top-k order; float adds don't commute, so verify
 * batches drifted from sequential decode. Under g_fw1 contributions are buffered per
 * (pos,k) and reduced in the position's own top-k order == bit-exact S=1 replay. */
static void moe(Model *m, Layer *l, int layer, float *x, int S, float *out){
    Cfg *c=&m->c; int D=c->hidden, E=c->n_experts, K=c->topk, I=c->moe_inter;
    float *logit=falloc(E), *choice=falloc(E);
    /* ---- FASE A: routing di tutte le S posizioni ---- */
    int *idxs=malloc((size_t)S*K*sizeof(int)); float *ws=malloc((size_t)S*K*sizeof(float));
    int *keff=malloc(S*sizeof(int));
    for(int s=0;s<S;s++){
        const float *xs=x+(int64_t)s*D;
        double tr0=now_s();
        matmul(logit, xs, l->router, 1, D, E);
        m->t_router += now_s()-tr0;
        /* in-place sigmoid (colibri #43): reuse logit[], drop separate sig[] buffer */
        for(int e=0;e<E;e++){ logit[e]=sigmoidf(logit[e]); choice[e]=logit[e]+l->router_bias[e]; }
        int *idx=idxs+(int64_t)s*K; float *w=ws+(int64_t)s*K;
        int Ksel = g_topk>0 ? (g_topk<K?g_topk:K) : K;
        for(int kk=0;kk<Ksel;kk++){ int best=-1; float bv=-1e30f;
            for(int e=0;e<E;e++){ int tk=0; for(int j=0;j<kk;j++) if(idx[j]==e){tk=1;break;}
                if(!tk && choice[e]>bv){bv=choice[e];best=e;} }
            idx[kk]=best; w[kk]=logit[best];
        }
        int Ke=Ksel;
        if(g_topp>0 && g_topp<1.f){
            for(int a=1;a<Ksel;a++){ int ii=idx[a]; float ww=w[a]; int b=a-1;
                while(b>=0 && w[b]<ww){ w[b+1]=w[b]; idx[b+1]=idx[b]; b--; } w[b+1]=ww; idx[b+1]=ii; }
            float tot=1e-20f; for(int kk=0;kk<Ksel;kk++) tot+=w[kk];
            float cum=0; for(int kk=0;kk<Ksel;kk++){ cum+=w[kk]; if(cum>=g_topp*tot){ Ke=kk+1; break; } }
        }
        keff[s]=Ke; m->ereq+=Ke;
        for(int kk=0;kk<Ke;kk++){
            m->eusage[layer][idx[kk]]++;
            if(m->eheat[layer][idx[kk]]<UINT32_MAX) m->eheat[layer][idx[kk]]++;
        }
        /* Trajectory: learn co-activations (I/O predictor; lossless) */
        if(g_traj>0 && s==S-1) traj_observe_layer(m, layer, idx, Ke);
        if(c->norm_topk){ float sm=0; for(int kk=0;kk<Ke;kk++) sm+=w[kk]; sm+=1e-20f; for(int kk=0;kk<Ke;kk++) w[kk]/=sm; }
        for(int kk=0;kk<Ke;kk++) w[kk]*=c->routed_scale;
        for(int d=0;d<D;d++) out[(int64_t)s*D+d]=0;
        /* LOOKA: compare this layer's true selection against the PILOT prediction
         * made by the previous layer (pilot_pred[layer]). */
        if(g_looka && pilot_pred_kt[layer]>0){
            int kt=pilot_pred_kt[layer];
            for(int kk=0;kk<Ke;kk++){ int e=idxs[(int64_t)s*K+kk];
                for(int j=0;j<kt;j++) if(pilot_pred[layer][j]==e){ looka_hit++; break; }
                looka_tot++; }
        }
    }
    m->enr[layer]=keff[S-1]; for(int kk=0;kk<keff[S-1];kk++) m->eroute[layer][kk]=idxs[(int64_t)(S-1)*K+kk];
    /* ---- FASE B: union degli expert del batch ---- */
    int *uniq=malloc((size_t)E*sizeof(int)); int nu=0;
    /* stack seen (E=256 on MiMo): avoids calloc/free per layer per token (colibri #43) */
    { unsigned char seen[E]; memset(seen,0,(size_t)E);
      for(int s=0;s<S;s++) for(int kk=0;kk<keff[s];kk++){ int e=idxs[(int64_t)s*K+kk];
          if(!seen[e]){ seen[e]=1; uniq[nu++]=e; } } }
    /* ---- FASE C/D: risolvi (pin/cache/disco) e calcola, a blocchi di 64 unici ---- */
    float *xg=falloc((int64_t)S*D), *gg=falloc((int64_t)S*I), *uu=falloc((int64_t)S*I), *hh=falloc((int64_t)S*D);
    int *rows=malloc(S*sizeof(int)); float *rw=malloc(S*sizeof(float));
    int *rk=malloc(S*sizeof(int));                            /* slot top-k della riga (per contrib) */
    float *contrib = g_fw1 ? falloc((int64_t)S*K*D) : NULL;   /* verifica: contributi per (pos,k) */
#ifdef COLI_CUDA
    /* Decode S=1: accumulate GPU experts on device (1 H2D + 1 D2H per layer).
     * Prefill S>1 keeps per-expert sticky path. */
    int cuda_x_live=0;
    int moe_dev=-1, moe_open=0;
    float *gpu_layer=NULL;   /* host buffer for moe_end result */
    if(g_cuda_enabled && g_cuda_ndev>0) coli_cuda_x_invalidate(g_cuda_devices[0]);
    /* S=1 + no contrib (normal decode): open device accumulate once with layer x. */
    if(g_cuda_enabled && S==1 && !contrib && !omp_in_parallel()){
        moe_dev = g_cuda_devices[0];
        if(coli_cuda_moe_begin(moe_dev, x, 1, D)){
            moe_open=1;
            gpu_layer=falloc(D);
            memset(gpu_layer,0,(size_t)D*sizeof(float));
        }
    }
#endif
    for(int base=0;base<nu;base+=64){
        int nb = nu-base<64 ? nu-base : 64;
        ESlot *use[64]; int missk[64], jq[64]; int nmiss=0;
        for(int j=0;j<nb;j++){ int eid=uniq[base+j]; use[j]=NULL; jq[j]=-1;
#ifdef COLI_CUDA
            if(g_cuda_enabled){                         /* VRAM-only tier: highest priority */
                ESlot *G=m->gpu_pin[layer];
                for(int z=0;z<m->ngpin[layer];z++) if(G[z].eid==eid){ m->hits++; use[j]=&G[z]; break; }
            }
#endif
            { ESlot *P=m->pin[layer];
            for(int z=0;z<m->npin[layer];z++) if(P[z].eid==eid){ m->hits++; use[j]=&P[z]; break; } }
            if(!use[j]){ ESlot *Sl=m->ecache[layer]; int nn=m->ecn[layer];
                for(int z=0;z<nn;z++) if(Sl[z].eid==eid){ m->hits++; Sl[z].used=++m->eclock; use[j]=&Sl[z]; break; } }
            if(!use[j]){ jq[j]=nmiss; use[j]=&m->ws[nmiss]; missk[nmiss++]=j; m->miss++; }
        }
        unsigned ovb=0;
        if(nmiss){
            if(g_overlap) ovb=ov_submit(m,layer,uniq+base,missk,nmiss);   /* load asincroni: si
                                                  * consumano in ordine nel loop di calcolo */
            else { double t0=now_s();
                #pragma omp parallel for schedule(dynamic,1)
                for(int q=0;q<nmiss;q++) expert_load(m,layer,uniq[base+missk[q]],&m->ws[q]);
                m->t_edisk += now_s()-t0; } }
        /* I/O ASINCRONO: readahead (WILLNEED) del blocco SUCCESSIVO mentre calcoliamo
         * questo — il kernel legge in background, le pread dopo trovano cache calda */
        if(base+64<nu){
            int nb2 = nu-(base+64)<64 ? nu-(base+64) : 64;
            for(int j=0;j<nb2;j++){
                int eid=uniq[base+64+j];
                if(!expert_resident(m,layer,eid)) expert_prefetch(m,layer,eid);
            }
        }
        /* ---- Compute: GPU first (queue), then CPU (runs while GPU works) ----
         * colibri #71: overlap resident-GPU FFN with host experts in same block.
         * Decode S=1 + moe_acc is async until layer moe_end.
         * Wait loads per-expert (not all-upfront) so OVERLAP load||compute stays alive.
         * When contrib/g_fw1 (exact order for DRAFT verify): sequential in phase B only. */
        char done[64]; memset(done,0,sizeof(done));
        int nr_arr[64]; float w1[64];
        for(int j=0;j<nb;j++){
            nr_arr[j]=0; w1[j]=0;
            for(int s=0;s<S;s++) for(int kk=0;kk<keff[s];kk++)
                if(idxs[(int64_t)s*K+kk]==uniq[base+j]){
                    w1[j]=ws[(int64_t)s*K+kk]; nr_arr[j]++; break;
                }
        }
#ifdef COLI_CUDA
        /* Phase A: wait+queue GPU experts only (async). Skip if contrib needs ordered sum. */
        if(moe_open && !contrib){
            for(int j=0;j<nb;j++){
                ESlot *e=use[j];
                if(nr_arr[j]<1){ done[j]=1; continue; }
                /* Resident GPU experts need no disk wait; miss-to-ws that is not GPU stays for B. */
                if(!(nr_arr[j]==1 && expert_gpu_ready(e) && e->g.cuda_device==moe_dev)) continue;
                if(g_overlap && jq[j]>=0){ double t0=now_s();
                    ov_wait(ovb+(unsigned)jq[j]); m->t_edisk += now_s()-t0; jq[j]=-1; }
                if(g_cuda_enabled && e->g.cuda_eligible) m->gpu_expert_calls++;
                double t0=now_s();
                if(moe_acc_qt(e, w1[j], 1)){ m->t_emm += now_s()-t0; done[j]=1; }
            }
        }
#endif
        /* Phase B: host / non-moe_acc path (overlaps GPU queue from phase A). */
        for(int j=0;j<nb;j++){
            if(done[j]) continue;
            if(g_overlap && jq[j]>=0){ double t0=now_s();
                ov_wait(ovb+(unsigned)jq[j]); m->t_edisk += now_s()-t0; jq[j]=-1; }
            ESlot *e=use[j];
            int nr=0;
            for(int s=0;s<S;s++) for(int kk=0;kk<keff[s];kk++)
                if(idxs[(int64_t)s*K+kk]==uniq[base+j]){
                    rows[nr]=s; rw[nr]=ws[(int64_t)s*K+kk]; rk[nr]=kk; nr++; break;
                }
            if(!nr) continue;
#ifdef COLI_CUDA
            if(g_cuda_enabled && e->g.cuda_eligible && !(moe_open && !contrib))
                m->gpu_expert_calls++;
#endif
            for(int r=0;r<nr;r++) memcpy(xg+(int64_t)r*D, x+(int64_t)rows[r]*D, D*sizeof(float));
            double t0=now_s();
#ifdef COLI_CUDA
            { int fl=0;
              if(cuda_x_live && (S==1 || nr==S)) fl=COLI_CUDA_SWIGLU_REUSE_X;
              if(swiglu_qt(hh, xg, &e->g, &e->u, &e->d, nr, fl)){
                  if(S==1 || nr==S) cuda_x_live=1;
              } else
#endif
            {
#ifdef COLI_CUDA
              cuda_x_live=0;
#endif
              matmul_qt(gg, xg, &e->g, nr);
              matmul_qt(uu, xg, &e->u, nr);
              for(int64_t z=0;z<(int64_t)nr*I;z++) gg[z]=siluf(gg[z])*uu[z];
              matmul_qt(hh, gg, &e->d, nr); }
#ifdef COLI_CUDA
            }
#endif
            for(int r=0;r<nr;r++){ float *hr=hh+(int64_t)r*D;
                if(contrib) memcpy(contrib+((int64_t)rows[r]*K+rk[r])*D, hr, D*sizeof(float));
                else { float *os=out+(int64_t)rows[r]*D, wgt=rw[r];
                       for(int d=0;d<D;d++) os[d]+=wgt*hr[d]; } }
            m->t_emm += now_s()-t0;
        }
        if(g_overlap && nmiss){ double t0=now_s();   /* ritardatari (expert rimasti senza righe):
                                                      * gli slot ws[] vanno quiescenti PRIMA dello
                                                      * swap LRU e del riuso al blocco successivo */
            for(int q=0;q<nmiss;q++) ov_wait(ovb+(unsigned)q);
            m->t_edisk += now_s()-t0; }
        { ESlot *Sl=m->ecache[layer]; int *nn=&m->ecn[layer];   /* promozione LRU (swap buffer) */
          int promo = nmiss<m->ecap ? nmiss : m->ecap;
          for(int a=0;a<promo;a++){ int q=nmiss-1-a; ESlot *dst;
              if(*nn<m->ecap) dst=&Sl[(*nn)++];
              else { int lru=0; for(int z=1;z<*nn;z++) if(Sl[z].used<Sl[lru].used) lru=z; dst=&Sl[lru]; }
              ESlot tmp=*dst; *dst=m->ws[q]; m->ws[q]=tmp; dst->used=++m->eclock; }
          if(promo>0) res_bits_rebuild_layer(m,layer);   /* bitmap: pin∪gpu∪LRU */
        }
    }
    if(contrib){                                  /* riduzione nell'ordine top-k della posizione */
        for(int s=0;s<S;s++){ float *os=out+(int64_t)s*D;
            for(int kk=0;kk<keff[s];kk++){ float wgt=ws[(int64_t)s*K+kk];
                const float *hr=contrib+((int64_t)s*K+kk)*D;
                for(int d=0;d<D;d++) os[d]+=wgt*hr[d]; } }
        free(contrib);
    }
#ifdef COLI_CUDA
    if(moe_open){
        double t0=now_s();
        if(coli_cuda_moe_end(gpu_layer, 1, D, moe_dev)){
            for(int d=0;d<D;d++) out[d]+=gpu_layer[d];
        }
        m->t_emm += now_s()-t0;
        free(gpu_layer);
    }
#endif
    free(logit); free(choice); free(idxs); free(ws); free(keff); free(uniq);
    free(xg); free(gg); free(uu); free(hh); free(rows); free(rw); free(rk);
}

static void dense_mlp(Layer *l, float *x, int S, int D, int I, float *out){
#ifdef COLI_CUDA
    if(swiglu_qt(out, x, &l->gate_proj, &l->up_proj, &l->down_proj, S, 0)) return;
#endif
    float *g=falloc((int64_t)S*I), *u=falloc((int64_t)S*I);
    matmul_qt(g, x, &l->gate_proj, S);
    matmul_qt(u, x, &l->up_proj,   S);
    for(int64_t i=0;i<(int64_t)S*I;i++) g[i]=siluf(g[i])*u[i];
    matmul_qt(out, g, &l->down_proj, S);
    free(g); free(u);
    (void)D;
}

/* forward di UN layer */
/* === PILOT: router-lookahead disk prefetch (ported from colibri/glm.c) ===============
 * While computing layer L, we apply layer L+1's router to the current state to
 * predict its top-K experts, and warm them with WILLNEED in a separate I/O thread.
 * This is a purely I/O hint: it touches no weights and no compute, so it NEVER
 * changes the output tokens. Measured recall on GLM ~71.6% of true top-8; the
 * misses are minor stalls. PILOT=1 enables the prefetch; LOOKA=1 (without PILOT)
 * measures its recall on MiMo. */
static int in_top(int *a,int n,int v){ for(int i=0;i<n;i++) if(a[i]==v) return 1; return 0; }
static void looka_print(void){
    if(g_looka && looka_tot>0)
        fprintf(stderr,"[looka] PILOT recall top-%d: %.1f%% (%ld/%ld)\n",
                g_pilot_k, 100.0*looka_hit/looka_tot, looka_hit, looka_tot);
}
static void *pilot_worker(void *arg){
    (void)arg;
    for(;;){
        unsigned r=__atomic_load_n(&pilot_r,__ATOMIC_ACQUIRE);
        unsigned w=__atomic_load_n(&pilot_w,__ATOMIC_ACQUIRE);
        if(r==w){ usleep(200); continue; }
        expert_prefetch((Model*)pilot_m, pilot_q[r&4095].l, pilot_q[r&4095].e);
        __atomic_store_n(&pilot_r,r+1,__ATOMIC_RELEASE);
    }
    return NULL;
}
static void pilot_prefetch(Model *m, int lnext, const float *x, int S){
    Cfg *c=&m->c; int D=c->hidden, E=c->n_experts;
    Layer *l = &m->L[lnext];
    int K = g_pilot_k<c->topk ? g_pilot_k : c->topk;
    if(g_pilot && !pilot_m){ pilot_m=(void*)m; pthread_t t; pthread_create(&t,NULL,pilot_worker,NULL); }
    float *nrm=falloc(D), *ch=falloc(E);
    unsigned char *seen = S>8 ? calloc(E,1) : NULL;   /* PREFILL: union dei top-K su tutte le S posizioni */
    for(int s=0;s<S;s++){
        double tr0=now_s();
        rmsnorm(nrm, x+(int64_t)s*D, l->post_ln, D, c->eps);
        matmul(ch, nrm, l->router, 1, D, E);
        m->t_router += now_s()-tr0;
        for(int e=0;e<E;e++) ch[e]=sigmoidf(ch[e])+l->router_bias[e];
        int top[64], kt=0;
        for(int e=0;e<E && kt<K;e++){ int b=-1; for(int f=0;f<E;f++) if((b<0||ch[f]>ch[b])&&!in_top(top,kt,f)) b=f; if(b>=0) top[kt++]=b; }
        for(int k=0;k<K;k++){ int e=top[k];
            if(seen){ seen[e]=1; continue; }           /* prefill: solo raccolta, prefetch dopo (dedup) */
            if(g_looka) pilot_pred[lnext][k]=e;
            if(g_pilot) expert_prefetch(m,lnext,e);   /* WILLNEED: idempotente, ignoriamo i duplicati */
        }
        if(g_looka && !seen) pilot_pred_kt[lnext]=K;
    }
    if(seen){
        /* PREFILL: enqueue l'unione (dedup) al ring del pilot_worker — la fadvise parte
         * dal thread I/O, non ruba tempo al compute. Ring pieno -> drop (e' solo un hint). */
        for(int e=0;e<E;e++) if(seen[e]){
            unsigned w=pilot_w;
            if(w-__atomic_load_n(&pilot_r,__ATOMIC_ACQUIRE)>=4096) break;
            pilot_q[w&4095].l=lnext; pilot_q[w&4095].e=e;
            __atomic_store_n(&pilot_w,w+1,__ATOMIC_RELEASE);
        }
        free(seen);
    }
    free(nrm); free(ch);
}
static void layer_forward(Model *m, Layer *l, int li, float *x, int S, int pos_base, float *nrm, float *tmp){
    Cfg *c=&m->c; int D=c->hidden;
    /* Sticky next-token: experts used for this layer on the previous token are a
     * strong prior for the same layer on the next token (~40%+ baseline; free). */
    if(g_spec && g_prefetch && l->sparse && m->enr[li]>0)
        for(int z=0;z<m->enr[li];z++) expert_prefetch(m,li,m->eroute[li][z]);
    for(int s=0;s<S;s++) rmsnorm(nrm+(int64_t)s*D, x+(int64_t)s*D, l->in_ln, D, c->eps);
    attention(m,l,li,nrm,S,pos_base,tmp);
    for(int64_t j=0;j<(int64_t)S*D;j++) x[j]+=tmp[j];
    /* PILOT: predict L+1 (and optionally L+2) routers while we compute this layer.
     * I/O hint only — tokens unchanged. LOOKA stays decode-only on L+1. */
    if((g_pilot || (g_looka && S<=8)) && li+1<c->n_layers && m->L[li+1].sparse)
        pilot_prefetch(m,li+1,x,S);
    if(g_pilot && g_pilot_depth>=2 && li+2<c->n_layers && m->L[li+2].sparse)
        pilot_prefetch(m,li+2,x,S);
    for(int s=0;s<S;s++) rmsnorm(nrm+(int64_t)s*D, x+(int64_t)s*D, l->post_ln, D, c->eps);
    if(l->sparse) moe(m,l,li,nrm,S,tmp); else dense_mlp(l,nrm,S,D,c->dense_inter,tmp);
    for(int64_t j=0;j<(int64_t)S*D;j++) x[j]+=tmp[j];
}
static void layers_forward(Model *m, float *x, int S, int pos_base){
    Cfg *c=&m->c; int D=c->hidden;
    float *nrm=falloc((int64_t)S*D), *tmp=falloc((int64_t)S*D);
    for(int i=0;i<c->n_layers;i++){
        /* progresso su stderr per i batch grossi (prefill): il primo byte di risposta
         * puo' arrivare dopo MINUTI di streaming — al buio sembra un blocco. */
        if(S>=8 && (i%4==0 || i==c->n_layers-1))
            fprintf(stderr,"[prefill] layer %d/%d · %d token\n", i+1, c->n_layers, S);
        layer_forward(m,&m->L[i],i,x,S,pos_base,nrm,tmp);
    }
    /* Snapshot routes for next-token Markov; warm predicted path for the next step. */
    if(g_traj>0){
        traj_commit_prev(m);
        if(S==1) traj_warm(m, g_traj_k);   /* decode: prepare next token */
    }
    free(nrm); free(tmp);
}

static void kv_alloc(Model *m, int max_t){
    Cfg *c=&m->c; int NR=c->n_layers+1;          /* riga extra: KV del layer MTP (se attivo) */
    if(m->K){ for(int i=0;i<NR;i++){ free(m->K[i]); free(m->V[i]); } free(m->K); free(m->V); }
    m->max_t=max_t;
    rope_build(m, max_t);                 /* precomputed RoPE tables up to max_t (decode/prefill) */
    m->K=calloc(NR,sizeof(float*)); m->V=calloc(NR,sizeof(float*));
    int rows = m->has_mtp ? NR : c->n_layers;    /* MTP KV row only if the head exists */
    double bytes_lin=0, bytes_ring=0;
    for(int i=0;i<rows;i++){
        int tlen=lyr_kv_rows(c,i,max_t);
        int kvh=lyr_kvh(c,i), hd=lyr_hd(c,i), vd=lyr_vd(c,i);
        m->K[i]=falloc((int64_t)tlen*kvh*hd);
        m->V[i]=falloc((int64_t)tlen*kvh*vd);
        bytes_lin  += (double)max_t*kvh*(hd+vd)*4.0;
        bytes_ring += (double)tlen *kvh*(hd+vd)*4.0;
    }
    if(bytes_lin>bytes_ring+1e6)
        fprintf(stderr,"[KV] SWA ring: %.2f GB -> %.2f GB (saved %.2f GB for expert cache budget)\n",
            bytes_lin/1e9, bytes_ring/1e9, (bytes_lin-bytes_ring)/1e9);
}

static void mtp_absorb(Model *m, const int *next_ids, const float *x, int S, int pos_base);
static float *step(Model *m, const int *ids, int S, int pos_base){
    Cfg *c=&m->c; int D=c->hidden;
    float *x=falloc((int64_t)S*D);
    for(int s=0;s<S;s++) embed_row(m, ids[s], x+(int64_t)s*D);
    layers_forward(m,x,S,pos_base);
    memcpy(m->hlast, x+(int64_t)(S-1)*D, D*sizeof(float));
    /* prefill della KV MTP: coppie (token@pos+1, hidden@pos) di tutto il batch */
    if(m->has_mtp && S>=2 && g_draft>0) mtp_absorb(m, ids+1, x, S-1, pos_base);
    float *last=falloc(D); rmsnorm(last, x+(int64_t)(S-1)*D, m->final_norm, D, c->eps);
    double th0=now_s();
    float *logit=falloc(c->vocab); matmul_qt(logit,last,&m->lm_head,1);
    m->t_head += now_s()-th0;
    free(x); free(last); return logit;
}

/* come step(), ma ritorna i logits di TUTTE le S posizioni [S,vocab] (per la verifica spec) */
static float *step_all(Model *m, const int *ids, int S, int pos_base){
    Cfg *c=&m->c; int D=c->hidden;
    float *x=falloc((int64_t)S*D);
    for(int s=0;s<S;s++) embed_row(m, ids[s], x+(int64_t)s*D);
    layers_forward(m,x,S,pos_base);
    if(S<=64) memcpy(m->h_all, x, (int64_t)S*D*sizeof(float));   /* hidden di TUTTE le pos (spec) */
    memcpy(m->hlast, x+(int64_t)(S-1)*D, D*sizeof(float));
    float *lo=falloc((int64_t)S*c->vocab), *row=falloc(D);
    for(int s=0;s<S;s++){ rmsnorm(row, x+(int64_t)s*D, m->final_norm, D, c->eps);
        matmul_qt(lo+(int64_t)s*c->vocab, row, &m->lm_head, 1); }
    free(x); free(row); return lo;
}

/* METODO E — prompt-lookup: cerca l'occorrenza piu' recente dell'ultimo bigramma nel
 * contesto e propone i token che la seguirono. Zero pesi extra, zero costo: e' solo
 * un'ipotesi che il modello verifichera'. */
static int ngram_draft(const int *ids, int len, int G, int *draft){
    if(len<4 || G<1) return 0;
    int a=ids[len-2], b=ids[len-1];
    for(int i=len-3;i>=1;i--)
        if(ids[i-1]==a && ids[i]==b){
            int n=0; for(int j=i+1;j<len && n<G;j++) draft[n++]=ids[j];
            return n;
        }
    return 0;
}

static inline int argmax_v(const float *lo, int V){
    int b=0; float bv=lo[0]; for(int i=1;i<V;i++) if(lo[i]>bv){bv=lo[i];b=i;} return b;
}

/* METODO MTP: propone fino a G draft con la testa multi-token nativa di MiMo-V2.5.
 * Input: next_tok (appena emesso, posizione kv) e hlast (hidden PRE-norm della pos kv-1).
 * Catena vLLM (mimo_v2_mtp.py):
 *   h' = final_layernorm( Blocco( eh_proj[ enorm(emb(tok)) ; hnorm(h) ] ) )
 *   draft = argmax(lm_head(h')); il passo successivo incatena h = h' (vLLM ritorna e
 *   incatena il hidden POST final_layernorm). Il primo h e' il hidden VERO del modello:
 *   pre-norm in hlast, model.norm applicato qui (convenzione vLLM, come per GLM).
 * La KV del layer MTP vive alla riga n_layers ed e' valida da kv_start (niente prefill
 * obbligatorio: la finestra di solo-decode basta per il draft). LOSSLESS comunque:
 * ogni draft viene verificato dal modello pieno in spec_decode. */
static int mtp_draft(Model *m, int next_tok, int kv, int G, int *draft){
    Cfg *c=&m->c; int D=c->hidden, li=c->n_layers, V=c->vocab;
    int p=kv-1; if(p<0||G<1) return 0;
    if(m->kv_start[li]<0 || m->kv_start[li]>p) m->kv_start[li]=p;
    float *x=falloc(D), *cat=falloc(2*D), *hx=falloc(D), *nrm=falloc(D), *tmp=falloc(D);
    float *row=falloc(D), *logit=falloc(V), *h=falloc(D);
    memcpy(h, m->hlast, D*sizeof(float));
    int tok=next_tok, n=0;
    int prenorm = getenv("MTP_PRENORM")!=NULL;   /* A/B: h senza model.norm (debug) */
    for(int g=0; g<G; g++){
        int pos=p+g; if(pos+2>=m->max_t) break;
        embed_row(m, tok, x);
        rmsnorm(x, x, m->enorm, D, c->eps);
        if(g==0 && !prenorm) rmsnorm(h, h, m->final_norm, D, c->eps);  /* h vero: post model.norm */
        rmsnorm(h, h, m->hnorm, D, c->eps);
        memcpy(cat, x, D*sizeof(float)); memcpy(cat+D, h, D*sizeof(float));
        matmul_qt(hx, cat, &m->eh_proj, 1);
        layer_forward(m, &m->mtpL, li, hx, 1, pos, nrm, tmp);
        rmsnorm(row, hx, m->mtp_norm, D, c->eps);
        matmul_qt(logit, row, &m->lm_head, 1);
        int t2=argmax_v(logit, V);
        draft[n++]=t2; tok=t2;
        memcpy(h, row, D*sizeof(float));         /* catena il hidden POST final_layernorm (vLLM) */
    }
    free(x); free(cat); free(hx); free(nrm); free(tmp); free(row); free(logit); free(h);
    return n;
}
/* assorbe nella KV della testa MTP le coppie VERIFICATE (emb(token@pos+1), h_vero@pos):
 * next_ids[i] = token alla posizione pos_base+i+1; x[i] = hidden VERO (pre-norm) a pos_base+i.
 * Un solo passaggio batch del layer MTP (denso: nessun expert da pagare). */
static void mtp_absorb(Model *m, const int *next_ids, const float *x, int S, int pos_base){
    if(!m->has_mtp || S<1) return;
    Cfg *c=&m->c; int D=c->hidden, li=c->n_layers;
    if(pos_base+S>m->max_t){ S=m->max_t-pos_base; if(S<1) return; }
    if(m->kv_start[li]<0 || m->kv_start[li]>pos_base) m->kv_start[li]=pos_base;
    float *hx=falloc((int64_t)S*D), *cat=falloc(2*D), *e=falloc(D), *hn=falloc(D), *hf=falloc(D);
    int prenorm = getenv("MTP_PRENORM")!=NULL;
    for(int i=0;i<S;i++){
        embed_row(m,next_ids[i],e);
        rmsnorm(e,e,m->enorm,D,c->eps);
        if(prenorm) rmsnorm(hn,x+(int64_t)i*D,m->hnorm,D,c->eps);
        else { rmsnorm(hf,x+(int64_t)i*D,m->final_norm,D,c->eps);   /* vLLM: h POST model.norm */
               rmsnorm(hn,hf,m->hnorm,D,c->eps); }
        memcpy(cat,e,D*sizeof(float)); memcpy(cat+D,hn,D*sizeof(float));
        matmul_qt(hx+(int64_t)i*D, cat, &m->eh_proj, 1);
    }
    float *nrm=falloc((int64_t)S*D), *tmp=falloc((int64_t)S*D);
    layer_forward(m,&m->mtpL,li,hx,S,pos_base,nrm,tmp);
    free(hx); free(cat); free(e); free(hn); free(hf); free(nrm); free(tmp);
}

/* ---- SAMPLING (temperatura + nucleus) con verifica speculativa LOSSLESS ----
 * Il draft (MTP/n-gram) e' DETERMINISTICO (argmax della testa): q = massa puntuale.
 * Rejection sampling di Leviathan: accetta il draft x_d con prob p(x_d); al rifiuto
 * ricampiona da p con x_d azzerato e rinormalizzato. La distribuzione risultante e'
 * ESATTAMENTE p: la speculazione resta invisibile all'output anche col sampling. */
static uint64_t g_rng=0x9E3779B97F4A7C15ULL;
static inline double rndu(void){ g_rng^=g_rng<<13; g_rng^=g_rng>>7; g_rng^=g_rng<<17;
    return (double)(g_rng>>11)*(1.0/9007199254740992.0); }
static float *g_pbuf=NULL; static int *g_pidx=NULL;   /* buffer riusati (decode single-thread) */
static int cmp_pdesc(const void *a,const void *b){
    float pa=g_pbuf[*(const int*)a], pb=g_pbuf[*(const int*)b];
    return pa<pb ? 1 : pa>pb ? -1 : 0; }
/* costruisce in g_pbuf la distribuzione target: softmax(lo/temp) troncata a top-p g_nuc */
static void dist_build(const float *lo, int V){
    if(!g_pbuf){ g_pbuf=falloc(V); g_pidx=malloc(V*sizeof(int)); }
    float mx=lo[0]; for(int i=1;i<V;i++) if(lo[i]>mx) mx=lo[i];
    double s=0; float invt=1.f/(g_temp>1e-4f?g_temp:1e-4f);
    for(int i=0;i<V;i++){ g_pbuf[i]=expf((lo[i]-mx)*invt); s+=g_pbuf[i]; }
    for(int i=0;i<V;i++) g_pbuf[i]/=(float)s;
    if(g_nuc>0 && g_nuc<1.f){
        for(int i=0;i<V;i++) g_pidx[i]=i;
        qsort(g_pidx,V,sizeof(int),cmp_pdesc);
        double cum=0; int keep=V;
        for(int i=0;i<V;i++){ cum+=g_pbuf[g_pidx[i]]; if(cum>=g_nuc){ keep=i+1; break; } }
        double s2=0; for(int i=keep;i<V;i++) g_pbuf[g_pidx[i]]=0;
        for(int i=0;i<keep;i++) s2+=g_pbuf[g_pidx[i]];
        for(int i=0;i<keep;i++) g_pbuf[g_pidx[i]]/=(float)s2;
    }
}
/* campiona da g_pbuf; ban>=0 -> quel token e' escluso (rinormalizzando al volo) */
static int dist_sample(int V, int ban){
    double z = 1.0 - (ban>=0 ? g_pbuf[ban] : 0.0); if(z<=1e-12) z=1e-12;
    double u = rndu()*z, cum=0;
    for(int i=0;i<V;i++){ if(i==ban) continue; cum+=g_pbuf[i]; if(cum>=u) return i; }
    for(int i=V-1;i>=0;i--) if(i!=ban && g_pbuf[i]>0) return i;
    return 0;
}
/* prossimo token dai logits: greedy se g_temp<=0, altrimenti sampling.
 * ban = token escluso perche' rifiutato dalla verifica speculativa precedente. */
static int pick_tok(const float *lo, int V, int ban){
    if(g_temp<=0) return argmax_v(lo,V);
    dist_build(lo,V);
    return dist_sample(V,ban);
}

/* stop-set attivo (popolato da run_text/run_serve dal config; vuoto in validazione,
 * dove si genera un numero fisso di token da confrontare con l'oracolo) */
static int g_stop[10], g_nstop=0;
static inline int is_stop(int t){ for(int i=0;i<g_nstop;i++) if(t==g_stop[i]) return 1; return 0; }
static void stops_add(int t){
    if(t>=0 && !is_stop(t) && g_nstop<(int)(sizeof g_stop/sizeof g_stop[0])) g_stop[g_nstop++]=t;
}
/* MiMo-V2.5 generation_config: eos_token_id=[151643 <|endoftext|>, 151645 <|im_end|>,
 * 151672 <|mimo_audio_eod|>] ma il config.json ne dichiara solo uno (151645). I due eos
 * testuali vengono armati per NOME dal tokenizer (robusto anche con config tiny/oracolo);
 * <|mimo_audio_eod|> e' solo per l'uscita audio, irrilevante per questo motore text-only. */
static void stops_arm(const Cfg *c, int tok_eos, int tok_eos2){
    g_nstop=0;
    for(int i=0;i<c->n_stop;i++) stops_add(c->stop_ids[i]);
    stops_add(tok_eos); stops_add(tok_eos2);
    fprintf(stderr,"[stop] %d token di stop:",g_nstop);
    for(int i=0;i<g_nstop;i++) fprintf(stderr," %d",g_stop[i]);
    fprintf(stderr,"\n");
}

/* decode greedy con SELF-SPECULATION n-gram: LOSSLESS (output identico al greedy puro).
 * Ogni forward verifica fino a g_draft token proposti dal contesto: i token accettati
 * costano UNA sola passata sui pesi -> disco e banda RAM ammortizzati su piu' token.
 * all: storia token (capacita' >= kv+n_new+g_draft+2), kv = token gia' in KV.
 * logit = logits della posizione kv-1 (dal prefill); viene liberato qui.
 * emit(tok,ud) per ogni token emesso. Ritorna i token emessi; *kv_out = nuova kv. */
static int spec_decode(Model *m, int *all, int kv, int n_new, int eos, float *logit,
                       void (*emit)(int,void*), void *ud, int *kv_out){
    Cfg *c=&m->c; int V=c->vocab; int emitted=0, done=0;
    int draft[64]; if(g_draft>63) g_draft=63;
    int carry_ban=-1;                    /* token rifiutato dalla verifica: escluso dal resample */
    while(emitted<n_new && !done){
        int next=pick_tok(logit,V,carry_ban); carry_ban=-1; free(logit); logit=NULL;
        if((eos>=0 && next==eos) || is_stop(next)) break;
        emit(next,ud); all[kv]=next; emitted++; m->n_emit++;
        if(emitted>=n_new) break;                       /* l'ultimo token non serve forwardarlo */
        int g = 0;
        /* auto-off adattivo: draft mai accettati = solo tassa di calcolo per forward */
        if(g_draft>0 && m->has_mtp && m->mtp_prop>=24 && m->mtp_acc*10 < m->mtp_prop){
            g_draft=0;
            fprintf(stderr,"[MTP] acceptance %.0f%% dopo %llu proposte: draft disattivati\n",
                100.0*m->mtp_acc/m->mtp_prop, (unsigned long long)m->mtp_prop);
        }
        if(g_draft>0){
            if(m->has_mtp){ g=mtp_draft(m,next,kv,g_draft,draft); m->mtp_prop+=g; }
            else g=ngram_draft(all,kv+1,g_draft,draft);
        }
        if(g>n_new-emitted) g=n_new-emitted;
        if(kv+1+g+1>m->max_t) g=m->max_t-kv-2;
        if(g<0) g=0;
        int S=1+g; int batch[64]; batch[0]=next; memcpy(batch+1,draft,g*sizeof(int));
        g_fw1=1;                                        /* verifica ≡ decode S=1, bit-esatto */
        float *lo=step_all(m,batch,S,kv); g_fw1=0; m->n_fw++;
        int k=0;                                        /* verifica: accetta finche' coincide */
        while(k<g && emitted<n_new){
            int accept;
            if(g_temp<=0) accept = (argmax_v(lo+(int64_t)k*V,V)==draft[k]);
            else { dist_build(lo+(int64_t)k*V,V);          /* rejection sampling: p(draft) */
                   accept = (rndu() < g_pbuf[draft[k]]); }
            if(!accept){ if(g_temp>0) carry_ban=draft[k]; break; }
            if((eos>=0 && draft[k]==eos) || is_stop(draft[k])){ done=1; break; }
            emit(draft[k],ud); all[kv+1+k]=draft[k]; emitted++; m->n_emit++; k++;
        }
        if(m->has_mtp){ m->mtp_acc+=k;
            if(k>=1) mtp_absorb(m, all+kv+1, m->h_all, k, kv);   /* KV MTP in sync coi verificati */
        }
        /* hlast deve corrispondere all'ultima posizione ACCETTATA (kv+k), non a fine batch
         * (per k==S-1 step_all ha gia' copiato la riga giusta) */
        if(k<S-1) memcpy(m->hlast, m->h_all+(int64_t)k*c->hidden, c->hidden*sizeof(float));
        kv += 1+k;                                      /* KV oltre kv e' stantia: verra' sovrascritta */
        logit=falloc(V); memcpy(logit, lo+(int64_t)k*V, V*sizeof(float)); free(lo);
    }
    if(logit) free(logit);
    if(kv_out) *kv_out=kv;
    return emitted;
}

/* emit callback: accumula in un array (validazione) */
typedef struct { int *dst; int n; } EmitStore;
static void emit_store(int t, void *ud){ EmitStore *e=(EmitStore*)ud; e->dst[e->n++]=t; }
/* emit callback: detokenizza e stampa in streaming (chat/run), con heartbeat */
typedef struct { Tok *T; Model *m; double t0; int count; int quiet; } EmitStream;
static void emit_stream(int t, void *ud){
    EmitStream *e=(EmitStream*)ud; char dec[64];
    int dn=tok_decode(e->T,&t,1,dec,63); dec[dn]=0; fputs(dec,stdout); fflush(stdout);
    if(!e->quiet && ++e->count%16==0){ double tt=e->m->hits+e->m->miss;
        fprintf(stderr,"\n[t=%d  RSS %.2f GB  hit %.0f%%  %.2f tok/s  %.2f tok/fw]\n", e->count,
            rss_gb(), tt?100.0*e->m->hits/tt:0.0, e->count/(now_s()-e->t0),
            e->m->n_fw?(double)e->m->n_emit/e->m->n_fw:1.0); }
}

/* teacher-forcing: un solo forward su ids[S], argmax per posizione in pred[S] */
static void forward_all(Model *m, const int *ids, int S, int *pred){
    Cfg *c=&m->c; int D=c->hidden;
    kv_alloc(m,S);
    float *x=falloc((int64_t)S*D);
    for(int s=0;s<S;s++) embed_row(m, ids[s], x+(int64_t)s*D);
    layers_forward(m,x,S,0);
    float *lo=falloc(c->vocab);
    float *row=falloc(D);
    for(int s=0;s<S;s++){
        rmsnorm(row, x+(int64_t)s*D, m->final_norm, D, c->eps);
        matmul_qt(lo, row, &m->lm_head, 1);
        int best=0; float bv=lo[0]; for(int i=1;i<c->vocab;i++) if(lo[i]>bv){bv=lo[i];best=i;}
        pred[s]=best;
    }
    free(row); free(x); free(lo);
}

/* log-prob (log-softmax) del token target dato il vettore di logit; *am=1 se e' l'argmax */
static double logprob_target(const float *lo, int V, int target, int *am){
    float mx=lo[0]; int best=0; for(int i=1;i<V;i++){ if(lo[i]>mx){mx=lo[i];best=i;} }
    double se=0; for(int i=0;i<V;i++) se+=exp((double)lo[i]-mx);
    if(am)*am=(best==target);
    return (double)(lo[target]-mx) - log(se);
}
/* modalita' SCORING per i benchmark (stile lm-eval, log-likelihood):
 * input: file con righe "<ctxlen> <contlen> <id0> .. <id_{T-1}>"  (T=ctxlen+contlen)
 * output: riga "<logprob_continuazione> <contlen> <greedy 0/1>" per richiesta.
 * Un solo forward per richiesta (teacher-forcing): niente generazione -> fattibile a bassa velocita'. */
static void run_score(Model *m, const char *path){
    Cfg *c=&m->c; int D=c->hidden;
    FILE *f=fopen(path,"rb"); if(!f){perror(path);exit(1);}
    int maxT=1; { char *ln=NULL; size_t cp=0;
        while(getline(&ln,&cp,f)>0){ int a,b; if(sscanf(ln,"%d %d",&a,&b)==2 && a+b>maxT) maxT=a+b; }
        free(ln); }
    kv_alloc(m,maxT);
    float *x=falloc((int64_t)maxT*D), *lo=falloc(c->vocab), *row=falloc(D);
    int *ids=malloc(maxT*sizeof(int));
    rewind(f); char *ln=NULL; size_t cp=0; int nreq=0; double t0=now_s();
    while(getline(&ln,&cp,f)>0){
        char *p=ln; int ctxlen=strtol(p,&p,10), contlen=strtol(p,&p,10), T=ctxlen+contlen;
        if(T<=0||ctxlen<1){ printf("0 0 0\n"); fflush(stdout); continue; }
        for(int i=0;i<T;i++) ids[i]=strtol(p,&p,10);
        for(int s=0;s<T;s++) embed_row(m, ids[s], x+(int64_t)s*D);
        layers_forward(m,x,T,0);
        double lp=0; int greedy=1;
        for(int pos=ctxlen-1; pos<T-1; pos++){
            rmsnorm(row, x+(int64_t)pos*D, m->final_norm, D, c->eps);
            matmul_qt(lo,row,&m->lm_head,1);
            int am; lp += logprob_target(lo,c->vocab,ids[pos+1],&am); if(!am) greedy=0;
        }
        printf("%.6f %d %d\n", lp, contlen, greedy); fflush(stdout);
        if(++nreq%5==0) fprintf(stderr,"[score %d req | %.1fs | RSS %.2f GB | hit %.0f%%]\n",
            nreq, now_s()-t0, rss_gb(), (m->hits+m->miss)?100.0*m->hits/(m->hits+m->miss):0.0);
    }
    free(ln); free(ids); free(x); free(lo); free(row); fclose(f);
}

static void generate(Model *m, const int *prompt, int np, int n_new, int *out){
    kv_alloc(m,np+n_new+g_draft+2);
    for(int i=0;i<np;i++) out[i]=prompt[i];
    float *logit=step(m,prompt,np,0);
    EmitStore es={out+np,0};
    spec_decode(m,out,np,n_new,-1,logit,emit_store,&es,NULL);
}

static void profile_print(Model *m, double elapsed){
    double accounted=m->t_edisk+m->t_emm+m->t_attn+m->t_head;
    printf("PROFILE: expert-disk %.3fs | expert-matmul %.3fs | attention %.3fs | lm_head %.3fs | other %.3fs\n",
        m->t_edisk,m->t_emm,m->t_attn,m->t_head,elapsed-accounted);
#ifdef COLI_CUDA
    if(g_cuda_enabled){
        double copy=coli_cuda_copy_seconds()-m->t_cuda_copy; if(copy<0) copy=0;
        double comp=m->t_emm-copy; if(comp<0) comp=0;
        printf("PROFILE-GPU: router %.3fs | cuda-copy(PCIe) %.3fs | cuda-compute %.3fs (matmul total %.3fs)\n",
            m->t_router, copy, comp, m->t_emm);
    }
#endif
}

/* Fixed-token decode benchmark: prefill all but the prompt's last token, then
 * replay the oracle sequence one token at a time. CPU and CUDA therefore see
 * identical hidden-state inputs even if their argmax predictions differ. */
static void run_replay(Model *m, const int *full, int nfull, int np){
    if(np<2||nfull<=np){ fprintf(stderr,"REPLAY richiede prompt e continuation non vuoti\n"); return; }
    kv_alloc(m,nfull+2);
    float *logit=step(m,full,np-1,0); free(logit);
    m->hits=m->miss=m->ereq=m->gpu_expert_calls=0;
    m->t_edisk=m->t_emm=m->t_attn=m->t_head=0;
    m->t_router=0;
#ifdef COLI_CUDA
    m->t_cuda_copy = g_cuda_enabled ? coli_cuda_copy_seconds() : 0;
#endif
    double t0=now_s(); int steps=0;
    for(int i=np-1;i<nfull-1;i++){
        logit=step(m,full+i,1,i); free(logit); steps++;
    }
    double dt=now_s()-t0, tot=m->hits+m->miss;
    printf("REPLAY decode: %d token in %.3fs | %.2f tok/s | expert hit %.1f%%\n",
        steps,dt,steps/dt,tot?100.0*m->hits/tot:0.0);
    profile_print(m,dt);
#ifdef COLI_CUDA
    if(m->gpu_expert_count) printf("CUDA expert tier: %d residenti (%.2f GB) | %llu chiamate servite da VRAM\n",
        m->gpu_expert_count,m->gpu_expert_bytes/1e9,(unsigned long long)m->gpu_expert_calls);
    if(g_cuda_enabled) cuda_stats_print();
#endif
}

/* generazione reale: tokenizza PROMPT, prefill + decode greedy con stop su EOS,
 * detokenizza e stampa il testo in streaming. */
static void run_text(Model *m, const char *snap, const char *prompt, int ngen){
    Cfg *c=&m->c; char tkp[2048]; snprintf(tkp,sizeof(tkp),"%s/tokenizer.json",snap);
    Tok T; tok_load(&T,tkp);
    int eos=tok_id_of(&T,"<|endoftext|>");
    stops_arm(&m->c, eos, tok_id_of(&T,"<|im_end|>"));
    if(g_temp<0) g_temp=1.0f;            /* auto: 1.0 = generation_config MiMo-V2.5 (top_p 0.95
                                          * taglia comunque la coda int4 rumorosa; TEMP per stringere) */
    int cap=(int)strlen(prompt)+16; int *pids=malloc(cap*sizeof(int));
    int np=tok_encode(&T,prompt,(int)strlen(prompt),pids,cap);
    tokens_check_vocab(pids,np,m->c.vocab,"run_text");
    if(np<1){ fprintf(stderr,"prompt vuoto dopo tokenizzazione\n"); return; }
    printf("prompt: %d token | genero fino a %d (stop EOS=%d) | draft %s=%d\n",
        np, ngen, eos, m->has_mtp?"MTP":"n-gram", g_draft);
    fputs(prompt,stdout); fflush(stdout);
    pref_bits_clear(m);                          /* new PROMPT epoch for WILLNEED dedupe */
    kv_alloc(m, np+ngen+g_draft+2);
    int *all=malloc((np+ngen+g_draft+2)*sizeof(int)); memcpy(all,pids,np*sizeof(int));
#ifdef COLI_CUDA
    m->t_cuda_copy = g_cuda_enabled ? coli_cuda_copy_seconds() : 0;
#endif
    m->t_router=0;
    double t=now_s();
    float *logit=step(m,pids,np,0);
    EmitStream es={&T,m,t,0,0};
    int produced=spec_decode(m,all,np,ngen,eos,logit,emit_stream,&es,NULL);
    double dt=now_s()-t;
    double tot=m->hits+m->miss;
    int nsp=0; for(int i=0;i<c->n_layers;i++) if(m->L[i].sparse) nsp++;
    printf("\n---\n%d token in %.2fs (%.2f tok/s) | hit-rate expert %.1f%% | RSS %.2f GB\n",
        produced, dt, produced/dt, tot?100.0*m->hits/tot:0.0, rss_gb());
    printf("expert caricati/token: %.1f (per-layer %.2f su %d; baseline topk=%d) | TOPK=%d TOPP=%.2f\n",
        produced?(double)m->ereq/produced:0.0, (produced&&nsp)?(double)m->ereq/produced/nsp:0.0, nsp, c->topk, g_topk, g_topp);
    printf("speculazione: %.2f token/forward (%llu fw per %llu tok)\n",
        m->n_fw?(double)m->n_emit/m->n_fw:1.0, (unsigned long long)m->n_fw, (unsigned long long)m->n_emit);
    if(m->has_mtp||m->mtp_prop) printf("MTP acceptance: %.1f%% (%llu accettati su %llu proposti)\n",
        m->mtp_prop?100.0*m->mtp_acc/m->mtp_prop:0.0,
        (unsigned long long)m->mtp_acc, (unsigned long long)m->mtp_prop);
#ifdef COLI_CUDA
    if(m->gpu_expert_count) printf("CUDA expert tier: %d residenti (%.2f GB) | %llu chiamate servite da VRAM\n",
        m->gpu_expert_count,m->gpu_expert_bytes/1e9,(unsigned long long)m->gpu_expert_calls);
    if(g_cuda_enabled) cuda_stats_print();
#endif
    profile_print(m,dt);
    free(pids); free(all);
    usage_save(m);
}

/* ---- Chat template ufficiale MiMo-V2.5 (da tokenizer_config.json, subset solo testo) ----
 * Rendering di riferimento (HF apply_chat_template, add_generation_prompt=True):
 *   <|im_start|>system\nYou are MiMo, a helpful AI assistant engineered by Xiaomi.<|im_end|>
 *   <|im_start|>user\n{msg}<|im_end|><|im_start|>assistant\n
 * NB: NESSUN newline tra <|im_end|> e il <|im_start|> successivo; nessun BOS.
 * Turni successivi: la risposta del modello resta in KV SENZA <|im_end|> (spec_decode si
 * ferma PRIMA di emettere lo stop token), quindi il turno n+1 apre chiudendo con <|im_end|>.
 * THINK: equivalente di enable_thinking del template (default 1 come l'ufficiale).
 * THINK=0 -> pre-riempie <think></think> e il modello risponde senza ragionamento.
 * SYSTEM: system prompt custom al posto del default Xiaomi (usato solo al primo turno).
 * In sync con tests/test_mimo_template.py (confronto eseguibile via TEMPLATE_DUMP=1).
 *
 * F-02: never treat snprintf's "would-be length" as bytes written. Returns the
 * number of bytes written (NUL-terminated), or -1 if the result does not fit
 * in [0, cap) (caller must not use buf contents as a full prompt). */
static int snappend(char *buf, int cap, int bl, const char *fmt, ...){
    if(bl<0 || cap<1 || bl>=cap) return -1;
    va_list ap; va_start(ap, fmt);
    int n=vsnprintf(buf+bl, (size_t)(cap-bl), fmt, ap);
    va_end(ap);
    if(n<0 || n>=cap-bl) return -1;   /* truncated or error: do not advance bl by n */
    return bl+n;
}
static int mimo_turn_render(char *buf, int cap, const char *user, int first){
    if(!buf || cap<1 || !user) return -1;
    int think = getenv("THINK") ? atoi(getenv("THINK")) : 1;
    const char *sys = getenv("SYSTEM");
    if(!sys || !*sys) sys = "You are MiMo, a helpful AI assistant engineered by Xiaomi.";
    int bl=0;
    if(first) bl=snappend(buf,cap,bl,"<|im_start|>system\n%s<|im_end|>",sys);
    else      bl=snappend(buf,cap,bl,"<|im_end|>");
    if(bl<0) return -1;
    bl=snappend(buf,cap,bl,"<|im_start|>user\n%s<|im_end|><|im_start|>assistant\n%s",
                user, think ? "" : "<think></think>");
    return bl;
}

/* modalita' SERVE (per la CLI 'coli'): carica il modello UNA volta, poi CHAT conversazionale.
 * KV-cache PERSISTENTE tra i turni: la storia resta in cache, si fa il prefill solo dei
 * token NUOVI -> il modello RICORDA la conversazione e non ri-processa il passato (lossless,
 * piu' umano, piu' veloce). Template chat MiMo-V2.5 con token speciali (CHAT_TEMPLATE=0 -> grezzo).
 * Protocollo: "\x01\x01" "READY" "\x01\x01\n" dopo il load; risposta in streaming; "\x01\x01" "END" "\x01\x01\n" a fine turno.
 * ":reset" (riga "\x02RESET") azzera la memoria. EOF -> esce. */
/* ---- LIVE RE-PIN (REPIN=n): after enough tokens, swap cold hot-store slots for
 * hot ones from the session heat map. Covers BOTH RAM pin and VRAM gpu_pin so the
 * adaptive cache chases the live workload (colibri #26 style). Hysteresis in
 * tier_pick_swap; max 4 RAM + 4 VRAM swaps per pass. .coli_usage stays persistent. */
static int g_repin=0;
static uint64_t g_last_repin=0;
typedef struct { long gain; int l, slot, eid; } RepinCand;
/* expert_resident: see bitmaps near expert_prefetch (O(1) res_bits, includes LRU). */
static int repin_pick_ram(Model *m, RepinCand *out, int maxc){
    Cfg *c=&m->c; int nb=0;
    for(int l=0;l<c->n_layers;l++){
        if(!m->npin || m->npin[l]<1 || !m->eheat[l]) continue;
        ESlot *P=m->pin[l]; int ids[4096], zp, eu; long g;
        int np=m->npin[l]; if(np>4096) np=4096;
        for(int z=0;z<np;z++) ids[z]=P[z].eid;
        if(!tier_pick_swap(m->eheat[l],c->n_experts,ids,np,&zp,&eu,&g)) continue;
        /* Don't pull an expert already on VRAM (or another pin slot). */
        if(expert_resident(m,l,eu)) continue;
        if(nb<maxc){ out[nb].gain=g; out[nb].l=l; out[nb].slot=zp; out[nb].eid=eu; nb++; }
        else { int w=0; for(int b=1;b<maxc;b++) if(out[b].gain<out[w].gain) w=b;
               if(g>out[w].gain){ out[w].gain=g; out[w].l=l; out[w].slot=zp; out[w].eid=eu; } }
    }
    return nb;
}
#ifdef COLI_CUDA
/* Same as tier_pick_swap but cold pool is only gpu_pin; hot excludes all residents. */
static int repin_pick_gpu(Model *m, RepinCand *out, int maxc){
    if(!g_cuda_enabled || !m->gpu_pin) return 0;
    Cfg *c=&m->c; int nb=0;
    for(int l=0;l<c->n_layers;l++){
        if(m->ngpin[l]<1 || !m->eheat[l]) continue;
        ESlot *G=m->gpu_pin[l]; int ng=m->ngpin[l]; if(ng>4096) ng=4096;
        int cold=0;
        for(int z=1;z<ng;z++) if(m->eheat[l][G[z].eid]<m->eheat[l][G[cold].eid]) cold=z;
        int hot=-1; uint32_t fh=0;
        for(int e=0;e<c->n_experts;e++){
            if(expert_resident(m,l,e)) continue;
            if(m->eheat[l][e]>fh){ fh=m->eheat[l][e]; hot=e; }
        }
        if(hot<0) continue;
        uint32_t fc=m->eheat[l][G[cold].eid];
        if(fh<=fc+(fc>>2)+4) continue;            /* same hysteresis as tier_pick_swap */
        long g=(long)fh-(long)fc;
        if(nb<maxc){ out[nb].gain=g; out[nb].l=l; out[nb].slot=cold; out[nb].eid=hot; nb++; }
        else { int w=0; for(int b=1;b<maxc;b++) if(out[b].gain<out[w].gain) w=b;
               if(g>out[w].gain){ out[w].gain=g; out[w].l=l; out[w].slot=cold; out[w].eid=hot; } }
    }
    return nb;
}
#endif
/* ---- TRAJECTORY predictor (I/O only, never changes tokens) ----------------------------
 * Learn co-activations: (1) same layer next token, (2) layer L → L+1 same token.
 * Bulk WILLNEED the predicted path so cold misses become warm hits.
 * Env: TRAJ=1 (default ON in SERVE), TRAJ_K=8, TRAJ_DEPTH=2, TRAJ=0 off. */
#define TRAJ_LMAX 128
#define TRAJ_EMAX 256
#define TRAJ_SUC  8
typedef struct { int16_t e; uint32_t c; } TrajEdge;
static TrajEdge g_traj_tok[TRAJ_LMAX][TRAJ_EMAX][TRAJ_SUC];
static TrajEdge g_traj_lay[TRAJ_LMAX][TRAJ_EMAX][TRAJ_SUC];
static int g_traj_prev[TRAJ_LMAX][64], g_traj_prev_n[TRAJ_LMAX];
static int g_traj_have_prev=0;
static long g_traj_warm_n=0;
/* Profile name shared with usage_path_set / mem_watch (must precede traj_path_set). */
static char g_profile[72];
static void traj_bump(TrajEdge suc[TRAJ_SUC], int e2){
    if(e2<0 || e2>=TRAJ_EMAX) return;
    int slot=-1, empty=-1;
    for(int i=0;i<TRAJ_SUC;i++){
        if(suc[i].e==e2){ slot=i; break; }
        if(suc[i].c==0 && empty<0) empty=i;
    }
    if(slot<0){
        if(empty>=0){ suc[empty].e=(int16_t)e2; suc[empty].c=1; return; }
        int w=0; for(int i=1;i<TRAJ_SUC;i++) if(suc[i].c<suc[w].c) w=i;
        suc[w].e=(int16_t)e2; suc[w].c=1; return;
    }
    if(suc[slot].c<UINT32_MAX) suc[slot].c++;
}
/* Insert/replace edge with absolute count (for disk load). */
static void traj_set_edge(TrajEdge suc[TRAJ_SUC], int e2, uint32_t cnt){
    if(e2<0 || e2>=TRAJ_EMAX || cnt==0) return;
    int slot=-1, empty=-1, weak=0;
    for(int i=0;i<TRAJ_SUC;i++){
        if(suc[i].e==e2){ slot=i; break; }
        if(suc[i].c==0 && empty<0) empty=i;
        if(suc[i].c<suc[weak].c) weak=i;
    }
    if(slot>=0){ if(cnt>suc[slot].c) suc[slot].c=cnt; return; }
    int dst = empty>=0 ? empty : weak;
    if(empty<0 && cnt<=suc[weak].c) return; /* table full of stronger edges */
    suc[dst].e=(int16_t)e2; suc[dst].c=cnt;
}
static char g_traj_path[2100]="";
static void traj_path_set(const char *snap){
    if(!snap||!*snap){ g_traj_path[0]=0; return; }
    if(g_profile[0])
        snprintf(g_traj_path,sizeof(g_traj_path),"%s/.coli_traj.%s",snap,g_profile);
    else
        snprintf(g_traj_path,sizeof(g_traj_path),"%s/.coli_traj",snap);
}
static int traj_save(void){
    if(g_traj<=0 || !g_traj_path[0]) return 0;
    char tmp[2200]; snprintf(tmp,sizeof(tmp),"%s.tmp",g_traj_path);
    FILE *f=fopen(tmp,"w"); if(!f) return 0;
    fprintf(f,"# coli_traj v1\n");
    int edges=0;
    for(int l=0;l<TRAJ_LMAX;l++)
        for(int e=0;e<TRAJ_EMAX;e++)
            for(int s=0;s<TRAJ_SUC;s++){
                if(g_traj_tok[l][e][s].c){
                    fprintf(f,"tok %d %d %d %u\n",l,e,(int)g_traj_tok[l][e][s].e,
                            g_traj_tok[l][e][s].c);
                    edges++;
                }
                if(g_traj_lay[l][e][s].c){
                    fprintf(f,"lay %d %d %d %u\n",l,e,(int)g_traj_lay[l][e][s].e,
                            g_traj_lay[l][e][s].c);
                    edges++;
                }
            }
    fclose(f);
    if(rename(tmp,g_traj_path)!=0){ remove(tmp); return 0; }
    if(edges>0)
        fprintf(stderr,"[TRAJ] saved %d edges -> %s (willneed_calls=%ld)\n",
                edges,g_traj_path,g_traj_warm_n);
    return edges;
}
static int traj_load(void){
    if(g_traj<=0 || !g_traj_path[0]) return 0;
    FILE *f=fopen(g_traj_path,"r"); if(!f) return 0;
    char line[256], kind[8]; int L,e0,e1; unsigned cnt; int n=0;
    while(fgets(line,sizeof(line),f)){
        if(line[0]=='#' || line[0]=='\n' || line[0]=='\r') continue;
        if(sscanf(line,"%7s %d %d %d %u",kind,&L,&e0,&e1,&cnt)!=5) continue;
        if(L<0||L>=TRAJ_LMAX||e0<0||e0>=TRAJ_EMAX||e1<0||e1>=TRAJ_EMAX||!cnt) continue;
        if(kind[0]=='t' && kind[1]=='o'){ traj_set_edge(g_traj_tok[L][e0],e1,cnt); n++; }
        else if(kind[0]=='l' && kind[1]=='a'){ traj_set_edge(g_traj_lay[L][e0],e1,cnt); n++; }
    }
    fclose(f);
    if(n>0) fprintf(stderr,"[TRAJ] loaded %d edges from %s\n",n,g_traj_path);
    return n;
}
static void traj_observe_layer(Model *m, int layer, const int *idx, int Ke){
    if(g_traj<=0 || !idx || Ke<1 || layer<0 || layer>=TRAJ_LMAX) return;
    if(Ke>64) Ke=64;
    if(layer>0 && m->L[layer-1].sparse && m->enr[layer-1]>0){
        int lp=layer-1; int n0=m->enr[lp]; if(n0>64) n0=64;
        for(int i=0;i<n0;i++){
            int e0=m->eroute[lp][i]; if(e0<0||e0>=TRAJ_EMAX) continue;
            for(int j=0;j<Ke;j++) traj_bump(g_traj_lay[lp][e0], idx[j]);
        }
    }
    if(g_traj_have_prev && g_traj_prev_n[layer]>0){
        int n0=g_traj_prev_n[layer]; if(n0>64) n0=64;
        for(int i=0;i<n0;i++){
            int e0=g_traj_prev[layer][i]; if(e0<0||e0>=TRAJ_EMAX) continue;
            for(int j=0;j<Ke;j++) traj_bump(g_traj_tok[layer][e0], idx[j]);
        }
    }
}
static void traj_commit_prev(Model *m){
    if(g_traj<=0 || !m) return;
    Cfg *c=&m->c;
    for(int l=0;l<c->n_layers && l<TRAJ_LMAX;l++){
        if(!m->L[l].sparse){ g_traj_prev_n[l]=0; continue; }
        int n=m->enr[l]; if(n<0) n=0; if(n>64) n=64;
        g_traj_prev_n[l]=n;
        for(int i=0;i<n;i++) g_traj_prev[l][i]=m->eroute[l][i];
    }
    g_traj_have_prev=1;
}
/* Cap posix_fadvise storm: each expert_prefetch ≈ 6 fadvise; WSL tax is real.
 * Measured: uncapped TRAJ_K=16 → ~1300 prefetches/token → "other" >> disk+matmul. */
static int g_traj_pref_budget=64;  /* max expert_prefetch calls per traj_warm */
static void traj_pref(Model *m, int layer, int eid, int *left){
    if(!left || *left<=0) return;
    if(expert_resident(m,layer,eid)) return;
    expert_prefetch(m,layer,eid);
    g_traj_warm_n++;
    (*left)--;
}
static void traj_warm(Model *m, int k_per_layer){
    if(g_traj<=0 || !m || k_per_layer<1) return;
    Cfg *c=&m->c;
    int kk=k_per_layer<8?k_per_layer:8;          /* hard cap: fan-out beyond 8 is syscall waste */
    int depth=g_traj_depth<1?1:(g_traj_depth>2?2:g_traj_depth); /* depth>2 rarely pays */
    int budget=g_traj_pref_budget;
    if(getenv("TRAJ_BUDGET")){
        budget=atoi(getenv("TRAJ_BUDGET"));
        if(budget<8) budget=8; if(budget>256) budget=256;
    }
    int left=budget;
    for(int l=0;l<c->n_layers && l<TRAJ_LMAX && left>0;l++){
        if(!m->L[l].sparse) continue;
        int E=c->n_experts; if(E>TRAJ_EMAX) E=TRAJ_EMAX;
        float score[TRAJ_EMAX];
        for(int e=0;e<E;e++){
            uint32_t h = (m->eheat && m->eheat[l]) ? m->eheat[l][e]
                        : (m->eusage && m->eusage[l] ? m->eusage[l][e] : 0);
            score[e]=(float)h;
        }
        int nsticky=0; int sticky[64];
        if(m->enr[l]>0){
            nsticky=m->enr[l]<64?m->enr[l]:64;
            for(int i=0;i<nsticky;i++){
                sticky[i]=m->eroute[l][i];
                if(sticky[i]>=0 && sticky[i]<E) score[sticky[i]] += 1e7f;
            }
        } else if(g_traj_prev_n[l]>0){
            nsticky=g_traj_prev_n[l]<64?g_traj_prev_n[l]:64;
            for(int i=0;i<nsticky;i++){
                sticky[i]=g_traj_prev[l][i];
                if(sticky[i]>=0 && sticky[i]<E) score[sticky[i]] += 1e7f;
            }
        }
        /* Cold boot / first token: Markov mass into scores (prefetch only top-K below). */
        if(nsticky==0){
            for(int e0=0;e0<E;e0++){
                for(int s=0;s<TRAJ_SUC;s++){
                    TrajEdge ed=g_traj_tok[l][e0][s];
                    if(ed.c && ed.e>=0 && ed.e<E){
                        score[ed.e] += 80.f * (float)ed.c;
                        score[e0]   += 8.f  * (float)ed.c;
                    }
                }
            }
            /* Strongest L→L+1 only (top 4), not all edges. */
            {
                int best_e[4]; uint32_t best_c[4]; int nb=0;
                for(int e0=0;e0<E;e0++) for(int s=0;s<TRAJ_SUC;s++){
                    TrajEdge el=g_traj_lay[l][e0][s];
                    if(el.c==0 || el.e<0 || el.e>=c->n_experts || el.e>=TRAJ_EMAX) continue;
                    if(nb<4){ best_e[nb]=el.e; best_c[nb]=(uint32_t)el.c; nb++; }
                    else {
                        int w=0; for(int i=1;i<4;i++) if(best_c[i]<best_c[w]) w=i;
                        if((uint32_t)el.c>best_c[w]){ best_e[w]=el.e; best_c[w]=(uint32_t)el.c; }
                    }
                }
                for(int i=0;i<nb && left>0;i++) traj_pref(m,l+1,best_e[i],&left);
            }
        }
        /* Sticky: boost scores; prefetch only strongest 2 L→L+1 successors per sticky. */
        for(int i=0;i<nsticky;i++){
            int e0=sticky[i]; if(e0<0||e0>=E) continue;
            for(int s=0;s<TRAJ_SUC;s++){
                TrajEdge ed=g_traj_tok[l][e0][s];
                if(ed.c==0 || ed.e<0 || ed.e>=E) continue;
                score[ed.e] += 100.f * (float)ed.c;
            }
            if(l+1<c->n_layers && m->L[l+1].sparse){
                /* pick top-2 lay edges by count for this source */
                int e_best[2]={-1,-1}; uint32_t c_best[2]={0,0};
                for(int s=0;s<TRAJ_SUC;s++){
                    TrajEdge ed=g_traj_lay[l][e0][s];
                    if(ed.c==0 || ed.e<0 || ed.e>=c->n_experts || ed.e>=TRAJ_EMAX) continue;
                    if(ed.c>c_best[0]){ c_best[1]=c_best[0]; e_best[1]=e_best[0];
                                        c_best[0]=(uint32_t)ed.c; e_best[0]=ed.e; }
                    else if(ed.c>c_best[1]){ c_best[1]=(uint32_t)ed.c; e_best[1]=ed.e; }
                }
                for(int j=0;j<2 && left>0;j++) if(e_best[j]>=0) traj_pref(m,l+1,e_best[j],&left);
            }
        }
        int frontier[8]; int nf=0;
        for(int t=0;t<kk && t<8;t++){
            int best=-1; float bv=-1.f;
            for(int e=0;e<E;e++){
                int skip=0; for(int j=0;j<nf;j++) if(frontier[j]==e){skip=1;break;}
                if(skip) continue;
                if(score[e]>bv){ bv=score[e]; best=e; }
            }
            if(best<0 || bv<=0) break;
            frontier[nf++]=best;
        }
        for(int d=1;d<depth;d++){
            int nadd=nf;
            for(int i=0;i<nadd;i++){
                int e0=frontier[i];
                for(int s=0;s<TRAJ_SUC;s++){
                    TrajEdge ed=g_traj_tok[l][e0][s];
                    if(ed.c==0 || ed.e<0 || ed.e>=E) continue;
                    score[ed.e] += (50.f/(float)d) * (float)ed.c;
                }
            }
            nf=0;
            for(int t=0;t<kk && t<8;t++){
                int best=-1; float bv=-1.f;
                for(int e=0;e<E;e++){
                    int skip=0; for(int j=0;j<nf;j++) if(frontier[j]==e){skip=1;break;}
                    if(skip) continue;
                    if(score[e]>bv){ bv=score[e]; best=e; }
                }
                if(best<0 || bv<=0) break;
                frontier[nf++]=best;
            }
        }
        for(int i=0;i<nf && left>0;i++) traj_pref(m,l,frontier[i],&left);
    }
}

/* Proactive cache warm (literature: ProMoE / FineMoE heat maps): WILLNEED the
 * hottest non-resident experts per layer so the next tokens hit page cache.
 * Pure I/O hint — free if already warm. Also runs trajectory bulk warm (TRAJ). */
static void heat_prefetch_top(Model *m, int k_per_layer){
    if(!m || k_per_layer<1) return;
    Cfg *c=&m->c;
    for(int l=0;l<c->n_layers;l++){
        if(!m->L[l].sparse) continue;
        uint32_t *h = m->eheat && m->eheat[l] ? m->eheat[l] : (m->eusage?m->eusage[l]:NULL);
        if(!h) continue;
        int picked[32]; int np=0;
        int kk = k_per_layer<32?k_per_layer:32;
        for(int t=0;t<kk;t++){
            int best=-1; uint32_t bh=0;
            for(int e=0;e<c->n_experts;e++){
                int skip=0; for(int j=0;j<np;j++) if(picked[j]==e){skip=1;break;}
                if(skip || expert_resident(m,l,e)) continue;
                if(h[e]>bh){ bh=h[e]; best=e; }
            }
            if(best<0 || bh==0) break;
            picked[np++]=best;
            expert_prefetch(m,l,best);
        }
    }
    /* Trajectory bulk: sticky + heat + Markov successors across layers/tokens */
    if(g_traj>0) traj_warm(m, g_traj_k>k_per_layer?g_traj_k:k_per_layer);
}

/* ---- RAM DINAMICA a runtime (colibri #71 / Fase 4) ----
 * Boot cap is a snapshot; at each SERVE turn boundary re-measure MemAvailable
 * and adapt the expert LRU: shrink under pressure (free real slabs, beat swap/OOM),
 * grow when free RAM returns. Dead band 3.5–6 GB free avoids thrashing.
 * MEMWATCH=0 disables. */
static int g_memwatch=1;
static double mem_available_gb(void);
static int64_t expert_bytes_probe(Model *m, int ebits);
static void eslot_release(ESlot *s){          /* free host backing of one LRU slot */
    QT *q[3]={&s->g,&s->u,&s->d};
#ifdef COLI_CUDA
    qt_cuda_reset(&s->g); qt_cuda_reset(&s->u); qt_cuda_reset(&s->d);
#endif
    if(s->slab){ free(s->slab); free(s->fslab); }
    else for(int k=0;k<3;k++){ free(q[k]->qf); free(q[k]->q8); free(q[k]->q4); free(q[k]->s); }
    memset(s,0,sizeof(*s));
}
static void mem_watch_pass(Model *m){
    if(!g_memwatch) return;
    double avail=mem_available_gb(); if(avail<=0) return;
    Cfg *c=&m->c; int nsp=0; for(int i=0;i<c->n_layers;i++) if(m->L[i].sparse) nsp++;
    if(nsp<1) return;
    int64_t eb=expert_bytes_probe(m,m->ebits);
    double slot_gb=(double)nsp*eb/1e9;        /* cost of ±1 cap across all sparse layers */
    if(slot_gb<=0) return;
    if(avail<3.5){                            /* pressure: page-cache / OS at risk */
        int drop=(int)((3.5-avail)/slot_gb)+1;
        int newcap=m->ecap-drop; if(newcap<1) newcap=1;
        if(newcap>=m->ecap) return;
        for(int i=0;i<c->n_layers;i++){
            if(!m->ecache[i]) continue;
            while(m->ecn[i]>newcap){          /* real LRU evict: free RAM, not just lower ecap */
                int lru=0;
                for(int z=1;z<m->ecn[i];z++) if(m->ecache[i][z].used<m->ecache[i][lru].used) lru=z;
                eslot_release(&m->ecache[i][lru]);
                m->ecache[i][lru]=m->ecache[i][m->ecn[i]-1];
                memset(&m->ecache[i][m->ecn[i]-1],0,sizeof(ESlot));
                m->ecn[i]--;
            }
        }
        fprintf(stderr,"[RAM] pressure: %.1f GB free -> cap %d->%d (LRU slots freed)%s\n",
            avail,m->ecap,newcap, g_profile[0]?g_profile:"");
        m->ecap=newcap;
        res_bits_rebuild_all(m);
        /* After shrink, warm hottest non-residents — next turn pays less disk. */
        heat_prefetch_top(m, 3);
    } else if(avail>6.0 && m->ecap<c->n_experts){   /* headroom: grow LRU */
        int grow=(int)((avail-6.0)*0.90/slot_gb);
        if(grow<1) return;
        int newcap=m->ecap+grow; if(newcap>c->n_experts) newcap=c->n_experts;
        if(newcap<=m->ecap) return;
        for(int i=0;i<c->n_layers;i++) if(m->ecache[i]){
            ESlot *p=realloc(m->ecache[i],(size_t)newcap*sizeof(ESlot));
            if(!p){ fprintf(stderr,"[RAM] headroom: realloc ecache failed (cap stays %d)\n",m->ecap); return; }
            m->ecache[i]=p;
            memset(m->ecache[i]+m->ecap,0,(size_t)(newcap-m->ecap)*sizeof(ESlot));
        }
        fprintf(stderr,"[RAM] headroom: %.1f GB free -> cap RAISED %d->%d (MEMWATCH=0 to disable)\n",
            avail,m->ecap,newcap);
        m->ecap=newcap;
    }
}

static void repin_pass(Model *m){
    if(g_repin<=0) return;
    if(m->n_emit - g_last_repin < (uint64_t)g_repin) return;
    g_last_repin = m->n_emit;
    /* ---- RAM pin swaps ---- */
    RepinCand cd[4]; int nb=repin_pick_ram(m,cd,4);
    for(int b=0;b<nb;b++){
        if(expert_resident(m,cd[b].l,cd[b].eid) &&
           !(m->pin[cd[b].l][cd[b].slot].eid==cd[b].eid)){
            /* target already elsewhere (e.g. VRAM) — skip */
            continue;
        }
        ESlot *s=&m->pin[cd[b].l][cd[b].slot];
        int old=s->eid;
        uint32_t old_heat=m->eheat[cd[b].l][old], new_heat=m->eheat[cd[b].l][cd[b].eid];
        double t0=now_s();
        expert_load(m,cd[b].l,cd[b].eid,s);
        m->t_edisk += now_s()-t0;                /* REPIN I/O belongs in disk, not "other" */
        fprintf(stderr,"[REPIN] RAM layer %d: out %d (heat=%u) <- in %d (heat=%u) in %.0f ms\n",
            cd[b].l,old,old_heat,cd[b].eid,new_heat,(now_s()-t0)*1e3);
    }
#ifdef COLI_CUDA
    /* ---- VRAM gpu_pin swaps (hottest non-residents replace coldest GPU slots) ---- */
    if(g_cuda_enabled && m->gpu_pin){
        RepinCand gd[4]; int ng=repin_pick_gpu(m,gd,4);
        for(int b=0;b<ng;b++){
            ESlot *s=&m->gpu_pin[gd[b].l][gd[b].slot];
            int old=s->eid;
            uint32_t old_heat=m->eheat[gd[b].l][old], new_heat=m->eheat[gd[b].l][gd[b].eid];
            int64_t old_gpu=(int64_t)coli_cuda_tensor_bytes(s->g.cuda)
                           +(int64_t)coli_cuda_tensor_bytes(s->u.cuda)
                           +(int64_t)coli_cuda_tensor_bytes(s->d.cuda);
            double t0=now_s();
            /* Drop old device tensors, load new expert to host, re-upload, free host. */
            qt_cuda_reset(&s->g); qt_cuda_reset(&s->u); qt_cuda_reset(&s->d);
            s->g.cuda_eligible=s->u.cuda_eligible=s->d.cuda_eligible=0;
            s->g.gpu_only=s->u.gpu_only=s->d.gpu_only=0;
            m->gpu_expert_bytes-=old_gpu;
            if(m->gpu_expert_count>0) m->gpu_expert_count--;
            expert_load(m,gd[b].l,gd[b].eid,s);
            s->g.cuda_device=s->u.cuda_device=s->d.cuda_device=
                (g_cuda_ndev>0?g_cuda_devices[0]:0);
            s->g.cuda_eligible=s->u.cuda_eligible=s->d.cuda_eligible=1;
            if(qt_cuda_upload(&s->g) && qt_cuda_upload(&s->u) && qt_cuda_upload(&s->d)){
                int64_t now_gpu=(int64_t)coli_cuda_tensor_bytes(s->g.cuda)
                               +(int64_t)coli_cuda_tensor_bytes(s->u.cuda)
                               +(int64_t)coli_cuda_tensor_bytes(s->d.cuda);
                m->gpu_expert_bytes+=now_gpu; m->gpu_expert_count++;
                expert_cpu_free(m,s);
                s->g.gpu_only=s->u.gpu_only=s->d.gpu_only=1;
                m->t_edisk += now_s()-t0;
                fprintf(stderr,"[REPIN] VRAM layer %d: out %d (heat=%u) <- in %d (heat=%u) in %.0f ms\n",
                    gd[b].l,old,old_heat,gd[b].eid,new_heat,(now_s()-t0)*1e3);
            } else {
                /* Keep host copy as degraded RAM-like slot inside gpu_pin array
                 * (still found by moe lookup); mark not gpu_only. */
                qt_cuda_reset(&s->g); qt_cuda_reset(&s->u); qt_cuda_reset(&s->d);
                s->g.cuda_eligible=s->u.cuda_eligible=s->d.cuda_eligible=0;
                fprintf(stderr,"[REPIN] VRAM layer %d: upload failed for e=%d; slot stays host-only\n",
                    gd[b].l,gd[b].eid);
            }
        }
    }
#endif
    for(int l=0;l<m->c.n_layers;l++) if(m->eheat[l]) tier_decay(m->eheat[l],m->c.n_experts);
    res_bits_rebuild_all(m);                     /* REPIN changed residency */
    /* After swaps, warm the new hot set into page cache. */
    heat_prefetch_top(m, 4);
}
static void run_serve(Model *m, const char *snap){
    char tkp[2048]; snprintf(tkp,sizeof(tkp),"%s/tokenizer.json",snap);
    Tok T; tok_load(&T,tkp);
    int eos=tok_id_of(&T,"<|endoftext|>");
    stops_arm(&m->c, eos, tok_id_of(&T,"<|im_end|>"));
    if(g_temp<0) g_temp=1.0f;            /* auto: 1.0 = generation_config MiMo-V2.5 (top_p 0.95
                                          * taglia comunque la coda int4 rumorosa; TEMP per stringere) */
    int ngen=getenv("NGEN")?atoi(getenv("NGEN")):256;
    int maxctx=getenv("CTX")?atoi(getenv("CTX")):4096;
    int templ=getenv("CHAT_TEMPLATE")?atoi(getenv("CHAT_TEMPLATE")):1;
    kv_alloc(m,maxctx);
    int len=0, first=1;                          /* len = contesto gia' in KV (persiste tra turni) */
    int *hist=malloc(maxctx*sizeof(int));        /* storia token (= contenuto della KV): serve
                                                  * al lookup n-gram e resta allineata a len */
    char *line=NULL; size_t cap=0; ssize_t nr; char *buf=malloc(1<<16);
    /* Kick page-cache for hottest experts before first token (ProMoE-style). */
    heat_prefetch_top(m, 6);
    printf("\x01\x01" "READY" "\x01\x01\n"); printf("STAT 0 0.00 0.0 %.2f\n", rss_gb()); fflush(stdout);
    while((nr=getline(&line,&cap,stdin))>0){
        if(nr>0 && line[nr-1]=='\n') line[--nr]=0;
        if(!strcmp(line,"\x02RESET")){ len=0; first=1;
            if(m->has_mtp) m->kv_start[m->c.n_layers]=-1;   /* la finestra MTP riparte da sola */
            printf("\x01\x01" "END" "\x01\x01\n"); printf("STAT 0 0.00 0.0 %.2f\n", rss_gb()); fflush(stdout); continue; }
        if(!strcmp(line,"\x02MORE")){                /* continua la risposta troncata da NGEN:
            la storia e' gia' in KV, basta ri-forwardare l'ULTIMO token per riavere i logits */
            if(len<1){ printf("\x01\x01" "END" "\x01\x01\n"); printf("STAT 0 0.00 0.0 %.2f\n", rss_gb()); fflush(stdout); continue; }
            int cur=ngen; if(len+cur+g_draft+2>=maxctx) cur=maxctx-len-g_draft-2;
            uint64_t h0=m->hits, ms0=m->miss; double tt0=now_s();
            float *logit=step(m,hist+len-1,1,len-1);
            EmitStream es={&T,m,now_s(),0,1};
            int prod=0;
            if(cur>0) prod=spec_decode(m,hist,len,cur,eos,logit,emit_stream,&es,&len);
            else free(logit);
            double tdt=now_s()-tt0; if(tdt<1e-6) tdt=1e-6;
            double dh=(double)(m->hits-h0), dm=(double)(m->miss-ms0);
            printf("\n\x01\x01" "END" "\x01\x01\n");
            printf("STAT %d %.2f %.1f %.2f\n", prod, prod/tdt, (dh+dm)>0?100.0*dh/(dh+dm):0.0, rss_gb());
            fflush(stdout); mem_watch_pass(m); repin_pass(m); continue; }   /* #71: RAM watch + live re-pin between turns */
        if(nr<1){ printf("\x01\x01" "END" "\x01\x01\n"); printf("STAT 0 0.00 0.0 %.2f\n", rss_gb()); fflush(stdout); continue; }
        /* API mode: an exact, length-prefixed prompt. Unlike the interactive
         * line protocol this accepts newlines. The tokenized prompt is matched
         * against hist so the common KV prefix survives stateless HTTP turns.
         * Per-request generation controls follow the byte count:
         *   \x02PROMPT <bytes> <max_tokens> <temperature> <top_p>\n<prompt>\n */
        char *raw=NULL, *input=line;
        int input_n=(int)nr, raw_mode=0, req_ngen=ngen, prompt_tokens=0;
        float base_temp=g_temp, base_nuc=g_nuc;
        if(!strncmp(line,"\x02PROMPT ",8)){
            unsigned long long nb=0; double rt=0, rp=0;
            if(sscanf(line+8,"%llu %d %lf %lf",&nb,&req_ngen,&rt,&rp)!=4 ||
               nb>(16u<<20) || req_ngen<1 || rt<0 || rt>2 || rp<=0 || rp>1){
                printf("\x01\x01" "END" "\x01\x01\n"); printf("STAT 0 0.00 0.0 %.2f 0 0\n",rss_gb()); fflush(stdout); continue;
            }
            raw=malloc((size_t)nb+1); if(!raw){fprintf(stderr,"OOM raw prompt\n");exit(1);}
            if(fread(raw,1,(size_t)nb,stdin)!=(size_t)nb){free(raw);break;}
            int delim=fgetc(stdin); if(delim!='\n' && delim!=EOF) ungetc(delim,stdin);
            if(memchr(raw,0,(size_t)nb)){free(raw); printf("\x01\x01" "END" "\x01\x01\n");
                printf("STAT 0 0.00 0.0 %.2f 0 0\n",rss_gb()); fflush(stdout); continue;}
            raw[nb]=0; input=raw; input_n=(int)nb; raw_mode=1;
            if(req_ngen>ngen) req_ngen=ngen;
            g_temp=(float)rt; g_nuc=(float)rp;
        }
        int bl=0, k=0;                           /* costruisce/tokenizza il turno */
        if(raw_mode){
            int *tmp=malloc(maxctx*sizeof(int)); if(!tmp){fprintf(stderr,"OOM raw tokens\n");exit(1);}
            prompt_tokens=tok_encode(&T,input,input_n,tmp,maxctx-8-g_draft);
            tokens_check_vocab(tmp,prompt_tokens,m->c.vocab,"serve_raw");
            int old_len=len, prefix=0;
            while(prefix<old_len && prefix<prompt_tokens && hist[prefix]==tmp[prefix]) prefix++;
            if(prefix<old_len){ len=prefix;
                if(m->has_mtp) m->kv_start[m->c.n_layers]=-1;  /* KV MTP oltre il prefisso: stantia */
            }
            k=prompt_tokens-len;
            if(k>0) memcpy(hist+len,tmp+len,k*sizeof(int));
            fprintf(stderr,"[API] KV prefix %d/%d token, prefill %d\n",len,prompt_tokens,k);
            free(tmp);
        } else {
            if(templ) bl=mimo_turn_render(buf,1<<16,input,first);
            else { bl=snappend(buf,1<<16,0,"%s",input); }
            if(bl<0){ free(raw); g_temp=base_temp; g_nuc=base_nuc;
                fprintf(stderr,"[chat] prompt too long for template buffer (max %d bytes)\n",1<<16);
                printf("\x01\x01" "END" "\x01\x01\n"); printf("STAT 0 0.00 0.0 %.2f 0 0\n", rss_gb());
                fflush(stdout); continue; }
            k=tok_encode(&T,buf,bl,hist+len,maxctx-len); prompt_tokens=k;
            tokens_check_vocab(hist+len,k,m->c.vocab,"serve_turn");
            if(len+k+8+g_draft>=maxctx){ len=0; first=1;
                if(m->has_mtp) m->kv_start[m->c.n_layers]=-1;  /* contesto azzerato: KV MTP stantia */
                if(templ) bl=mimo_turn_render(buf,1<<16,input,1);
                else bl=snappend(buf,1<<16,0,"%s",input);
                if(bl<0){ free(raw); g_temp=base_temp; g_nuc=base_nuc;
                    fprintf(stderr,"[chat] prompt too long for template buffer (max %d bytes)\n",1<<16);
                    printf("\x01\x01" "END" "\x01\x01\n"); printf("STAT 0 0.00 0.0 %.2f 0 0\n", rss_gb());
                    fflush(stdout); continue; }
                k=tok_encode(&T,buf,bl,hist,maxctx); if(k>maxctx-8-g_draft) k=maxctx-8-g_draft;
                prompt_tokens=k;
                tokens_check_vocab(hist,k,m->c.vocab,"serve_turn_reset");
            }
        }
        if(prompt_tokens<1){ free(raw); g_temp=base_temp; g_nuc=base_nuc;
            printf("\x01\x01" "END" "\x01\x01\n"); printf("STAT 0 0.00 0.0 %.2f 0 0\n", rss_gb()); fflush(stdout); continue; }
        first=0;
        pref_bits_clear(m);                      /* new turn epoch: allow WILLNEED again */
        heat_prefetch_top(m, 4);                 /* warm likely experts before this turn's prefill */
        int cur=req_ngen; if(len+k+cur+g_draft+2>=maxctx) cur=maxctx-len-k-g_draft-2;
        uint64_t h0=m->hits, ms0=m->miss; double tt0=now_s();
        float *logit;
        if(k>0){ logit=step(m,hist+len,k,len); len+=k; }
        else logit=step(m,hist+len-1,1,len-1);   /* prompt identico/prefisso: rigenera i logits */
        EmitStream es={&T,m,now_s(),0,1};
        int prod=0;
        if(cur>0) prod=spec_decode(m,hist,len,cur,eos,logit,emit_stream,&es,&len);
        else free(logit);
        double tdt=now_s()-tt0; if(tdt<1e-6) tdt=1e-6;
        double dh=(double)(m->hits-h0), dm=(double)(m->miss-ms0);
        printf("%s\x01\x01" "END" "\x01\x01\n",raw_mode?"":"\n");
        printf("STAT %d %.2f %.1f %.2f %d %d\n", prod, prod/tdt,
            (dh+dm)>0?100.0*dh/(dh+dm):0.0, rss_gb(), prompt_tokens, prod>=cur);
        fflush(stdout);
        free(raw); g_temp=base_temp; g_nuc=base_nuc;
        usage_save(m);                   /* la cache che impara: storia aggiornata a ogni turno */
        mem_watch_pass(m);               /* #71: dynamic LRU to free/grow RAM between turns */
        repin_pass(m);                   /* also re-pin after full turns (was MORE-only) */
    }
    free(line); free(hist); free(buf);
    usage_save(m);
}

static int *read_arr(jval*o,const char*k,int*n){ jval*a=json_get(o,k); int*r=malloc(a->len*sizeof(int));
    for(int i=0;i<a->len;i++) r[i]=(int)a->kids[i]->num; *n=a->len; return r; }

/* byte residenti di un tensore [O,I] al numero di bit dato (specchio di qt_bytes) */
static int64_t tbytes(int O,int I,int bits){
    if(bits>=16) return (int64_t)O*I*4;
    if(bits>=5)  return (int64_t)O*I + (int64_t)O*4;
    return (int64_t)O*((I+1)/2) + (int64_t)O*4;
}
/* byte VERI di un expert: dal container se pre-quantizzato, altrimenti stima da ebits */
static int64_t expert_bytes_probe(Model *m, int ebits){
    Cfg *c=&m->c; int64_t eb=0; char nm[256];
    int fm=0; while(fm<c->n_layers-1 && !c->is_moe[fm]) fm++;   /* primo layer MoE */
    snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.0.gate_proj.weight",fm);
    if(st_nbytes(&m->S,nm)>0){
        const char *suf[3]={"gate_proj","up_proj","down_proj"};
        for(int k=0;k<3;k++){
            snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.0.%s.weight",fm,suf[k]);
            eb+=st_nbytes(&m->S,nm);
            snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.0.%s.weight.qs",fm,suf[k]);
            int64_t q=st_nbytes(&m->S,nm); if(q>0) eb+=q;
        }
    }
    if(eb<=0) eb = tbytes(c->moe_inter,c->hidden,ebits)*2 + tbytes(c->hidden,c->moe_inter,ebits);
    return eb;
}

/* scarica su file l'istogramma d'uso degli expert: righe "layer eid count" (per PIN).
 * Scrittura atomica (tmp+rename): viene chiamata anche a ogni turno di serve e il
 * processo puo' morire in qualsiasi momento. */
static void stats_dump_q(Model *m, const char *path, int quiet){
    char tmp[2100]; snprintf(tmp,sizeof(tmp),"%s.tmp",path);
    FILE *f=fopen(tmp,"w"); if(!f){ if(!quiet) perror(tmp); return; }
    Cfg *c=&m->c; int64_t tot=0, nz=0;
    for(int i=0;i<c->n_layers;i++){ if(!m->eusage[i]) continue;
        for(int e=0;e<c->n_experts;e++) if(m->eusage[i][e]){ fprintf(f,"%d %d %u\n",i,e,m->eusage[i][e]); tot+=m->eusage[i][e]; nz++; } }
    fclose(f); rename(tmp,path);
    if(!quiet) fprintf(stderr,"[STATS] %lld selezioni su %lld expert distinti -> %s\n",(long long)tot,(long long)nz,path);
}
static void stats_dump(Model *m, const char *path){ stats_dump_q(m,path,0); }

/* CACHE CHE IMPARA: istogramma d'uso PERSISTENTE in <SNAP>/.coli_usage
 * (or .coli_usage.<profile> when COLI_PROFILE / PENG_PROFILE is set — colibri #71).
 * Caricato all'avvio, salvato a ogni turno: pin/REPIN chase the user's workload. */
static char g_usage_path[2100]="";
/* Build usage path; profile keeps chat/code/etc heat maps from polluting each other. */
static void usage_path_set(const char *snap){
    const char *prof=getenv("COLI_PROFILE");
    if(!prof||!*prof) prof=getenv("PENG_PROFILE");
    g_profile[0]=0;
    if(prof && *prof){
        int j=0;
        for(const char *p=prof; *p && j<(int)sizeof(g_profile)-1; p++){
            unsigned char c=(unsigned char)*p;
            if((c>='a'&&c<='z')||(c>='A'&&c<='Z')||(c>='0'&&c<='9')||c=='_'||c=='-'||c=='.')
                g_profile[j++]=(char)c;
        }
        g_profile[j]=0;
    }
    if(g_profile[0])
        snprintf(g_usage_path,sizeof(g_usage_path),"%s/.coli_usage.%s",snap,g_profile);
    else
        snprintf(g_usage_path,sizeof(g_usage_path),"%s/.coli_usage",snap);
    traj_path_set(snap);   /* same profile → .coli_traj[.name] */
}
static int64_t usage_load(Model *m, const char *path){
    FILE *f=fopen(path,"r"); if(!f) return 0;
    Cfg *c=&m->c; int l,e; uint32_t cnt; int64_t tot=0;
    while(fscanf(f,"%d %d %u",&l,&e,&cnt)==3)
        if(l>=0&&l<c->n_layers&&e>=0&&e<c->n_experts&&m->eusage[l]){ m->eusage[l][e]+=cnt; tot+=cnt; }
    fclose(f); return tot;
}
static void usage_save(Model *m){
    if(g_usage_path[0]) stats_dump_q(m,g_usage_path,1);
    traj_save();   /* persist Markov edges with the same cadence as usage */
}

/* HOT-STORE ("il redis del colibri'"): carica in RAM, UNA VOLTA e per sempre, i top expert
 * per frequenza d'uso misurata (file STATS di un run precedente), entro un budget in GB.
 * Ogni hit evita una lettura dal disco lento. */
/* MLOCK: inchioda in RAM fisica gli expert pinnati cosi' il compressore di memoria di
 * macOS non li comprime/evacua (visto: RSS reale < residente previsto -> "hit" lenti).
 * -1 = auto (ON su macOS dove serve e RLIMIT_MEMLOCK e' permissivo; OFF altrove, dove
 * il limite e' spesso minuscolo e va alzato a mano), 0 = off, 1 = force.
 * EN: MLOCK: wire pinned experts into physical RAM so macOS's memory compressor can't
 * compress/evict them (we saw actual RSS < intended resident -> slow "hits"). -1 = auto
 * (ON on macOS where it matters and RLIMIT_MEMLOCK is permissive; OFF elsewhere, where the
 * limit is often tiny and must be raised by hand), 0 = off, 1 = force. */
static int g_mlock=-1;
static int mem_should_wire(void){
    if(g_mlock>=0) return g_mlock;
#if defined(__APPLE__)
    return 1;                                     /* macOS: default ON */
#else
    return 0;                                     /* Linux/altri: opt-in via MLOCK=1 / opt-in */
#endif
}
/* Inchioda [addr,addr+len) in RAM fisica. No-op fuori da POSIX (Windows ecc.).
 * EN: wire [addr,addr+len) into physical RAM. No-op off POSIX (Windows, etc.). */
static int mem_wire(void *addr, size_t len){
#if defined(__APPLE__) || defined(__linux__)
    return mlock(addr, len);
#else
    (void)addr; (void)len; return 0;
#endif
}
/* Inchioda tutti gli slab degli expert pinnati (pesi + scale). Non fatale se fallisce.
 * EN: wire all pinned-expert slabs (weights + scales). Non-fatal on failure. */
static void pin_wire(Model *m){
    if(!mem_should_wire()) return;
    Cfg *c=&m->c; double t0=now_s(); int64_t wired=0; long failed=0;
    for(int i=0;i<c->n_layers;i++) for(int z=0;z<m->npin[i];z++){
        ESlot *s=&m->pin[i][z];
        if(s->slab){  if(mem_wire(s->slab, s->slab_cap)==0) wired+=s->slab_cap; else failed++; }
        if(s->fslab){ size_t fl=(size_t)s->fslab_cap*sizeof(float);
                      if(mem_wire(s->fslab, fl)==0) wired+=fl; else failed++; }
    }
    if(failed)
        fprintf(stderr,"[PIN] mlock: %.1f GB inchiodati/wired, %ld alloc fallite/failed "
            "(alza il limite / raise it: ulimit -l unlimited) in %.0fs\n", wired/1e9, failed, now_s()-t0);
    else
        fprintf(stderr,"[PIN] mlock: %.1f GB inchiodati in RAM fisica / wired in physical RAM "
            "(niente compressione/no compression) in %.0fs\n", wired/1e9, now_s()-t0);
}

static void pin_load(Model *m, const char *statspath, double gb){
    FILE *f=fopen(statspath,"r"); if(!f){ perror(statspath); return; }
    typedef struct { int l,e; uint32_t c; } Rec;
    Cfg *c=&m->c; int cap=c->n_layers*c->n_experts;
    Rec *r=malloc((size_t)cap*sizeof(Rec)); int n=0;
    int l,e; uint32_t cnt;
    while(n<cap && fscanf(f,"%d %d %u",&l,&e,&cnt)==3){
        int ok = l>=0 && e>=0 && e<c->n_experts && l<c->n_layers && m->L[l].sparse;
        if(ok) r[n++]=(Rec){l,e,cnt};
    }
    fclose(f);
    for(int a=0;a<n;a++){ int best=a;                       /* selection sort parziale, poi taglio */
        for(int b=a+1;b<n;b++) if(r[b].c>r[best].c) best=b;
        Rec t=r[a]; r[a]=r[best]; r[best]=t;
        if(a>4095) break;                                    /* bastano i top ~4k */
    }
    int64_t eb=expert_bytes_probe(m,m->ebits);
    int npin=(int)(gb*1e9/eb); if(npin>n) npin=n; if(npin>4096) npin=4096;
    if(npin<1){ free(r); return; }
    /* on_gpu[a]=1 if r[a] placed in VRAM (GPU-first hot store when CUDA). */
    char *on_gpu=calloc((size_t)n,1);
#ifdef COLI_CUDA
    /* GPU-FIRST hot store: put the MOST frequent experts in VRAM (faster path +
     * moe_acc), then pin the next band in RAM. Old order (RAM then complementary
     * VRAM) left the hottest experts on CPU IDOT while cooler ones sat on GPU. */
    if(g_cuda_enabled && g_cuda_expert_gb!=0){
        double remaining[COLI_CUDA_MAX_DEVICES]={0}, placed_b[COLI_CUDA_MAX_DEVICES]={0};
        int placed_n[COLI_CUDA_MAX_DEVICES]={0};
        double safe_total=0;
        /* 0.35 GB headroom (was 0.5): pack more hot experts on 12 GB cards; override
         * via CUDA_HEADROOM_GB. OOM still guarded by cudaMalloc fail → remaining=0. */
        double headroom = 0.35e9;
        if(getenv("CUDA_HEADROOM_GB")){
            headroom = atof(getenv("CUDA_HEADROOM_GB"))*1e9;
            if(headroom<0.1e9) headroom=0.1e9;
            if(headroom>2e9) headroom=2e9;
        }
        for(int i=0;i<g_cuda_ndev;i++){
            size_t free_b=0,total_b=0;
            if(coli_cuda_mem_info(g_cuda_devices[i],&free_b,&total_b)){
                remaining[i]=(double)free_b-headroom;
                if(remaining[i]<0) remaining[i]=0;
                safe_total+=remaining[i];
            }
        }
        double budget = g_cuda_expert_gb>0 ? g_cuda_expert_gb*1e9 : safe_total;
        if(budget>safe_total) budget=safe_total;
        for(int i=0;i<c->n_layers;i++){
            if(m->gpu_pin[i]){
                for(int z=0;z<m->ngpin[i];z++){
                    ESlot *old=&m->gpu_pin[i][z];
                    qt_cuda_reset(&old->g); qt_cuda_reset(&old->u); qt_cuda_reset(&old->d);
                    free(old->slab); free(old->fslab);
                }
                free(m->gpu_pin[i]); m->gpu_pin[i]=NULL; m->ngpin[i]=0;
            }
        }
        m->gpu_expert_count=0; m->gpu_expert_bytes=0;
        double tvg=now_s(); int gtotal=0;
        for(int a=0;a<n && m->gpu_expert_bytes<budget;a++){
            int li=r[a].l;
            if(m->ngpin[li]>=4096) continue;
            m->gpu_pin[li]=realloc(m->gpu_pin[li],(m->ngpin[li]+1)*sizeof(ESlot));
            ESlot *s=&m->gpu_pin[li][m->ngpin[li]];
            memset(s,0,sizeof(ESlot));
            expert_load(m,li,r[a].e,s);
            int64_t need=qt_bytes(&s->g)+qt_bytes(&s->u)+qt_bytes(&s->d);
            if(m->gpu_expert_bytes+need>budget){   /* would exceed VRAM: drop */
                free(s->slab); free(s->fslab); s->slab=NULL; s->fslab=NULL; break;
            }
            int tried[COLI_CUDA_MAX_DEVICES]={0}, placed=0;
            for(int attempt=0;attempt<g_cuda_ndev && !placed;attempt++){
                int best=-1;
                for(int i=0;i<g_cuda_ndev;i++) if(!tried[i] && remaining[i]>=need &&
                    (best<0||placed_b[i]<placed_b[best])) best=i;
                if(best<0) break;
                tried[best]=1;
                s->g.cuda_device=s->u.cuda_device=s->d.cuda_device=g_cuda_devices[best];
                s->g.cuda_eligible=s->u.cuda_eligible=s->d.cuda_eligible=1;
                if(qt_cuda_upload(&s->g) && qt_cuda_upload(&s->u) && qt_cuda_upload(&s->d)){
                    int64_t actual=(int64_t)coli_cuda_tensor_bytes(s->g.cuda)
                                      +(int64_t)coli_cuda_tensor_bytes(s->u.cuda)
                                      +(int64_t)coli_cuda_tensor_bytes(s->d.cuda);
                    m->gpu_expert_count++; m->gpu_expert_bytes+=actual;
                    remaining[best]-=actual; placed_b[best]+=actual; placed_n[best]++;
                    placed=1;
                } else {
                    qt_cuda_reset(&s->g); qt_cuda_reset(&s->u); qt_cuda_reset(&s->d);
                    s->g.cuda_eligible=s->u.cuda_eligible=s->d.cuda_eligible=0;
                    remaining[best]=0;             /* device rejected its projected capacity */
                }
            }
            if(placed){
                expert_cpu_free(m,s);              /* CPU slab freed: VRAM-only now */
                s->g.gpu_only=s->u.gpu_only=s->d.gpu_only=1;
                m->ngpin[li]++; gtotal++;
                on_gpu[a]=1;
            } else {                               /* upload rejected: skip this slot */
                free(s->slab); free(s->fslab); s->slab=NULL; s->fslab=NULL;
                /* shrink realloc footprint: leave empty slot uncounted */
            }
        }
        fprintf(stderr,"[CUDA] GPU-first hot tier: %d expert, VRAM %.2f GB in %.0fs (budget %.1f GB%s)\n",
            gtotal,m->gpu_expert_bytes/1e9,now_s()-tvg,budget/1e9,
            g_cuda_expert_gb<0?" auto":"");
        for(int i=0;i<g_cuda_ndev;i++) fprintf(stderr,"[CUDA]   device %d: %d expert, %.2f GB free-left %.2f GB\n",
            g_cuda_devices[i],placed_n[i],placed_b[i]/1e9,remaining[i]/1e9);
    }
#endif
    /* RAM pin: next most-frequent experts not already on GPU. */
    {
        int *cnt_l=calloc(c->n_layers,sizeof(int));
        int want=npin;
        for(int a=0;a<n && want>0;a++) if(!on_gpu[a]){ cnt_l[r[a].l]++; want--; }
        for(int i=0;i<c->n_layers;i++) if(cnt_l[i]) m->pin[i]=calloc(cnt_l[i],sizeof(ESlot));
        double t0=now_s(); int nloaded=0;
        for(int a=0;a<n && nloaded<npin;a++){
            if(on_gpu[a]) continue;
            int li=r[a].l, slot=m->npin[li]++;
            expert_load(m,li,r[a].e,&m->pin[li][slot]);
            nloaded++;
        }
        m->resident_bytes += (int64_t)nloaded*eb;
        fprintf(stderr,"[PIN] hot-store: %d expert in RAM (%.1f GB) in %.0fs da %s%s\n",
            nloaded, nloaded*eb/1e9, now_s()-t0, statspath,
#ifdef COLI_CUDA
            (g_cuda_enabled && g_cuda_expert_gb!=0)?" (after GPU-first)":""
#else
            ""
#endif
            );
        free(cnt_l);
    }
    pin_wire(m);
    res_bits_rebuild_all(m);                     /* bitmap: pin ∪ gpu after hot-store load */
    free(on_gpu); free(r);
}

static double g_mem_avail_boot=0;   /* MemAvailable all'avvio, prima di caricare il modello */
/* RAM disponibile ADESSO (GB): e' il tetto vero, non il totale. Linux: MemAvailable
 * da /proc/meminfo. macOS: pagine free+inactive+purgeable da host_statistics64
 * (stessa semantica: recuperabili senza swap). Senza questo ramo il fallback
 * "assumo 8 GB" castrava la cache expert proprio sulle macchine con piu' RAM. */
static double mem_available_gb(void){
#ifdef __APPLE__
    mach_msg_type_number_t cnt=HOST_VM_INFO64_COUNT;
    vm_statistics64_data_t vm;
    if(host_statistics64(mach_host_self(),HOST_VM_INFO64,(host_info64_t)&vm,&cnt)!=KERN_SUCCESS) return 0;
    return ((double)vm.free_count+(double)vm.inactive_count+(double)vm.purgeable_count)
           * (double)sysconf(_SC_PAGESIZE) / 1e9;
#else
    FILE *f=fopen("/proc/meminfo","r"); if(!f) return 0;
    char ln[256]; double kb=0;
    while(fgets(ln,sizeof(ln),f)) if(sscanf(ln,"MemAvailable: %lf",&kb)==1) break;
    fclose(f); return kb/1e6;
#endif
}

/* byte della KV cache GQA a max_ctx: SWA/MTP ring = sliding_window rows (F-11). */
static double kv_bytes_at(const Cfg *c, int max_ctx, int has_mtp){
    double b=0; int rows=c->n_layers+(has_mtp?1:0);
    for(int i=0;i<rows;i++){
        int tlen=lyr_kv_rows(c,i,max_ctx);
        b += (double)tlen*lyr_kvh(c,i)*(lyr_hd(c,i)+lyr_vd(c,i))*4.0;
    }
    return b;
}
/* byte disponibili per gli expert (pin + LRU) nel budget — specchio del conto di cap_for_ram */
static double expert_avail(Model *m, double ram_gb, int ebits, int max_ctx){
    Cfg *c=&m->c; int64_t eb=expert_bytes_probe(m,ebits);
    if(ram_gb<=0){ ram_gb=g_mem_avail_boot*0.88; if(ram_gb<4) ram_gb=8; }
    double slack = 1.2e9 + 2.5e9 + 64.0*(double)eb + kv_bytes_at(c,max_ctx,m->has_mtp);
    return ram_gb*1e9 - (double)m->resident_bytes - slack;
}

/* clampa la cache expert a un budget RAM (GB): cap t.c. residente + cache + slack <= budget.
 * ram_gb<=0 -> budget AUTO = 88% della RAM disponibile adesso (lascia respiro a OS+wrapper:
 * sforare = OOM-kill del kernel a meta' generazione, molto peggio di una cache piu' piccola). */
static void cap_for_ram(Model *m, double ram_gb, int ebits, int max_ctx){
    Cfg *c=&m->c; int nsp=0; for(int i=0;i<c->n_layers;i++) if(m->L[i].sparse) nsp++;
    int64_t eb=expert_bytes_probe(m,ebits);
    int auto_b = ram_gb<=0;
    if(auto_b){ ram_gb = g_mem_avail_boot*0.92;   /* more budget for expert LRU when CUDA densas freed */
        if(ram_gb<4){ fprintf(stderr,"[RAM] MemAvailable illeggibile/troppo bassa, assumo 8 GB\n"); ram_gb=8; } }
    /* slack ONESTO, non forfettario (l'OOM del 2026-07-04 veniva da qui):
     *  ws[64] slab del working-set (si materializzano TUTTI nel prefill batch-union),
     *  KV cache GQA a max_ctx, attivazioni+logits+overhead ~1.2 GB */
    double ws_b  = 64.0*(double)eb;
    double kv_b  = kv_bytes_at(c,max_ctx,m->has_mtp);
    /* Page-cache reserve: 2.5 GB was conservative. With O_DIRECT+CUDA densas off host,
     * 1.5 GB leaves more room for expert LRU (hit-rate lever). PC_GB= overrides. */
    /* Page-cache reserve: 1.0 GB default (was 1.5). SWA ring already reclaims ~1.5 GB
     * of KV budget; lower PC leaves more for expert LRU hit-rate (override PC_GB=). */
    double pc_b  = getenv("PC_GB") ? atof(getenv("PC_GB"))*1e9 : 1.0e9;
    if(pc_b<0.5e9) pc_b=0.5e9;
    double slack = 1.0e9 + pc_b + ws_b + kv_b;
    double avail = ram_gb*1e9 - (double)m->resident_bytes - slack;
    int capmax = (avail>0 && nsp>0) ? (int)(avail/((double)nsp*eb)) : 0;
    if(capmax<1) capmax=1;
    if(capmax < m->ecap){
        fprintf(stderr,"[RAM_GB=%.1f%s] residente %.1f GB + slack %.1f GB (ws %.1f, KV@%d %.1f), "
            "expert %.1f MB x %d layer -> cap abbassato %d->%d (proiezione picco %.1f GB)\n",
            ram_gb, auto_b?" auto":"", m->resident_bytes/1e9, slack/1e9, ws_b/1e9, max_ctx, kv_b/1e9,
            eb/1e6, nsp, m->ecap, capmax,
            (m->resident_bytes + (double)capmax*nsp*eb + slack)/1e9);
        m->ecap=capmax;
    } else {
        /* AUTO-RAISE (issue #12): il budget consente PIU' cache di quella chiesta.
         * Senza questo, una macchina da 128 GB girava con la LRU di una da 16
         * (cap=8 di default in coli): hit 23-28% con decine di GB inutilizzati.
         * Tetto a n_experts: oltre, ogni layer avrebbe slot che non puo' riempire.
         * CAP_RAISE=0 ripristina il comportamento fisso. */
        int raise_on = getenv("CAP_RAISE")?atoi(getenv("CAP_RAISE")):1;
        int newcap = capmax>c->n_experts ? c->n_experts : capmax;
        if(raise_on && newcap>m->ecap){
            for(int i=0;i<c->n_layers;i++) if(m->ecache[i]){
                m->ecache[i]=realloc(m->ecache[i],(size_t)newcap*sizeof(ESlot));
                memset(m->ecache[i]+m->ecap,0,(size_t)(newcap-m->ecap)*sizeof(ESlot));
            }
            fprintf(stderr,"[RAM_GB=%.1f%s] cap ALZATO %d->%d: il budget lo consente "
                "(proiezione picco %.1f GB; CAP_RAISE=0 per disattivare)\n",
                ram_gb, auto_b?" auto":"", m->ecap, newcap,
                (m->resident_bytes + (double)newcap*nsp*eb + slack)/1e9);
            m->ecap=newcap;
        } else
            fprintf(stderr,"[RAM_GB=%.1f%s] cap=%d ok (proiezione picco %.1f GB)\n", ram_gb, auto_b?" auto":"", m->ecap,
                (m->resident_bytes + (double)m->ecap*nsp*eb + slack)/1e9);
    }
}

int main(int argc, char **argv){
#ifdef __linux__
    signal(SIGSEGV, segv_handler);   /* backtrace on host crash (no debugger) */
#endif
    /* OMP threads should not busy-spin while main waits on disk */
    if(!getenv("OMP_WAIT_POLICY")) setenv("OMP_WAIT_POLICY","passive",1);
    /* TEMPLATE_DUMP=1: modalita' debug per tests/test_mimo_template.py. Legge da stdin
     * righe "U <msg>" (turno utente -> renderizzato col template chat) e "A <testo>"
     * (cio' che il modello AVREBBE generato -> resta in KV cosi' com'e', senza stop token),
     * stampa il prompt esatto che verrebbe tokenizzato ed esce. Nessun modello caricato:
     * valida SOLO la costruzione del template (confrontata con HF apply_chat_template). */
    if(getenv("TEMPLATE_DUMP") && atoi(getenv("TEMPLATE_DUMP"))){
        char *ln=NULL; size_t cp=0; ssize_t nr; int first=1;
        const int tcap=1<<16;
        char *tbuf=malloc((size_t)tcap); if(!tbuf) return 1;
        while((nr=getline(&ln,&cp,stdin))>0){
            if(ln[nr-1]=='\n') ln[--nr]=0;
            if(nr<2 || ln[1]!=' ') continue;
            if(ln[0]=='U'){
                int bl=mimo_turn_render(tbuf,tcap,ln+2,first);
                if(bl<0){ fprintf(stderr,"TEMPLATE_DUMP: prompt does not fit %d-byte buffer\n",tcap); free(ln); free(tbuf); return 2; }
                fwrite(tbuf,1,(size_t)bl,stdout); first=0;
            } else if(ln[0]=='A') fputs(ln+2,stdout);
        }
        free(ln); free(tbuf); return 0;
    }
    const char *snap=getenv("SNAP"); if(!snap){fprintf(stderr,"SNAP=<dir>\n");return 1;}
    /* SERVE/chat: speed-oriented defaults (override with env). Oracle/TF keep exact path. */
    int serve_mode = getenv("SERVE") && atoi(getenv("SERVE"));
    /* SPEED=1: unity profile — fewer experts/token, tighter I/O anticipatory stack.
     * Does not change matmul math (TOPK only shrinks routed set). Chat --fast sets it. */
    int speed_mode = getenv("SPEED") && atoi(getenv("SPEED"));
    if(speed_mode)
        fprintf(stderr,"[SPEED] profile on (TOPK/TRAJ/REPIN/TOPP defaults tightened; env overrides win)\n");
    g_nopack = getenv("NOPACK")?1:0;
    g_drop = getenv("DROP")?1:0;
    /* sticky next-token WILLNEED: auto ON for SERVE / when PILOT is on */
    if(getenv("PREFETCH")) g_prefetch=atoi(getenv("PREFETCH"));
    else g_prefetch = (serve_mode || (getenv("PILOT") && atoi(getenv("PILOT")))) ? 1 : 0;
    if(getenv("TOPK")) g_topk=atoi(getenv("TOPK"));
    else g_topk = 0;                             /* full topk; TOPK=6 optional (quality trade) */
    if(getenv("TOPP")) g_topp=atof(getenv("TOPP"));
    else g_topp = speed_mode ? 0.55 : 0;         /* SPEED: nucleus trim (measured stack) */
    g_mlock  = getenv("MLOCK")?atoi(getenv("MLOCK")):-1;   /* -1 auto (ON macOS), 0 off, 1 force / auto (ON macOS), 0 off, 1 force */
    g_spec = getenv("SPEC")?atoi(getenv("SPEC")):1;
    g_draft = getenv("DRAFT")?atoi(getenv("DRAFT")):-1;   /* draft per forward; -1 = auto dopo il
                                                           * load (testa MTP presente -> 3, no -> 0) */
    g_direct = getenv("DIRECT")?atoi(getenv("DIRECT")):(serve_mode?1:0);
    g_overlap = getenv("OVERLAP")?atoi(getenv("OVERLAP")):1;   /* 0 = load/compute a fasi (A/B) */
    { const char *e=getenv("OVERLAP_T"); if(e){ g_overlap_t=atoi(e);
        if(g_overlap_t<1)g_overlap_t=1; if(g_overlap_t>16)g_overlap_t=16; } }
    g_idot = getenv("IDOT")?atoi(getenv("IDOT")):1;        /* 0 = kernel f32 esatti (A/B) */
    { const char *e=getenv("I4S");
      if(e) g_i4s=atoi(e);
      else if(serve_mode) g_i4s=1;   /* chat default; CUDA path may force I4S=1 below */
    }
    /* REPIN: SERVE/SPEED every 32 tok (mid-gen load is expensive on WSL if too eager) */
    g_repin = getenv("REPIN")?atoi(getenv("REPIN")):(serve_mode||speed_mode?32:0);
    g_memwatch = getenv("MEMWATCH")?atoi(getenv("MEMWATCH")):1;  /* #71: adapt ecap each SERVE turn */
    /* Trajectory bulk WILLNEED: default ON for SERVE (multi-turn hit); off for oracle/TF */
    if(getenv("TRAJ")) g_traj=atoi(getenv("TRAJ"));
    else g_traj = (serve_mode || speed_mode) ? 1 : 0;
    if(getenv("TRAJ_K")){ g_traj_k=atoi(getenv("TRAJ_K")); if(g_traj_k<1)g_traj_k=1; if(g_traj_k>8)g_traj_k=8; }
    else g_traj_k = (speed_mode || serve_mode) ? 6 : 8;  /* lean: more K was fadvise storm */
    if(getenv("TRAJ_DEPTH")){ g_traj_depth=atoi(getenv("TRAJ_DEPTH")); if(g_traj_depth<1)g_traj_depth=1; if(g_traj_depth>4)g_traj_depth=4; }
    if(g_traj>0)
        fprintf(stderr,"[TRAJ] bulk expert path WILLNEED on (K=%d depth=%d; TRAJ=0 off)\n",
                g_traj_k, g_traj_depth);
    g_temp = getenv("TEMP")?atof(getenv("TEMP")):-1;       /* -1 = auto (1.0 chat/testo, greedy altrove) */
    g_nuc  = getenv("NUCLEUS")?atof(getenv("NUCLEUS")):0.95f;  /* 0.95 = generation_config MiMo-V2.5 */
    g_looka = getenv("LOOKA")?1:0;                    /* PILOT accuracy measurement (no prefetch) */
    /* PILOT helps disk-bound cold tokens; default ON in SERVE when not set. */
    g_pilot = getenv("PILOT")?atoi(getenv("PILOT")):(serve_mode?1:0);
    { const char *e=getenv("PILOT_K"); if(e) g_pilot_k=atoi(e); }
    if(getenv("PILOT_DEPTH")){
        g_pilot_depth=atoi(getenv("PILOT_DEPTH"));
        if(g_pilot_depth<1) g_pilot_depth=1;
        if(g_pilot_depth>3) g_pilot_depth=3;
    } else if(serve_mode) g_pilot_depth=1;           /* L+2 optional: PILOT_DEPTH=2 (costs CPU) */
    if(g_looka) atexit(looka_print);
    if(getenv("SEED")) g_rng = (uint64_t)atoll(getenv("SEED"))*0x9E3779B97F4A7C15ULL+1;
    else { struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); g_rng ^= (uint64_t)ts.tv_nsec<<20 ^ (uint64_t)getpid(); }
    if(g_draft<-1) g_draft=-1;
    if(g_draft>63) g_draft=63;
    int cap  = argc>1?atoi(argv[1]):64;
    int ebits= argc>2?atoi(argv[2]):8;
    int dbits= argc>3?atoi(argv[3]):ebits;
#ifdef COLI_CUDA
    if(getenv("COLI_CUDA") && atoi(getenv("COLI_CUDA"))){
        const char *one=getenv("COLI_GPU"), *many=getenv("COLI_GPUS");
        if(one&&many){ fprintf(stderr,"usa COLI_GPU oppure COLI_GPUS, non entrambi\n"); return 2; }
        if(many) g_cuda_ndev=parse_cuda_devices(many,g_cuda_devices);
        else if(one) g_cuda_ndev=parse_cuda_devices(one,g_cuda_devices);
        else { g_cuda_ndev=1; g_cuda_devices[0]=0; }
        if(g_cuda_ndev<1){ fprintf(stderr,"COLI_GPUS non valido: usa una lista come 0,1,2\n"); return 2; }
        g_cuda_enabled=coli_cuda_init(g_cuda_devices,g_cuda_ndev);
        if(!g_cuda_enabled){ fprintf(stderr,"[CUDA] backend richiesto ma non disponibile\n"); return 2; }
    }
    g_cuda_dense=getenv("CUDA_DENSE")?atoi(getenv("CUDA_DENSE")):0;
    /* CUDA_EXPERT_GB: unset+CUDA → auto (-1); 0=off; >0=explicit GB cap. */
    if(getenv("CUDA_EXPERT_GB")) g_cuda_expert_gb=atof(getenv("CUDA_EXPERT_GB"));
    else g_cuda_expert_gb = g_cuda_enabled ? -1.0 : 0.0;
    if((getenv("COLI_GPU")||getenv("COLI_GPUS"))&&!g_cuda_enabled){ fprintf(stderr,"COLI_GPU(S) requires COLI_CUDA=1\n"); return 2; }
    if(g_cuda_dense&&!g_cuda_enabled){ fprintf(stderr,"CUDA_DENSE requires COLI_CUDA=1\n"); return 2; }
    if(g_cuda_expert_gb!=0 && !g_cuda_enabled){ fprintf(stderr,"CUDA_EXPERT_GB requires COLI_CUDA=1\n"); return 2; }
    if(g_cuda_enabled){
        fprintf(stderr,"[CUDA] mode: routed experts%s | expert-GB %s\n",
            g_cuda_dense?" + resident dense tensors":" only (resident dense on CPU)",
            g_cuda_expert_gb<0?"auto":(g_cuda_expert_gb==0?"off":"cap"));
        /* Fast CPU expert path for S=1 int4 when env I4S unset (chat-speed, not oracle-exact). */
        if(!getenv("I4S")) g_i4s=1;
    }
#else
    if((getenv("COLI_CUDA") && atoi(getenv("COLI_CUDA"))) ||
       getenv("COLI_GPU") || getenv("COLI_GPUS") ||
       (getenv("CUDA_DENSE") && atoi(getenv("CUDA_DENSE"))) ||
       getenv("CUDA_EXPERT_GB")){
        fprintf(stderr,"CUDA requested but this binary is CPU-only; rebuild with: make CUDA=1\n");
        return 2;
    }
#endif
    printf("== Motore C MiMo (mimo_v2), cache=%d expert/layer | expert@%d-bit densa@%d-bit | idot: " IDOT_KERNEL " ==\n", cap, ebits, dbits);
    g_mem_avail_boot = mem_available_gb();
    Model m; double t0=now_s(); model_init(&m,snap,cap,ebits,dbits);
    /* DRAFT auto = 0 anche con la testa MTP: misurato sul modello reale (2026-07-11,
     * 32 token chat, greedy) l'acceptance e' buona (64% a DRAFT=2, 37% a DRAFT=4) e i
     * forward calano 31->14, ma il tok/s NON migliora (0.16-0.20 vs 0.17-0.19; DRAFT=4
     * 0.14; con TOPP=0.7 DIRECT=1: 0.45 vs 0.37): su questo host disk-bound ogni
     * posizione di draft paga i SUOI expert dal disco (batch-union piu' larga, hit-rate
     * 15-17%->10-12%) e la tassa I/O mangia il guadagno di compute — stesso fenomeno
     * visto su colibri'/GLM. Con cache calda (expert residenti/pinnati o NVMe veloce)
     * DRAFT=2 conviene: opt-in esplicito. LOSSLESS in ogni caso (gate byte-identico
     * DRAFT=0 vs 2 vs 4 verificato). */
    if(g_draft<0) g_draft = 0;
#ifdef COLI_CUDA
    /* Eager dense upload before pin_load so the expert VRAM budget sees real free. */
    if(g_cuda_enabled && g_cuda_dense) cuda_upload_dense_all(&m);
#endif
    printf("caricato in %.2fs | densa residente: %.2f MB | layers=%d experts=%d (draft=%d)\n",
           now_s()-t0, m.resident_bytes/(1024.0*1024.0), m.c.n_layers, m.c.n_experts, g_draft);
    fprintf(stderr,"[MTP] %s (draft=%d%s)\n",
        m.has_mtp?"attiva: decodifica speculativa nativa":"assente (fallback n-gram)", g_draft,
        (m.has_mtp && g_draft==0)?", DRAFT=2 per attivarla":"");
    if(!strncmp(snap,"/mnt/",5))
        fprintf(stderr,"ATTENZIONE: il modello e' su %s (filesystem 9p/Windows, lento e fadvise inefficace).\n"
                       "            Per RAM e velocita' tienilo su ext4 (es. /home/...).\n", snap);
    /* HOT-STORE: PIN=<statsfile> [PIN_GB=g] -> top expert per frequenza fissi in RAM.
     * Va PRIMA di cap_for_ram: i pinnati contano nel residente. */
    if(getenv("PIN")) pin_load(&m, getenv("PIN"), getenv("PIN_GB")?atof(getenv("PIN_GB")):10.0);
    /* CACHE CHE IMPARA: usage file under SNAP; optional COLI_PROFILE / PENG_PROFILE
     * isolates heat maps (chat vs code vs …). AUTOPIN=0 disables auto pin from history. */
    { double ram_env = getenv("RAM_GB")?atof(getenv("RAM_GB")):0.0;
      int est_ctx = getenv("CTX")?atoi(getenv("CTX")):4096;   /* stesso default di run_serve */
      usage_path_set(snap);
      if(g_profile[0])
          fprintf(stderr,"[PROFILE] COLI_PROFILE=%s -> %s\n", g_profile, g_usage_path);
      int64_t hist = usage_load(&m,g_usage_path);
      if(hist>0) fprintf(stderr,"[USAGE] storia expert: %lld selezioni (%s)\n",(long long)hist,g_usage_path);
      if(g_traj>0) traj_load();   /* warm Markov from previous sessions */
      int autopin = getenv("AUTOPIN")?atoi(getenv("AUTOPIN")):1;
      if(!getenv("PIN") && autopin && hist>=5000){
          /* quota pin ∝ confidence; at full history use 85% of expert budget (colibri #71;
           * was 50% — left LRU starving the learned hot set on 20+ GB boxes). */
          double conf = (double)hist/200000.0; if(conf>1) conf=1;
          double pin_frac = getenv("PIN_FRAC")?atof(getenv("PIN_FRAC")):0.85;
          if(pin_frac<0.1) pin_frac=0.1; if(pin_frac>0.95) pin_frac=0.95;
          double pin_gb = expert_avail(&m,ram_env,ebits,est_ctx)*pin_frac*conf/1e9;
          if(pin_gb>=0.5) pin_load(&m, g_usage_path, pin_gb);
      }
      /* SEMPRE: senza clamp la LRU cresce fino a cap*76 layer = decine di GB -> OOM-kill.
       * RAM_GB assente o <=0 = budget automatico da MemAvailable. */
      cap_for_ram(&m, ram_env, ebits, est_ctx);
      /* After pin+cap: async WILLNEED from usage heat + loaded TRAJ while we still have
       * wall-clock before first prefill (SERVE READY / PROMPT encode). Free if cold. */
      if(g_traj>0){
          long w0=g_traj_warm_n;
          int bud0=g_traj_pref_budget; g_traj_pref_budget=128; /* boot: one-shot, allow more */
          pref_bits_clear(&m);
          heat_prefetch_top(&m, g_traj_k>4?g_traj_k:4);
          g_traj_pref_budget=bud0;
          if(g_traj_warm_n>w0)
              fprintf(stderr,"[TRAJ] boot warm +%ld hints (total willneed_calls=%ld)\n",
                      g_traj_warm_n-w0, g_traj_warm_n);
      }
    }
    const char *stats=getenv("STATS");   /* STATS=<file> -> istogramma uso expert a fine run */

    /* modo scoring per benchmark: SCORE=<requests.txt> -> log-likelihood per riga */
    if(getenv("SCORE")){ run_score(&m, getenv("SCORE")); if(stats) stats_dump(&m,stats); return 0; }

    /* modo serve persistente per la CLI 'coli': SERVE=1 */
    if(getenv("SERVE")){ run_serve(&m, snap); if(stats) stats_dump(&m,stats); return 0; }

    /* modo testo reale: PROMPT="..." [NGEN=n] -> tokenizza, genera, detokenizza */
    if(getenv("PROMPT")){
        int ngen=getenv("NGEN")?atoi(getenv("NGEN")):64;
        run_text(&m, snap, getenv("PROMPT"), ngen);
        if(stats) stats_dump(&m,stats);
        return 0;
    }

    /* altrimenti: validazione contro l'oracolo (ref_mimo.json) */
    const char *refpath=getenv("REF")?getenv("REF"):"ref_mimo.json";
    FILE *f=fopen(refpath,"rb"); if(!f){perror(refpath);return 1;}
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char *b=malloc(n+1); if(fread(b,1,n,f)!=(size_t)n){} b[n]=0; fclose(f);
    char *ar=NULL; jval *ref=json_parse(b,&ar);
    int np,nfull; int *prompt=read_arr(ref,"prompt_ids",&np); int *full=read_arr(ref,"full_ids",&nfull);
    tokens_check_vocab(prompt,np,m.c.vocab,"ref.prompt_ids");
    tokens_check_vocab(full,nfull,m.c.vocab,"ref.full_ids");
    int n_new=nfull-np;

    if(getenv("REPLAY")){
        run_replay(&m,full,nfull,np);
        if(stats) stats_dump(&m,stats);
        return 0;
    }

    if(getenv("TF")){
        int *tf=read_arr(ref,"tf_pred",&(int){0});
        int *pred=malloc(nfull*sizeof(int));
#ifdef COLI_CUDA
        m.t_cuda_copy = g_cuda_enabled ? coli_cuda_copy_seconds() : 0;
#endif
        m.t_router=0;
        double tt=now_s();
        forward_all(&m, full, nfull, pred); double tdt=now_s()-tt;
        int ok=0; for(int i=0;i<nfull;i++) ok+=(pred[i]==tf[i]);
        printf("PREFILL (teacher-forcing) C vs oracolo: %d/%d posizioni | %.1f pos/s\n",
            ok,nfull,nfull/tdt);
        profile_print(&m,tdt);
#ifdef COLI_CUDA
        if(g_cuda_enabled) cuda_stats_print();
#endif
        return 0;
    }
    int *out=malloc((np+n_new)*sizeof(int));
#ifdef COLI_CUDA
    m.t_cuda_copy = g_cuda_enabled ? coli_cuda_copy_seconds() : 0;
#endif
    m.t_router=0;
    double t=now_s(); generate(&m,prompt,np,n_new,out); double dt=now_s()-t;
    int match=0;
    printf("\nRiferimento (oracolo): "); for(int i=np;i<nfull;i++) printf("%d ", full[i]);
    printf("\nMotore C MiMo        : "); for(int i=np;i<nfull;i++){ printf("%d ", out[i]); if(out[i]==full[i])match++; }
    printf("\nToken coincidenti: %d/%d\n", match, n_new);
    double tot=m.hits+m.miss;
    printf("Speculazione %s (DRAFT=%d): %.2f token/forward (%llu fw per %llu tok)\n",
        m.has_mtp?"MTP":"n-gram", g_draft,
        m.n_fw?(double)m.n_emit/m.n_fw:1.0, (unsigned long long)m.n_fw, (unsigned long long)m.n_emit);
    if(m.mtp_prop) printf("MTP acceptance: %.1f%% (%llu/%llu)\n",
        100.0*m.mtp_acc/m.mtp_prop, (unsigned long long)m.mtp_acc, (unsigned long long)m.mtp_prop);
    printf("Hit-rate cache expert: %.1f%% (hit=%llu miss=%llu) | RSS: %.2f GB | %.1f tok/s\n",
           tot?100.0*m.hits/tot:0.0, (unsigned long long)m.hits, (unsigned long long)m.miss, rss_gb(), n_new/dt);
    profile_print(&m,dt);
#ifdef COLI_CUDA
    if(m.gpu_expert_count) printf("CUDA expert tier: %d residenti (%.2f GB) | %llu chiamate servite da VRAM\n",
        m.gpu_expert_count,m.gpu_expert_bytes/1e9,(unsigned long long)m.gpu_expert_calls);
    if(g_cuda_enabled) cuda_stats_print();
#endif
    if(stats) stats_dump(&m,stats);
    return 0;
}
