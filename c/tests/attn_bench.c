/* Synthetic attention() benchmark + correctness harness with real MiMo-V2.5
 * dims (48 layers: 9 full GQA + 39 SWA ring, H=64, hd=192, vd=128) — no model
 * snapshot needed. Times attention() at S=1 decode and checks the fused CUDA
 * path against the classic one.
 * Modes (argv[1]): 0=CPU | 1=CUDA dense 2-sync | 2=CUDA_ATTN fused | 3=correctness
 * argv[2]=pos (0 for mode 3), argv[3]=iters.
 * Build CPU-only: gcc -O3 -mavx2 -mfma -mf16c -fopenmp -I.. attn_bench.c -o /tmp/attn_bench -lm
 * Build CUDA: same + -DCOLI_CUDA -I<cudainc> ../backend_cuda.o -lcudart -lstdc++
 */
#define main mimo_main
#include "../mimo.c"
#undef main

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned rng_state = 999;
static float frand(void){
    rng_state = rng_state*1664525u + 1013904223u;
    return ((rng_state>>8)&0xFFFFFF)/(float)0x1000000*2.f-1.f;
}
static void fill_qt(QT *w,int fmt,int O,int I){
    memset(w,0,sizeof *w); w->fmt=fmt; w->O=O; w->I=I;
    w->s=malloc((size_t)O*sizeof(float));
    for(int o=0;o<O;o++) w->s[o]=0.02f+0.03f*frand();
    if(fmt==1){ w->q8=malloc((size_t)O*I);
        for(size_t i=0;i<(size_t)O*I;i++) w->q8[i]=(int8_t)(frand()*60);
    } else { w->q4=malloc((size_t)O*((I+1)/2));
        for(size_t i=0;i<(size_t)O*((I+1)/2);i++) w->q4[i]=(uint8_t)(frand()*255); }
}

#define NL 48
#define HID 4096
#define NH 64
#define KVF 4
#define KVS 8
#define HDF 192
#define VDF 128
#define WIN 128

int main(int argc, char **argv){
    int use_cuda = argc>1 ? atoi(argv[1]) : 0;
    int pos = argc>2 ? atoi(argv[2]) : 1000;
    int iters = argc>3 ? atoi(argv[3]) : 20;
    Model m; memset(&m,0,sizeof m);
    Cfg *c=&m.c;
    c->hidden=HID; c->n_layers=NL; c->n_heads=NH;
    c->kv_heads_full=KVF; c->kv_heads_swa=KVS;
    c->head_dim=HDF; c->v_head_dim=VDF;
    c->swa_head_dim=HDF; c->swa_v_head_dim=VDF;
    c->rope_dim=64; c->sliding_window=WIN;
    c->theta_full=10000.f; c->theta_swa=10000.f;
    c->v_scale=0.707f; c->eps=1e-5f; c->vocab=152576;
    /* real pattern: 9 full layers (0,5,11,17,23,29,35,41,47), rest SWA */
    for(int i=0;i<NL;i++) c->is_swa[i]=1;
    int full_idx[9]={0,5,11,17,23,29,35,41,47};
    for(int i=0;i<9;i++) c->is_swa[full_idx[i]]=0;
    m.rope_rd_full=c->rope_dim;
    m.rope_rd_swa=(int)((int64_t)c->swa_head_dim*c->rope_dim/c->head_dim);
    m.L=calloc(NL,sizeof(Layer));
    for(int i=0;i<NL;i++){
        Layer *l=&m.L[i];
        int kvh=c->is_swa[i]?KVS:KVF;
        int rowsz=NH*HDF+kvh*HDF+kvh*VDF;
        fill_qt(&l->qkv,1,rowsz,HID);
        fill_qt(&l->o,1,HID,NH*VDF);
        if(c->is_swa[i]){ l->sink=malloc(NH*sizeof(float));
            for(int h=0;h<NH;h++) l->sink[h]=frand()*2.f; }
    }
    float *x=malloc(HID*sizeof(float)), *out=malloc(HID*sizeof(float));
    for(int i=0;i<HID;i++) x[i]=frand()*0.1f;

#ifdef COLI_CUDA
    if(use_cuda){
        int devs[1]={0};
        g_cuda_enabled=coli_cuda_init(devs,1);
        g_cuda_ndev=1; g_cuda_devices[0]=0;
        g_cuda_dense=1;
        g_cuda_attn = use_cuda>=2 ? 1 : 0;
        if(!g_cuda_enabled){ fprintf(stderr,"no cuda\n"); return 1; }
        kv_alloc(&m, 4096);          /* after cuda init: device KV hook runs here */
        for(int i=0;i<NL;i++){ m.L[i].qkv.cuda_eligible=1; m.L[i].o.cuda_eligible=1; }
        cuda_upload_dense_all(&m);
    } else {
        kv_alloc(&m, 4096);
    }
#else
    if(use_cuda){ fprintf(stderr,"built without COLI_CUDA\n"); return 1; }
    kv_alloc(&m, 4096);
#endif
    m.kv_start=calloc(NL+1,sizeof(int));

    /* warmup + timed passes over all 48 layers (1 token each) */
    if(use_cuda==3){
        /* correctness: fused (attn_dev on) vs classic CUDA dense (attn_dev off),
         * same GEMV kernels both sides -> isolates the fused kernel math. */
        double maxd=0, maxa=0; int worst_li=-1, worst_p=-1;
        for(int p=0;p<140;p++){
            for(int li=0;li<NL;li++){
                m.attn_dev=1;
                attention(&m,&m.L[li],li,x,1,p,out);
                float outa[HID]; memcpy(outa,out,HID*sizeof(float));
                m.attn_dev=0;                       /* classic path: qkv/o GEMV + CPU scoring */
                attention(&m,&m.L[li],li,x,1,p,out);
                for(int i=0;i<HID;i++){
                    double d=fabs((double)outa[i]-out[i]);
                    double mag=fabs((double)out[i])+1e-3;
                    if(d>maxa) maxa=d;
                    if(d/mag>maxd){ maxd=d/mag; worst_li=li; worst_p=p; }
                }
            }
        }
        printf("fused vs classic: max abs %.3g | max rel %.3g (layer %d pos %d) %s\n",
               maxa, maxd, worst_li, worst_p, (maxa<2e-3) ? "OK" : "FAIL");
        return (maxa<2e-3) ? 0 : 1;
    }
    double best=1e9, tot=0;
    for(int it=0;it<iters+2;it++){
        double t0=now_s();
        for(int li=0;li<NL;li++)
            attention(&m,&m.L[li],li,x,1,pos,out);
        double dt=now_s()-t0;
        if(it>=2){ tot+=dt; if(dt<best)best=dt; }
    }
    printf("attention 48-layer pass @pos=%d cuda=%d: avg %.2f ms  best %.2f ms  (t_attn accum %.2f ms)\n",
        pos,use_cuda,tot/iters*1e3,best*1e3,m.t_attn/iters*1e3);
    printf("  -> per 24 tok: %.2f s\n", tot/iters*24);
    return 0;
}
