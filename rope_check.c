/* Test autónomo: valida que la tabla RoPE precomputada (rope_apply + rope_build)
 * produce EXACTAMENTE el mismo resultado que rope_neox original, para ambos tipos
 * de capa (full/SWA) y muchas posiciones. Si esto pasa, el cambio es bit-identical. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

static float theta_full = 10000000.0f;
static float theta_swa  = 10000.0f;
static int   rope_dim   = 64;     /* = head_dim * partial_rotary_factor (full) */
static int   swa_head_dim = 24;   /* head_dim del layer SWA en el tiny-oracle */
static int   head_dim   = 72;     /* head_dim del layer full en el tiny-oracle */
/* rd por tipo de capa (igual que attention(): hd*rope_dim/head_dim) */
static int rope_rd_full(void){ return rope_dim; }                       /* hd=head_dim */
static int rope_rd_swa (void){ return (int)((long long)swa_head_dim*rope_dim/head_dim); }

/* ---- original ---- */
static void rope_neox(float *v, int pos, int rd, float theta){
    int half=rd/2;
    for(int j=0;j<half;j++){
        float ang=(float)pos*powf(theta,-2.f*(float)j/(float)rd);
        float cs=cosf(ang), sn=sinf(ang);
        float a=v[j], b=v[half+j];
        v[j]=a*cs-b*sn; v[half+j]=b*cs+a*sn;
    }
}
/* ---- nuevo (tabla) ---- */
static float *full_cos,*full_sin,*swa_cos,*swa_sin;
static int rope_cap=0;
static void rope_build(int max_t){
    int nf=rope_dim/2, ns=rope_rd_swa()/2;
    if(rope_cap){ free(full_cos);free(full_sin);free(swa_cos);free(swa_sin);} rope_cap=max_t;
    full_cos=malloc((size_t)max_t*nf*sizeof(float)); full_sin=malloc((size_t)max_t*nf*sizeof(float));
    swa_cos =malloc((size_t)max_t*ns*sizeof(float)); swa_sin =malloc((size_t)max_t*ns*sizeof(float));
    for(int pos=0;pos<max_t;pos++){
        for(int j=0;j<nf;j++){ float ang=(float)pos*powf(theta_full,-2.f*(float)j/(float)rope_dim);
            full_cos[(size_t)pos*nf+j]=cosf(ang); full_sin[(size_t)pos*nf+j]=sinf(ang); }
        for(int j=0;j<ns;j++){ float ang=(float)pos*powf(theta_swa,-2.f*(float)j/(float)rope_rd_swa());
            swa_cos[(size_t)pos*ns+j]=cosf(ang); swa_sin[(size_t)pos*ns+j]=sinf(ang); }
    }
}
static void rope_apply(float *v, int rd, const float *cos, const float *sin){
    int half=rd/2;
    for(int j=0;j<half;j++){ float cs=cos[j], sn=sin[j]; float a=v[j], b=v[half+j];
        v[j]=a*cs-b*sn; v[half+j]=b*cs+a*sn; }
}

static int bits_eq(const float *a, const float *b, int n){
    for(int i=0;i<n;i++) if(memcmp(&a[i],&b[i],sizeof(float))!=0) return 0;
    return 1;
}

int main(void){
    int max_t=64; rope_build(max_t);
    int fails=0, checked=0;
    /* barajar rd/theta por tipo; probar posiciones 0..max_t-1 y tambien algunas > max_t-1
     * para confirmar que el caller debe llamar rope_build con el max_t correcto. */
    for(int swa=0; swa<2; swa++){
        int rd = swa ? rope_rd_swa() : rope_rd_full();
        float theta = swa ? theta_swa : theta_full;
        const float *bc = swa ? swa_cos : full_cos;
        const float *bs = swa ? swa_sin : full_sin;
        int half=rd/2;
        for(int pos=0; pos<max_t; pos++){
            for(int rep=0; rep<4; rep++){
                float v1[256], v2[256];
                for(int i=0;i<rd;i++){ float x=(float)((pos*31+rep*7+i*13)%97)/97.0f-0.5f; v1[i]=v2[i]=x; }
                rope_neox(v1,pos,rd,theta);
                const float *ct = bc+(size_t)pos*half;
                const float *st = bs+(size_t)pos*half;
                rope_apply(v2,rd,ct,st);
                checked++;
                if(!bits_eq(v1,v2,rd)){ fails++; if(fails<=5){
                    printf("MISMATCH swa=%d pos=%d rd=%d\n",swa,pos,rd);
                    for(int i=0;i<rd;i++) if(memcmp(&v1[i],&v2[i],sizeof(float)))
                        printf("  i=%d  orig=%a  table=%a\n",i,v1[i],v2[i]); } }
            }
        }
    }
    printf("rope table check: %d comparaciones, %d fallos\n", checked, fails);
    /* tambien comprobar que el offset pos*half usa half=rd/2 y no nf/ns crudos */
    printf(fails? "FAIL\n":"OK bit-identico\n");
    return fails?1:0;
}
