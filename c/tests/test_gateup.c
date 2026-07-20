/* Bit-exactness gate for gate_up_qt (shared gate+up activation quantization).
 * The fused path must produce BIT-IDENTICAL results to the classic 2x matmul_qt
 * path — same qrow_i8 quantization, same integer dot kernels, computed once
 * instead of twice. Uses the upstream `#include "../mimo.c"` test pattern. */
#define main mimo_main
#include "../mimo.c"
#undef main

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned rng_state = 12345;
static float frand(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return ((rng_state >> 8) & 0xFFFFFF) / (float)0x1000000 * 2.f - 1.f;
}

static void fill_qt(QT *w, int fmt, int O, int I) {
    memset(w, 0, sizeof *w);
    w->fmt = fmt; w->O = O; w->I = I;
    w->s = malloc((size_t)O * sizeof(float));
    for (int o = 0; o < O; o++) w->s[o] = 0.01f + 0.05f * frand();
    if (fmt == 1) {
        w->q8 = malloc((size_t)O * I);
        for (size_t i = 0; i < (size_t)O * (size_t)I; i++) w->q8[i] = (int8_t)(frand() * 127);
    } else {
        w->q4 = malloc((size_t)O * ((size_t)(I + 1) / 2));
        for (size_t i = 0; i < (size_t)O * (size_t)((I + 1) / 2); i++) w->q4[i] = (uint8_t)(frand() * 255);
    }
}

static void free_qt(QT *w) { free(w->s); free(w->q8); free(w->q4); }

int main(void) {
    int fails = 0;
    const int fmts[2] = {1, 2};                 /* int8 and int4 (IDOT-eligible) */
    const int dims[3][2] = {{256, 96}, {200, 40}, {130, 18}};  /* aligned + ragged tails */
    for (int fi = 0; fi < 2; fi++)
    for (int di = 0; di < 3; di++)
    for (int S = 1; S <= 5; S++)                /* decode + speculative-verify batch */
    for (int fw = 0; fw < 2; fw++)              /* g_fw1 pinned-kernel verify mode */
    for (int i4s = 1; i4s <= 2; i4s++) {        /* int4 IDOT threshold on/off */
        int fmt = fmts[fi], I = dims[di][0], O = dims[di][1];
        QT g, u; fill_qt(&g, fmt, O, I); fill_qt(&u, fmt, O, I);
        float *x = malloc((size_t)S * I * sizeof(float));
        for (int i = 0; i < S * I; i++) x[i] = frand() * 3.f;
        size_t nb = (size_t)S * O * sizeof(float);
        float *g1 = malloc(nb), *u1 = malloc(nb), *g2 = malloc(nb), *u2 = malloc(nb);
        g_fw1 = fw; g_i4s = i4s;
        memset(g1, 0xAA, nb); memset(u1, 0xAA, nb);
        memset(g2, 0x55, nb); memset(u2, 0x55, nb);
        matmul_qt(g1, x, &g, S); matmul_qt(u1, x, &u, S);   /* classic pair */
        gate_up_qt(g2, u2, x, &g, &u, S);                    /* fused */
        if (memcmp(g1, g2, nb) || memcmp(u1, u2, nb)) {
            printf("MISMATCH fmt=%d I=%d O=%d S=%d fw1=%d i4s=%d\n", fmt, I, O, S, fw, i4s);
            fails++;
        }
        g_fw1 = 0;
        free(x); free(g1); free(u1); free(g2); free(u2);
        free_qt(&g); free_qt(&u);
    }
    if (!fails) printf("gate_up_qt: bit-identical to 2x matmul_qt (fmt 1/2, 3 shapes, S 1..5, fw1, i4s)\n");
    return fails ? 1 : 0;
}
