/* Micro-bench: dist_build top-p heap select vs the old full qsort, on the real
 * vocab size (V=152576, temp=1.0, nuc=0.95 = chat defaults). Not a pass/fail
 * test — prints ms/call for both implementations. */
#define main mimo_main
#include "../mimo.c"
#undef main

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static unsigned rng_state = 4242;
static float frand(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return ((rng_state >> 8) & 0xFFFFFF) / (float)0x1000000 * 2.f - 1.f;
}

static int cmp_desc_bench(const void *a, const void *b) {
    float pa = g_pbuf[*(const int *)a], pb = g_pbuf[*(const int *)b];
    return pa < pb ? 1 : pa > pb ? -1 : 0;
}
/* the old implementation: full qsort of V indices + linear scan */
static void dist_build_qsort(const float *lo, int V) {
    if (!g_pbuf) { g_pbuf = falloc(V); g_pidx = malloc(V * sizeof(int)); }
    float mx = lo[0]; for (int i = 1; i < V; i++) if (lo[i] > mx) mx = lo[i];
    double s = 0; float invt = 1.f / (g_temp > 1e-4f ? g_temp : 1e-4f);
    for (int i = 0; i < V; i++) { g_pbuf[i] = expf((lo[i] - mx) * invt); s += g_pbuf[i]; }
    for (int i = 0; i < V; i++) g_pbuf[i] /= (float)s;
    if (g_nuc > 0 && g_nuc < 1.f) {
        for (int i = 0; i < V; i++) g_pidx[i] = i;
        qsort(g_pidx, V, sizeof(int), cmp_desc_bench);
        double cum = 0; int keep = V;
        for (int i = 0; i < V; i++) { cum += g_pbuf[g_pidx[i]]; if (cum >= g_nuc) { keep = i + 1; break; } }
        double s2 = 0; for (int i = keep; i < V; i++) g_pbuf[g_pidx[i]] = 0;
        for (int i = 0; i < keep; i++) s2 += g_pbuf[g_pidx[i]];
        for (int i = 0; i < keep; i++) g_pbuf[g_pidx[i]] /= (float)s2;
    }
}

static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

int main(void) {
    const int V = 152576, REP = 60;
    float *lo = malloc((size_t)V * sizeof(float));
    g_temp = 1.0f; g_nuc = 0.95f;

    for (int i = 0; i < V; i++) lo[i] = frand() * 24 - 12;
    dist_build(lo, V); dist_build_qsort(lo, V);   /* warmup + lazy alloc */

    double t0 = now_ms();
    for (int r = 0; r < REP; r++) { lo[0] += 1e-3f; dist_build_qsort(lo, V); }
    double t1 = now_ms();
    for (int r = 0; r < REP; r++) { lo[0] += 1e-3f; dist_build(lo, V); }
    double t2 = now_ms();

    printf("top-p select (V=%d, nuc=%.2f): qsort %.2f ms/call -> heap %.2f ms/call  (%.1fx)\n",
           V, g_nuc, (t1 - t0) / REP, (t2 - t1) / REP, (t1 - t0) / (t2 - t1));
    free(lo);
    return 0;
}
