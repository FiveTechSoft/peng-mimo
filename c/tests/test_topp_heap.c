/* Correctness gate for the partial top-p heap select in dist_build (colibri #354
 * port: Floyd heapify + k pops replaces the full-vocab qsort). Verifies against an
 * independent double-precision full-sort reference:
 *   1. the kept set equals the reference head (tie-free random logits),
 *   2. the tail is fully zeroed,
 *   3. the head renormalizes to 1,
 *   4. tie-heavy input keeps the right COUNT and mass (set is unspecified). */
#define main mimo_main
#include "../mimo.c"
#undef main

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned rng_state = 777;
static float frand(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return ((rng_state >> 8) & 0xFFFFFF) / (float)0x1000000 * 2.f - 1.f;
}

static double ref_p[1 << 18];
static int ref_cmp(const void *a, const void *b) {
    double pa = ref_p[*(const int *)a], pb = ref_p[*(const int *)b];
    return pa < pb ? 1 : pa > pb ? -1 : 0;
}

int main(void) {
    const int V = 1009;                          /* odd, non-power-of-two */
    const float nucs[5] = {0.05f, 0.5f, 0.9f, 0.95f, 0.99f};
    int fails = 0;
    float *lo = malloc((size_t)V * sizeof(float));
    int *ridx = malloc((size_t)V * sizeof(int));
    double *rp = malloc((size_t)V * sizeof(double));

    for (int trial = 0; trial < 300; trial++) {
        int ties = (trial % 10 == 9);            /* every 10th: constant logits */
        for (int i = 0; i < V; i++) lo[i] = ties ? 0.5f : frand() * 24.f - 12.f;
        for (int n = 0; n < 5; n++) {
            float nuc = nucs[n];
            g_temp = 0.7f; g_nuc = nuc;
            dist_build(lo, V);

            /* double-precision reference: softmax, full sort desc, truncate at cum>=nuc */
            double mx = lo[0]; for (int i = 1; i < V; i++) if (lo[i] > mx) mx = lo[i];
            double s = 0;
            for (int i = 0; i < V; i++) { ref_p[i] = rp[i] = exp((lo[i] - mx) / 0.7); s += rp[i]; }
            for (int i = 0; i < V; i++) { rp[i] /= s; ref_p[i] = rp[i]; }
            for (int i = 0; i < V; i++) ridx[i] = i;
            qsort(ridx, V, sizeof(int), ref_cmp);
            double cum = 0; int keep = V;
            for (int i = 0; i < V; i++) { cum += rp[ridx[i]]; if (cum >= nuc) { keep = i + 1; break; } }

            int nonzero = 0; double gsum = 0;
            for (int i = 0; i < V; i++) if (g_pbuf[i] > 0) { nonzero++; gsum += g_pbuf[i]; }
            if (nonzero != keep) {
                printf("trial %d nuc %.2f: kept %d want %d\n", trial, nuc, nonzero, keep);
                fails++; continue;
            }
            if (fabs(gsum - 1.0) > 1e-4) {
                printf("trial %d nuc %.2f: renorm sum %.6f\n", trial, nuc, gsum);
                fails++; continue;
            }
            if (!ties) {
                /* set equality with the reference head + weight agreement */
                double rsum = 0;
                for (int i = 0; i < keep; i++) rsum += rp[ridx[i]];
                for (int i = 0; i < keep; i++) {
                    int id = ridx[i];
                    double want = rp[id] / rsum;
                    if (g_pbuf[id] <= 0 || fabs(g_pbuf[id] - want) > 1e-4 * (want > 1e-6 ? want : 1e-6)) {
                        printf("trial %d nuc %.2f: id %d got %g want %g\n",
                               trial, nuc, id, g_pbuf[id], want);
                        fails++; break;
                    }
                }
            }
        }
    }
    /* nuc outside (0,1): no truncation, distribution untouched */
    g_temp = 1.0f; g_nuc = 0;
    for (int i = 0; i < V; i++) lo[i] = frand() * 10 - 5;
    dist_build(lo, V);
    double s = 0; for (int i = 0; i < V; i++) s += g_pbuf[i];
    if (fabs(s - 1.0) > 1e-4) { printf("nuc=0: sum %.6f\n", s); fails++; }

    if (!fails) printf("dist_build heap top-p: matches full-sort reference (300 trials x 5 nuc)\n");
    free(lo); free(ridx); free(rp);
    return fails ? 1 : 0;
}
