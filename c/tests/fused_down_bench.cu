/* fused_down_bench — kernel-level benchmark del path MoE decode (S=1) sin modelo.
 *
 * Mide el bucle real del engine: moe_begin (H2D x + zero yacc) -> NE x moe_acc
 * (gate_up_silu + down[ + axpy]) -> moe_end (D2H + sync), con pesos int4
 * sinteticos en las shapes reales de MiMo (gate/up 4096->2048, down 2048->4096).
 *
 * A/B via env:
 *   COLI_CUDA_NO_FUSED_DOWN=1   down GEMV + axpy separados (baseline)
 *   COLI_CUDA_PINNED=0          staging pageable (baseline)
 *   COLI_CUDA_PROF=1            desglose por eventos (ms por tipo de kernel/copy)
 *
 * Uso: ./fused_down_bench [reps] [n_experts]
 */
#include "../backend_cuda.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

static double now_s(void) {
    using namespace std::chrono;
    return duration_cast<duration<double>>(steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char **argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 50;
    const int NE = argc > 2 ? std::atoi(argv[2]) : 8;   /* top-8 = regimen decode real */
    const int D = 4096, I = 2048, S = 1, FMT = 2;       /* int4 */

    int dev = 0;
    if (!coli_cuda_init(&dev, 1)) { std::fprintf(stderr, "no CUDA\n"); return 77; }

    unsigned lcg = 12345;
    auto fr = [&]() {
        lcg = lcg * 1664525u + 1013904223u;
        return ((lcg >> 8) & 0xFFFFFF) / (float)0x1000000;
    };

    const size_t rb_g = (size_t)(D + 1) / 2, rb_d = (size_t)(I + 1) / 2;
    float *x = (float *)std::malloc((size_t)D * sizeof(float));
    for (int i = 0; i < D; i++) x[i] = fr() - 0.5f;

    ColiCudaTensor *tg[64], *tu[64], *td[64];
    if (NE > 64) return 2;
    float *gs[64], *us[64], *ds[64];
    for (int e = 0; e < NE; e++) {
        uint8_t *gw = (uint8_t *)std::malloc(rb_g * I);
        uint8_t *uw = (uint8_t *)std::malloc(rb_g * I);
        uint8_t *dw = (uint8_t *)std::malloc(rb_d * D);
        for (size_t i = 0; i < rb_g * I; i++) gw[i] = (uint8_t)(fr() * 255);
        for (size_t i = 0; i < rb_g * I; i++) uw[i] = (uint8_t)(fr() * 255);
        for (size_t i = 0; i < rb_d * D; i++) dw[i] = (uint8_t)(fr() * 255);
        gs[e] = (float *)std::malloc((size_t)I * sizeof(float));
        us[e] = (float *)std::malloc((size_t)I * sizeof(float));
        ds[e] = (float *)std::malloc((size_t)D * sizeof(float));
        for (int i = 0; i < I; i++) { gs[e][i] = 0.01f + fr() * 0.1f; us[e][i] = 0.01f + fr() * 0.1f; }
        for (int i = 0; i < D; i++) ds[e][i] = 0.01f + fr() * 0.1f;
        tg[e] = tu[e] = td[e] = nullptr;
        /* upload una vez (pesos residentes: aislamos kernel+activaciones del PCIe) */
        if (!coli_cuda_tensor_upload(&tg[e], gw, gs[e], FMT, D, I, dev)) return 1;
        if (!coli_cuda_tensor_upload(&tu[e], uw, us[e], FMT, D, I, dev)) return 1;
        if (!coli_cuda_tensor_upload(&td[e], dw, ds[e], FMT, I, D, dev)) return 1;
        std::free(gw); std::free(uw); std::free(dw);
    }

    float *y = (float *)std::malloc((size_t)D * sizeof(float));
    const float wexp = 0.125f;

    /* warmup */
    for (int r = 0; r < 3; r++) {
        coli_cuda_moe_begin(dev, x, S, D);
        for (int e = 0; e < NE; e++)
            coli_cuda_moe_acc(&tg[e], &tu[e], &td[e], nullptr, nullptr, FMT,
                              nullptr, nullptr, FMT, nullptr, nullptr, FMT, wexp, S, D, I, dev);
        coli_cuda_moe_end(y, S, D, dev);
    }

    coli_cuda_prof_reset(dev);
    double t0 = now_s();
    for (int r = 0; r < reps; r++) {
        coli_cuda_moe_begin(dev, x, S, D);
        for (int e = 0; e < NE; e++)
            coli_cuda_moe_acc(&tg[e], &tu[e], &td[e], nullptr, nullptr, FMT,
                              nullptr, nullptr, FMT, nullptr, nullptr, FMT, wexp, S, D, I, dev);
        coli_cuda_moe_end(y, S, D, dev);
    }
    double wall = now_s() - t0;

    /* trafico VRAM por experto: 3 pesos int4 + mid write/read + y write/read */
    const double wbytes = (double)NE * (2.0 * rb_g * I + (double)rb_d * D);
    const double mb = wbytes / 1e6;
    std::printf("fused_down_bench: reps %d x %d experts (D %d I %d int4)\n", reps, NE, D, I);
    std::printf("  wall %.3f s | %.3f ms/expert-call | %.1f GB/s VRAM (pesos %.1f MB/rep)\n",
                wall, wall * 1e3 / ((double)reps * NE), wbytes * reps / wall / 1e9, mb);

    if (coli_cuda_prof_enabled()) {
        coli_cuda_prof_flush(dev);
        ColiCudaProf p;
        if (coli_cuda_prof_snapshot(dev, &p)) {
            double per = 1.0 / ((double)reps * NE);
            std::printf("  PROF kernels %llu | gate_up %.1f ms | gemv %.1f ms | axpy %.1f ms | other %.1f ms\n",
                        (unsigned long long)p.n_launch, p.ms_gate_up, p.ms_gemv, p.ms_axpy,
                        p.ms_kernel_other);
            std::printf("  PROF per-expert-call: gate_up %.1f us | down %.1f us | axpy %.1f us\n",
                        p.ms_gate_up * per * 1e3, p.ms_gemv * per * 1e3, p.ms_axpy * per * 1e3);
            std::printf("  PROF H2D %llu ops %.1f KB %.2f ms | D2H %llu ops %.1f KB %.2f ms | sync-wait %.2f ms\n",
                        (unsigned long long)p.n_h2d, p.h2d_bytes / 1e3, p.ms_h2d,
                        (unsigned long long)p.n_d2h, p.d2h_bytes / 1e3, p.ms_d2h, p.ms_sync_wait);
        }
    }

    for (int e = 0; e < NE; e++) {
        coli_cuda_tensor_free(tg[e]); coli_cuda_tensor_free(tu[e]); coli_cuda_tensor_free(td[e]);
        std::free(gs[e]); std::free(us[e]); std::free(ds[e]);
    }
    std::free(x); std::free(y);
    coli_cuda_shutdown();
    return 0;
}
