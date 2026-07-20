#include "../backend_cuda.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>

static int close_enough(const float *got, const float *want, int n) {
    for (int i = 0; i < n; i++) {
        if (std::fabs(got[i] - want[i]) > 1e-4f) {
            std::fprintf(stderr, "mismatch %d: got %.6f want %.6f\n", i, got[i], want[i]);
            return 0;
        }
    }
    return 1;
}

int main(int argc, char **argv) {
    int devices[COLI_CUDA_MAX_DEVICES], ndev = argc > 1 ? argc - 1 : 1;
    if (ndev > COLI_CUDA_MAX_DEVICES) return 2;
    for (int i = 0; i < ndev; i++) devices[i] = argc > 1 ? std::atoi(argv[i + 1]) : 0;
    if (!coli_cuda_init(devices, ndev)) return 77;
    if (coli_cuda_device_count() != ndev) return 1;
    int d0 = devices[0], d1 = devices[ndev > 1 ? 1 : 0];
    size_t count = 99, bytes = 99;
    coli_cuda_stats(-1, &count, &bytes);
    if (count || bytes) return 1;
    const float x[8] = {1, -2, 3, -4, 2, 1, -1, 0.5f};
    float got[4];

    const int8_t q8[8] = {1, 2, 3, 4, -1, 2, -3, 4};
    const float s8[2] = {0.5f, 2.0f};
    const float want8[4] = {-5.0f, -60.0f, 1.5f, 10.0f};
    ColiCudaTensor *t8 = nullptr;
    if (!coli_cuda_tensor_upload(&t8, q8, s8, 1, 4, 2, d0)) return 1;
    if (coli_cuda_tensor_upload(&t8, q8, s8, 1, 5, 2, d0)) return 1;
    if (ndev > 1 && coli_cuda_tensor_upload(&t8, q8, s8, 1, 4, 2, d1)) return 1;
    if (!coli_cuda_matmul(&t8, got, x, q8, s8, 1, 2, 4, 2, d0) || !close_enough(got, want8, 4)) return 1;

    /* Rows [-8,-1,0,7] and [1,2,3,4], packed low nibble first. */
    const uint8_t q4[4] = {0x70, 0xf8, 0xa9, 0xcb};
    const float s4[2] = {1.0f, 0.25f};
    const float want4[2] = {-34.0f, -2.5f};
    ColiCudaTensor *t4 = nullptr;
    if (!coli_cuda_matmul(&t4, got, x, q4, s4, 2, 1, 4, 2, d1) || !close_enough(got, want4, 2)) return 1;

    const uint8_t q2[2] = {0xe4, 0x1b};
    const float s2[2] = {0.5f, 2.0f};
    const float want2[2] = {-2.0f, 12.0f};
    ColiCudaTensor *t2 = nullptr;
    if (!coli_cuda_matmul(&t2, got, x, q2, s2, 3, 1, 4, 2, d1) || !close_enough(got, want2, 2)) return 1;

    /* Vectorized uint4 paths: I=32 -> rb=16 (i4) / 16B rows (i8), aligned chunks.
     * Integer-valued x and unit scales keep every partial sum exactly
     * representable in fp32, so the reference below is order-independent. */
    {
        const int I = 32, O = 2;
        float xv[I]; int8_t w8[O * I]; uint8_t w4[O * (I / 2)];
        float unit[O] = {1.0f, 1.0f};
        for (int i = 0; i < I; i++) xv[i] = (float)((i * 7 % 11) - 5);          /* -5..5 ints */
        for (int i = 0; i < O * I; i++) w8[i] = (int8_t)((i * 13 % 17) - 8);    /* -8..8 */
        for (int i = 0; i < O * (I / 2); i++) w4[i] = (uint8_t)(i * 37 % 251);
        double acc8[2] = {0, 0}, acc4[2] = {0, 0};                              /* CPU reference */
        for (int o = 0; o < O; o++)
            for (int i = 0; i < I; i++) {
                acc8[o] += xv[i] * w8[o * I + i];
                uint8_t b = w4[o * (I / 2) + i / 2];
                acc4[o] += xv[i] * (double)(((i & 1) ? (b >> 4) : (b & 15)) - 8);
            }
        float want8v[2] = {(float)acc8[0], (float)acc8[1]};
        float want4v[2] = {(float)acc4[0], (float)acc4[1]};
        float gotv[2];
        ColiCudaTensor *tv8 = nullptr, *tv4 = nullptr;
        if (!coli_cuda_matmul(&tv8, gotv, xv, w8, unit, 1, 1, I, O, d0) || !close_enough(gotv, want8v, 2)) return 1;
        if (!coli_cuda_matmul(&tv4, gotv, xv, w4, unit, 2, 1, I, O, d0) || !close_enough(gotv, want4v, 2)) return 1;
        /* Fused gate+up on the vectorized path: silu(g(x))*u(x), D=32, Iffn=2.
         * down is [O=D=32, I=Iffn=2]; rows packed int4 [1,1] (0x99) so every
         * output y[d] = mid[0]+mid[1]. int4 value 1 = nibble 9, low nibble first.
         * The fused kernel requires gate/up same fmt: case A fmt 2/2, case B fmt 1/1. */
        uint8_t downw[32]; float units32[32];
        for (int d = 0; d < 32; d++) { downw[d] = 0x99; units32[d] = 1.0f; }
        for (int fmt = 1; fmt <= 2; fmt++) {
            float gu_out[32];
            ColiCudaTensor *tg = nullptr, *tu = nullptr, *td = nullptr;
            double mid[2];
            for (int o = 0; o < 2; o++) {
                double g8 = 0, u8 = 0;
                for (int i = 0; i < I; i++) {
                    double wg, wu;
                    if (fmt == 2) {
                        uint8_t b = w4[o * (I / 2) + i / 2];
                        wg = wu = (double)(((i & 1) ? (b >> 4) : (b & 15)) - 8);
                    } else {
                        wg = wu = (double)w8[o * I + i];
                    }
                    g8 += xv[i] * wg; u8 += xv[i] * wu;
                }
                double sil = g8 / (1.0 + std::exp(-g8));
                mid[o] = sil * u8;
            }
            float wantgu[32];
            for (int d = 0; d < 32; d++) wantgu[d] = (float)(mid[0] + mid[1]);
            const void *gw = fmt == 2 ? (const void *)w4 : (const void *)w8;
            if (!coli_cuda_swiglu(&tg, &tu, &td, gu_out, xv,
                                 gw, unit, fmt, gw, unit, fmt, downw, units32, 2,
                                 1, I, 2, d0, 0) || !close_enough(gu_out, wantgu, 32)) return 1;
            coli_cuda_tensor_free(tg);
            coli_cuda_tensor_free(tu);
            coli_cuda_tensor_free(td);
        }
        coli_cuda_tensor_free(tv8);
        coli_cuda_tensor_free(tv4);
    }

    const float wf[8] = {1, 0, -1, 2, 0.5f, 0.5f, 0.5f, 0.5f};
    const float wantf[2] = {-10.0f, -1.0f};
    ColiCudaTensor *tf = nullptr;
    if (!coli_cuda_matmul(&tf, got, x, wf, nullptr, 0, 1, 4, 2, d0) || !close_enough(got, wantf, 2)) return 1;

    coli_cuda_stats(-1, &count, &bytes);
    if (count != 4 || bytes != 70) {
        std::fprintf(stderr, "unexpected CUDA stats: %zu tensors, %zu bytes\n", count, bytes);
        return 1;
    }
    if (coli_cuda_tensor_device(t8) != d0 || coli_cuda_tensor_device(tf) != d0 ||
        coli_cuda_tensor_device(t4) != d1 || coli_cuda_tensor_device(t2) != d1) return 1;
    coli_cuda_stats(d0, &count, &bytes);
    if (ndev > 1) {
        if (count != 2 || bytes != 48) return 1;
        coli_cuda_stats(d1, &count, &bytes);
        if (count != 2 || bytes != 22) return 1;
    } else if (count != 4 || bytes != 70) return 1;

    /* Fused SwiGLU: y = down(silu(gate(x))*up(x)). Tiny f32 dims D=2 I=2 S=1. */
    {
        const float x1[2] = {1.0f, -1.0f};
        /* gate rows: [1,0], [0,1]  → gate(x)=x; up same; down identity */
        const float Wg[4] = {1, 0, 0, 1};
        const float Wu[4] = {1, 0, 0, 1};
        const float Wd[4] = {1, 0, 0, 1};
        float yf[2];
        ColiCudaTensor *tg = nullptr, *tu = nullptr, *td = nullptr;
        if (!coli_cuda_swiglu(&tg, &tu, &td, yf, x1,
                             Wg, nullptr, 0, Wu, nullptr, 0, Wd, nullptr, 0,
                             1, 2, 2, d0, 0)) return 1;
        /* silu(1)*1 = 1/(1+e^-1), silu(-1)*(-1) = (-1)/(1+e) * -1 */
        float s0 = 1.0f / (1.0f + std::exp(-1.0f));
        float s1 = (-1.0f) / (1.0f + std::exp(1.0f));
        float want_s[2] = {s0 * 1.0f, s1 * (-1.0f)};
        if (!close_enough(yf, want_s, 2)) return 1;
        /* Second call: sticky REUSE_X + null host weights (already on device). */
        if (!coli_cuda_swiglu(&tg, &tu, &td, yf, x1,
                             nullptr, nullptr, 0, nullptr, nullptr, 0, nullptr, nullptr, 0,
                             1, 2, 2, d0, COLI_CUDA_SWIGLU_REUSE_X)) return 1;
        if (!close_enough(yf, want_s, 2)) return 1;
        coli_cuda_tensor_free(tg);
        coli_cuda_tensor_free(tu);
        coli_cuda_tensor_free(td);
    }

    /* ---- matmul_ex flags: Y_DEV chaining + X_DEV consumption ---- */
    {
        float *qdev = nullptr;
        float out2[2];
        /* y stays on device... */
        if (!coli_cuda_matmul_ex(&t8, nullptr, &qdev, x, q8, s8, 1, 1, 4, 2, d0,
                                 COLI_CUDA_MM_Y_DEV)) return 1;
        if (!qdev) return 1;
        /* ...then consumed on device by a second matmul (t8 is [2,4] so a [4,2]
         * chain needs another tensor: reuse t8 transposed semantics via f32 id) */
        const float I2[4] = {1, 0, 0, 1};
        ColiCudaTensor *tid = nullptr;
        if (!coli_cuda_matmul_ex(&tid, out2, nullptr, qdev, I2, nullptr, 0, 1, 2, 2, d0,
                                 COLI_CUDA_MM_X_DEV)) return 1;
        if (std::fabs(out2[0] - want8[0]) > 1e-4f || std::fabs(out2[1] - want8[1]) > 1e-4f) return 1;
        coli_cuda_tensor_free(tid);
    }

    /* ---- fused decode attention vs CPU reference (full + SWA ring + sink) ---- */
    {
        const int NL = 2, H = 4, KVH = 2, HD = 32, VD = 16, RD = 16, WIN = 8, MAXT = 32;
        const int GROUP = H / KVH, QS = H * HD, KS = KVH * HD, VS = KVH * VD;
        const float theta = 10000.f, v_scale = 0.707f, scale = 1.f / std::sqrt((float)HD);
        int rows[NL] = {MAXT, WIN}, kvh[NL] = {KVH, KVH}, hd[NL] = {HD, HD},
            vd[NL] = {VD, VD}, swa[NL] = {0, 1};
        if (!coli_cuda_attn_kv_alloc(d0, NL, rows, kvh, hd, vd, swa, WIN, MAXT, H)) return 1;
        /* rope tables (same formula as rope_build) */
        const int half = RD / 2;
        float *cos_t = (float *)std::malloc((size_t)MAXT * half * sizeof(float));
        float *sin_t = (float *)std::malloc((size_t)MAXT * half * sizeof(float));
        for (int p = 0; p < MAXT; p++)
            for (int j = 0; j < half; j++) {
                float ang = (float)p * std::pow(theta, -2.f * (float)j / (float)RD);
                cos_t[p * half + j] = std::cos(ang);
                sin_t[p * half + j] = std::sin(ang);
            }
        if (!coli_cuda_attn_rope(d0, 0, cos_t, sin_t, MAXT, half)) return 1;
        if (!coli_cuda_attn_rope(d0, 1, cos_t, sin_t, MAXT, half)) return 1;
        float sink1[H] = {0.25f, -0.5f, 1.0f, 0.0f};
        if (!coli_cuda_attn_layer_cfg(d0, 0, RD, nullptr, H)) return 1;
        if (!coli_cuda_attn_layer_cfg(d0, 1, RD, sink1, H)) return 1;
        /* host caches: Kref/Vref = CPU reference state; Kmir/Vmir = mirror targets */
        float *Kh[NL], *Vh[NL], *Kmir[NL], *Vmir[NL];
        for (int li = 0; li < NL; li++) {
            Kh[li] = (float *)std::calloc((size_t)rows[li] * KS, sizeof(float));
            Vh[li] = (float *)std::calloc((size_t)rows[li] * VS, sizeof(float));
            Kmir[li] = (float *)std::calloc((size_t)rows[li] * KS, sizeof(float));
            Vmir[li] = (float *)std::calloc((size_t)rows[li] * VS, sizeof(float));
            if (!coli_cuda_attn_mirror(d0, li, Kmir[li], Vmir[li])) return 1;
        }
        float *qkv_dev = nullptr;
        if (cudaMalloc(&qkv_dev, (QS + KS + VS) * sizeof(float)) != cudaSuccess) return 1;
        unsigned lcg = 42;
        auto frand = [&]() {
            lcg = lcg * 1664525u + 1013904223u;
            return ((lcg >> 8) & 0xFFFFFF) / (float)0x1000000 - 0.5f;
        };
        float *qkv = (float *)std::malloc((QS + KS + VS) * sizeof(float));
        float ctx_h[H * VD];
        int bad = 0;
        for (int pos = 0; pos < 24 && !bad; pos++) {
            for (int li = 0; li < NL && !bad; li++) {
                for (int i = 0; i < QS + KS + VS; i++) qkv[i] = frand();
                if (cudaMemcpy(qkv_dev, qkv, (QS + KS + VS) * sizeof(float),
                               cudaMemcpyHostToDevice) != cudaSuccess) return 1;
                float *ctxd = coli_cuda_attn_decode(d0, li, qkv_dev, pos, 0, v_scale);
                if (!ctxd) return 1;
                /* backend runs on a non-blocking stream: device-sync before host reads */
                if (cudaDeviceSynchronize() != cudaSuccess) return 1;
                if (cudaMemcpy(ctx_h, ctxd, H * VD * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
                    return 1;
                /* CPU reference: rope+v_scale+append, then 2-pass softmax */
                int lo = swa[li] ? (pos - WIN + 1 > 0 ? pos - WIN + 1 : 0) : 0;
                int pphy = swa[li] ? pos % rows[li] : pos;
                for (int g = 0; g < KVH; g++) {
                    float k[HD];
                    for (int d = 0; d < HD; d++) k[d] = qkv[QS + g * HD + d];
                    for (int j = 0; j < half; j++) {
                        float cs = cos_t[pos * half + j], sn = sin_t[pos * half + j];
                        float a = k[j], b = k[half + j];
                        k[j] = a * cs - b * sn; k[half + j] = b * cs + a * sn;
                    }
                    for (int d = 0; d < HD; d++) Kh[li][(size_t)pphy * KS + g * HD + d] = k[d];
                    for (int d = 0; d < VD; d++)
                        Vh[li][(size_t)pphy * VS + g * VD + d] = qkv[QS + KS + g * VD + d] * v_scale;
                }
                for (int h = 0; h < H && !bad; h++) {
                    int g = h / GROUP;
                    float q[HD];
                    for (int d = 0; d < HD; d++) q[d] = qkv[h * HD + d];
                    for (int j = 0; j < half; j++) {
                        float cs = cos_t[pos * half + j], sn = sin_t[pos * half + j];
                        float a = q[j], b = q[half + j];
                        q[j] = a * cs - b * sn; q[half + j] = b * cs + a * sn;
                    }
                    int nt = pos + 1 - lo;
                    float mx = li == 1 ? sink1[h] : -1e30f;
                    float sc[64];
                    for (int j = 0; j < nt; j++) {
                        int t = lo + j, tp = swa[li] ? t % rows[li] : t;
                        float a = 0;
                        for (int d = 0; d < HD; d++) a += q[d] * Kh[li][(size_t)tp * KS + g * HD + d];
                        sc[j] = a * scale;
                        if (sc[j] > mx) mx = sc[j];
                    }
                    float den = li == 1 ? std::exp(sink1[h] - mx) : 0.f;
                    for (int j = 0; j < nt; j++) { sc[j] = std::exp(sc[j] - mx); den += sc[j]; }
                    for (int d = 0; d < VD; d++) {
                        float acc = 0;
                        for (int j = 0; j < nt; j++) {
                            int t = lo + j, tp = swa[li] ? t % rows[li] : t;
                            acc += sc[j] * Vh[li][(size_t)tp * VS + g * VD + d];
                        }
                        float want = acc / den;
                        if (std::fabs(ctx_h[h * VD + d] - want) > 5e-4f) {
                            std::fprintf(stderr, "attn mismatch pos %d li %d h %d d %d: got %.6f want %.6f\n",
                                         pos, li, h, d, ctx_h[h * VD + d], want);
                            bad = 1; break;
                        }
                    }
                }
                /* host mirror must equal the CPU reference row just appended
                 * (kernel rope/v_scale math is elementwise: near bit-exact) */
                if (cudaDeviceSynchronize() != cudaSuccess) return 1;
                for (int i = 0; i < KS && !bad; i++)
                    if (std::fabs(Kmir[li][(size_t)pphy * KS + i] - Kh[li][(size_t)pphy * KS + i]) > 1e-5f) {
                        std::fprintf(stderr, "mirror K mismatch pos %d li %d i %d\n", pos, li, i);
                        bad = 1;
                    }
                for (int i = 0; i < VS && !bad; i++)
                    if (std::fabs(Vmir[li][(size_t)pphy * VS + i] - Vh[li][(size_t)pphy * VS + i]) > 1e-5f) {
                        std::fprintf(stderr, "mirror V mismatch pos %d li %d i %d\n", pos, li, i);
                        bad = 1;
                    }
            }
        }
        if (bad) return 1;
        /* upload_rows handoff: overwrite device SWA cache from host, decode one more */
        if (!coli_cuda_attn_upload_rows(d0, 1, 0, WIN, Kh[1], Vh[1])) return 1;
        for (int i = 0; i < QS + KS + VS; i++) qkv[i] = frand();
        if (cudaMemcpy(qkv_dev, qkv, (QS + KS + VS) * sizeof(float),
                       cudaMemcpyHostToDevice) != cudaSuccess) return 1;
        if (!coli_cuda_attn_decode(d0, 1, qkv_dev, 24, 0, v_scale)) return 1;
        std::free(qkv);
        cudaFree(qkv_dev);
        for (int li = 0; li < NL; li++) {
            std::free(Kh[li]); std::free(Vh[li]);
            std::free(Kmir[li]); std::free(Vmir[li]);
        }
        std::free(cos_t); std::free(sin_t);
    }

    coli_cuda_tensor_free(t8);
    coli_cuda_tensor_free(t4);
    coli_cuda_tensor_free(t2);
    coli_cuda_tensor_free(tf);
    coli_cuda_stats(-1, &count, &bytes);
    if (count || bytes) return 1;
    coli_cuda_shutdown();
    std::printf("cuda backend: q8/q4/q2/f32 + fused SwiGLU ok on %d device(s)\n", ndev);
    return 0;
}
