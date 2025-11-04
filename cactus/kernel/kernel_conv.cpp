#include "kernel.h"
#include "kernel_utils.h"
#include <arm_neon.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>
#include <cstddef>
#include <iostream>

constexpr size_t T_TILE_F16 = 2;

void cactus_conv1d_causal_depthwise_f16(
    const __fp16* input, 
    const __fp16* weight, 
    __fp16* output,      
    size_t N, size_t L, size_t C, size_t K, size_t dilation)
{
    const size_t in_bs  = L * C;
    const size_t out_bs = L * C;

    for (size_t n = 0; n < N; ++n) {
        const __fp16* Xb = input  + n * in_bs;
        __fp16*       Yb = output + n * out_bs;

        for (size_t c = 0; c < C; ++c) {
            std::vector<float> wrev(K);
            const __fp16* Wc = weight + c * K;
            for (size_t k = 0; k < K; ++k) wrev[k] = (float)Wc[K - 1 - k];

            for (size_t t0 = 0; t0 < L; t0 += T_TILE_F16) {
                const size_t t1 = std::min(t0 + 1, L - 1);

                float32x4_t vacc0 = vdupq_n_f32(0.f);
                float32x4_t vacc1 = vdupq_n_f32(0.f);

                size_t k = 0;
                for (; k + 8 <= K; k += 8) {
                 
                    float x0_0=0, x1_0=0, x2_0=0, x3_0=0;
                    float x0_1=0, x1_1=0, x2_1=0, x3_1=0;
                    {
                        ptrdiff_t a0=(ptrdiff_t)t0-(ptrdiff_t)((k+0)*dilation);
                        ptrdiff_t a1=(ptrdiff_t)t0-(ptrdiff_t)((k+1)*dilation);
                        ptrdiff_t a2=(ptrdiff_t)t0-(ptrdiff_t)((k+2)*dilation);
                        ptrdiff_t a3=(ptrdiff_t)t0-(ptrdiff_t)((k+3)*dilation);
                        if (a0>=0) x0_0 = (float)Xb[(size_t)a0*C + c];
                        if (a1>=0) x1_0 = (float)Xb[(size_t)a1*C + c];
                        if (a2>=0) x2_0 = (float)Xb[(size_t)a2*C + c];
                        if (a3>=0) x3_0 = (float)Xb[(size_t)a3*C + c];

                        ptrdiff_t b0=(ptrdiff_t)t1-(ptrdiff_t)((k+0)*dilation);
                        ptrdiff_t b1=(ptrdiff_t)t1-(ptrdiff_t)((k+1)*dilation);
                        ptrdiff_t b2=(ptrdiff_t)t1-(ptrdiff_t)((k+2)*dilation);
                        ptrdiff_t b3=(ptrdiff_t)t1-(ptrdiff_t)((k+3)*dilation);
                        if (b0>=0) x0_1 = (float)Xb[(size_t)b0*C + c];
                        if (b1>=0) x1_1 = (float)Xb[(size_t)b1*C + c];
                        if (b2>=0) x2_1 = (float)Xb[(size_t)b2*C + c];
                        if (b3>=0) x3_1 = (float)Xb[(size_t)b3*C + c];
                    }
                    float32x4_t xv0 = {x0_0,x1_0,x2_0,x3_0};
                    float32x4_t yv0 = {x0_1,x1_1,x2_1,x3_1};
                    float32x4_t wv0 = {wrev[k+0],wrev[k+1],wrev[k+2],wrev[k+3]};
                    vacc0 = vfmaq_f32(vacc0, xv0, wv0);
                    vacc1 = vfmaq_f32(vacc1, yv0, wv0);

                    float a0_0=0, a1_0=0, a2_0=0, a3_0=0;
                    float a0_1=0, a1_1=0, a2_1=0, a3_1=0;
                    {
                        ptrdiff_t a0i=(ptrdiff_t)t0-(ptrdiff_t)((k+4)*dilation);
                        ptrdiff_t a1i=(ptrdiff_t)t0-(ptrdiff_t)((k+5)*dilation);
                        ptrdiff_t a2i=(ptrdiff_t)t0-(ptrdiff_t)((k+6)*dilation);
                        ptrdiff_t a3i=(ptrdiff_t)t0-(ptrdiff_t)((k+7)*dilation);
                        if (a0i>=0) a0_0 = (float)Xb[(size_t)a0i*C + c];
                        if (a1i>=0) a1_0 = (float)Xb[(size_t)a1i*C + c];
                        if (a2i>=0) a2_0 = (float)Xb[(size_t)a2i*C + c];
                        if (a3i>=0) a3_0 = (float)Xb[(size_t)a3i*C + c];

                        ptrdiff_t b0i=(ptrdiff_t)t1-(ptrdiff_t)((k+4)*dilation);
                        ptrdiff_t b1i=(ptrdiff_t)t1-(ptrdiff_t)((k+5)*dilation);
                        ptrdiff_t b2i=(ptrdiff_t)t1-(ptrdiff_t)((k+6)*dilation);
                        ptrdiff_t b3i=(ptrdiff_t)t1-(ptrdiff_t)((k+7)*dilation);
                        if (b0i>=0) a0_1 = (float)Xb[(size_t)b0i*C + c];
                        if (b1i>=0) a1_1 = (float)Xb[(size_t)b1i*C + c];
                        if (b2i>=0) a2_1 = (float)Xb[(size_t)b2i*C + c];
                        if (b3i>=0) a3_1 = (float)Xb[(size_t)b3i*C + c];
                    }
                    float32x4_t xv1 = {a0_0,a1_0,a2_0,a3_0};
                    float32x4_t yv1 = {a0_1,a1_1,a2_1,a3_1};
                    float32x4_t wv1 = {wrev[k+4],wrev[k+5],wrev[k+6],wrev[k+7]};
                    vacc0 = vfmaq_f32(vacc0, xv1, wv1);
                    vacc1 = vfmaq_f32(vacc1, yv1, wv1);
                }

                float acc0 = vaddvq_f32(vacc0);
                float acc1 = vaddvq_f32(vacc1);

                for (; k < K; ++k) {
                    ptrdiff_t a=(ptrdiff_t)t0-(ptrdiff_t)(k*dilation);
                    if (a>=0) acc0 += wrev[k] * (float)Xb[(size_t)a*C + c];
                    ptrdiff_t b=(ptrdiff_t)t1-(ptrdiff_t)(k*dilation);
                    if (b>=0) acc1 += wrev[k] * (float)Xb[(size_t)b*C + c];
                }

                Yb[t0*C + c] = (__fp16)acc0;
                if (t0 + 1 < L) Yb[t1*C + c] = (__fp16)acc1;
            }
        }
    }
}


void cactus_conv1d_causal_depthwise_f32(
    const float* input, 
    const float* weight,
    float* output,      
    size_t N, size_t L, size_t C, size_t K, size_t dilation)
{
    const size_t T_TILE_F32 = 2;
    const size_t in_bs  = L * C;
    const size_t out_bs = L * C;

    for (size_t n = 0; n < N; ++n) {
        const float* Xb = input  + n * in_bs;
        float*       Yb = output + n * out_bs;

        for (size_t c = 0; c < C; ++c) {
            std::vector<float> wrev(K);
            const float* Wc = weight + c * K;
            for (size_t k = 0; k < K; ++k) wrev[k] = Wc[K - 1 - k];

            for (size_t t0 = 0; t0 < L; t0 += T_TILE_F32) {
                const size_t t1 = std::min(t0 + 1, L - 1);

                float32x4_t vacc0_0 = vdupq_n_f32(0.f);
                float32x4_t vacc0_1 = vdupq_n_f32(0.f);

                size_t k = 0;
                for (; k + 8 <= K; k += 8) {
                    if (t0 + 4 < L) __builtin_prefetch(Xb + (t0 + 4) * C + c);

                    float x0_0 = 0.f, x1_0 = 0.f, x2_0 = 0.f, x3_0 = 0.f; 
                    float x0_1 = 0.f, x1_1 = 0.f, x2_1 = 0.f, x3_1 = 0.f; 
                    {
                        ptrdiff_t xt0 = (ptrdiff_t)t0 - (ptrdiff_t)((k + 0) * dilation);
                        ptrdiff_t xt1 = (ptrdiff_t)t0 - (ptrdiff_t)((k + 1) * dilation);
                        ptrdiff_t xt2 = (ptrdiff_t)t0 - (ptrdiff_t)((k + 2) * dilation);
                        ptrdiff_t xt3 = (ptrdiff_t)t0 - (ptrdiff_t)((k + 3) * dilation);

                        if (xt0 >= 0) x0_0 = Xb[(size_t)xt0 * C + c];
                        if (xt1 >= 0) x1_0 = Xb[(size_t)xt1 * C + c];
                        if (xt2 >= 0) x2_0 = Xb[(size_t)xt2 * C + c];
                        if (xt3 >= 0) x3_0 = Xb[(size_t)xt3 * C + c];

                        ptrdiff_t yu0 = (ptrdiff_t)t1 - (ptrdiff_t)((k + 0) * dilation);
                        ptrdiff_t yu1 = (ptrdiff_t)t1 - (ptrdiff_t)((k + 1) * dilation);
                        ptrdiff_t yu2 = (ptrdiff_t)t1 - (ptrdiff_t)((k + 2) * dilation);
                        ptrdiff_t yu3 = (ptrdiff_t)t1 - (ptrdiff_t)((k + 3) * dilation);
                        if (yu0 >= 0) x0_1 = Xb[(size_t)yu0 * C + c];
                        if (yu1 >= 0) x1_1 = Xb[(size_t)yu1 * C + c];
                        if (yu2 >= 0) x2_1 = Xb[(size_t)yu2 * C + c];
                        if (yu3 >= 0) x3_1 = Xb[(size_t)yu3 * C + c];
                    }

                    float32x4_t xv0 = {x0_0, x1_0, x2_0, x3_0};
                    float32x4_t yv0 = {x0_1, x1_1, x2_1, x3_1};

                    float w0 = wrev[k+0], w1 = wrev[k+1], w2 = wrev[k+2], w3 = wrev[k+3];
                    float32x4_t wv0 = {w0, w1, w2, w3};

                    vacc0_0 = vfmaq_f32(vacc0_0, xv0, wv0);
                    vacc0_1 = vfmaq_f32(vacc0_1, yv0, wv0);

                    float a0_0 = 0.f, a1_0 = 0.f, a2_0 = 0.f, a3_0 = 0.f;
                    float a0_1 = 0.f, a1_1 = 0.f, a2_1 = 0.f, a3_1 = 0.f;
                    {
                        ptrdiff_t xt0 = (ptrdiff_t)t0 - (ptrdiff_t)((k + 4) * dilation);
                        ptrdiff_t xt1 = (ptrdiff_t)t0 - (ptrdiff_t)((k + 5) * dilation);
                        ptrdiff_t xt2 = (ptrdiff_t)t0 - (ptrdiff_t)((k + 6) * dilation);
                        ptrdiff_t xt3 = (ptrdiff_t)t0 - (ptrdiff_t)((k + 7) * dilation);

                        if (xt0 >= 0) a0_0 = Xb[(size_t)xt0 * C + c];
                        if (xt1 >= 0) a1_0 = Xb[(size_t)xt1 * C + c];
                        if (xt2 >= 0) a2_0 = Xb[(size_t)xt2 * C + c];
                        if (xt3 >= 0) a3_0 = Xb[(size_t)xt3 * C + c];

                        ptrdiff_t yu0 = (ptrdiff_t)t1 - (ptrdiff_t)((k + 4) * dilation);
                        ptrdiff_t yu1 = (ptrdiff_t)t1 - (ptrdiff_t)((k + 5) * dilation);
                        ptrdiff_t yu2 = (ptrdiff_t)t1 - (ptrdiff_t)((k + 6) * dilation);
                        ptrdiff_t yu3 = (ptrdiff_t)t1 - (ptrdiff_t)((k + 7) * dilation);
                        if (yu0 >= 0) a0_1 = Xb[(size_t)yu0 * C + c];
                        if (yu1 >= 0) a1_1 = Xb[(size_t)yu1 * C + c];
                        if (yu2 >= 0) a2_1 = Xb[(size_t)yu2 * C + c];
                        if (yu3 >= 0) a3_1 = Xb[(size_t)yu3 * C + c];
                    }

                    float32x4_t xv1 = {a0_0, a1_0, a2_0, a3_0};
                    float32x4_t yv1 = {a0_1, a1_1, a2_1, a3_1};

                    float u0 = wrev[k+4], u1 = wrev[k+5], u2 = wrev[k+6], u3 = wrev[k+7];
                    float32x4_t wv1 = {u0, u1, u2, u3};

                    vacc0_0 = vfmaq_f32(vacc0_0, xv1, wv1);
                    vacc0_1 = vfmaq_f32(vacc0_1, yv1, wv1);
                }

                float acc0 = vaddvq_f32(vacc0_0) + vaddvq_f32(vacc0_1);
                float acc1 = vaddvq_f32(vacc0_1); 
                acc0 = vaddvq_f32(vacc0_0);
                acc1 = vaddvq_f32(vacc0_1);

                for (; k < K; ++k) {
                    ptrdiff_t x0 = (ptrdiff_t)t0 - (ptrdiff_t)(k * dilation);
                    if (x0 >= 0) acc0 += wrev[k] * Xb[(size_t)x0 * C + c];
                    ptrdiff_t x1 = (ptrdiff_t)t1 - (ptrdiff_t)(k * dilation);
                    if (x1 >= 0) acc1 += wrev[k] * Xb[(size_t)x1 * C + c];
                }

                Yb[t0 * C + c] = acc0;
                if (t0 + 1 < L) {
                    Yb[t1 * C + c] = acc1;
                }
            }
        }
    }
}

void cactus_conv1d_f32_k3(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    size_t N, size_t L, size_t C_in, size_t C_out, size_t K, size_t dilation
){
    const size_t T_TILE_F32 = 2;
    const size_t in_bs  = L * C_in;
    const size_t out_bs = L * C_out;
    const size_t center = K / 2;

    for (size_t n = 0; n < N; ++n) {
        const float* Xb = input  + n * in_bs;
        float* Yb = output + n * out_bs;

        for (size_t t0 = 0; t0 < L; t0 += T_TILE_F32) {
            const size_t t1 = std::min(t0 + 1, L - 1);

            for (size_t oc = 0; oc < C_out; ++oc) {

                float32x4_t vacc0_0 = vdupq_n_f32(0.f);
                float32x4_t vacc0_1 = vdupq_n_f32(0.f);

                const float* Woc = weight + oc * (C_in * K);

                size_t ic = 0;
                for (; ic + 8 <= C_in; ic += 8) {

                    if (t0 + 4 < L) __builtin_prefetch(Xb + (t0 + 4) * C_in + ic);

                    float x_t0_m[8], x_t0_0[8], x_t0_p[8];
                    float x_t1_m[8], x_t1_0[8], x_t1_p[8];

                    for (size_t u = 0; u < 8; u++) {
                        const size_t chid = ic + u;

                        ptrdiff_t tm0 = (ptrdiff_t)t0 - (ptrdiff_t)center;
                        ptrdiff_t tp0 = (ptrdiff_t)t0 + (ptrdiff_t)center;

                        ptrdiff_t tm1 = (ptrdiff_t)t1 - (ptrdiff_t)center;
                        ptrdiff_t tp1 = (ptrdiff_t)t1 + (ptrdiff_t)center;

                        x_t0_m[u] = (tm0 >= 0) ? Xb[(size_t)tm0 * C_in + chid] : 0.f;
                        x_t0_0[u] = Xb[t0 * C_in + chid];
                        x_t0_p[u] = (tp0 < (ptrdiff_t)L) ? Xb[(size_t)tp0 * C_in + chid] : 0.f;

                        x_t1_m[u] = (tm1 >= 0) ? Xb[(size_t)tm1 * C_in + chid] : 0.f;
                        x_t1_0[u] = Xb[t1 * C_in + chid];
                        x_t1_p[u] = (tp1 < (ptrdiff_t)L) ? Xb[(size_t)tp1 * C_in + chid] : 0.f;
                    }

                    for (size_t u = 0; u < 8; u++) {
                        const float* Wc = Woc + (ic + u) * K;

                        float32x4_t xv0 = {x_t0_m[u], x_t0_0[u], x_t0_p[u], 0.f};
                        float32x4_t xv1 = {x_t1_m[u], x_t1_0[u], x_t1_p[u], 0.f};

                        float32x4_t wv = {Wc[0], Wc[1], Wc[2], 0.f};

                        vacc0_0 = vfmaq_f32(vacc0_0, xv0, wv);
                        vacc0_1 = vfmaq_f32(vacc0_1, xv1, wv);
                    }
                }

                for (; ic < C_in; ic++) {
                    const float* Wc = Woc + ic * K;

                    ptrdiff_t tm0 = (ptrdiff_t)t0 - (ptrdiff_t)center;
                    ptrdiff_t tp0 = (ptrdiff_t)t0 + (ptrdiff_t)center;

                    ptrdiff_t tm1 = (ptrdiff_t)t1 - (ptrdiff_t)center;
                    ptrdiff_t tp1 = (ptrdiff_t)t1 + (ptrdiff_t)center;

                    float x0m = (tm0 >= 0) ? Xb[(size_t)tm0 * C_in + ic] : 0.f;
                    float x00 = Xb[t0 * C_in + ic];
                    float x0p = (tp0 < (ptrdiff_t)L) ? Xb[(size_t)tp0 * C_in + ic] : 0.f;

                    float x1m = (tm1 >= 0) ? Xb[(size_t)tm1 * C_in + ic] : 0.f;
                    float x10 = Xb[t1 * C_in + ic];
                    float x1p = (tp1 < (ptrdiff_t)L) ? Xb[(size_t)tp1 * C_in + ic] : 0.f;

                    float32x4_t xv0 = {x0m, x00, x0p, 0.f};
                    float32x4_t xv1 = {x1m, x10, x1p, 0.f};

                    float32x4_t wv = {Wc[0], Wc[1], Wc[2], 0.f};

                    vacc0_0 = vfmaq_f32(vacc0_0, xv0, wv);
                    vacc0_1 = vfmaq_f32(vacc0_1, xv1, wv);
                }

                float acc0 = vaddvq_f32(vacc0_0);
                float acc1 = vaddvq_f32(vacc0_1);

                acc0 += bias[oc];
                acc1 += bias[oc];

                Yb[t0 * C_out + oc] = acc0;
                if (t0 + 1 < L)
                    Yb[t1 * C_out + oc] = acc1;
            }
        }
    }
}