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

    CactusThreading::parallel_for_2d(N, C, 4, [&](size_t n, size_t c) {
        const __fp16* Xb = input  + n * in_bs;
        __fp16*       Yb = output + n * out_bs;

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
    });
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

    CactusThreading::parallel_for_2d(N, C, 4, [&](size_t n, size_t c) {
        const float* Xb = input  + n * in_bs;
        float*       Yb = output + n * out_bs;

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
    });
}

void cactus_conv1d_f32_k3(
    const float* input,
    const float* weight,
    float* output,
    size_t N, size_t L,
    size_t C_in, size_t C_out,
    size_t stride
){
    const size_t out_len = ((L - 1) / stride) + 1;

    const size_t in_bs = C_in * L;
    const size_t out_bs = C_out * out_len;

    for (size_t n = 0; n < N; ++n) {
        const float* Xb = input  + n * in_bs;
        float* Yb = output + n * out_bs;

        for (size_t out_idx = 0; out_idx < out_len; out_idx += 2) {
            const size_t out_t0  = out_idx;
            const bool have_t1 = (out_idx + 1) < out_len;
            const size_t out_t1  = have_t1 ? (out_idx + 1) : 0;

            const size_t t0 = out_t0 * stride;
            const size_t t1 = have_t1 ? (out_t1 * stride) : 0;

            for (size_t oc = 0; oc < C_out; ++oc) {
                float32x4_t acc0 = vdupq_n_f32(0.f);
                float32x4_t acc1 = vdupq_n_f32(0.f);

                const float* Woc = weight + oc * (C_in * 3);
                size_t ic = 0;

                for (; ic + 8 <= C_in; ic += 8) {
                    float x0m[8], x00[8], x0p[8];
                    float x1m[8], x10[8], x1p[8];

                    for (size_t u = 0; u < 8; ++u) {
                        const size_t ch  = ic + u;
                        const float* Xc  = Xb + ch * L;

                        const ptrdiff_t tm0 = (ptrdiff_t)t0 - 1;
                        const ptrdiff_t tp0 = (ptrdiff_t)t0 + 1;
                        x0m[u] = (tm0 >= 0) ? Xc[tm0] : 0.f;
                        x00[u] = Xc[t0];
                        x0p[u] = (tp0 < (ptrdiff_t)L) ? Xc[tp0] : 0.f;

                        if (have_t1) {
                            const ptrdiff_t tm1 = (ptrdiff_t)t1 - 1;
                            const ptrdiff_t tp1 = (ptrdiff_t)t1 + 1;
                            x1m[u] = (tm1 >= 0) ? Xc[tm1] : 0.f;
                            x10[u] = Xc[t1];
                            x1p[u] = (tp1 < (ptrdiff_t)L) ? Xc[tp1] : 0.f;
                        } else {
                            x1m[u] = x10[u] = x1p[u] = 0.f;
                        }
                    }

                    for (size_t u = 0; u < 8; ++u) {
                        const float* Wc = Woc + (ic + u) * 3;

                        const float32x4_t xv0 = {x0m[u], x00[u], x0p[u], 0.f};
                        const float32x4_t wv  = {Wc[0], Wc[1], Wc[2], 0.f};
                        acc0 = vfmaq_f32(acc0, xv0, wv);

                        if (have_t1) {
                            const float32x4_t xv1 = {x1m[u], x10[u], x1p[u], 0.f};
                            acc1 = vfmaq_f32(acc1, xv1, wv);
                        }
                    }
                }

                for (; ic < C_in; ++ic) {
                    const float* Xc = Xb + ic * L;
                    const float* Wc = Woc + ic * 3;

                    const ptrdiff_t tm0 = (ptrdiff_t)t0 - 1;
                    const ptrdiff_t tp0 = (ptrdiff_t)t0 + 1;

                    const float x0m = (tm0 >= 0) ? Xc[tm0] : 0.f;
                    const float x00 = Xc[t0];
                    const float x0p = (tp0 < (ptrdiff_t)L) ? Xc[tp0] : 0.f;

                    const float32x4_t xv0 = {x0m, x00, x0p, 0.f};
                    const float32x4_t wv = {Wc[0], Wc[1], Wc[2], 0.f};
                    acc0 = vfmaq_f32(acc0, xv0, wv);

                    if (have_t1) {
                        const ptrdiff_t tm1 = (ptrdiff_t)t1 - 1;
                        const ptrdiff_t tp1 = (ptrdiff_t)t1 + 1;

                        const float x1m = (tm1 >= 0) ? Xc[tm1] : 0.f;
                        const float x10 = Xc[t1];
                        const float x1p = (tp1 < (ptrdiff_t)L) ? Xc[tp1] : 0.f;

                        const float32x4_t xv1 = {x1m, x10, x1p, 0.f};
                        acc1 = vfmaq_f32(acc1, xv1, wv);
                    }
                }

                float32x2_t s0 = vadd_f32(vget_low_f32(acc0), vget_high_f32(acc0));
                float sum0 = vget_lane_f32(s0, 0) + vget_lane_f32(s0, 1);
                float* Yoc = Yb + oc * out_len;
                Yoc[out_t0] = sum0;

                if (have_t1) {
                    float32x2_t s1 = vadd_f32(vget_low_f32(acc1), vget_high_f32(acc1));
                    float sum1 = vget_lane_f32(s1, 0) + vget_lane_f32(s1, 1);
                    Yoc[out_t1] = sum1;
                }
            }
        }
    }
}

void cactus_conv1d_f16_k3(
    const __fp16* input,
    const __fp16* weight,
    __fp16* output,
    size_t N, size_t L,
    size_t C_in, size_t C_out,
    size_t stride
){
    const size_t out_len = ((L - 1) / stride) + 1;

    const size_t in_bs  = C_in * L;
    const size_t out_bs = C_out * out_len;

    for (size_t n = 0; n < N; ++n) {
        const __fp16* Xb = input  + n * in_bs;
        __fp16* Yb = output + n * out_bs;

        for (size_t out_idx = 0; out_idx < out_len; out_idx += 2) {
            const size_t out_t0  = out_idx;
            const bool   have_t1 = (out_idx + 1) < out_len;
            const size_t out_t1  = have_t1 ? (out_idx + 1) : out_idx;

            const size_t t0 = out_t0 * stride;
            const size_t t1 = have_t1 ? (out_t1 * stride) : t0;

            for (size_t oc = 0; oc < C_out; ++oc) {
                float32x4_t acc0 = vdupq_n_f32(0.f);
                float32x4_t acc1 = vdupq_n_f32(0.f);

                const __fp16* Woc = weight + oc * (C_in * 3);

                size_t ic = 0;

                for (; ic + 16 <= C_in; ic += 16) {
                    for (size_t u = 0; u < 16; ++u) {
                        const __fp16* Xc = Xb + (ic + u) * L;
                        const __fp16* Wc = Woc + (ic + u) * 3;

                        const float16x8_t wv = {
                            Wc[0], Wc[1], Wc[2], (__fp16)0,
                            Wc[0], Wc[1], Wc[2], (__fp16)0
                        };

                        const ptrdiff_t tm0 = (ptrdiff_t)t0 - 1;
                        const ptrdiff_t tp0 = (ptrdiff_t)t0 + 1;
                        const ptrdiff_t tm1 = (ptrdiff_t)t1 - 1;
                        const ptrdiff_t tp1 = (ptrdiff_t)t1 + 1;

                        const __fp16 x0m = (tm0 >= 0) ? Xc[tm0] : (__fp16)0;
                        const __fp16 x00 = Xc[t0];
                        const __fp16 x0p = (tp0 < (ptrdiff_t)L) ? Xc[tp0] : (__fp16)0;

                        __fp16 x1m = 0, x10 = 0, x1p = 0;
                        if (have_t1) {
                            x1m = (tm1 >= 0) ? Xc[tm1] : (__fp16)0;
                            x10 = Xc[t1];
                            x1p = (tp1 < (ptrdiff_t)L) ? Xc[tp1] : (__fp16)0;
                        }

                        const float16x8_t xv = {
                            x0m, x00, x0p, (__fp16)0,
                            x1m, x10, x1p, (__fp16)0
                        };

                        const float16x4_t xv0_h = vget_low_f16(xv);
                        const float16x4_t wv0_h = vget_low_f16(wv);
                        acc0 = vfmaq_f32(acc0, vcvt_f32_f16(xv0_h), vcvt_f32_f16(wv0_h));

                        if (have_t1) {
                            const float16x4_t xv1_h = vget_high_f16(xv);
                            const float16x4_t wv1_h = vget_high_f16(wv);
                            acc1 = vfmaq_f32(acc1, vcvt_f32_f16(xv1_h), vcvt_f32_f16(wv1_h));
                        }
                    }
                }

                for (; ic < C_in; ++ic) {
                    const __fp16* Xc = Xb + ic * L;
                    const __fp16* Wc = Woc + ic * 3;

                    const float16x8_t wv = {
                        Wc[0], Wc[1], Wc[2], (__fp16)0,
                        Wc[0], Wc[1], Wc[2], (__fp16)0
                    };

                    const ptrdiff_t tm0 = (ptrdiff_t)t0 - 1;
                    const ptrdiff_t tp0 = (ptrdiff_t)t0 + 1;
                    const ptrdiff_t tm1 = (ptrdiff_t)t1 - 1;
                    const ptrdiff_t tp1 = (ptrdiff_t)t1 + 1;

                    const __fp16 x0m = (tm0 >= 0) ? Xc[tm0] : (__fp16)0;
                    const __fp16 x00 = Xc[t0];
                    const __fp16 x0p = (tp0 < (ptrdiff_t)L) ? Xc[tp0] : (__fp16)0;

                    __fp16 x1m = 0, x10 = 0, x1p = 0;
                    if (have_t1) {
                        x1m = (tm1 >= 0) ? Xc[tm1] : (__fp16)0;
                        x10 = Xc[t1];
                        x1p = (tp1 < (ptrdiff_t)L) ? Xc[tp1] : (__fp16)0;
                    }

                    const float16x8_t xv = {
                        x0m, x00, x0p, (__fp16)0,
                        x1m, x10, x1p, (__fp16)0
                    };

                    const float16x4_t xv0_h = vget_low_f16(xv);
                    const float16x4_t wv0_h = vget_low_f16(wv);
                    acc0 = vfmaq_f32(acc0, vcvt_f32_f16(xv0_h), vcvt_f32_f16(wv0_h));

                    if (have_t1) {
                        const float16x4_t xv1_h = vget_high_f16(xv);
                        const float16x4_t wv1_h = vget_high_f16(wv);
                        acc1 = vfmaq_f32(acc1, vcvt_f32_f16(xv1_h), vcvt_f32_f16(wv1_h));
                    }
                }

                float32x2_t s0 = vadd_f32(vget_low_f32(acc0), vget_high_f32(acc0));
                float sum0 = vget_lane_f32(s0, 0) + vget_lane_f32(s0, 1);

                __fp16* Yoc = Yb + oc * out_len;
                Yoc[out_t0] = (__fp16)sum0;

                if (have_t1) {
                    float32x2_t s1 = vadd_f32(vget_low_f32(acc1), vget_high_f32(acc1));
                    float sum1 = vget_lane_f32(s1, 0) + vget_lane_f32(s1, 1);
                    Yoc[out_t1] = (__fp16)sum1;
                }
            }
        }
    }
}

void cactus_bilinear_interpolation_fp32(const float* input, float* output, size_t src_height, size_t src_width, size_t embed_dim,
                                        size_t dst_height, size_t dst_width)
{
            
    float scale_h = (src_height > 1 && dst_height > 1) 
                    ? static_cast<float>(src_height - 1) / static_cast<float>(dst_height - 1)
                    : 0.0f;
    float scale_w = (src_width > 1 && dst_width > 1)
                    ? static_cast<float>(src_width - 1) / static_cast<float>(dst_width - 1)
                            : 0.0f;
            
    for (size_t dst_y = 0; dst_y < dst_height; ++dst_y) {
        for (size_t dst_x = 0; dst_x < dst_width; ++dst_x) {
            float src_y_float = dst_y * scale_h;
            float src_x_float = dst_x * scale_w;
            
            int y0 = static_cast<int>(std::floor(src_y_float));
            int x0 = static_cast<int>(std::floor(src_x_float));

            int y1 = ((y0 + 1) < static_cast<int>(src_height)) ? (y0 + 1) : (static_cast<int>(src_height) - 1);
            int x1 = ((x0 + 1) < static_cast<int>(src_width)) ? (x0 + 1) : (static_cast<int>(src_width) - 1);

            float dy = src_y_float - y0;
            float dx = src_x_float - x0;
            
            float w00 = (1.0f - dx) * (1.0f - dy);
            float w01 = dx * (1.0f - dy);
            float w10 = (1.0f - dx) * dy;
            float w11 = dx * dy;
            
            size_t idx00 = (y0 * static_cast<int>(src_width) + x0) * static_cast<int>(embed_dim);
            size_t idx01 = (y0 * static_cast<int>(src_width) + x1) * static_cast<int>(embed_dim);
            size_t idx10 = (y1 * static_cast<int>(src_width) + x0) * static_cast<int>(embed_dim);
            size_t idx11 = (y1 * static_cast<int>(src_width) + x1) * static_cast<int>(embed_dim);

            size_t out_idx = (dst_y * dst_width + dst_x) * embed_dim;
            
            for (size_t d = 0; d < embed_dim; ++d) {
                output[out_idx + d] = 
                    input[idx00 + d] * w00 +
                    input[idx01 + d] * w01 +
                    input[idx10 + d] * w10 +
                    input[idx11 + d] * w11;
            }
        }
    }
}

void cactus_resize_nearest_asymmetric_fp32(const float* input,
                                           float* output,
                                           size_t outer_count,
                                           size_t src_height,
                                           size_t src_width,
                                           size_t dst_height,
                                           size_t dst_width)
{
    if (src_height == 0 || src_width == 0 || dst_height == 0 || dst_width == 0) {
        throw std::runtime_error("Resize nearest asymmetric: invalid input dimensions");
    }

    const float scale_h = static_cast<float>(src_height) / static_cast<float>(dst_height);
    const float scale_w = static_cast<float>(src_width) / static_cast<float>(dst_width);
    const size_t src_plane = src_height * src_width;
    const size_t dst_plane = dst_height * dst_width;

    CactusThreading::parallel_for(
        outer_count * dst_plane,
        CactusThreading::Thresholds::ELEMENT_WISE,
        [&](size_t start, size_t end) {
            for (size_t idx = start; idx < end; ++idx) {
                size_t o = idx / dst_plane;
                size_t rem = idx % dst_plane;
                size_t dst_y = rem / dst_width;
                size_t dst_x = rem % dst_width;

                int src_y = static_cast<int>(std::floor(static_cast<float>(dst_y) * scale_h));
                src_y = std::max(0, std::min(src_y, static_cast<int>(src_height) - 1));

                int src_x = static_cast<int>(std::floor(static_cast<float>(dst_x) * scale_w));
                src_x = std::max(0, std::min(src_x, static_cast<int>(src_width) - 1));

                size_t src_idx = o * src_plane + static_cast<size_t>(src_y) * src_width + static_cast<size_t>(src_x);
                size_t dst_idx = o * dst_plane + dst_y * dst_width + dst_x;
                output[dst_idx] = input[src_idx];
            }
        });
}

void cactus_resize_nearest_asymmetric_fp16(const __fp16* input,
                                           __fp16* output,
                                           size_t outer_count,
                                           size_t src_height,
                                           size_t src_width,
                                           size_t dst_height,
                                           size_t dst_width)
{
    if (src_height == 0 || src_width == 0 || dst_height == 0 || dst_width == 0) {
        throw std::runtime_error("Resize nearest asymmetric: invalid input dimensions");
    }

    const float scale_h = static_cast<float>(src_height) / static_cast<float>(dst_height);
    const float scale_w = static_cast<float>(src_width) / static_cast<float>(dst_width);
    const size_t src_plane = src_height * src_width;
    const size_t dst_plane = dst_height * dst_width;

    CactusThreading::parallel_for(
        outer_count * dst_plane,
        CactusThreading::Thresholds::ELEMENT_WISE,
        [&](size_t start, size_t end) {
            for (size_t idx = start; idx < end; ++idx) {
                size_t o = idx / dst_plane;
                size_t rem = idx % dst_plane;
                size_t dst_y = rem / dst_width;
                size_t dst_x = rem % dst_width;

                int src_y = static_cast<int>(std::floor(static_cast<float>(dst_y) * scale_h));
                src_y = std::max(0, std::min(src_y, static_cast<int>(src_height) - 1));

                int src_x = static_cast<int>(std::floor(static_cast<float>(dst_x) * scale_w));
                src_x = std::max(0, std::min(src_x, static_cast<int>(src_width) - 1));

                size_t src_idx = o * src_plane + static_cast<size_t>(src_y) * src_width + static_cast<size_t>(src_x);
                size_t dst_idx = o * dst_plane + dst_y * dst_width + dst_x;
                output[dst_idx] = input[src_idx];
            }
        });
}

void cactus_resize_nearest_asymmetric_int8(const int8_t* input,
                                           int8_t* output,
                                           size_t outer_count,
                                           size_t src_height,
                                           size_t src_width,
                                           size_t dst_height,
                                           size_t dst_width)
{
    if (src_height == 0 || src_width == 0 || dst_height == 0 || dst_width == 0) {
        throw std::runtime_error("Resize nearest asymmetric: invalid input dimensions");
    }

    const float scale_h = static_cast<float>(src_height) / static_cast<float>(dst_height);
    const float scale_w = static_cast<float>(src_width) / static_cast<float>(dst_width);
    const size_t src_plane = src_height * src_width;
    const size_t dst_plane = dst_height * dst_width;

    CactusThreading::parallel_for(
        outer_count * dst_plane,
        CactusThreading::Thresholds::ELEMENT_WISE,
        [&](size_t start, size_t end) {
            for (size_t idx = start; idx < end; ++idx) {
                size_t o = idx / dst_plane;
                size_t rem = idx % dst_plane;
                size_t dst_y = rem / dst_width;
                size_t dst_x = rem % dst_width;

                int src_y = static_cast<int>(std::floor(static_cast<float>(dst_y) * scale_h));
                src_y = std::max(0, std::min(src_y, static_cast<int>(src_height) - 1));

                int src_x = static_cast<int>(std::floor(static_cast<float>(dst_x) * scale_w));
                src_x = std::max(0, std::min(src_x, static_cast<int>(src_width) - 1));

                size_t src_idx = o * src_plane + static_cast<size_t>(src_y) * src_width + static_cast<size_t>(src_x);
                size_t dst_idx = o * dst_plane + dst_y * dst_width + dst_x;
                output[dst_idx] = input[src_idx];
            }
        });
}

void cactus_maxpool2d_f32(
    const float* input,
    float* output,
    size_t N, size_t C, size_t H, size_t W,
    size_t kernel_h, size_t kernel_w,
    size_t stride_h, size_t stride_w,
    size_t pad_top, size_t pad_left,
    size_t pad_bottom, size_t pad_right,
    size_t dilation_h, size_t dilation_w)
{
    const size_t k_eff_h = (kernel_h - 1) * dilation_h + 1;
    const size_t k_eff_w = (kernel_w - 1) * dilation_w + 1;

    const size_t out_h = (H + pad_top + pad_bottom - k_eff_h) / stride_h + 1;
    const size_t out_w = (W + pad_left + pad_right - k_eff_w) / stride_w + 1;

    const size_t in_stride_c = H * W;
    const size_t in_stride_n = C * in_stride_c;
    const size_t out_stride_c = out_h * out_w;
    const size_t out_stride_n = C * out_stride_c;

    CactusThreading::parallel_for_2d(N, C, 4, [&](size_t n, size_t c) {
        const float* in_nc = input + n * in_stride_n + c * in_stride_c;
        float* out_nc = output + n * out_stride_n + c * out_stride_c;

        for (size_t oh = 0; oh < out_h; ++oh) {
            for (size_t ow = 0; ow < out_w; ++ow) {
                const ptrdiff_t in_h_start = static_cast<ptrdiff_t>(oh * stride_h) - static_cast<ptrdiff_t>(pad_top);
                const ptrdiff_t in_w_start = static_cast<ptrdiff_t>(ow * stride_w) - static_cast<ptrdiff_t>(pad_left);

                float max_val = -std::numeric_limits<float>::infinity();

                for (size_t kh = 0; kh < kernel_h; ++kh) {
                    const ptrdiff_t in_h = in_h_start + static_cast<ptrdiff_t>(kh * dilation_h);
                    if (in_h < 0 || in_h >= static_cast<ptrdiff_t>(H)) continue;

                    for (size_t kw = 0; kw < kernel_w; ++kw) {
                        const ptrdiff_t in_w = in_w_start + static_cast<ptrdiff_t>(kw * dilation_w);
                        if (in_w < 0 || in_w >= static_cast<ptrdiff_t>(W)) continue;

                        const size_t in_idx = static_cast<size_t>(in_h) * W + static_cast<size_t>(in_w);
                        const float val = in_nc[in_idx];
                        if (val > max_val) max_val = val;
                    }
                }

                out_nc[oh * out_w + ow] = max_val;
            }
        }
    });
}

void cactus_maxpool2d_f16(
    const __fp16* input,
    __fp16* output,
    size_t N, size_t C, size_t H, size_t W,
    size_t kernel_h, size_t kernel_w,
    size_t stride_h, size_t stride_w,
    size_t pad_top, size_t pad_left,
    size_t pad_bottom, size_t pad_right,
    size_t dilation_h, size_t dilation_w)
{
    const size_t k_eff_h = (kernel_h - 1) * dilation_h + 1;
    const size_t k_eff_w = (kernel_w - 1) * dilation_w + 1;

    const size_t out_h = (H + pad_top + pad_bottom - k_eff_h) / stride_h + 1;
    const size_t out_w = (W + pad_left + pad_right - k_eff_w) / stride_w + 1;

    const size_t in_stride_c = H * W;
    const size_t in_stride_n = C * in_stride_c;
    const size_t out_stride_c = out_h * out_w;
    const size_t out_stride_n = C * out_stride_c;

    CactusThreading::parallel_for_2d(N, C, 4, [&](size_t n, size_t c) {
        const __fp16* in_nc = input + n * in_stride_n + c * in_stride_c;
        __fp16* out_nc = output + n * out_stride_n + c * out_stride_c;

        for (size_t oh = 0; oh < out_h; ++oh) {
            for (size_t ow = 0; ow < out_w; ++ow) {
                const ptrdiff_t in_h_start = static_cast<ptrdiff_t>(oh * stride_h) - static_cast<ptrdiff_t>(pad_top);
                const ptrdiff_t in_w_start = static_cast<ptrdiff_t>(ow * stride_w) - static_cast<ptrdiff_t>(pad_left);

                float max_val = -std::numeric_limits<float>::infinity();

                for (size_t kh = 0; kh < kernel_h; ++kh) {
                    const ptrdiff_t in_h = in_h_start + static_cast<ptrdiff_t>(kh * dilation_h);
                    if (in_h < 0 || in_h >= static_cast<ptrdiff_t>(H)) continue;

                    for (size_t kw = 0; kw < kernel_w; ++kw) {
                        const ptrdiff_t in_w = in_w_start + static_cast<ptrdiff_t>(kw * dilation_w);
                        if (in_w < 0 || in_w >= static_cast<ptrdiff_t>(W)) continue;

                        const size_t in_idx = static_cast<size_t>(in_h) * W + static_cast<size_t>(in_w);
                        const float val = static_cast<float>(in_nc[in_idx]);
                        if (val > max_val) max_val = val;
                    }
                }

                out_nc[oh * out_w + ow] = static_cast<__fp16>(max_val);
            }
        }
    });
}

void cactus_maxpool2d_int8(
    const int8_t* input,
    int8_t* output,
    size_t N, size_t C, size_t H, size_t W,
    size_t kernel_h, size_t kernel_w,
    size_t stride_h, size_t stride_w,
    size_t pad_top, size_t pad_left,
    size_t pad_bottom, size_t pad_right,
    size_t dilation_h, size_t dilation_w)
{
    const size_t k_eff_h = (kernel_h - 1) * dilation_h + 1;
    const size_t k_eff_w = (kernel_w - 1) * dilation_w + 1;

    const size_t out_h = (H + pad_top + pad_bottom - k_eff_h) / stride_h + 1;
    const size_t out_w = (W + pad_left + pad_right - k_eff_w) / stride_w + 1;

    const size_t in_stride_c = H * W;
    const size_t in_stride_n = C * in_stride_c;
    const size_t out_stride_c = out_h * out_w;
    const size_t out_stride_n = C * out_stride_c;

    CactusThreading::parallel_for_2d(N, C, 4, [&](size_t n, size_t c) {
        const int8_t* in_nc = input + n * in_stride_n + c * in_stride_c;
        int8_t* out_nc = output + n * out_stride_n + c * out_stride_c;

        for (size_t oh = 0; oh < out_h; ++oh) {
            for (size_t ow = 0; ow < out_w; ++ow) {
                const ptrdiff_t in_h_start = static_cast<ptrdiff_t>(oh * stride_h) - static_cast<ptrdiff_t>(pad_top);
                const ptrdiff_t in_w_start = static_cast<ptrdiff_t>(ow * stride_w) - static_cast<ptrdiff_t>(pad_left);

                int8_t max_val = std::numeric_limits<int8_t>::lowest();

                for (size_t kh = 0; kh < kernel_h; ++kh) {
                    const ptrdiff_t in_h = in_h_start + static_cast<ptrdiff_t>(kh * dilation_h);
                    if (in_h < 0 || in_h >= static_cast<ptrdiff_t>(H)) continue;

                    for (size_t kw = 0; kw < kernel_w; ++kw) {
                        const ptrdiff_t in_w = in_w_start + static_cast<ptrdiff_t>(kw * dilation_w);
                        if (in_w < 0 || in_w >= static_cast<ptrdiff_t>(W)) continue;

                        const size_t in_idx = static_cast<size_t>(in_h) * W + static_cast<size_t>(in_w);
                        const int8_t val = in_nc[in_idx];
                        if (val > max_val) max_val = val;
                    }
                }

                out_nc[oh * out_w + ow] = max_val;
            }
        }
    });
}

void cactus_global_avg_pool2d_f32(
    const float* input,
    float* output,
    size_t N, size_t C, size_t H, size_t W)
{
    const size_t spatial_size = H * W;
    const float inv_spatial = 1.0f / static_cast<float>(spatial_size);
    const size_t in_stride_c = H * W;
    const size_t in_stride_n = C * in_stride_c;

    CactusThreading::parallel_for_2d(N, C, 4, [&](size_t n, size_t c) {
        const float* in_nc = input + n * in_stride_n + c * in_stride_c;

        float sum = 0.0f;
        for (size_t i = 0; i < spatial_size; ++i) {
            sum += in_nc[i];
        }

        output[n * C + c] = sum * inv_spatial;
    });
}

void cactus_global_avg_pool2d_f16(
    const __fp16* input,
    __fp16* output,
    size_t N, size_t C, size_t H, size_t W)
{
    const size_t spatial_size = H * W;
    const float inv_spatial = 1.0f / static_cast<float>(spatial_size);
    const size_t in_stride_c = H * W;
    const size_t in_stride_n = C * in_stride_c;

    CactusThreading::parallel_for_2d(N, C, 4, [&](size_t n, size_t c) {
        const __fp16* in_nc = input + n * in_stride_n + c * in_stride_c;

        float sum = 0.0f;
        for (size_t i = 0; i < spatial_size; ++i) {
            sum += static_cast<float>(in_nc[i]);
        }

        output[n * C + c] = static_cast<__fp16>(sum * inv_spatial);
    });
}

void cactus_global_avg_pool2d_int8(
    const int8_t* input,
    int8_t* output,
    size_t N, size_t C, size_t H, size_t W,
    float input_scale, float output_scale)
{
    const size_t spatial_size = H * W;
    const float inv_spatial = 1.0f / static_cast<float>(spatial_size);
    const size_t in_stride_c = H * W;
    const size_t in_stride_n = C * in_stride_c;

    CactusThreading::parallel_for_2d(N, C, 4, [&](size_t n, size_t c) {
        const int8_t* in_nc = input + n * in_stride_n + c * in_stride_c;

        int32_t sum = 0;
        for (size_t i = 0; i < spatial_size; ++i) {
            sum += static_cast<int32_t>(in_nc[i]);
        }

        float avg = static_cast<float>(sum) * input_scale * inv_spatial;
        float quantized = avg / output_scale;
        quantized = std::max(-128.0f, std::min(127.0f, std::round(quantized)));
        output[n * C + c] = static_cast<int8_t>(quantized);
    });
}

void cactus_conv2d_f32(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    size_t N, size_t C_in, size_t H, size_t W_in,
    size_t C_out,
    size_t kernel_h, size_t kernel_w,
    size_t stride_h, size_t stride_w,
    size_t pad_top, size_t pad_left,
    size_t pad_bottom, size_t pad_right,
    size_t groups)
{
    const size_t C_in_per_group = C_in / groups;
    const size_t C_out_per_group = C_out / groups;

    const size_t H_out = (H + pad_top + pad_bottom - kernel_h) / stride_h + 1;
    const size_t W_out = (W_in + pad_left + pad_right - kernel_w) / stride_w + 1;

    const size_t stride_n_in = C_in * H * W_in;
    const size_t stride_c_in = H * W_in;
    const size_t stride_n_out = C_out * H_out * W_out;
    const size_t stride_c_out = H_out * W_out;

    const size_t stride_co_w = C_in_per_group * kernel_h * kernel_w;
    const size_t stride_ci_w = kernel_h * kernel_w;

    CactusThreading::parallel_for_2d(N, C_out, 4, [&](size_t n, size_t c_out_index) {
        const size_t g = c_out_index / C_out_per_group;
        const size_t co = c_out_index % C_out_per_group;

        const size_t c_in_group_start = g * C_in_per_group;
        const float* in_n = input + n * stride_n_in;
        float* out_nc = output + n * stride_n_out + c_out_index * stride_c_out;

        const float* w_co = weights + c_out_index * stride_co_w;
        const float b_val = bias ? bias[c_out_index] : 0.0f;

        for (size_t oh = 0; oh < H_out; ++oh) {
            const ptrdiff_t in_h_start = static_cast<ptrdiff_t>(oh * stride_h) - static_cast<ptrdiff_t>(pad_top);

            for (size_t ow = 0; ow < W_out; ++ow) {
                const ptrdiff_t in_w_start = static_cast<ptrdiff_t>(ow * stride_w) - static_cast<ptrdiff_t>(pad_left);

                float acc = b_val;

                for (size_t ci = 0; ci < C_in_per_group; ++ci) {
                    const size_t c_in_index = c_in_group_start + ci;
                    const float* in_nc = in_n + c_in_index * stride_c_in;
                    const float* w_ci = w_co + ci * stride_ci_w;

                    for (size_t kh = 0; kh < kernel_h; ++kh) {
                        const ptrdiff_t in_h = in_h_start + static_cast<ptrdiff_t>(kh);
                        if (in_h < 0 || in_h >= static_cast<ptrdiff_t>(H)) continue;

                        for (size_t kw = 0; kw < kernel_w; ++kw) {
                            const ptrdiff_t in_w = in_w_start + static_cast<ptrdiff_t>(kw);
                            if (in_w < 0 || in_w >= static_cast<ptrdiff_t>(W_in)) continue;

                            const size_t in_idx = static_cast<size_t>(in_h) * W_in + static_cast<size_t>(in_w);
                            const size_t w_idx = kh * kernel_w + kw;
                            acc += in_nc[in_idx] * w_ci[w_idx];
                        }
                    }
                }

                out_nc[oh * W_out + ow] = acc;
            }
        }
    });
}

void cactus_conv2d_f16(
    const __fp16* input,
    const __fp16* weights,
    const __fp16* bias,
    __fp16* output,
    size_t N, size_t C_in, size_t H, size_t W_in,
    size_t C_out,
    size_t kernel_h, size_t kernel_w,
    size_t stride_h, size_t stride_w,
    size_t pad_top, size_t pad_left,
    size_t pad_bottom, size_t pad_right,
    size_t groups)
{
    const size_t C_in_per_group = C_in / groups;
    const size_t C_out_per_group = C_out / groups;

    const size_t H_out = (H + pad_top + pad_bottom - kernel_h) / stride_h + 1;
    const size_t W_out = (W_in + pad_left + pad_right - kernel_w) / stride_w + 1;

    const size_t stride_n_in = C_in * H * W_in;
    const size_t stride_c_in = H * W_in;
    const size_t stride_n_out = C_out * H_out * W_out;
    const size_t stride_c_out = H_out * W_out;

    const size_t stride_co_w = C_in_per_group * kernel_h * kernel_w;
    const size_t stride_ci_w = kernel_h * kernel_w;

    CactusThreading::parallel_for_2d(N, C_out, 4, [&](size_t n, size_t c_out_index) {
        const size_t g = c_out_index / C_out_per_group;
        const size_t co = c_out_index % C_out_per_group;

        const size_t c_in_group_start = g * C_in_per_group;
        const __fp16* in_n = input + n * stride_n_in;
        __fp16* out_nc = output + n * stride_n_out + c_out_index * stride_c_out;

        const __fp16* w_co = weights + c_out_index * stride_co_w;
        const float b_val = bias ? static_cast<float>(bias[c_out_index]) : 0.0f;

        for (size_t oh = 0; oh < H_out; ++oh) {
            const ptrdiff_t in_h_start = static_cast<ptrdiff_t>(oh * stride_h) - static_cast<ptrdiff_t>(pad_top);

            for (size_t ow = 0; ow < W_out; ++ow) {
                const ptrdiff_t in_w_start = static_cast<ptrdiff_t>(ow * stride_w) - static_cast<ptrdiff_t>(pad_left);

                float acc = b_val;

                for (size_t ci = 0; ci < C_in_per_group; ++ci) {
                    const size_t c_in_index = c_in_group_start + ci;
                    const __fp16* in_nc = in_n + c_in_index * stride_c_in;
                    const __fp16* w_ci = w_co + ci * stride_ci_w;

                    for (size_t kh = 0; kh < kernel_h; ++kh) {
                        const ptrdiff_t in_h = in_h_start + static_cast<ptrdiff_t>(kh);
                        if (in_h < 0 || in_h >= static_cast<ptrdiff_t>(H)) continue;

                        for (size_t kw = 0; kw < kernel_w; ++kw) {
                            const ptrdiff_t in_w = in_w_start + static_cast<ptrdiff_t>(kw);
                            if (in_w < 0 || in_w >= static_cast<ptrdiff_t>(W_in)) continue;

                            const size_t in_idx = static_cast<size_t>(in_h) * W_in + static_cast<size_t>(in_w);
                            const size_t w_idx = kh * kernel_w + kw;
                            acc += static_cast<float>(in_nc[in_idx]) * static_cast<float>(w_ci[w_idx]);
                        }
                    }
                }

                out_nc[oh * W_out + ow] = static_cast<__fp16>(acc);
            }
        }
    });
}

void cactus_conv_transpose2d_f32(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    size_t N, size_t C_in, size_t H_in, size_t W_in,
    size_t C_out,
    size_t kernel_h, size_t kernel_w,
    size_t stride,
    size_t pad,
    size_t groups,
    size_t H_out, size_t W_out)
{
    const size_t C_in_per_group = C_in / groups;
    const size_t C_out_per_group = C_out / groups;

    const size_t stride_n_in = C_in * H_in * W_in;
    const size_t stride_c_in = H_in * W_in;
    const size_t stride_n_out = C_out * H_out * W_out;
    const size_t stride_c_out = H_out * W_out;

    // Weight layout: [C_in, C_out_per_group, kH, kW]
    const size_t stride_ci_w = C_out_per_group * kernel_h * kernel_w;
    const size_t stride_co_w = kernel_h * kernel_w;

    // Initialize output with bias or zeros
    CactusThreading::parallel_for_2d(N, C_out, 4, [&](size_t n, size_t co) {
        float* out_nc = output + n * stride_n_out + co * stride_c_out;
        const float b_val = bias ? bias[co] : 0.0f;
        for (size_t i = 0; i < stride_c_out; ++i) {
            out_nc[i] = b_val;
        }
    });

    // Main loop: scatter input contributions into output
    for (size_t n = 0; n < N; ++n) {
        const float* in_n = input + n * stride_n_in;
        float* out_n = output + n * stride_n_out;

        for (size_t g = 0; g < groups; ++g) {
            const size_t c_in_group_start = g * C_in_per_group;
            const size_t c_out_group_start = g * C_out_per_group;

            const float* w_group = weights + c_in_group_start * stride_ci_w;

            for (size_t ci = 0; ci < C_in_per_group; ++ci) {
                const size_t c_in_index = c_in_group_start + ci;
                const float* in_nc = in_n + c_in_index * stride_c_in;
                const float* w_ci = w_group + ci * stride_ci_w;

                for (size_t ih = 0; ih < H_in; ++ih) {
                    for (size_t iw = 0; iw < W_in; ++iw) {
                        const float x_val = in_nc[ih * W_in + iw];
                        if (x_val == 0.0f) continue;

                        const ptrdiff_t out_h_base = static_cast<ptrdiff_t>(ih * stride) - static_cast<ptrdiff_t>(pad);
                        const ptrdiff_t out_w_base = static_cast<ptrdiff_t>(iw * stride) - static_cast<ptrdiff_t>(pad);

                        for (size_t kh = 0; kh < kernel_h; ++kh) {
                            const ptrdiff_t oh = out_h_base + static_cast<ptrdiff_t>(kh);
                            if (oh < 0 || oh >= static_cast<ptrdiff_t>(H_out)) continue;

                            for (size_t kw = 0; kw < kernel_w; ++kw) {
                                const ptrdiff_t ow = out_w_base + static_cast<ptrdiff_t>(kw);
                                if (ow < 0 || ow >= static_cast<ptrdiff_t>(W_out)) continue;

                                const size_t w_idx = kh * kernel_w + kw;

                                for (size_t co = 0; co < C_out_per_group; ++co) {
                                    const size_t c_out_index = c_out_group_start + co;
                                    float* out_nc = out_n + c_out_index * stride_c_out;
                                    const float* w_co = w_ci + co * stride_co_w;

                                    out_nc[static_cast<size_t>(oh) * W_out + static_cast<size_t>(ow)] +=
                                        x_val * w_co[w_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void cactus_conv_transpose2d_f16(
    const __fp16* input,
    const __fp16* weights,
    const __fp16* bias,
    __fp16* output,
    size_t N, size_t C_in, size_t H_in, size_t W_in,
    size_t C_out,
    size_t kernel_h, size_t kernel_w,
    size_t stride,
    size_t pad,
    size_t groups,
    size_t H_out, size_t W_out)
{
    const size_t C_in_per_group = C_in / groups;
    const size_t C_out_per_group = C_out / groups;

    const size_t stride_n_in = C_in * H_in * W_in;
    const size_t stride_c_in = H_in * W_in;
    const size_t stride_n_out = C_out * H_out * W_out;
    const size_t stride_c_out = H_out * W_out;

    const size_t stride_ci_w = C_out_per_group * kernel_h * kernel_w;
    const size_t stride_co_w = kernel_h * kernel_w;

    // Initialize output with bias or zeros
    CactusThreading::parallel_for_2d(N, C_out, 4, [&](size_t n, size_t co) {
        __fp16* out_nc = output + n * stride_n_out + co * stride_c_out;
        const float b_val = bias ? static_cast<float>(bias[co]) : 0.0f;
        for (size_t i = 0; i < stride_c_out; ++i) {
            out_nc[i] = static_cast<__fp16>(b_val);
        }
    });

    // Main loop: scatter input contributions into output
    for (size_t n = 0; n < N; ++n) {
        const __fp16* in_n = input + n * stride_n_in;
        __fp16* out_n = output + n * stride_n_out;

        for (size_t g = 0; g < groups; ++g) {
            const size_t c_in_group_start = g * C_in_per_group;
            const size_t c_out_group_start = g * C_out_per_group;

            const __fp16* w_group = weights + c_in_group_start * stride_ci_w;

            for (size_t ci = 0; ci < C_in_per_group; ++ci) {
                const size_t c_in_index = c_in_group_start + ci;
                const __fp16* in_nc = in_n + c_in_index * stride_c_in;
                const __fp16* w_ci = w_group + ci * stride_ci_w;

                for (size_t ih = 0; ih < H_in; ++ih) {
                    for (size_t iw = 0; iw < W_in; ++iw) {
                        const float x_val = static_cast<float>(in_nc[ih * W_in + iw]);
                        if (x_val == 0.0f) continue;

                        const ptrdiff_t out_h_base = static_cast<ptrdiff_t>(ih * stride) - static_cast<ptrdiff_t>(pad);
                        const ptrdiff_t out_w_base = static_cast<ptrdiff_t>(iw * stride) - static_cast<ptrdiff_t>(pad);

                        for (size_t kh = 0; kh < kernel_h; ++kh) {
                            const ptrdiff_t oh = out_h_base + static_cast<ptrdiff_t>(kh);
                            if (oh < 0 || oh >= static_cast<ptrdiff_t>(H_out)) continue;

                            for (size_t kw = 0; kw < kernel_w; ++kw) {
                                const ptrdiff_t ow = out_w_base + static_cast<ptrdiff_t>(kw);
                                if (ow < 0 || ow >= static_cast<ptrdiff_t>(W_out)) continue;

                                const size_t w_idx = kh * kernel_w + kw;

                                for (size_t co = 0; co < C_out_per_group; ++co) {
                                    const size_t c_out_index = c_out_group_start + co;
                                    __fp16* out_nc = out_n + c_out_index * stride_c_out;
                                    const __fp16* w_co = w_ci + co * stride_co_w;

                                    float current = static_cast<float>(out_nc[static_cast<size_t>(oh) * W_out + static_cast<size_t>(ow)]);
                                    current += x_val * static_cast<float>(w_co[w_idx]);
                                    out_nc[static_cast<size_t>(oh) * W_out + static_cast<size_t>(ow)] = static_cast<__fp16>(current);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

