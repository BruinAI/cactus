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

// Layouts:
// X: [N, L, C_in]
// W: [C_in * M, 1, K]   (packed as [C_out, K], where C_out = C_in*M)
// Y: [N, L, C_out]
//
// y[n,t,c*M+m] = sum_{k=0..K-1, t-k*d>=0} X[n, t-k*d, c] * W[c*M+m, k]

static inline ptrdiff_t safe_ti(size_t t, size_t k, size_t dilation) {
    return (ptrdiff_t)t - (ptrdiff_t)k * (ptrdiff_t)dilation;
}


constexpr size_t T_TILE_F16 = 2;

void cactus_conv1d_causal_depthwise_f16(
    const __fp16* input,    // [N, L, C]
    const __fp16* weight,   // [C, 1, K] forward order (fp16)
    __fp16* output,         // [N, L, C]
    size_t N, size_t L, size_t C, size_t K, size_t dilation)
{
    const size_t in_bs  = L * C;
    const size_t out_bs = L * C;

    for (size_t n = 0; n < N; ++n) {
        const __fp16* Xb = input  + n * in_bs;
        __fp16*       Yb = output + n * out_bs;

        for (size_t c = 0; c < C; ++c) {
            // Pre-reverse weights (convert to f32 here to avoid repeated casts)
            std::vector<float> wrev(K);
            const __fp16* Wc = weight + c * K;
            for (size_t k = 0; k < K; ++k) wrev[k] = (float)Wc[K - 1 - k];

            for (size_t t0 = 0; t0 < L; t0 += T_TILE_F16) {
                const size_t t1 = std::min(t0 + 1, L - 1);

                float32x4_t vacc0 = vdupq_n_f32(0.f);
                float32x4_t vacc1 = vdupq_n_f32(0.f);

                size_t k = 0;
                for (; k + 8 <= K; k += 8) {
                    // taps k..k+3
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

                    // taps k+4..k+7
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
    const float* input,     // [N, L, C] (inner stride C)
    const float* weight,    // [C, 1, K] forward order
    float* output,          // [N, L, C]
    size_t N, size_t L, size_t C, size_t K, size_t dilation)
{
    const size_t T_TILE_F32 = 2;
    const size_t in_bs  = L * C;
    const size_t out_bs = L * C;

    // process batch, channel
    for (size_t n = 0; n < N; ++n) {
        const float* Xb = input  + n * in_bs;
        float*       Yb = output + n * out_bs;

        for (size_t c = 0; c < C; ++c) {
            // Pre-reverse weights for this channel (one-time, cheap, improves inner-loop)
            std::vector<float> wrev(K);
            const float* Wc = weight + c * K;
            for (size_t k = 0; k < K; ++k) wrev[k] = Wc[K - 1 - k];

            // Walk time in tiles of 2
            for (size_t t0 = 0; t0 < L; t0 += T_TILE_F32) {
                const size_t t1 = std::min(t0 + 1, L - 1);

                // two accumulators for the two time points
                float32x4_t vacc0_0 = vdupq_n_f32(0.f);
                float32x4_t vacc0_1 = vdupq_n_f32(0.f);
                float       tail0   = 0.f;
                float       tail1   = 0.f;

                // Tap blocks: 8 at a time (two 4-lane vfma blocks)
                size_t k = 0;
                for (; k + 8 <= K; k += 8) {
                    // prefetch upcoming input rows lightly
                    if (t0 + 4 < L) __builtin_prefetch(Xb + (t0 + 4) * C + c);

                    // -------- taps k..k+3
                    float x0_0 = 0.f, x1_0 = 0.f, x2_0 = 0.f, x3_0 = 0.f; // for t0
                    float x0_1 = 0.f, x1_1 = 0.f, x2_1 = 0.f, x3_1 = 0.f; // for t1
                    {
                        ptrdiff_t xt0 = (ptrdiff_t)t0 - (ptrdiff_t)((k + 0) * dilation);
                        ptrdiff_t xt1 = (ptrdiff_t)t0 - (ptrdiff_t)((k + 1) * dilation);
                        ptrdiff_t xt2 = (ptrdiff_t)t0 - (ptrdiff_t)((k + 2) * dilation);
                        ptrdiff_t xt3 = (ptrdiff_t)t0 - (ptrdiff_t)((k + 3) * dilation);

                        if (xt0 >= 0) x0_0 = Xb[(size_t)xt0 * C + c];
                        if (xt1 >= 0) x1_0 = Xb[(size_t)xt1 * C + c];
                        if (xt2 >= 0) x2_0 = Xb[(size_t)xt2 * C + c];
                        if (xt3 >= 0) x3_0 = Xb[(size_t)xt3 * C + c];

                        // second time point (t1)
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

                    // -------- taps k+4..k+7
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

                // reduce vector accumulators
                float acc0 = vaddvq_f32(vacc0_0) + vaddvq_f32(vacc0_1);
                float acc1 = vaddvq_f32(vacc0_1); // (vacc0_1 already included above for t1 vector parts)
                // fix: we summed vacc0_1 twice; correct way:
                acc0 = vaddvq_f32(vacc0_0);
                acc1 = vaddvq_f32(vacc0_1);

                // scalar tail taps
                for (; k < K; ++k) {
                    ptrdiff_t x0 = (ptrdiff_t)t0 - (ptrdiff_t)(k * dilation);
                    if (x0 >= 0) acc0 += wrev[k] * Xb[(size_t)x0 * C + c];
                    ptrdiff_t x1 = (ptrdiff_t)t1 - (ptrdiff_t)(k * dilation);
                    if (x1 >= 0) acc1 += wrev[k] * Xb[(size_t)x1 * C + c];
                }

                // store t0
                Yb[t0 * C + c] = acc0;
                // store t1 if it belongs to this tile
                if (t0 + 1 < L) {
                    Yb[t1 * C + c] = acc1;
                }
            }
        }
    }
}


constexpr size_t T_TILE_S8 = 2;

static void cactus_conv1d_depthwise_causal_int8_impl(
    const int8_t* input,     // [N,L,C]
    const int8_t* weight,    // [C,1,K] forward
    int32_t* output,         // [N,L,C]
    size_t N, size_t L, size_t C, size_t K, size_t dilation)
{
    const size_t in_bs  = L*C;
    const size_t out_bs = L*C;

    for (size_t n=0;n<N;++n){
        const int8_t* Xb = input  + n*in_bs;
        int32_t*      Yb = output + n*out_bs;

        for (size_t c=0;c<C;++c){
            // Pre-reverse weights
            std::vector<int8_t> wrev(K);
            const int8_t* Wc = weight + c*K;
            for (size_t k=0;k<K;++k) wrev[k] = Wc[K-1-k];

            for (size_t t0=0;t0<L; t0+=T_TILE_S8){
                const size_t t1 = std::min(t0+1, L-1);

                int32x4_t vacc0 = vdupq_n_s32(0);
                int32x4_t vacc1 = vdupq_n_s32(0);

                size_t k=0;

#if defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)
                // 16-tap SDOT blocks
                for (; k+16<=K; k+=16){
                    int8_t xv0_buf[16];
                    int8_t xv1_buf[16];
                    for (int i = 0; i < 16; ++i) {
                        ptrdiff_t a = (ptrdiff_t)t0 - (ptrdiff_t)((k + i) * dilation);
                        xv0_buf[i] = (a >= 0) ? Xb[(size_t)a * C + c] : 0;
                        ptrdiff_t b = (ptrdiff_t)t1 - (ptrdiff_t)((k + i) * dilation);
                        xv1_buf[i] = (b >= 0) ? Xb[(size_t)b * C + c] : 0;
                    }

                    int8x16_t xv0 = vld1q_s8(xv0_buf);
                    int8x16_t xv1 = vld1q_s8(xv1_buf);
                    int8x16_t wv  = vld1q_s8(&wrev[k]); // contiguous

                    vacc0 = vdotq_s32(vacc0, xv0, wv);
                    vacc1 = vdotq_s32(vacc1, xv1, wv);
                }
#endif
                // 8-tap fallback blocks (portable)
                for (; k+8<=K; k+=8){
                    int8_t xv0_buf[8];
                    int8_t xv1_buf[8];
                    for (int i = 0; i < 8; ++i) {
                        ptrdiff_t a = (ptrdiff_t)t0 - (ptrdiff_t)((k + i) * dilation);
                        xv0_buf[i] = (a >= 0) ? Xb[(size_t)a * C + c] : 0;
                        ptrdiff_t b = (ptrdiff_t)t1 - (ptrdiff_t)((k + i) * dilation);
                        xv1_buf[i] = (b >= 0) ? Xb[(size_t)b * C + c] : 0;
                    }

                    int8x8_t xv0 = vld1_s8(xv0_buf);
                    int8x8_t xv1 = vld1_s8(xv1_buf);
                    int8x8_t wv  = vld1_s8(&wrev[k]);

                    // vmull_s8 -> int16x8_t, then pairwise add+ widen
                    int16x8_t p0 = vmull_s8(xv0, wv);
                    int16x8_t p1 = vmull_s8(xv1, wv);
                    int32x4_t s0 = vpaddlq_s16(p0);
                    int32x4_t s1 = vpaddlq_s16(p1);
                    vacc0 = vaddq_s32(vacc0, s0);
                    vacc1 = vaddq_s32(vacc1, s1);
                }

                // scalar tail
                int32_t acc0 = vaddvq_s32(vacc0);
                int32_t acc1 = vaddvq_s32(vacc1);
                for (; k<K; ++k){
                    ptrdiff_t a=(ptrdiff_t)t0-(ptrdiff_t)(k*dilation);
                    if (a>=0) acc0 += (int32_t)Xb[(size_t)a*C + c] * (int32_t)wrev[k];
                    ptrdiff_t b=(ptrdiff_t)t1-(ptrdiff_t)(k*dilation);
                    if (b>=0) acc1 += (int32_t)Xb[(size_t)b*C + c] * (int32_t)wrev[k];
                }

                Yb[t0*C + c] = acc0;
                if (t0+1<L) Yb[t1*C + c] = acc1;
            }
        }
    }
}

// Requantized s8 output with uniform scale:
//   acc_scale = (input_scale * weight_scale) / output_scale
void cactus_conv1d_causal_depthwise_int8(
    const int8_t* input,      // [N,L,C]
    const int8_t* weight,     // [C,1,K]
    int8_t* output,           // [N,L,C]
    size_t N, size_t L, size_t C, size_t K, size_t dilation,
    float input_scale, float weight_scale, float output_scale)
{
    std::vector<int32_t> tmp(static_cast<size_t>(N) * L * C);

    cactus_conv1d_depthwise_causal_int8_impl(
        input,
        weight,
        tmp.data(),
        N,
        L,
        C,
        K,
        dilation);

    // uniform requantize
    const float acc_scale = (input_scale * weight_scale) / output_scale;
    for (size_t n=0;n<N;++n){
        int32_t* src = tmp.data() + n*(L*C);
        int8_t*  dst = output     + n*(L*C);
        for (size_t t=0;t<L;++t){
            for (size_t c=0;c<C;++c){
                float scaled = (float)src[t*C + c] * acc_scale;
                int32_t q = (int32_t)lrintf(scaled);
                q = std::min(127, std::max(-128, q));
                dst[t*C + c] = (int8_t)q;
            }
        }
    }
}


