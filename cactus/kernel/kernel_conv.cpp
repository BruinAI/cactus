#include "kernel.h"
#include "kernel_utils.h"
#include <arm_neon.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

// Causal conv1d kernels
// Layouts: input [N, L, C_in], weight [C_out, C_in, K], output [N, L, C_out]
// Only past taps (t - k*dilation >= 0) are included.
void cactus_conv1d_causal_int8(
    const int8_t* input,
    const int8_t* weight,
    int8_t* output,
    size_t batch_size,
    size_t length,
    size_t in_channels,
    size_t out_channels,
    size_t kernel_size,
    size_t dilation,
    float input_scale,
    float weight_scale,
    float output_scale
) {
    constexpr size_t VEC = 16;
    constexpr size_t UNROLL = 2;
    const size_t in_aligned = (in_channels / (VEC * UNROLL)) * (VEC * UNROLL);

    const size_t in_batch_stride  = length * in_channels;
    const size_t out_batch_stride = length * out_channels;
    const size_t w_oc_stride      = in_channels * kernel_size;
    const size_t w_k_stride       = in_channels;
    const float  prod_scale       = input_scale * weight_scale;

    CactusThreading::parallel_for(batch_size * out_channels, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
        [=](size_t start_idx, size_t end_idx) {
            for (size_t bo = start_idx; bo < end_idx; ++bo) {
                const size_t b  = bo / out_channels;
                const size_t oc = bo % out_channels;

                const int8_t* Xb = input  + b * in_batch_stride;
                const int8_t* Woc = weight + oc * w_oc_stride;
                int8_t*      Yb = output + b * out_batch_stride;

                for (size_t t = 0; t < length; ++t) {
                    int32_t acc_total = 0;

                    for (size_t k = 0; k < kernel_size; ++k) {
                        const ptrdiff_t x_t = static_cast<ptrdiff_t>(t) - static_cast<ptrdiff_t>(k * dilation);
                        if (x_t < 0) break;

                        const int8_t* x_ptr = Xb + static_cast<size_t>(x_t) * in_channels;
                        const int8_t* w_ptr = Woc + k * w_k_stride;

                        int32x4_t acc_v = vdupq_n_s32(0);
                        for (size_t c = 0; c < in_aligned; c += VEC * UNROLL) {
                            int8x16_t x0 = vld1q_s8(&x_ptr[c]);
                            int8x16_t w0 = vld1q_s8(&w_ptr[c]);
                            int8x16_t x1 = vld1q_s8(&x_ptr[c + VEC]);
                            int8x16_t w1 = vld1q_s8(&w_ptr[c + VEC]);

                            acc_v = accum_i8mm(acc_v, x0, w0);
                            acc_v = accum_i8mm(acc_v, x1, w1);
                        }
                        acc_total += vaddvq_s32(acc_v);

                        for (size_t c = in_aligned; c < in_channels; c += VEC) {
                            size_t rem = std::min(VEC, in_channels - c);
                            int8_t x_tmp[VEC] = {};
                            int8_t w_tmp[VEC] = {};
                            memcpy(x_tmp, &x_ptr[c], rem);
                            memcpy(w_tmp, &w_ptr[c], rem);

                            int8x16_t xv = vld1q_s8(x_tmp);
                            int8x16_t wv = vld1q_s8(w_tmp);

                            int32x4_t acc_r = vdupq_n_s32(0);
                            acc_r = accum_i8mm(acc_r, xv, wv);
                            acc_total += vaddvq_s32(acc_r);
                        }
                    }

                    float y_fp = static_cast<float>(acc_total) * prod_scale;
                    float y_q = y_fp / output_scale;
                    int32_t q = static_cast<int32_t>(y_q >= 0.f ? (y_q + 0.5f) : (y_q - 0.5f));
                    q = std::max(-128, std::min(127, q));
                    Yb[t * out_channels + oc] = static_cast<int8_t>(q);
                }
            }
        });
}

void cactus_conv1d_causal_f32(
    const float* input,
    const float* weight,
    float* output,
    size_t batch_size,
    size_t length,
    size_t in_channels,
    size_t out_channels,
    size_t kernel_size,
    size_t dilation
) {
    constexpr size_t VEC = 4;  // float32x4
    const size_t in_aligned = (in_channels / VEC) * VEC;

    const size_t in_batch_stride  = length * in_channels;
    const size_t out_batch_stride = length * out_channels;
    const size_t w_oc_stride      = in_channels * kernel_size;
    const size_t w_k_stride       = in_channels;

    CactusThreading::parallel_for(batch_size * out_channels, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
        [=](size_t start_idx, size_t end_idx) {
            for (size_t bo = start_idx; bo < end_idx; ++bo) {
                const size_t b  = bo / out_channels;
                const size_t oc = bo % out_channels;

                const float* Xb = input  + b * in_batch_stride;
                const float* Woc = weight + oc * w_oc_stride;
                float*       Yb = output + b * out_batch_stride;

                for (size_t t = 0; t < length; ++t) {
                    float32x4_t acc_v = vdupq_n_f32(0.0f);
                    float acc_tail = 0.0f;

                    for (size_t k = 0; k < kernel_size; ++k) {
                        const ptrdiff_t x_t = static_cast<ptrdiff_t>(t) - static_cast<ptrdiff_t>(k * dilation);
                        if (x_t < 0) break;

                        const float* x_ptr = Xb  + static_cast<size_t>(x_t) * in_channels;
                        const float* w_ptr = Woc + k * w_k_stride;

                        for (size_t c = 0; c < in_aligned; c += VEC) {
                            float32x4_t xv = vld1q_f32(&x_ptr[c]);
                            float32x4_t wv = vld1q_f32(&w_ptr[c]);
                            acc_v = vfmaq_f32(acc_v, xv, wv);
                        }
                        for (size_t c = in_aligned; c < in_channels; ++c) {
                            acc_tail += x_ptr[c] * w_ptr[c];
                        }
                    }

                    float y = vaddvq_f32(acc_v) + acc_tail;
                    Yb[t * out_channels + oc] = y;
                }
            }
        });
}

void cactus_conv1d_causal_f16(
    const __fp16* input,
    const __fp16* weight,
    __fp16* output,
    size_t batch_size,
    size_t length,
    size_t in_channels,
    size_t out_channels,
    size_t kernel_size,
    size_t dilation
) {
    constexpr size_t VECH = 8;
    constexpr size_t VECF = 4;
    const size_t in_aligned = (in_channels / VECH) * VECH;

    const size_t in_batch_stride  = length * in_channels;
    const size_t out_batch_stride = length * out_channels;
    const size_t w_oc_stride      = in_channels * kernel_size;
    const size_t w_k_stride       = in_channels;

    CactusThreading::parallel_for(batch_size * out_channels, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
        [=](size_t start_idx, size_t end_idx) {
            for (size_t bo = start_idx; bo < end_idx; ++bo) {
                const size_t b  = bo / out_channels;
                const size_t oc = bo % out_channels;

                const __fp16* Xb  = input  + b * in_batch_stride;
                const __fp16* Woc = weight + oc * w_oc_stride;
                __fp16*       Yb  = output + b * out_batch_stride;

                for (size_t t = 0; t < length; ++t) {
                    float32x4_t acc0 = vdupq_n_f32(0.0f);
                    float32x4_t acc1 = vdupq_n_f32(0.0f);
                    float tail = 0.0f;

                    for (size_t k = 0; k < kernel_size; ++k) {
                        const ptrdiff_t x_t = static_cast<ptrdiff_t>(t) - static_cast<ptrdiff_t>(k * dilation);
                        if (x_t < 0) break;

                        const __fp16* x_ptr = Xb  + static_cast<size_t>(x_t) * in_channels;
                        const __fp16* w_ptr = Woc + k * w_k_stride;

                        size_t c = 0;
                        for (; c + VECH <= in_aligned; c += VECH) {
                            float16x8_t xv16 = vld1q_f16(&x_ptr[c]);
                            float16x8_t wv16 = vld1q_f16(&w_ptr[c]);

                            float32x4_t xv_lo = vcvt_f32_f16(vget_low_f16(xv16));
                            float32x4_t xv_hi = vcvt_f32_f16(vget_high_f16(xv16));
                            float32x4_t wv_lo = vcvt_f32_f16(vget_low_f16(wv16));
                            float32x4_t wv_hi = vcvt_f32_f16(vget_high_f16(wv16));

                            acc0 = vfmaq_f32(acc0, xv_lo, wv_lo);
                            acc1 = vfmaq_f32(acc1, xv_hi, wv_hi);
                        }
                        for (; c < in_channels; ++c) {
                            tail += static_cast<float>(x_ptr[c]) * static_cast<float>(w_ptr[c]);
                        }
                    }

                    float y = vaddvq_f32(acc0) + vaddvq_f32(acc1) + tail;
                    Yb[t * out_channels + oc] = static_cast<__fp16>(y);
                }
            }
        });
}

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

inline void cactus_conv1d_depthwise_causal_f16_impl(
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


#include <cstddef>

inline void cactus_conv1d_depthwise_causal_f32_impl(
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

inline void cactus_conv1d_depthwise_causal_int8_impl(
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

// Requantized s8 output with **per-channel** scales:
//   acc_scale[c] = (input_scale[c] * weight_scale[c]) / output_scale[c]
inline void cactus_conv1d_depthwise_causal_s8_neon_s8out_tiled(
    const int8_t* input,      // [N,L,C]
    const int8_t* weight,     // [C,1,K]
    int8_t* output,           // [N,L,C]
    const float* acc_scale,   // [C] per-channel
    size_t N, size_t L, size_t C, size_t K, size_t dilation)
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

    // per-channel requantize
    for (size_t n=0;n<N;++n){
        int32_t* src = tmp.data() + n*(L*C);
        int8_t*  dst = output     + n*(L*C);
        for (size_t t=0;t<L;++t){
            for (size_t c=0;c<C;++c){
                float scaled = (float)src[t*C + c] * acc_scale[c];
                int32_t q = (int32_t)lrintf(scaled);
                q = std::min(127, std::max(-128, q));
                dst[t*C + c] = (int8_t)q;
            }
        }
    }
}

template <typename T>
static void cactus_conv1d_causal_depthwise_naive_float(
    const T* input,
    const T* weight,
    T* output,
    size_t N,
    size_t L,
    size_t C_in,
    size_t K,
    size_t dilation,
    size_t M)
{
    const size_t C_out = C_in * M;
    const size_t in_batch_stride  = L * C_in;
    const size_t out_batch_stride = L * C_out;

    for (size_t n = 0; n < N; ++n) {
        const T* Xb = input + n * in_batch_stride;
        T*       Yb = output + n * out_batch_stride;

        for (size_t t = 0; t < L; ++t) {
            for (size_t co = 0; co < C_out; ++co) {
                const size_t c = co / M;
                const T* Wc = weight + co * K;

                float acc = 0.0f;
                for (size_t k = 0; k < K; ++k) {
                    const ptrdiff_t xt = static_cast<ptrdiff_t>(t) - static_cast<ptrdiff_t>(k * dilation);
                    if (xt < 0) break;

                    const float x_val = static_cast<float>(Xb[static_cast<size_t>(xt) * C_in + c]);
                    const float w_val = static_cast<float>(Wc[K - 1 - k]);
                    acc += x_val * w_val;
                }

                Yb[t * C_out + co] = static_cast<T>(acc);
            }
        }
    }
}

static void cactus_conv1d_causal_depthwise_naive_int8(
    const int8_t* input,
    const int8_t* weight,
    int8_t* output,
    size_t N,
    size_t L,
    size_t C_in,
    size_t K,
    size_t dilation,
    size_t M,
    float input_scale,
    float weight_scale,
    float output_scale)
{
    const size_t C_out = C_in * M;
    const size_t in_batch_stride  = L * C_in;
    const size_t out_batch_stride = L * C_out;
    const float acc_scale = (input_scale * weight_scale) / output_scale;

    for (size_t n = 0; n < N; ++n) {
        const int8_t* Xb = input + n * in_batch_stride;
        int8_t*       Yb = output + n * out_batch_stride;

        for (size_t t = 0; t < L; ++t) {
            for (size_t co = 0; co < C_out; ++co) {
                const size_t c = co / M;
                const int8_t* Wc = weight + co * K;

                int32_t acc = 0;
                for (size_t k = 0; k < K; ++k) {
                    const ptrdiff_t xt = static_cast<ptrdiff_t>(t) - static_cast<ptrdiff_t>(k * dilation);
                    if (xt < 0) break;

                    acc += static_cast<int32_t>(Xb[static_cast<size_t>(xt) * C_in + c]) *
                           static_cast<int32_t>(Wc[K - 1 - k]);
                }

                int32_t q = static_cast<int32_t>(lrintf(static_cast<float>(acc) * acc_scale));
                q = std::min(127, std::max(-128, q));
                Yb[t * C_out + co] = static_cast<int8_t>(q);
            }
        }
    }
}

void cactus_conv1d_depthwise_causal_f32(
    const float* input,
    const float* weight,
    float* output,
    size_t N,
    size_t L,
    size_t C,
    size_t K,
    size_t dilation)
{
    cactus_conv1d_depthwise_causal_f32_impl(input, weight, output, N, L, C, K, dilation);
}

void cactus_conv1d_causal_depthwise_f32(
    const float* x,
    const float* w,
    float* y,
    size_t N,
    size_t L,
    size_t C_in,
    size_t K,
    size_t dilation,
    size_t M)
{
    if (M == 1) {
        cactus_conv1d_depthwise_causal_f32_impl(x, w, y, N, L, C_in, K, dilation);
        return;
    }

    cactus_conv1d_causal_depthwise_naive_float(x, w, y, N, L, C_in, K, dilation, M);
}

void cactus_conv1d_causal_depthwise_f16(
    const __fp16* x,
    const __fp16* w,
    __fp16* y,
    size_t N,
    size_t L,
    size_t C_in,
    size_t K,
    size_t dilation,
    size_t M)
{
    if (M == 1) {
        cactus_conv1d_depthwise_causal_f16_impl(x, w, y, N, L, C_in, K, dilation);
        return;
    }

    cactus_conv1d_causal_depthwise_naive_float(x, w, y, N, L, C_in, K, dilation, M);
}

void cactus_conv1d_causal_depthwise_int8(
    const int8_t* x,
    const int8_t* w,
    int8_t* y,
    size_t N,
    size_t L,
    size_t C_in,
    size_t K,
    size_t dilation,
    size_t M,
    float input_scale,
    float weight_scale,
    float output_scale)
{
    if (M == 1) {
        std::vector<float> acc_scale(C_in);
        const float factor = (input_scale * weight_scale) / output_scale;
        std::fill(acc_scale.begin(), acc_scale.end(), factor);
        cactus_conv1d_depthwise_causal_s8_neon_s8out_tiled(
            x,
            w,
            y,
            acc_scale.data(),
            N,
            L,
            C_in,
            K,
            dilation);
        return;
    }

    cactus_conv1d_causal_depthwise_naive_int8(
        x,
        w,
        y,
        N,
        L,
        C_in,
        K,
        dilation,
        M,
        input_scale,
        weight_scale,
        output_scale);
}
