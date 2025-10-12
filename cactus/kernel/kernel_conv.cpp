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
