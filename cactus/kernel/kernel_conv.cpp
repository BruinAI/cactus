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

// ---------------- FP32 (NEON-accelerated for M==1) ----------------
void cactus_conv1d_causal_depthwise_f32(
    const float* x, const float* w, float* y,
    size_t N, size_t L, size_t C_in,
    size_t K, size_t dilation, size_t M)
{
    const size_t CinM = C_in * M;               // = C_out
    const size_t in_batch_stride  = L * C_in;
    const size_t out_batch_stride = L * CinM;

#if defined(__ARM_NEON)
    if (M == 1) {
        // Vectorize across channels in 4-lane blocks (x is contiguous over channels).
        const size_t Cblk = 4;
        const size_t Cvec = (C_in / Cblk) * Cblk;

        for (size_t n = 0; n < N; ++n) {
            const float* Xn = x + n * in_batch_stride;
            float*       Yn = y + n * out_batch_stride;

            for (size_t t = 0; t < L; ++t) {
                // Vector blocks of 4 channels
                for (size_t c = 0; c < Cvec; c += Cblk) {
                    float32x4_t acc = vdupq_n_f32(0.0f);
                    // Sum over taps with causal guard
                    for (size_t k = 0; k < K; ++k) {
                        const ptrdiff_t ti = safe_ti(t, k, dilation);
                        if (ti < 0) break;

                        // inputs: contiguous across channels
                        const float* xptr = &Xn[ (size_t)ti * C_in + c ];
                        float32x4_t xv = vld1q_f32(xptr);

                        // weights: W is laid out [C_in, K], stride K between channels
                        // Gather 4 scalars with stride K, assemble into a vector
                        float wt[4];
                        wt[0] = w[(c+0)*K + k];
                        wt[1] = w[(c+1)*K + k];
                        wt[2] = w[(c+2)*K + k];
                        wt[3] = w[(c+3)*K + k];
                        float32x4_t wv = vld1q_f32(wt);

                        acc = vfmaq_f32(acc, xv, wv);
                    }
                    vst1q_f32(&Yn[t * CinM + c], acc);
                }

                // Tail channels (scalar)
                for (size_t c = Cvec; c < C_in; ++c) {
                    float acc = 0.f;
                    const float* wc = w + c * K;
                    for (size_t k = 0; k < K; ++k) {
                        const ptrdiff_t ti = safe_ti(t, k, dilation);
                        if (ti < 0) break;
                        acc += Xn[(size_t)ti * C_in + c] * wc[k];
                    }
                    Yn[t * CinM + c] = acc;
                }
            }
        }
        return;
    }
#endif

    // Fallback (handles any M, any platform)
    for (size_t n = 0; n < N; ++n) {
        const float* Xn = x + n * in_batch_stride;
        float*       Yn = y + n * out_batch_stride;

        for (size_t c = 0; c < C_in; ++c) {
            const float* Wc = w + c * (K * M); // [M,K]
            for (size_t t = 0; t < L; ++t) {
                for (size_t m = 0; m < M; ++m) {
                    const float* wcm = Wc + m * K;
                    float acc = 0.0f;
                    for (size_t k = 0; k < K; ++k) {
                        const ptrdiff_t ti = safe_ti(t, k, dilation);
                        if (ti < 0) break;
                        acc += Xn[(size_t)ti * C_in + c] * wcm[k];
                    }
                    Yn[t * CinM + c * M + m] = acc;
                }
            }
        }
    }
}

// ---------------- FP16 (NEON-accelerated for M==1; accumulate in f32) ----------------
void cactus_conv1d_causal_depthwise_f16(
    const __fp16* x, const __fp16* w, __fp16* y,
    size_t N, size_t L, size_t C_in,
    size_t K, size_t dilation, size_t M)
{
    const size_t CinM = C_in * M;
    const size_t in_batch_stride  = L * C_in;
    const size_t out_batch_stride = L * CinM;

#if defined(__ARM_NEON)
    if (M == 1) {
        const size_t Cblk = 4;                  // process 4 channels per step
        const size_t Cvec = (C_in / Cblk) * Cblk;

        for (size_t n = 0; n < N; ++n) {
            const __fp16* Xn = x + n * in_batch_stride;
            __fp16*       Yn = y + n * out_batch_stride;

            for (size_t t = 0; t < L; ++t) {
                for (size_t c = 0; c < Cvec; c += Cblk) {
                    float32x4_t acc = vdupq_n_f32(0.0f);
                    for (size_t k = 0; k < K; ++k) {
                        const ptrdiff_t ti = safe_ti(t, k, dilation);
                        if (ti < 0) break;

                        // load 4 half inputs, convert to f32
                        const __fp16* xptr_h = &Xn[(size_t)ti * C_in + c];
                        float16x4_t xv_h4 = vld1_f16(xptr_h);
                        float32x4_t xv = vcvt_f32_f16(xv_h4);

                        // gather 4 half weights (stride K), convert to f32 vector
                        __fp16 wt_h[4];
                        wt_h[0] = w[(c+0)*K + k];
                        wt_h[1] = w[(c+1)*K + k];
                        wt_h[2] = w[(c+2)*K + k];
                        wt_h[3] = w[(c+3)*K + k];
                        float16x4_t wv_h4 = vld1_f16(wt_h);
                        float32x4_t wv = vcvt_f32_f16(wv_h4);

                        acc = vfmaq_f32(acc, xv, wv);
                    }
                    // store back as f16
                    float16x4_t out_h4 = vcvt_f16_f32(acc);
                    vst1_f16(&Yn[t * CinM + c], out_h4);
                }

                // Tail channels (scalar)
                for (size_t c = Cvec; c < C_in; ++c) {
                    float acc = 0.f;
                    const __fp16* wc = w + c * K;
                    for (size_t k = 0; k < K; ++k) {
                        const ptrdiff_t ti = safe_ti(t, k, dilation);
                        if (ti < 0) break;
                        acc += (float)Xn[(size_t)ti * C_in + c] * (float)wc[k];
                    }
                    Yn[t * CinM + c] = (__fp16)acc;
                }
            }
        }
        return;
    }
#endif

    // Fallback (any M)
    for (size_t n = 0; n < N; ++n) {
        const __fp16* Xn = x + n * in_batch_stride;
        __fp16*       Yn = y + n * out_batch_stride;

        for (size_t c = 0; c < C_in; ++c) {
            const __fp16* Wc = w + c * (K * M);
            for (size_t t = 0; t < L; ++t) {
                for (size_t m = 0; m < M; ++m) {
                    const __fp16* wcm = Wc + m * K;
                    float acc = 0.0f;
                    for (size_t k = 0; k < K; ++k) {
                        const ptrdiff_t ti = safe_ti(t, k, dilation);
                        if (ti < 0) break;
                        acc += (float)Xn[(size_t)ti * C_in + c] * (float)wcm[k];
                    }
                    Yn[t * CinM + c * M + m] = (__fp16)acc;
                }
            }
        }
    }
}

// ---------------- INT8 (scalar; K small so this is fine) ----------------
void cactus_conv1d_causal_depthwise_int8(
    const int8_t* x, const int8_t* w, int8_t* y,
    size_t N, size_t L, size_t C_in,
    size_t K, size_t dilation, size_t M,
    float input_scale, float weight_scale, float output_scale)
{
    const float prod_scale = input_scale * weight_scale;
    const size_t CinM = C_in * M;
    const size_t in_batch_stride  = L * C_in;
    const size_t out_batch_stride = L * CinM;

    for (size_t n = 0; n < N; ++n) {
        const int8_t* Xn = x + n * in_batch_stride;
        int8_t*       Yn = y + n * out_batch_stride;

        for (size_t c = 0; c < C_in; ++c) {
            const int8_t* Wc = w + c * (K * M);
            for (size_t t = 0; t < L; ++t) {
                for (size_t m = 0; m < M; ++m) {
                    const int8_t* wcm = Wc + m * K;
                    float acc_i8 = 0.0f;
                    for (size_t k = 0; k < K; ++k) {
                        const ptrdiff_t ti = safe_ti(t, k, dilation);
                        if (ti < 0) break;
                        acc_i8 += (float)Xn[(size_t)ti * C_in + c] * (float)wcm[k];
                    }
                    const float y_fp = acc_i8 * prod_scale;
                    float y_qf = y_fp / output_scale;
                    int q = (int)(y_qf >= 0.f ? y_qf + 0.5f : y_qf - 0.5f);
                    if (q < -128) q = -128; else if (q > 127) q = 127;
                    Yn[t * CinM + c * M + m] = (int8_t)q;
                }
            }
        }
    }
}

// scalar, depthwise, causal, CORRECT reference
void ref_causal_dw_conv1d_f32(
    const float* x,     // [N, L, C]
    const float* w,     // [C*M, 1, K]  (often M==1)
    float* y,           // [N, L, C*M]
    size_t N, size_t L, size_t C, size_t K, size_t dilation, size_t M)
{
    const size_t CinM = C * M;
    for (size_t n = 0; n < N; ++n) {
        const float* Xn = x + n * L * C;
        float*       Yn = y + n * L * CinM;
        for (size_t t = 0; t < L; ++t) {
            for (size_t c = 0; c < C; ++c) {
                for (size_t m = 0; m < M; ++m) {
                    float acc = 0.f;
                    const float* wcm = w + (c*M + m) * K; // [K]
                    for (size_t k = 0; k < K; ++k) {
                        ptrdiff_t ti = (ptrdiff_t)t - (ptrdiff_t)k * (ptrdiff_t)dilation;
                        if (ti < 0) break;
                        acc += Xn[(size_t)ti * C + c] * wcm[k];
                    }
                    Yn[t * CinM + (c*M + m)] = acc;
                }
            }
        }
    }
}

// Depthwise, causal, 1D conv, FP32
// Input  [N, L, C], Weight [C, 1, K], Output [N, L, C]
// y[n,t,c] = sum_{k=0..K-1} w[c,0,k] * x[n, t - k*d, c], with causal guard (t - k*d >= 0)
// IMPORTANT: tap k=K-1 is the "current" sample to match PyTorch/Dao correlation convention.
void cactus_conv1d_depthwise_causal_f32(
    const float* input,
    const float* weight,  // [C, 1, K]
    float* output,
    size_t N,
    size_t L,
    size_t C,
    size_t K,
    size_t dilation
) {
    const size_t in_batch_stride  = L * C;
    const size_t out_batch_stride = L * C;
    const size_t w_c_stride       = K;   // since Cin=1
    // NOTE: we’ll read taps in reversed k (k_rev = K-1-k) to match Dao/PyTorch causal behavior.
    for (size_t n = 0; n < N; ++n) {
        const float* Xb = input  + n * in_batch_stride;
        float*       Yb = output + n * out_batch_stride;
        for (size_t c = 0; c < C; ++c) {
            const float* Wc = weight + c * w_c_stride;
            for (size_t t = 0; t < L; ++t) {
                float acc = 0.f;
                // iterate causal taps
                for (size_t k = 0; k < K; ++k) {
                    const ptrdiff_t xt = static_cast<ptrdiff_t>(t) - static_cast<ptrdiff_t>(k * dilation);
                    if (xt < 0) break; // causal guard
                    // reversed tap to align current sample with Wc[K-1]
                    const size_t k_rev = K - 1 - k;
                    acc += Wc[k_rev] * Xb[static_cast<size_t>(xt) * C + c];
                }
                Yb[t * C + c] = acc;
            }
        }
    }
}

