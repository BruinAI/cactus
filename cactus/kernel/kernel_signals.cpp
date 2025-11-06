#include "kernel.h"
#include "kernel_utils.h"
#include <arm_neon.h>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// void spectrogram(
//     const float* input, 
//     const size_t hop_length)
// {

// }

void cactus_rfft_f32_1d(const float* input, float* output, const size_t n, const char* norm) {
    const size_t out_len = n / 2 + 1;
    const float two_pi_over_n = 2.0f * M_PI / static_cast<float>(n);
    
    float norm_factor = 1.0f;
    if (norm) {
        if (std::strcmp(norm, "backward") == 0) {
            norm_factor = 1.0f;
        } else if (std::strcmp(norm, "forward") == 0) {
            norm_factor = 1.0f / static_cast<float>(n);
        } else if (std::strcmp(norm, "ortho") == 0) {
            norm_factor = 1.0f / std::sqrt(static_cast<float>(n));
        } else {
            throw std::invalid_argument("norm must be one of {\"backward\",\"forward\",\"ortho\"}");
        }
    }

    for (size_t i = 0; i < out_len; i++) {
        float re = 0.0f;
        float im = 0.0f;
        const float base = -two_pi_over_n * static_cast<float>(i);
        for (size_t j = 0; j < n; j++) {
            const float angle = base * static_cast<float>(j);
            const float input_val = input[j];
            re += input_val * std::cos(angle);
            im += input_val * std::sin(angle);
        }
        
        output[i * 2] = re * norm_factor;
        output[i * 2 + 1] = im * norm_factor;
    }
}