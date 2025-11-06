#include "kernel.h"
#include "kernel_utils.h"
#include <arm_neon.h>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>
#include <stdexcept>
#include <numbers>
#include <limits>

void cactus_spectrogram_f32(
    const float* waveform,
    size_t waveform_length,
    const float* window,
    size_t window_length,
    size_t frame_length,
    size_t hop_length,
    const size_t* fft_length,
    float* spectrogram,
    float power = 1.0f,
    bool center = true,
    const char* pad_mode = "reflect",
    bool onesided = true,
    float dither = 0.0f,
    const float* preemphasis = nullptr,
    const float* mel_filters = nullptr,
    size_t mel_filters_size = 0,
    float mel_floor = 1e-10f,
    const char* log_mel = nullptr,
    float reference = 1.0f,
    float min_value = 1e-10f,
    const float* db_range = nullptr,
    bool remove_dc_offset = false)
{
    size_t actual_fft_length;
    if (fft_length == nullptr) {
        actual_fft_length = frame_length;
    } else {
        actual_fft_length = *fft_length;
    }

    if (frame_length > actual_fft_length) {
        throw std::invalid_argument(
            "frame_length (" + std::to_string(frame_length) + 
            ") may not be larger than fft_length (" + 
            std::to_string(actual_fft_length) + ")");
    }

    if (window_length != frame_length) {
        throw std::invalid_argument(
            "Length of the window (" + std::to_string(window_length) + 
            ") must equal frame_length (" + std::to_string(frame_length) + ")");
    }


    if (hop_length <= 0) {
        throw std::invalid_argument("hop_length must be greater than zero");
    }


    if (power == 0.0f && mel_filters != nullptr) {
        throw std::invalid_argument(
            "You have provided `mel_filters` but `power` is `None`. "
            "Mel spectrogram computation is not yet supported for complex-valued spectrogram. "
            "Specify `power` to fix this issue.");
    }

    //TODO: implement padding

    const size_t num_frames = 1 + (waveform_length - frame_length) / hop_length;
    const size_t num_frequency_bins = (actual_fft_length / 2) + 1;

    std::vector<float> buffer(actual_fft_length);
    std::vector<float> raw_complex_frequencies(num_frequency_bins * 2);

    const size_t num_mel_bins = mel_filters != nullptr ? mel_filters_size / num_frequency_bins : 0;
    const size_t spectrogram_bins = mel_filters != nullptr ? num_mel_bins : num_frequency_bins;

    std::vector<float> temp_spectrogram(num_frames * num_frequency_bins);

    size_t timestep = 0;
    for (size_t frame_idx = 0; frame_idx < num_frames; frame_idx++) {
        std::fill(buffer.begin(), buffer.end(), 0.0f);

        size_t available_length = std::min(frame_length, waveform_length - timestep);
        std::copy(waveform + timestep, waveform + timestep + available_length, buffer.data());

        // need to add if conditions

        for (size_t i = 0; i < frame_length; i++) {
            buffer[i] *= window[i];
        }
        
        cactus_rfft_f32_1d(buffer.data(), raw_complex_frequencies.data(), actual_fft_length, "backward");

        for (size_t i = 0; i < num_frequency_bins; i++) {
            float real = raw_complex_frequencies[i * 2];
            float imag = raw_complex_frequencies[i * 2 + 1];
            float magnitude = std::hypot(real, imag);
            temp_spectrogram[frame_idx * num_frequency_bins + i] = std::pow(magnitude, power);
        }

        timestep += hop_length;
    }

    if (mel_filters != nullptr) {
        for (size_t m = 0; m < num_mel_bins; m++) {
            for (size_t t = 0; t < num_frames; t++) {
                float sum = 0.0f;
                for (size_t f = 0; f < num_frequency_bins; f++) {
                    sum += mel_filters[m * num_frequency_bins + f] * temp_spectrogram[t * num_frequency_bins + f];
                }
                spectrogram[m * num_frames + t] = std::max(mel_floor, sum);
            }
        }
    } else {
        for (size_t t = 0; t < num_frames; t++) {
            for (size_t f = 0; f < num_frequency_bins; f++) {
                spectrogram[f * num_frames + t] = temp_spectrogram[t * num_frequency_bins + f];
            }
        }
    }

    if (power != 0.0f && log_mel != nullptr) {
        const size_t total_elements = spectrogram_bins * num_frames;

        if (std::strcmp(log_mel, "log") == 0) {
            for (size_t i = 0; i < total_elements; i++) {
                spectrogram[i] = std::log(spectrogram[i]);
            }
        } else if (std::strcmp(log_mel, "log10") == 0) {
            for (size_t i = 0; i < total_elements; i++) {
                spectrogram[i] = std::log10(spectrogram[i]);
            }
        } else if (std::strcmp(log_mel, "dB") == 0) {
            if (power == 1.0f) {
                to_db(spectrogram, total_elements, reference, min_value, db_range, 20.0f);
            } else if (power == 2.0f) {
                to_db(spectrogram, total_elements, reference, min_value, db_range, 10.0f);
            } else {
                throw std::invalid_argument(
                    "Cannot use log_mel option 'dB' with power " + std::to_string(power));
            }
        } else {
            throw std::invalid_argument("Unknown log_mel option: " + std::string(log_mel));
        }
    }
}

void cactus_rfft_f32_1d(const float* input, float* output, const size_t n, const char* norm = "backward") {
    const size_t out_len = n / 2 + 1;
    const float two_pi_over_n = 2.0f * std::numbers::pi / static_cast<float>(n);
    
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

static void to_db(
    float* spectrogram,
    size_t size,
    float reference,
    float min_value,
    const float* db_range,
    float multiplier)
{
    if (reference <= 0.0f) {
        throw std::invalid_argument("reference must be greater than zero");
    }
    if (min_value <= 0.0f) {
        throw std::invalid_argument("min_value must be greater than zero");
    }

    reference = std::max(min_value, reference);
    const float log_ref = std::log10(reference);

    for (size_t i = 0; i < size; i++) {
        float value = std::max(min_value, spectrogram[i]);
        spectrogram[i] = multiplier * (std::log10(value) - log_ref);
    }

    if (db_range != nullptr) {
        if (*db_range <= 0.0f) {
            throw std::invalid_argument("db_range must be greater than zero");
        }

        float max_db = -std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < size; i++) {
            max_db = std::max(max_db, spectrogram[i]);
        }

        float min_db = max_db - *db_range;
        for (size_t i = 0; i < size; i++) {
            spectrogram[i] = std::max(min_db, spectrogram[i]);
        }
    }
}