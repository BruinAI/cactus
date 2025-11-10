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

void cactus_spectrogram_f32(
    const float* waveform,
    size_t waveform_length,
    const float* window,
    size_t window_length,
    size_t frame_length,
    size_t hop_length,
    const size_t* fft_length,
    float* spectrogram,
    float power,
    bool center,
    const char* pad_mode,
    bool onesided,
    float dither,
    const float* preemphasis,
    const float* mel_filters,
    size_t mel_filters_size,
    float mel_floor,
    const char* log_mel,
    float reference,
    float min_value,
    const float* db_range,
    bool remove_dc_offset)
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

    std::vector<float> padded_waveform;
    const float* input_waveform = waveform;
    size_t input_length = waveform_length;

    if (center) {
        size_t pad_length = frame_length / 2;
        size_t padded_length = waveform_length + 2 * pad_length;
        padded_waveform.resize(padded_length);

        if (std::strcmp(pad_mode, "reflect") == 0) {
            for (size_t i = 0; i < pad_length; i++) {
                padded_waveform[i] = waveform[pad_length - i];
            }

            std::copy(waveform, waveform + waveform_length, padded_waveform.data() + pad_length);

            for (size_t i = 0; i < pad_length; i++) {
                padded_waveform[pad_length + waveform_length + i] = waveform[waveform_length - 2 - i];
            }
        } else {
            throw std::invalid_argument("Unsupported pad_mode: " + std::string(pad_mode));
        }

        input_waveform = padded_waveform.data();
        input_length = padded_length;
    }

    const size_t num_frames = 1 + (input_length - frame_length) / hop_length;
    const size_t num_frequency_bins = (actual_fft_length / 2) + 1;

    std::vector<float> buffer(actual_fft_length);
    std::vector<float> raw_complex_frequencies(num_frequency_bins * 2);

    const size_t num_mel_bins = mel_filters != nullptr ? mel_filters_size / num_frequency_bins : 0;
    const size_t spectrogram_bins = mel_filters != nullptr ? num_mel_bins : num_frequency_bins;

    std::vector<float> temp_spectrogram(num_frames * num_frequency_bins);

    size_t timestep = 0;
    for (size_t frame_idx = 0; frame_idx < num_frames; frame_idx++) {
        std::fill(buffer.begin(), buffer.end(), 0.0f);

        size_t available_length = std::min(frame_length, input_length - timestep);
        std::copy(input_waveform + timestep, input_waveform + timestep + available_length, buffer.data());

        if (dither != 0.0f) {
            for (size_t i = 0; i < frame_length; i++) {
                float u1 = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
                float u2 = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
                float randn = std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * std::numbers::pi * u2);
                buffer[i] += dither * randn;
            }
        }

        if (remove_dc_offset) {
            float mean = 0.0f;
            for (size_t i = 0; i < frame_length; i++) {
                mean += buffer[i];
            }
            mean /= static_cast<float>(frame_length);

            for (size_t i = 0; i < frame_length; i++) {
                buffer[i] -= mean;
            }
        }

        if (preemphasis != nullptr) {
            float preemph_coef = *preemphasis;
            for (size_t i = frame_length - 1; i > 0; i--) {
                buffer[i] -= preemph_coef * buffer[i - 1];
            }
            buffer[0] *= (1.0f - preemph_coef);
        }

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

void cactus_rfft_f32_1d(const float* input, float* output, const size_t n, const char* norm) {
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