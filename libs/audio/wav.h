#ifndef WAV_LOADER_H
#define WAV_LOADER_H

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <algorithm>
#include <cmath>

// We assume ARM / a compiler that supports __fp16
// (Apple Clang, GCC on ARM, Clang on ARM, etc.)

enum class Precision {
    FP32,
    FP16
};

struct AudioFP32 {
    int sample_rate;
    std::vector<float> samples;
};

struct AudioFP16 {
    int sample_rate;
    std::vector<__fp16> samples;
};

// =========================================================
// 1. WAV loader → always loads to FP32 first
// =========================================================
inline AudioFP32 load_wav_fp32(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file)
        throw std::runtime_error("Could not open WAV file: " + path);

    // ---- RIFF ----
    char riff[4];
    file.read(riff, 4);
    if (std::string(riff, 4) != "RIFF") throw std::runtime_error("Not RIFF");

    uint32_t chunk_size;
    file.read(reinterpret_cast<char*>(&chunk_size), 4);

    char wave[4];
    file.read(wave, 4);
    if (std::string(wave, 4) != "WAVE") throw std::runtime_error("Not WAVE");

    // ---- fmt ----
    char fmt_id[4];
    uint32_t fmt_size;
    file.read(fmt_id, 4);
    file.read(reinterpret_cast<char*>(&fmt_size), 4);
    if (std::string(fmt_id, 4) != "fmt ")
        throw std::runtime_error("Missing fmt chunk");

    uint16_t audio_format, num_channels, bits_per_sample;
    uint32_t sample_rate, byte_rate;
    uint16_t block_align;

    file.read(reinterpret_cast<char*>(&audio_format), 2);
    file.read(reinterpret_cast<char*>(&num_channels), 2);
    file.read(reinterpret_cast<char*>(&sample_rate), 4);
    file.read(reinterpret_cast<char*>(&byte_rate), 4);
    file.read(reinterpret_cast<char*>(&block_align), 2);
    file.read(reinterpret_cast<char*>(&bits_per_sample), 2);

    if (audio_format != 1 || bits_per_sample != 16)
        throw std::runtime_error("Only 16-bit PCM WAV supported");

    if (fmt_size > 16)
        file.seekg(fmt_size - 16, std::ios::cur);

    // ---- find "data" chunk ----
    char data_id[4];
    uint32_t data_size;

    while (true) {
        file.read(data_id, 4);
        file.read(reinterpret_cast<char*>(&data_size), 4);
        if (!file) throw std::runtime_error("Malformed WAV: missing data chunk");

        if (std::string(data_id, 4) == "data")
            break;

        // skip other chunks
        file.seekg(data_size, std::ios::cur);
    }

    size_t num_samples = data_size / 2;  // 16-bit
    std::vector<float> tmp(num_samples);

    // read PCM samples
    for (size_t i = 0; i < num_samples; i++) {
        int16_t s;
        file.read(reinterpret_cast<char*>(&s), 2);
        tmp[i] = float(s) / 32768.0f;
    }

    // ---- stereo → mono ----
    std::vector<float> mono;

    if (num_channels == 1) {
        mono = std::move(tmp);
    } else if (num_channels == 2) {
        mono.reserve(num_samples / 2);
        for (size_t i = 0; i < num_samples; i += 2)
            mono.push_back(0.5f * (tmp[i] + tmp[i + 1]));
    } else {
        throw std::runtime_error("Unsupported channel count");
    }

    return AudioFP32{ (int)sample_rate, std::move(mono) };
}

// =========================================================
// 2. Convert FP32 → __fp16
// =========================================================
inline std::vector<__fp16> fp32_to_fp16(const std::vector<float>& v) {
    std::vector<__fp16> out(v.size());
    for (size_t i = 0; i < v.size(); i++)
        out[i] = (__fp16)v[i];
    return out;
}

// =========================================================
// 3. WAV loader → FP16 output
// =========================================================
inline AudioFP16 load_wav_fp16(const std::string& path) {
    AudioFP32 a = load_wav_fp32(path);
    return AudioFP16{
        a.sample_rate,
        fp32_to_fp16(a.samples)
    };
}

// =========================================================
// 4. Unified API
// =========================================================
inline AudioFP32 load_wav(const std::string& path) {
    return load_wav_fp32(path);
}

inline AudioFP16 load_wav16(const std::string& path) {
    return load_wav_fp16(path);
}

// =========================================================
// 5. Resampling (same algorithm, different output type)
// =========================================================

inline std::vector<float> resample_to_16k_fp32(
    const std::vector<float>& in, int sr_in)
{
    const int sr_out = 16000;
    if (sr_in == sr_out) return in;

    double ratio = double(sr_out) / double(sr_in);
    size_t out_len = size_t(in.size() * ratio);

    std::vector<float> out(out_len);

    for (size_t i = 0; i < out_len; i++) {
        double pos = i / ratio;
        size_t i0 = (size_t)pos;
        double frac = pos - i0;

        out[i] = (i0 + 1 < in.size())
            ? float((1.0 - frac) * in[i0] + frac * in[i0 + 1])
            : in.back();
    }
    return out;
}

inline std::vector<__fp16> resample_to_16k_fp16(
    const std::vector<float>& in, int sr_in)
{
    auto fp32 = resample_to_16k_fp32(in, sr_in);

    std::vector<__fp16> out(fp32.size());
    for (size_t i = 0; i < fp32.size(); i++)
        out[i] = (__fp16)fp32[i];
    return out;
}

#endif // WAV_LOADER_H
