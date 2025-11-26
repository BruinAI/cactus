// test_depthwise_causal.cpp
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <limits>
#include <cstddef>
#include <cassert>

void cactus_conv1d_depthwise_causal_f32(
    const float* input,
    const float* weight,  // [C, 1, K] flattened as [C, K]
    float* output,
    size_t N,
    size_t L,
    size_t C,
    size_t K,
    size_t dilation
) {
    const size_t in_bs  = L * C;
    const size_t out_bs = L * C;
    const size_t w_c_stride = K; // Cin==1

    for (size_t n = 0; n < N; ++n) {
        const float* Xb = input  + n * in_bs;
        float*       Yb = output + n * out_bs;
        for (size_t c = 0; c < C; ++c) {
            const float* Wc = weight + c * w_c_stride;   // taps in forward order
            for (size_t t = 0; t < L; ++t) {
                float acc = 0.f;
                for (size_t k = 0; k < K; ++k) {
                    const ptrdiff_t xt = (ptrdiff_t)t - (ptrdiff_t)(k * dilation);
                    if (xt < 0) break;                   // causal guard
                    acc += Wc[K - 1 - k] * Xb[(size_t)xt * C + c];
                }
                Yb[t * C + c] = acc;
            }
        }
    }
}

static bool load_bin(const std::string& path, std::vector<float>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.seekg(0, std::ios::end);
    const auto nbytes = f.tellg();
    if (nbytes % sizeof(float) != 0) return false;
    const size_t n = static_cast<size_t>(nbytes / sizeof(float));
    out.resize(n);
    f.seekg(0, std::ios::beg);
    f.read(reinterpret_cast<char*>(out.data()), nbytes);
    return f.good();
}

static bool load_shapes(const std::string& path, size_t& N, size_t& L, size_t& C, size_t& K) {
    std::ifstream f(path);
    if (!f) return false;
    std::string key;
    size_t val;
    int seen = 0;
    while (f >> key >> val) {
        if      (key == "N") { N = val; ++seen; }
        else if (key == "L") { L = val; ++seen; }
        else if (key == "C") { C = val; ++seen; }
        else if (key == "K") { K = val; ++seen; }
    }
    return seen == 4;
}

int main(int argc, char** argv) {
    const std::string dir = (argc > 1) ? argv[1] : "fixtures";

    size_t N=0,L=0,C=0,K=0;
    if (!load_shapes(dir + "/shapes.txt", N, L, C, K)) {
        std::cerr << "Failed to read shapes.txt\n";
        return 1;
    }

    std::vector<float> x, w, y_ref;
    if (!load_bin(dir + "/input_nlc.bin", x))   { std::cerr << "Failed to read input\n"; return 1; }
    if (!load_bin(dir + "/weight_ck.bin", w))   { std::cerr << "Failed to read weight\n"; return 1; }
    if (!load_bin(dir + "/out_ref_nlc.bin", y_ref)) { std::cerr << "Failed to read out_ref\n"; return 1; }

    const size_t numel_x = N * L * C;
    const size_t numel_w = C * K;
    const size_t numel_y = N * L * C;
    if (x.size() != numel_x || w.size() != numel_w || y_ref.size() != numel_y) {
        std::cerr << "Size mismatch in loaded buffers\n";
        return 1;
    }

    std::vector<float> y(numel_y, 0.f);
    size_t dilation = 1; // HF reference uses dilation=1
    cactus_conv1d_depthwise_causal_f32(x.data(), w.data(), y.data(), N, L, C, K, dilation);

    // Compare
    double max_abs_diff = 0.0;
    size_t bad_idx = std::numeric_limits<size_t>::max();
    for (size_t i = 0; i < numel_y; ++i) {
        double d = std::abs(double(y[i]) - double(y_ref[i]));
        if (d > max_abs_diff) {
            max_abs_diff = d;
            bad_idx = i;
        }
    }

    const double atol = 1e-6; // fp32 exact math should match; small tol anyway
    if (max_abs_diff <= atol) {
        std::cout << "PASS. max |diff| = " << max_abs_diff << "\n";
        return 0;
    } else {
        size_t t = (bad_idx / C) % L;
        size_t c = bad_idx % C;
        size_t n = bad_idx / (L * C);
        std::cerr << "FAIL. max |diff| = " << max_abs_diff
                  << " at n=" << n << " t=" << t << " c=" << c << "\n";
        std::cerr << "got=" << y[bad_idx] << " ref=" << y_ref[bad_idx] << "\n";
        return 2;
    }
}
