#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstring>
#include <filesystem>

namespace fs = std::filesystem;

enum class Precision { F32 };

struct PrecisionTraits {
    static size_t size_of(Precision p) {
        switch (p) { case Precision::F32: return 4; }
        throw std::runtime_error("Unsupported precision");
    }
};

struct TensorBuffer {
    void* data = nullptr;
    std::vector<size_t> shape;   // row-major
    Precision precision = Precision::F32;
    float quantization_scale = 1.f;

    void* get_data() const { return data; }
    void  set_external(void* p) { data = p; }
};

struct SliceParams { int axis = 0; size_t slice_start = 0; };
struct Node {
    TensorBuffer output_buffer;
    SliceParams params;
    std::vector<int> input_ids;
};

// --------- IO helpers ---------
static bool load_bin(const std::string& path, std::vector<float>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.seekg(0, std::ios::end);
    auto nbytes = f.tellg();
    if (nbytes % sizeof(float)) return false;
    size_t n = static_cast<size_t>(nbytes / sizeof(float));
    out.resize(n);
    f.seekg(0, std::ios::beg);
    f.read(reinterpret_cast<char*>(out.data()), nbytes);
    return f.good();
}

static bool load_meta(const std::string& path,
                      Precision& prec,
                      std::vector<size_t>& shape,
                      size_t& axis,
                      size_t& slice_start,
                      std::string& mode) {
    std::ifstream f(path);
    if (!f) return false;
    std::string key;
    prec = Precision::F32;
    int rank = -1;
    mode = "identity";
    while (f >> key) {
        if (key == "DTYPE") {
            std::string v; f >> v;
            if (v != "f32") { std::cerr << "Only f32 supported\n"; return false; }
        } else if (key == "RANK") {
            f >> rank;
        } else if (key == "SHAPE") {
            shape.resize(rank);
            for (int i=0;i<rank;++i) f >> shape[i];
        } else if (key == "AXIS") {
            f >> axis;
        } else if (key == "SLICE_START") {
            f >> slice_start;
        } else if (key == "MODE") {
            f >> mode; // not required for this verifier
        }
    }
    return rank > 0 && !shape.empty();
}

static size_t numel_of(const std::vector<size_t>& s) {
    size_t n = 1; for (auto d: s) n *= d; return n;
}

// Row-major input strides: s[i] = product(shape[i+1:])
static std::vector<size_t> row_major_strides(const std::vector<size_t>& shape) {
    std::vector<size_t> st(shape.size(), 1);
    for (int i = int(shape.size()) - 2; i >= 0; --i) {
        st[size_t(i)] = st[size_t(i+1)] * shape[size_t(i+1)];
    }
    return st;
}

// Iterate multi-index (last dim fastest).
template <class Fn>
static void for_each_index(const std::vector<size_t>& shape, Fn&& fn) {
    const size_t rank = shape.size();
    if (rank == 0) { fn(std::vector<size_t>{}); return; }
    std::vector<size_t> idx(rank, 0);
    while (true) {
        fn(idx);
        for (int i = int(rank) - 1; i >= 0; --i) {
            if (++idx[size_t(i)] < shape[size_t(i)]) break;
            idx[size_t(i)] = 0;
            if (i == 0) return;
        }
    }
}

// --------- Your slice logic (unchanged semantics; corrected cast) ---------
static void run_slice_like_repo(const TensorBuffer& input_tb,
                                TensorBuffer& output_tb,
                                int axis_param,
                                size_t slice_start_param) {
    auto tensor_buffer = input_tb;

    const size_t axis_index = static_cast<size_t>(axis_param);
    const size_t slice_start = slice_start_param;

    size_t stride = 1;
    for (size_t i = axis_index + 1; i < tensor_buffer.shape.size(); ++i) {
        stride *= tensor_buffer.shape[i];
    }

    const size_t element_size = PrecisionTraits::size_of(tensor_buffer.precision);
    const size_t byte_stride  = stride * element_size;
    const size_t byte_offset  = slice_start * byte_stride;

    auto* base_ptr = reinterpret_cast<char*>(tensor_buffer.data); // <-- reinterpret_cast
    if (!base_ptr) { throw std::runtime_error("Slice input buffer is not available"); }

    output_tb.set_external(reinterpret_cast<void*>(base_ptr + byte_offset));
    output_tb.precision = tensor_buffer.precision;
    output_tb.quantization_scale = tensor_buffer.quantization_scale;

    // For this standalone test we set output shape here
    output_tb.shape = tensor_buffer.shape;
    if (slice_start > output_tb.shape[axis_index])
        throw std::runtime_error("slice_start beyond dimension");
    output_tb.shape[axis_index] -= slice_start;
}

// --------- Core check: address parity (no transforms) ---------
// For each output multi-index m, compute:
//   - src_flat_orig: flat index in original input
//   - rel_off_rebased: flat offset from rebased pointer
// Then compare:  orig[src_flat_orig]  ==  rebased[rel_off_rebased]
static int verify_by_address_parity(const fs::path& dir,
                                    const std::vector<size_t>& in_shape,
                                    size_t axis, size_t slice_start,
                                    const float* orig_base,
                                    const float* rebased_base,
                                    bool verbose) {
    const size_t rank = in_shape.size();
    const auto in_strides = row_major_strides(in_shape);

    std::vector<size_t> out_shape = in_shape;
    out_shape[axis] -= slice_start;

    // base_flat = slice_start * product(in_shape[axis+1:])
    size_t base_flat = slice_start;
    for (size_t i = axis + 1; i < rank; ++i) base_flat *= in_shape[i];

    double max_abs = 0.0; size_t maxi = 0; size_t flat_counter = 0;

    for_each_index(out_shape, [&](const std::vector<size_t>& m){
        // original multi-index = m with axis shifted by +slice_start
        // src_flat_orig = sum_i src_i * in_strides[i]
        size_t src_flat_orig = 0;
        for (size_t i = 0; i < rank; ++i) {
            const size_t src_i = (i == axis) ? (m[i] + slice_start) : m[i];
            src_flat_orig += src_i * in_strides[i];
        }

        // rebased relative offset = sum_i m[i] * in_strides[i]
        size_t rel_off_rebased = 0;
        for (size_t i = 0; i < rank; ++i) {
            rel_off_rebased += m[i] * in_strides[i];
        }

        float got    = rebased_base[rel_off_rebased];
        float expect = orig_base[src_flat_orig];

        double d = std::abs(double(got) - double(expect));
        if (d > max_abs) { max_abs = d; maxi = flat_counter; }
        ++flat_counter;
    });

    const bool pass = (max_abs <= 1e-6);
    if (verbose) {
        std::cout << dir.filename().string()
                  << "  axis=" << axis << " start=" << slice_start
                  << "  -> " << (pass ? "PASS" : "FAIL")
                  << "  max|diff|=" << max_abs << "\n";
    }
    return pass ? 0 : 3;
}

static int run_case_dir(const fs::path& dir, bool verbose) {
    Precision prec;
    std::vector<size_t> shape;
    size_t axis=0, slice_start=0;
    std::string mode;

    if (!load_meta((dir/"meta.txt").string(), prec, shape, axis, slice_start, mode)) {
        std::cerr << "[FAIL] " << dir << " : could not read meta.txt\n";
        return 1;
    }

    // Load the input buffer (values can be arange or random; it doesn't matter)
    std::vector<float> x_raw;
    if (!load_bin((dir/"input.bin").string(), x_raw)) {
        std::cerr << "[FAIL] " << dir << " : input.bin\n"; return 1;
    }
    if (x_raw.size() != numel_of(shape)) {
        std::cerr << "[FAIL] " << dir << " : input size mismatch\n"; return 1;
    }

    // Build minimal nodes
    Node in, sl;
    in.output_buffer.data = x_raw.data();
    in.output_buffer.shape = shape;
    in.output_buffer.precision = prec;

    sl.params.axis = static_cast<int>(axis);
    sl.params.slice_start = slice_start;

    // Rebind pointer using your slicing logic
    run_slice_like_repo(in.output_buffer, sl.output_buffer, sl.params.axis, sl.params.slice_start);

    // Compare by address parity (no transforms)
    const float* orig_base   = static_cast<const float*>(in.output_buffer.get_data());
    const float* rebased_base= static_cast<const float*>(sl.output_buffer.get_data());
    return verify_by_address_parity(dir, shape, axis, slice_start, orig_base, rebased_base, verbose);
}

int main(int argc, char** argv) {
    fs::path path = (argc > 1) ? fs::path(argv[1]) : fs::path("slice_fixtures_identity");
    if (!fs::exists(path)) {
        std::cerr << "No test cases found: directory doesn't exist -> " << path << "\n";
        return 1;
    }

    std::vector<fs::path> case_dirs;
    if (fs::exists(path / "meta.txt")) {
        case_dirs.push_back(path);
    } else {
        for (auto& p : fs::directory_iterator(path)) {
            if (p.is_directory() && fs::exists(p.path() / "meta.txt")) {
                case_dirs.push_back(p.path());
            }
        }
    }
    if (case_dirs.empty()) {
        std::cerr << "No test cases under: " << path << "\n";
        return 1;
    }

    int failures = 0, total = 0;
    for (auto& d : case_dirs) {
        int rc = run_case_dir(d, /*verbose=*/true);
        ++total;
        if (rc != 0) ++failures;
    }
    std::cout << "\nSummary: " << (total - failures) << "/" << total << " cases passed.\n";
    return failures == 0 ? 0 : 2;
}
