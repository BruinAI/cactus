#include "model.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>

namespace cactus {
namespace engine {

namespace {

std::string read_text_file(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        return {};
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

size_t tensor_numel(const std::vector<size_t>& shape) {
    size_t count = 1;
    for (size_t dim : shape) {
        count *= dim;
    }
    return count;
}

} // namespace

float WhisperCloudHandoffModel::gelu_tanh(float x) {
    constexpr float kPi = 3.14159265358979323846f;
    const float k = std::sqrt(2.0f / kPi);
    const float x3 = x * x * x;
    return 0.5f * x * (1.0f + std::tanh(k * (x + 0.044715f * x3)));
}

bool WhisperCloudHandoffModel::parse_json_float(const std::string& json, const std::string& key, float& out_value) {
    const std::regex re("\"" + key + "\"\\s*:\\s*([-+0-9.eE]+)");
    std::smatch m;
    if (!std::regex_search(json, m, re) || m.size() < 2) {
        return false;
    }
    try {
        out_value = std::stof(m[1].str());
        return true;
    } catch (...) {
        return false;
    }
}

bool WhisperCloudHandoffModel::parse_json_string(const std::string& json, const std::string& key, std::string& out_value) {
    const std::regex re("\"" + key + "\"\\s*:\\s*\"([^\"]+)\"");
    std::smatch m;
    if (!std::regex_search(json, m, re) || m.size() < 2) {
        return false;
    }
    out_value = m[1].str();
    return true;
}

void WhisperCloudHandoffModel::set_error(std::string* error, const std::string& message) const {
    if (error != nullptr) {
        *error = message;
    }
}

bool WhisperCloudHandoffModel::load_tensor_f32(
    const std::string& path,
    std::vector<float>& out_data,
    std::vector<size_t>& out_shape,
    std::string* error) {

    try {
        GraphFile::MappedFile mapped(path);
        out_shape = mapped.shape();
        const size_t numel = tensor_numel(out_shape);
        out_data.resize(numel, 0.0f);

        if (mapped.precision() == Precision::FP32) {
            const float* src = static_cast<const float*>(mapped.data());
            std::copy(src, src + numel, out_data.begin());
            return true;
        }

        if (mapped.precision() == Precision::FP16) {
            const __fp16* src = static_cast<const __fp16*>(mapped.data());
            Quantization::fp16_to_fp32(src, out_data.data(), numel);
            return true;
        }

        set_error(
            error,
            "Unsupported tensor precision in " + path +
                ". Re-convert cloud_handoff weights with --precision FP16.");
        return false;
    } catch (const std::exception& e) {
        set_error(error, "Failed to load tensor " + path + ": " + e.what());
        return false;
    }
}

bool WhisperCloudHandoffModel::init(const std::string& model_folder, std::string* error) {
    initialized_ = false;
    fc1_weight_.clear();
    fc1_bias_.clear();
    fc2_weight_.clear();
    fc2_bias_.clear();
    feature_mean_.clear();
    feature_std_.clear();
    feature_names_.clear();
    input_dim_ = 0;
    hidden_dim_ = 0;
    output_dim_ = 0;
    threshold_ = 0.5f;
    high_freq_cutoff_hz_ = 3000.0f;
    activation_ = "relu";

    const std::filesystem::path base(model_folder);
    const std::filesystem::path fc1w_path = base / "cloud_handoff_fc1.weights";
    const std::filesystem::path fc1b_path = base / "cloud_handoff_fc1.bias";
    const std::filesystem::path fc2w_path = base / "cloud_handoff_fc2.weights";
    const std::filesystem::path fc2b_path = base / "cloud_handoff_fc2.bias";
    const std::filesystem::path fmean_path = base / "cloud_handoff_feature_mean.weights";
    const std::filesystem::path fstd_path = base / "cloud_handoff_feature_std.weights";
    const std::filesystem::path features_path = base / "classifier_features.json";
    const std::filesystem::path meta_path = base / "classifier_meta.json";

    if (!std::filesystem::exists(fc1w_path) || !std::filesystem::exists(fc1b_path) ||
        !std::filesystem::exists(fc2w_path) || !std::filesystem::exists(fc2b_path)) {
        set_error(error, "Missing required cloud_handoff weight files in: " + model_folder);
        return false;
    }

    std::vector<size_t> fc1w_shape;
    if (!load_tensor_f32(fc1w_path.string(), fc1_weight_, fc1w_shape, error)) {
        return false;
    }
    if (fc1w_shape.size() != 2) {
        set_error(error, "cloud_handoff_fc1.weights must be rank-2");
        return false;
    }
    hidden_dim_ = fc1w_shape[0];
    input_dim_ = fc1w_shape[1];

    std::vector<size_t> fc1b_shape;
    if (!load_tensor_f32(fc1b_path.string(), fc1_bias_, fc1b_shape, error)) {
        return false;
    }
    if (fc1b_shape.size() != 1 || fc1b_shape[0] != hidden_dim_) {
        set_error(error, "cloud_handoff_fc1.bias shape mismatch");
        return false;
    }

    std::vector<size_t> fc2w_shape;
    if (!load_tensor_f32(fc2w_path.string(), fc2_weight_, fc2w_shape, error)) {
        return false;
    }
    if (fc2w_shape.size() != 2 || fc2w_shape[1] != hidden_dim_) {
        set_error(error, "cloud_handoff_fc2.weights shape mismatch");
        return false;
    }
    output_dim_ = fc2w_shape[0];

    std::vector<size_t> fc2b_shape;
    if (!load_tensor_f32(fc2b_path.string(), fc2_bias_, fc2b_shape, error)) {
        return false;
    }
    if (fc2b_shape.size() != 1 || fc2b_shape[0] != output_dim_) {
        set_error(error, "cloud_handoff_fc2.bias shape mismatch");
        return false;
    }

    if (std::filesystem::exists(fmean_path) && std::filesystem::exists(fstd_path)) {
        std::vector<size_t> mean_shape;
        std::vector<size_t> std_shape;
        if (!load_tensor_f32(fmean_path.string(), feature_mean_, mean_shape, error)) {
            return false;
        }
        if (!load_tensor_f32(fstd_path.string(), feature_std_, std_shape, error)) {
            return false;
        }
        if (feature_mean_.size() != input_dim_ || feature_std_.size() != input_dim_) {
            feature_mean_.clear();
            feature_std_.clear();
        }
    }

    if (std::filesystem::exists(meta_path)) {
        const std::string meta_json = read_text_file(meta_path.string());
        float parsed = 0.0f;
        if (parse_json_float(meta_json, "threshold", parsed)) {
            threshold_ = parsed;
        }
        if (parse_json_float(meta_json, "hf_cutoff_hz", parsed) ||
            parse_json_float(meta_json, "high_freq_cutoff_hz", parsed) ||
            parse_json_float(meta_json, "hf_energy_cutoff_hz", parsed)) {
            high_freq_cutoff_hz_ = parsed;
        }
        std::string activation;
        if (parse_json_string(meta_json, "activation", activation)) {
            std::transform(activation.begin(), activation.end(), activation.begin(), [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });
            if (activation == "relu" || activation == "gelu") {
                activation_ = activation;
            }
        }
    }

    if (std::filesystem::exists(features_path)) {
        const std::string json = read_text_file(features_path.string());
        size_t key_pos = json.find("\"features\"");
        if (key_pos == std::string::npos) {
            key_pos = json.find("\"feature_names\"");
        }
        size_t arr_begin = key_pos == std::string::npos ? std::string::npos : json.find('[', key_pos);
        if (arr_begin != std::string::npos) {
            int depth = 1;
            size_t arr_end = arr_begin + 1;
            while (arr_end < json.size() && depth > 0) {
                if (json[arr_end] == '[') {
                    depth++;
                } else if (json[arr_end] == ']') {
                    depth--;
                }
                arr_end++;
            }
            if (depth == 0 && arr_end > arr_begin + 1) {
                const std::string arr_payload = json.substr(arr_begin + 1, arr_end - arr_begin - 2);
                const std::regex str_re("\"([^\"]+)\"");
                auto begin = std::sregex_iterator(arr_payload.begin(), arr_payload.end(), str_re);
                auto end = std::sregex_iterator();
                for (auto it = begin; it != end; ++it) {
                    feature_names_.push_back((*it)[1].str());
                }
            }
        }
    }

    if (feature_names_.size() != input_dim_) {
        feature_names_.clear();
        feature_names_.reserve(input_dim_);
        for (size_t i = 0; i < input_dim_; i++) {
            feature_names_.push_back("whisper_encoder_mean_" + std::to_string(i));
        }
    }

    initialized_ = true;
    return true;
}

float WhisperCloudHandoffModel::predict_probability(const std::vector<float>& features) const {
    if (!initialized_ || features.size() != input_dim_) {
        return 0.0f;
    }

    std::vector<float> x(features.begin(), features.end());
    if (feature_mean_.size() == input_dim_ && feature_std_.size() == input_dim_) {
        for (size_t i = 0; i < input_dim_; i++) {
            const float denom = std::fabs(feature_std_[i]) > 1e-8f ? feature_std_[i] : 1.0f;
            x[i] = (x[i] - feature_mean_[i]) / denom;
        }
    }

    std::vector<float> h(hidden_dim_, 0.0f);
    for (size_t row = 0; row < hidden_dim_; row++) {
        float acc = fc1_bias_[row];
        const size_t offset = row * input_dim_;
        for (size_t col = 0; col < input_dim_; col++) {
            acc += fc1_weight_[offset + col] * x[col];
        }
        if (activation_ == "gelu") {
            h[row] = gelu_tanh(acc);
        } else {
            h[row] = std::max(0.0f, acc);
        }
    }

    float logit = fc2_bias_[0];
    for (size_t col = 0; col < hidden_dim_; col++) {
        logit += fc2_weight_[col] * h[col];
    }

    const float clamped = std::max(-40.0f, std::min(40.0f, logit));
    return 1.0f / (1.0f + std::exp(-clamped));
}

bool WhisperCloudHandoffModel::predict_handoff(const std::vector<float>& features, float* out_probability) const {
    const float prob = predict_probability(features);
    if (out_probability != nullptr) {
        *out_probability = prob;
    }
    return prob >= threshold_;
}

} // namespace engine
} // namespace cactus
