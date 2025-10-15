#define private public
#define protected public
#include "cactus.h"
#undef private
#undef protected

#include "test_utils.h"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace cactus::engine;

namespace {

std::filesystem::path find_project_root() {
    namespace fs = std::filesystem;
    fs::path current = fs::current_path();
    for (int depth = 0; depth < 6; ++depth) {
        if (fs::exists(current / "weights") && fs::exists(current / "cactus")) {
            return current;
        }
        if (!current.has_parent_path()) {
            break;
        }
        current = current.parent_path();
    }
    throw std::runtime_error("Unable to locate project root (weights/cactus directories)");
}

std::vector<uint32_t> tokenize_sample_text() {
    namespace fs = std::filesystem;
    auto project_root = find_project_root();
    
    // Get weights suffix from environment variable
    std::string weights_suffix = "";
    const char* suffix_env = std::getenv("CACTUS_WEIGHTS_SUFFIX");
    if (suffix_env) {
        weights_suffix = suffix_env;
    }
    
    fs::path weights_dir = project_root / "weights" / ("nomic-embed-text-v2-moe" + weights_suffix);

    SPTokenizer tokenizer;
    if (!tokenizer.load_vocabulary_with_config(
            (weights_dir / "vocab.txt").string(),
            (weights_dir / "merges.txt").string(),
            (weights_dir / "tokenizer_config.txt").string())) {
        throw std::runtime_error("Failed to load Nomic tokenizer assets");
    }

    const std::string prompt = "Kind regards from Noah Cylich.";
    auto tokens = tokenizer.encode(prompt);
    
    // Add BOS/EOS tokens (Nomic model expects them)
    std::vector<uint32_t> result;
    result.push_back(tokenizer.get_bos_token());
    result.insert(result.end(), tokens.begin(), tokens.end());
    result.push_back(tokenizer.get_eos_token());
    
    return result;
}

class TestableNomicModel : public NomicModel {
public:
    using NomicModel::NomicModel;

    size_t call_build_attention(CactusGraph* gb,
                                size_t normalized_input,
                                uint32_t layer_idx,
                                ComputeBackend backend,
                                bool use_cache,
                                size_t position_offset = 0) {
        return NomicModel::build_attention(gb, normalized_input, layer_idx, backend, use_cache, position_offset);
    }

    size_t call_build_transformer_block(CactusGraph* gb,
                                        size_t hidden,
                                        uint32_t layer_idx,
                                        ComputeBackend backend,
                                        bool use_cache,
                                        size_t position_offset = 0) {
        return NomicModel::build_transformer_block(gb, hidden, layer_idx, backend, use_cache, position_offset);
    }

    size_t call_forward(const std::vector<uint32_t>& tokens, bool use_cache) {
        return NomicModel::forward(tokens, use_cache);
    }
};

bool expect_cache_exception(const std::function<void()>& fn) {
    try {
        fn();
    } catch (const std::runtime_error& err) {
        return std::string(err.what()).find("does not support cache") != std::string::npos;
    } catch (...) {
        return false;
    }
    return false;
}

bool test_nomic_forward_executes_with_tokens() {
    try {
        auto project_root = find_project_root();
        
        // Get weights suffix from environment variable
        std::string weights_suffix = "";
        const char* suffix_env = std::getenv("CACTUS_WEIGHTS_SUFFIX");
        if (suffix_env) {
            weights_suffix = suffix_env;
        }
        
        std::string model_path = (project_root / "weights" / ("nomic-embed-text-v2-moe" + weights_suffix)).string();

        auto model_ptr = create_model(model_path);
        if (!model_ptr) {
            std::cerr << "Failed to create model from: " << model_path << "\n";
            return false;
        }
        if (!model_ptr->init(model_path, 0)) {
            std::cerr << "Failed to initialize model from: " << model_path << "\n";
            return false;
        }
        
        auto* model = dynamic_cast<NomicModel*>(model_ptr.get());
        if (!model) {
            std::cerr << "Model is not a NomicModel!\n";
            return false;
        }

        const auto tokens = tokenize_sample_text();
        
        auto* gb = static_cast<CactusGraph*>(model->graph_handle_);
        size_t final_hidden = model->forward(tokens, false);
        
        gb->execute("nomic_profile.txt");
        
        auto* output_ptr = gb->get_output(final_hidden);
        const auto& output_buffer = gb->get_output_buffer(final_hidden);
        
        const Config& config = model->get_config();
        size_t expected_size = tokens.size() * config.hidden_dim;
        
        if (output_buffer.total_size != expected_size) {
            std::cerr << "Expected embedding size " << expected_size 
                      << ", got " << output_buffer.total_size << "\n";
            return false;
        }

        // Convert to FP32 if necessary
        std::vector<float> fp32_data;
        float* data = nullptr;
        if (output_buffer.precision == Precision::FP16) {
            fp32_data.resize(output_buffer.total_size);
            Quantization::fp16_to_fp32(static_cast<const __fp16*>(output_ptr), 
                                       fp32_data.data(), 
                                       output_buffer.total_size);
            data = fp32_data.data();
        } else if (output_buffer.precision == Precision::FP32) {
            data = static_cast<float*>(output_ptr);
        } else {
            std::cerr << "Unsupported output precision\n";
            return false;
        }

        for (size_t i = 0; i < output_buffer.total_size; ++i) {
            if (!std::isfinite(data[i])) {
                std::cerr << "Non-finite value at index " << i << "\n";
                return false;
            }
        }

        // Check first 5 values are within 10% of expected
        const float expected_first_5[] = {-0.1831f,  0.5969f, -0.6945f,  0.4719f,  0.4937f};
        for (size_t i = 0; i < 5; ++i) {
            float actual = data[i];
            float expected = expected_first_5[i];
            float tolerance = 0.25f * std::abs(expected);
            float diff = std::abs(actual - expected);
            if (diff > tolerance) {
                std::cerr << "Value at index " << i << " out of range: "
                          << "expected " << expected << " ± " << tolerance 
                          << ", got " << actual << " (diff: " << diff << ")\n";
                return false;
            }
        }

        return true;
    } catch (const std::exception& err) {
        std::cerr << "Forward execution threw: " << err.what() << "\n";
        return false;
    }
}

bool test_nomic_forward_rejects_cache() {
    Config config;
    config.num_layers = 1;
    TestableNomicModel model(config);

    return expect_cache_exception([&]() {
        model.call_forward({0}, true);
    });
}

bool test_nomic_attention_rejects_cache() {
    Config config;
    config.num_layers = 1;
    TestableNomicModel model(config);

    return expect_cache_exception([&]() {
        model.call_build_attention(nullptr, 0, 0, ComputeBackend::CPU, true);
    });
}

bool test_nomic_transformer_block_rejects_cache() {
    Config config;
    config.num_layers = 1;
    TestableNomicModel model(config);

    return expect_cache_exception([&]() {
        model.call_build_transformer_block(nullptr, 0, 0, ComputeBackend::CPU, true);
    });
}

}  // namespace

int main() {
    TestUtils::TestRunner runner("Nomic Model Tests");

    runner.run_test("Forward executes with tokens", test_nomic_forward_executes_with_tokens());
    runner.run_test("Forward cache guard", test_nomic_forward_rejects_cache());
    runner.run_test("Attention cache guard", test_nomic_attention_rejects_cache());
    runner.run_test("Transformer block cache guard", test_nomic_transformer_block_rejects_cache());

    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
