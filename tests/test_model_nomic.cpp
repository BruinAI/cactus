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

std::string op_type_to_string(OpType type) {
    switch (type) {
        case OpType::INPUT: return "INPUT";
        case OpType::PRECISION_CAST: return "PRECISION_CAST";
        case OpType::ADD: return "ADD";
        case OpType::ADD_CLIPPED: return "ADD_CLIPPED";
        case OpType::SUBTRACT: return "SUBTRACT";
        case OpType::MULTIPLY: return "MULTIPLY";
        case OpType::DIVIDE: return "DIVIDE";
        case OpType::MATMUL: return "MATMUL";
        case OpType::TRANSPOSE: return "TRANSPOSE";
        case OpType::RESHAPE: return "RESHAPE";
        case OpType::GATHER: return "GATHER";
        case OpType::EMBEDDING: return "EMBEDDING";
        case OpType::SUM: return "SUM";
        case OpType::MEAN: return "MEAN";
        case OpType::VARIANCE: return "VARIANCE";
        case OpType::MIN: return "MIN";
        case OpType::MAX: return "MAX";
        case OpType::RMS_NORM: return "RMS_NORM";
        case OpType::ROPE: return "ROPE";
        case OpType::SOFTMAX: return "SOFTMAX";
        case OpType::ATTENTION: return "ATTENTION";
        case OpType::SCALAR_ADD: return "SCALAR_ADD";
        case OpType::SCALAR_SUBTRACT: return "SCALAR_SUBTRACT";
        case OpType::SCALAR_MULTIPLY: return "SCALAR_MULTIPLY";
        case OpType::SCALAR_DIVIDE: return "SCALAR_DIVIDE";
        case OpType::SCALAR_EXP: return "SCALAR_EXP";
        case OpType::SCALAR_SQRT: return "SCALAR_SQRT";
        case OpType::SCALAR_COS: return "SCALAR_COS";
        case OpType::SCALAR_SIN: return "SCALAR_SIN";
        case OpType::SILU: return "SILU";
        case OpType::GELU: return "GELU";
        case OpType::SAMPLE: return "SAMPLE";
        case OpType::CONCAT: return "CONCAT";
        case OpType::SCATTER_TOPK: return "SCATTER_TOPK";
        case OpType::TOPK: return "TOPK";
        case OpType::LAYERNORM: return "LAYERNORM";
        case OpType::INDEX: return "INDEX";
    }
    return "UNKNOWN";
}

std::string precision_to_string(Precision prec) {
    switch (prec) {
        case Precision::INT8: return "INT8";
        case Precision::FP16: return "FP16";
        case Precision::FP32: return "FP32";
    }
    return "UNKNOWN";
}

void dump_graph_activations(CactusGraph* gb) {
    std::cout << "\n--- Activation Dump (first 5 values per node) ---\n";
    for (const auto& node_ptr : gb->nodes_) {
        const auto& node = *node_ptr;
        const auto& buffer = node.output_buffer;
        size_t total_elements = 1;
        if (buffer.shape.empty()) {
            total_elements = 0;
        } else {
            for (size_t dim : buffer.shape) {
                total_elements *= dim;
            }
        }
        std::cout << "[Node " << node.id << "] "
                  << op_type_to_string(node.op_type)
                  << " | precision=" << precision_to_string(buffer.precision)
                  << " | shape=[";
        for (size_t i = 0; i < buffer.shape.size(); ++i) {
            if (i > 0) std::cout << ",";
            std::cout << buffer.shape[i];
        }
        std::cout << "] | values:";

        if (total_elements == 0) {
            std::cout << " <empty>\n";
            continue;
        }

        void* data_ptr = gb->get_output(node.id);
        size_t sample_count = std::min<size_t>(5, total_elements);

        if (buffer.precision == Precision::FP32) {
            const float* data = static_cast<const float*>(data_ptr);
            for (size_t i = 0; i < sample_count; ++i) {
                std::cout << (i == 0 ? " " : ", ") << data[i];
            }
        } else if (buffer.precision == Precision::FP16) {
            const __fp16* data = static_cast<const __fp16*>(data_ptr);
            for (size_t i = 0; i < sample_count; ++i) {
                std::cout << (i == 0 ? " " : ", ") << static_cast<float>(data[i]);
            }
        } else {
            const int8_t* data = static_cast<const int8_t*>(data_ptr);
            float scale = buffer.quantization_scale;
            for (size_t i = 0; i < sample_count; ++i) {
                // float dequantized = static_cast<float>(data[i]) * scale;
                // std::cout << (i == 0 ? " " : ", ") << dequantized;
                std::cout << (i == 0 ? " " : ", ") << static_cast<int>(data[i]);
            }
        }
        if (total_elements > sample_count) {
            std::cout << ", ...";
        }
        std::cout << "\n";
    }
    std::cout << "--- End Activation Dump ---\n\n";
}

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

    const std::string prompt = "Cactus activation inspection test.";
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
        
        gb->execute("profile.txt");
        
        if (std::getenv("CACTUS_DUMP_ACTIVATIONS")) {
            dump_graph_activations(gb);
        }
        
        auto* output_ptr = gb->get_output(final_hidden);
        const auto& output_buffer = gb->get_output_buffer(final_hidden);
        
        const Config& config = model->get_config();
        size_t expected_size = tokens.size() * config.hidden_dim;
        
        if (output_buffer.total_size != expected_size) {
            std::cerr << "Expected embedding size " << expected_size 
                      << ", got " << output_buffer.total_size << "\n";
            return false;
        }

        float* data = static_cast<float*>(output_ptr);
        for (size_t i = 0; i < output_buffer.total_size; ++i) {
            if (!std::isfinite(data[i])) {
                std::cerr << "Non-finite value at index " << i << "\n";
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
