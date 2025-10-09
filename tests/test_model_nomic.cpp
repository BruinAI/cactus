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
#include <memory>
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
        case OpType::TOPK: return "TOPK";
        case OpType::LAYERNORM: return "LAYERNORM";
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
    fs::path weights_dir = project_root / "weights" / "nomic-embed-text-v2-moe";

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
    result.push_back(tokenizer.get_bos_token());  // BOS
    result.insert(result.end(), tokens.begin(), tokens.end());
    result.push_back(tokenizer.get_eos_token());  // EOS
    
    return result;
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
            for (size_t i = 0; i < sample_count; ++i) {
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

class TestableNomicModel : public NomicModel {
public:
    using NomicModel::NomicModel;

    void setup_dummy_graph(uint32_t vocab_size, uint32_t hidden_dim, uint32_t ffn_dim) {
        if (graph_handle_) {
            delete static_cast<CactusGraph*>(graph_handle_);
            graph_handle_ = nullptr;
        }

        config_.vocab_size = vocab_size;
        config_.hidden_dim = hidden_dim;
        config_.attention_heads = 1;
        config_.attention_kv_heads = 1;
        config_.attention_head_dim = hidden_dim;
        config_.ffn_intermediate_dim = ffn_dim;
        config_.precision = Config::Precision::FP32;
        config_.default_backend = Config::Backend::CPU;
        config_.rope_theta = 0.0f;
        if (config_.moe_every_n_layers == 0) {
            config_.moe_every_n_layers = 2;
        }

        attention_scale_ = 1.0f / std::sqrt(static_cast<float>(config_.attention_head_dim));

        auto* gb = new CactusGraph();
        graph_handle_ = gb;

        owned_buffers_.clear();
        auto set_node = [&](size_t node_id, const std::vector<float>& values, const char*) {
            auto buffer = std::make_shared<std::vector<float>>(values);
            gb->set_external_input(node_id, buffer->data(), Precision::FP32);
            owned_buffers_.push_back(std::move(buffer));
        };

        embedding_node_id_ = gb->input({vocab_size, hidden_dim}, Precision::FP32);
        std::vector<float> embedding(vocab_size * hidden_dim);
        for (uint32_t row = 0; row < vocab_size; ++row) {
            for (uint32_t col = 0; col < hidden_dim; ++col) {
                embedding[row * hidden_dim + col] = static_cast<float>((row + col) % 7 + 1);
            }
        }
        set_node(embedding_node_id_, embedding, "embedding");

        weight_nodes_.embedding_layernorm_weight = gb->input({hidden_dim}, Precision::FP32);
        set_node(weight_nodes_.embedding_layernorm_weight, std::vector<float>(hidden_dim, 1.0f), "embed_ln_weight");

        weight_nodes_.embedding_layernorm_bias = gb->input({hidden_dim}, Precision::FP32);
        set_node(weight_nodes_.embedding_layernorm_bias, std::vector<float>(hidden_dim, 0.0f), "embed_ln_bias");

        std::vector<float> zeros_hidden(hidden_dim, 0.0f);
        for (auto& layer : weight_nodes_.layers) {
            layer.attn_qkv_weight = gb->input({hidden_dim, hidden_dim}, Precision::FP32);
            std::vector<float> qkv(hidden_dim * hidden_dim, 0.0f);
            for (uint32_t h = 0; h < hidden_dim; ++h) {
                qkv[h * hidden_dim + h] = 1.0f;
            }
            set_node(layer.attn_qkv_weight, qkv, "attn_qkv_weight");

            layer.attn_qkv_bias = gb->input({hidden_dim}, Precision::FP32);
            set_node(layer.attn_qkv_bias, std::vector<float>(hidden_dim, 0.0f), "attn_qkv_bias");

            layer.attn_output_weight = gb->input({hidden_dim, hidden_dim}, Precision::FP32);
            std::vector<float> attn_out(hidden_dim * hidden_dim, 0.0f);
            for (uint32_t h = 0; h < hidden_dim; ++h) {
                attn_out[h * hidden_dim + h] = 1.0f;
            }
            set_node(layer.attn_output_weight, attn_out, "attn_output_weight");

            layer.attn_output_bias = gb->input({hidden_dim}, Precision::FP32);
            set_node(layer.attn_output_bias, zeros_hidden, "attn_output_bias");

            layer.ffn_norm_1_weight = gb->input({hidden_dim}, Precision::FP32);
            set_node(layer.ffn_norm_1_weight, std::vector<float>(hidden_dim, 1.0f), "ffn_norm1_weight");

            layer.ffn_norm_1_bias = gb->input({hidden_dim}, Precision::FP32);
            set_node(layer.ffn_norm_1_bias, zeros_hidden, "ffn_norm1_bias");

            layer.ffn_norm_2_weight = gb->input({hidden_dim}, Precision::FP32);
            set_node(layer.ffn_norm_2_weight, std::vector<float>(hidden_dim, 1.0f), "ffn_norm2_weight");

            layer.ffn_norm_2_bias = gb->input({hidden_dim}, Precision::FP32);
            set_node(layer.ffn_norm_2_bias, zeros_hidden, "ffn_norm2_bias");

            layer.ffn_up_weight = gb->input({ffn_dim, hidden_dim}, Precision::FP32);
            std::vector<float> ffn_up(ffn_dim * hidden_dim, 0.0f);
            for (uint32_t h = 0; h < hidden_dim && h < ffn_dim; ++h) {
                ffn_up[h * hidden_dim + h] = 1.0f;
            }
            set_node(layer.ffn_up_weight, ffn_up, "ffn_up_weight");

            layer.ffn_up_bias = gb->input({ffn_dim}, Precision::FP32);
            set_node(layer.ffn_up_bias, std::vector<float>(ffn_dim, 0.0f), "ffn_up_bias");

            layer.ffn_down_weight = gb->input({hidden_dim, ffn_dim}, Precision::FP32);
            std::vector<float> ffn_down(hidden_dim * ffn_dim, 0.0f);
            for (uint32_t h = 0; h < hidden_dim && h < ffn_dim; ++h) {
                ffn_down[h * ffn_dim + h] = 1.0f;
            }
            set_node(layer.ffn_down_weight, ffn_down, "ffn_down_weight");

            layer.ffn_down_bias = gb->input({hidden_dim}, Precision::FP32);
            set_node(layer.ffn_down_bias, zeros_hidden, "ffn_down_bias");
        }
    }

    size_t forward_with_tokens(const std::vector<uint32_t>& tokens, bool dump_activations) {
        auto* gb = graph();
        gb->soft_reset();

        size_t seq_len = tokens.size();
        auto backend = ComputeBackend::CPU;

        size_t input_node_id = gb->input({seq_len}, Precision::FP32);
        std::vector<float> input_data(tokens.begin(), tokens.end());
        gb->set_input(input_node_id, const_cast<float*>(input_data.data()), Precision::FP32);

        size_t hidden = gb->embedding(embedding_node_id_, input_node_id);
        hidden = gb->layernorm(hidden, weight_nodes_.embedding_layernorm_weight, weight_nodes_.embedding_layernorm_bias, config_.layer_norm_eps);

        for (uint32_t layer_idx = 0; layer_idx < config_.num_layers; ++layer_idx) {
            hidden = build_transformer_block(gb, hidden, layer_idx, backend);
        }

        if (dump_activations) {
            gb->execute();
            dump_graph_activations(gb);
        }

        return hidden;
    }

    CactusGraph* graph() {
        return static_cast<CactusGraph*>(graph_handle_);
    }

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

private:
    std::vector<std::shared_ptr<std::vector<float>>> owned_buffers_;
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
        std::string model_path = (project_root / "weights" / "nomic-embed-text-v2-moe").string();

        std::cout << "Initializing model from: " << model_path << std::endl;
        auto model_ptr = create_model(model_path);
        if (!model_ptr) {
            std::cerr << "Failed to create model from: " << model_path << "\n";
            return false;
        }
        if (!model_ptr->init(model_path, 0)) {
            std::cerr << "Failed to initialize model from: " << model_path << "\n";
            return false;
        }
        std::cout << "Model initialized successfully" << std::endl;
        
        auto* model = dynamic_cast<NomicModel*>(model_ptr.get());
        if (!model) {
            std::cerr << "Model is not a NomicModel!\n";
            return false;
        }

        const auto tokens = tokenize_sample_text();
        std::cout << "Tokenized " << tokens.size() << " tokens" << std::endl;
        
        bool dump_requested = std::getenv("CACTUS_DUMP_ACTIVATIONS") != nullptr;
        
        // Call forward manually to have access to the graph before execute
        auto* gb = static_cast<CactusGraph*>(model->graph_handle_);
        std::cout << "Calling forward..." << std::endl;
        size_t final_hidden = model->forward(tokens, false);
        std::cout << "Forward complete, graph has " << gb->nodes_.size() << " nodes" << std::endl;
        std::cout << "Executing graph..." << std::endl;
        
        try {
            gb->execute();
            std::cout << "Execution complete" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Execution failed: " << e.what() << std::endl;
            throw;
        }
        
        if (dump_requested) {
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

        std::cout << "All validations passed" << std::endl;
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
