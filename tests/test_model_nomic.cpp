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
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <limits>
#include <unordered_map>
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

class InspectableNomicModel : public NomicModel {
public:
    using NomicModel::NomicModel;

    struct MoETrace {
        size_t normalized_input = 0;
        size_t router_logits = 0;
        size_t router_probs = 0;
        size_t top_indices = 0;
        size_t top_weights = 0;
        size_t expert_outputs = 0;
        size_t selected_experts = 0;
        size_t weights = 0;
        size_t weighted = 0;
        size_t final_output = 0;
    };

    struct LayerTrace {
        size_t block_input = 0;
        size_t attention_output = 0;
        size_t residual_pre_norm1 = 0;
        size_t norm1_output = 0;
        size_t mlp_output = 0;
        size_t residual_post_mlp = 0;
        size_t norm2_output = 0;
    };

    struct AttentionTrace {
        size_t normalized_input = 0;
        size_t q_proj = 0;
        size_t k_proj = 0;
        size_t v_proj = 0;
        size_t q_rope = 0;
        size_t k_rope = 0;
        size_t context = 0;
        size_t output = 0;
    };

    mutable std::unordered_map<uint32_t, MoETrace> moe_traces;
    mutable std::unordered_map<uint32_t, LayerTrace> layer_traces;
    mutable std::unordered_map<uint32_t, AttentionTrace> attention_traces;

protected:
    size_t build_attention(CactusGraph* gb, size_t normalized_input, uint32_t layer_idx,
                          ComputeBackend backend, bool use_cache = false, size_t position_offset = 0) override {
        if (use_cache) {
            throw std::runtime_error("NomicModel does not support cache, it's an encoder model");
        }
        (void)position_offset;

        auto& trace = attention_traces[layer_idx];
        trace.normalized_input = normalized_input;

        const auto& layer = weight_nodes_.layers[layer_idx];
        const auto& weight_shape = gb->get_output_buffer(layer.attn_qkv_weight).shape;
        if (weight_shape.size() != 2) {
            throw std::runtime_error("QKV weight must be 2D");
        }

        const size_t hidden_dim = weight_shape[1];
        const size_t total_qkv_dim = weight_shape[0];
        if (total_qkv_dim % 3 != 0) {
            throw std::runtime_error("QKV weight first dimension must be divisible by 3");
        }

        const auto input_precision = gb->get_output_buffer(normalized_input).precision;
        size_t normalized_fp32 = normalized_input;
        if (input_precision != Precision::FP32) {
            normalized_fp32 = gb->precision_cast(normalized_input, Precision::FP32);
        }

        auto make_range_input = [&](size_t start, size_t length) {
            auto indices_node = gb->input({length}, Precision::FP32);
            std::vector<float> indices(length);
            for (size_t i = 0; i < length; ++i) {
                indices[i] = static_cast<float>(start + i);
            }
            gb->set_input(indices_node, indices.data(), Precision::FP32);
            return indices_node;
        };

        const size_t segment = total_qkv_dim / 3;
        auto q_indices = make_range_input(0, segment);
        auto k_indices = make_range_input(segment, segment);
        auto v_indices = make_range_input(segment * 2, segment);

        auto q_weight = gb->gather(layer.attn_qkv_weight, q_indices);
        auto k_weight = gb->gather(layer.attn_qkv_weight, k_indices);
        auto v_weight = gb->gather(layer.attn_qkv_weight, v_indices);
        if (gb->get_output_buffer(q_weight).precision != Precision::FP32) {
            q_weight = gb->precision_cast(q_weight, Precision::FP32);
        }
        if (gb->get_output_buffer(k_weight).precision != Precision::FP32) {
            k_weight = gb->precision_cast(k_weight, Precision::FP32);
        }
        if (gb->get_output_buffer(v_weight).precision != Precision::FP32) {
            v_weight = gb->precision_cast(v_weight, Precision::FP32);
        }

        auto q_bias = gb->gather(layer.attn_qkv_bias, q_indices);
        auto k_bias = gb->gather(layer.attn_qkv_bias, k_indices);
        auto v_bias = gb->gather(layer.attn_qkv_bias, v_indices);
        if (gb->get_output_buffer(q_bias).precision != Precision::FP32) {
            q_bias = gb->precision_cast(q_bias, Precision::FP32);
        }
        if (gb->get_output_buffer(k_bias).precision != Precision::FP32) {
            k_bias = gb->precision_cast(k_bias, Precision::FP32);
        }
        if (gb->get_output_buffer(v_bias).precision != Precision::FP32) {
            v_bias = gb->precision_cast(v_bias, Precision::FP32);
        }

        auto q_proj = gb->matmul(normalized_fp32, q_weight, true, backend);
        q_proj = gb->add(q_proj, q_bias);
        trace.q_proj = q_proj;

        auto k_proj = gb->matmul(normalized_fp32, k_weight, true, backend);
        k_proj = gb->add(k_proj, k_bias);
        trace.k_proj = k_proj;

        auto v_proj = gb->matmul(normalized_fp32, v_weight, true, backend);
        v_proj = gb->add(v_proj, v_bias);
        trace.v_proj = v_proj;

        const auto& q_shape = gb->get_output_buffer(q_proj).shape;
        if (q_shape.size() != 2) {
            throw std::runtime_error("Projected queries must be 2D");
        }
        const size_t seq_len = q_shape[0];
        const size_t num_heads = config_.attention_heads;
        const size_t head_dim = config_.attention_head_dim;

        if (num_heads == 0 || head_dim == 0 || num_heads * head_dim != hidden_dim) {
            throw std::runtime_error("Invalid attention head configuration for Nomic model");
        }

        auto reshape_to_heads = [&](size_t tensor) {
            return gb->reshape(tensor, {1, seq_len, num_heads, head_dim});
        };

        auto q_proj_4d = reshape_to_heads(q_proj);
        auto k_proj_4d = reshape_to_heads(k_proj);
        auto v_proj_4d = reshape_to_heads(v_proj);

        if (config_.rope_theta > 0) {
            q_proj_4d = gb->rope(q_proj_4d, config_.rope_theta, 0);
            k_proj_4d = gb->rope(k_proj_4d, config_.rope_theta, 0);
        }
        trace.q_rope = q_proj_4d;
        trace.k_rope = k_proj_4d;

        auto attn_output_4d = gb->attention(q_proj_4d, k_proj_4d, v_proj_4d, attention_scale_, 0, false);
        auto attn_output = gb->reshape(attn_output_4d, {seq_len, num_heads * head_dim});
        trace.context = attn_output;

        size_t out_weight = layer.attn_output_weight;
        if (gb->get_output_buffer(out_weight).precision != Precision::FP32) {
            out_weight = gb->precision_cast(out_weight, Precision::FP32);
        }
        size_t out_bias = layer.attn_output_bias;
        if (gb->get_output_buffer(out_bias).precision != Precision::FP32) {
            out_bias = gb->precision_cast(out_bias, Precision::FP32);
        }

        auto output = gb->matmul(attn_output, out_weight, true, backend);
        output = gb->add(output, out_bias);
        if (gb->get_output_buffer(output).precision != input_precision) {
            output = gb->precision_cast(output, input_precision);
        }
        trace.output = output;
        return output;
    }

    size_t build_mlp(CactusGraph* gb, size_t normalized_h, uint32_t layer_idx,
                    ComputeBackend backend) const override {
        if ((layer_idx + 1) % config_.moe_every_n_layers != 0) {
            return NomicModel::build_standard_mlp(gb, normalized_h, layer_idx, backend);
        }

        auto& trace = moe_traces[layer_idx];
        trace.normalized_input = normalized_h;

        const auto& layer = weight_nodes_.layers[layer_idx];
        const auto& router_shape = gb->get_output_buffer(layer.mlp_router_layer_weight).shape;
        if (router_shape.size() != 2) {
            throw std::runtime_error("MoE router weight must be 2D");
        }

        const size_t num_experts = config_.num_experts != 0 ? config_.num_experts : router_shape[0];
        const auto& w1_shape = gb->get_output_buffer(layer.mlp_experts_mlp_w1).shape;
        const size_t expert_dim = w1_shape[0] / num_experts;
        const size_t hidden_dim = w1_shape[1];
        const size_t seq_len = gb->get_output_buffer(normalized_h).shape[0];

        auto make_range = [&](size_t start, size_t length) {
            auto node = gb->input({length}, Precision::FP32);
            std::vector<float> indices(length);
            for (size_t i = 0; i < length; ++i) indices[i] = static_cast<float>(start + i);
            gb->set_input(node, indices.data(), Precision::FP32);
            return node;
        };

        const auto input_precision = gb->get_output_buffer(normalized_h).precision;

        size_t normalized_fp32 = normalized_h;
        if (input_precision != Precision::FP32) {
            normalized_fp32 = gb->precision_cast(normalized_h, Precision::FP32);
        }

        size_t router_weight = layer.mlp_router_layer_weight;
        if (gb->get_output_buffer(router_weight).precision != Precision::FP32) {
            router_weight = gb->precision_cast(router_weight, Precision::FP32);
        }

        auto router_logits = gb->matmul(normalized_fp32, router_weight, true, backend);
        trace.router_logits = router_logits;
        auto router_probs = gb->softmax(router_logits);
        trace.router_probs = router_probs;
        const size_t top_k = std::min<size_t>(2, num_experts);
        auto [top_indices, top_weights] = gb->topk(router_probs, top_k);
        trace.top_indices = top_indices;
        trace.top_weights = top_weights;

        std::vector<size_t> expert_outputs;
        expert_outputs.reserve(num_experts);
        for (size_t e = 0; e < num_experts; ++e) {
            auto indices = make_range(e * expert_dim, expert_dim);
            auto w1 = gb->gather(layer.mlp_experts_mlp_w1, indices);
            if (gb->get_output_buffer(w1).precision != Precision::FP32) {
                w1 = gb->precision_cast(w1, Precision::FP32);
            }
            auto hidden = gb->matmul(normalized_fp32, w1, true, backend);
            if (gb->get_output_buffer(hidden).precision != Precision::FP32) {
                hidden = gb->precision_cast(hidden, Precision::FP32);
            }
            hidden = gb->gelu(hidden);
            auto w2 = gb->gather(layer.mlp_experts_mlp_w2, indices);
            if (gb->get_output_buffer(w2).precision != Precision::FP32) {
                w2 = gb->precision_cast(w2, Precision::FP32);
            }
            w2 = gb->transpose(w2, backend);
            auto expert_out = gb->matmul(hidden, w2, true, backend);
            if (gb->get_output_buffer(expert_out).precision != Precision::FP32) {
                expert_out = gb->precision_cast(expert_out, Precision::FP32);
            }
            expert_outputs.push_back(gb->reshape(expert_out, {seq_len, 1, hidden_dim}));
        }
        size_t all_experts = expert_outputs[0];
        for (size_t i = 1; i < expert_outputs.size(); ++i) {
            all_experts = gb->concat(all_experts, expert_outputs[i], 1);
        }
        trace.expert_outputs = all_experts;

        auto all_experts_flat = gb->reshape(all_experts, {seq_len * num_experts, hidden_dim});

        std::vector<float> offsets(seq_len * top_k);
        for (size_t t = 0; t < seq_len; ++t) {
            float base = static_cast<float>(t * num_experts);
            for (size_t k = 0; k < top_k; ++k) {
                offsets[t * top_k + k] = base;
            }
        }
        auto offsets_node = gb->input({seq_len * top_k}, Precision::FP32);
        gb->set_input(offsets_node, offsets.data(), Precision::FP32);

        auto top_indices_flat = gb->reshape(top_indices, {seq_len * top_k});
        auto gather_indices = gb->add(top_indices_flat, offsets_node);

        auto selected = gb->gather(all_experts_flat, gather_indices);
        selected = gb->reshape(selected, {seq_len, top_k, hidden_dim});
        trace.selected_experts = selected;

        auto weights = gb->reshape(top_weights, {seq_len, top_k, 1});
        if (gb->get_output_buffer(weights).precision != Precision::FP32) {
            weights = gb->precision_cast(weights, Precision::FP32);
        }
        trace.weights = weights;

        auto weighted = gb->multiply(selected, weights);
        trace.weighted = weighted;
        auto output = gb->sum(weighted, 1);
        if (gb->get_output_buffer(output).precision != Precision::FP32) {
            output = gb->precision_cast(output, Precision::FP32);
        }

        size_t bias = layer.mlp_experts_bias;
        if (gb->get_output_buffer(bias).precision != Precision::FP32) {
            bias = gb->precision_cast(bias, Precision::FP32);
        }

        output = gb->add(output, bias);
        if (gb->get_output_buffer(output).precision != input_precision) {
            output = gb->precision_cast(output, input_precision);
        }
        trace.final_output = output;
        return output;
    }

    size_t build_transformer_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx,
                                  ComputeBackend backend, bool use_cache, size_t position_offset = 0) override {
        if (use_cache) {
            throw std::runtime_error("NomicModel does not support cache, it's an encoder model");
        }
        (void)position_offset;
        auto& trace = layer_traces[layer_idx];
        trace.block_input = hidden;

        auto attn_output = build_attention(gb, hidden, layer_idx, backend);
        trace.attention_output = attn_output;

        auto residual = gb->add(hidden, attn_output);
        trace.residual_pre_norm1 = residual;

        auto normalized_residual = gb->layernorm(residual, weight_nodes_.layers[layer_idx].ffn_norm_1_weight,
                                                 weight_nodes_.layers[layer_idx].ffn_norm_1_bias, config_.layer_norm_eps);
        trace.norm1_output = normalized_residual;

        auto mlp_output = build_mlp(gb, normalized_residual, layer_idx, backend);
        trace.mlp_output = mlp_output;

        auto final_residual = gb->add(normalized_residual, mlp_output);
        trace.residual_post_mlp = final_residual;

        auto normalized_final_residual = gb->layernorm(final_residual,
                                                       weight_nodes_.layers[layer_idx].ffn_norm_2_weight,
                                                       weight_nodes_.layers[layer_idx].ffn_norm_2_bias,
                                                       config_.layer_norm_eps);
        trace.norm2_output = normalized_final_residual;
        return normalized_final_residual;
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

struct CapturedTensor {
    std::vector<size_t> shape;
    std::vector<float> data;
    Precision precision;
};

CapturedTensor capture_tensor(CactusGraph* gb, size_t node_id) {
    CapturedTensor result;
    if (node_id == 0) {
        throw std::runtime_error("capture_tensor received invalid node id");
    }
    const auto& buffer = gb->get_output_buffer(node_id);
    result.shape = buffer.shape;
    result.precision = buffer.precision;
    size_t total = buffer.total_size;
    result.data.resize(total);

    if (total == 0) {
        return result;
    }

    void* raw_ptr = gb->get_output(node_id);
    switch (buffer.precision) {
        case Precision::FP32: {
            const float* src = static_cast<const float*>(raw_ptr);
            std::copy(src, src + total, result.data.begin());
            break;
        }
        case Precision::FP16: {
            const __fp16* src = static_cast<const __fp16*>(raw_ptr);
            for (size_t i = 0; i < total; ++i) {
                result.data[i] = static_cast<float>(src[i]);
            }
            break;
        }
        case Precision::INT8: {
            const int8_t* src = static_cast<const int8_t*>(raw_ptr);
            float scale = buffer.quantization_scale;
            for (size_t i = 0; i < total; ++i) {
                result.data[i] = static_cast<float>(src[i]) * scale;
            }
            break;
        }
    }
    return result;
}

void write_tensor_json(std::ofstream& out, const std::string& name, const CapturedTensor& tensor, bool is_last) {
    out << "  \"" << name << "\": {\n";
    out << "    \"shape\": [";
    for (size_t i = 0; i < tensor.shape.size(); ++i) {
        if (i) out << ", ";
        out << tensor.shape[i];
    }
    out << "],\n";
    out << "    \"data\": [";
    for (size_t i = 0; i < tensor.data.size(); ++i) {
        if (i) out << ", ";
        out << tensor.data[i];
    }
    out << "]\n  }" << (is_last ? "\n" : ",\n");
}

bool capture_moe_layer1_activations() {
    try {
        auto project_root = find_project_root();
        auto model_path = project_root / "weights" / "nomic-embed-text-v2-moe";

        Config config;
        if (!config.from_json((model_path / "config.txt").string())) {
            std::cerr << "Failed to load config from " << (model_path / "config.txt") << "\n";
            return false;
        }

        InspectableNomicModel model(config);
        if (!model.init(model_path.string(), 0)) {
            std::cerr << "Failed to initialize InspectableNomicModel\n";
            return false;
        }

        const auto tokens = tokenize_sample_text();
        auto* gb = static_cast<CactusGraph*>(model.graph_handle_);
        gb->soft_reset();
        model.forward(tokens, false);
        gb->execute();

        auto trace_it = model.moe_traces.find(1);
        if (trace_it == model.moe_traces.end()) {
            std::cerr << "Did not capture MoE trace for layer 1\n";
            return false;
        }
        const auto& trace = trace_it->second;
        const auto& layer_trace_it = model.layer_traces.find(1);
        if (layer_trace_it == model.layer_traces.end()) {
            std::cerr << "Did not capture layer trace for layer 1\n";
            return false;
        }
        const auto& layer_trace = layer_trace_it->second;
        const auto& attn_trace_it = model.attention_traces.find(1);
        if (attn_trace_it == model.attention_traces.end()) {
            std::cerr << "Did not capture attention trace for layer 1\n";
            return false;
        }
        const auto& attn_trace = attn_trace_it->second;

        CapturedTensor norm = capture_tensor(gb, trace.normalized_input);
        CapturedTensor router_logits = capture_tensor(gb, trace.router_logits);
        CapturedTensor router_probs = capture_tensor(gb, trace.router_probs);
        CapturedTensor top_indices = capture_tensor(gb, trace.top_indices);
        CapturedTensor top_weights = capture_tensor(gb, trace.top_weights);
        CapturedTensor expert_outputs = capture_tensor(gb, trace.expert_outputs);
        CapturedTensor selected = capture_tensor(gb, trace.selected_experts);
        CapturedTensor weights = capture_tensor(gb, trace.weights);
        CapturedTensor weighted = capture_tensor(gb, trace.weighted);
        CapturedTensor moe_output = capture_tensor(gb, trace.final_output);
        CapturedTensor attn_q = capture_tensor(gb, attn_trace.q_proj);
        CapturedTensor attn_k = capture_tensor(gb, attn_trace.k_proj);
        CapturedTensor attn_v = capture_tensor(gb, attn_trace.v_proj);
        CapturedTensor attn_q_rope = capture_tensor(gb, attn_trace.q_rope);
        CapturedTensor attn_k_rope = capture_tensor(gb, attn_trace.k_rope);
        CapturedTensor attn_context = capture_tensor(gb, attn_trace.context);
        CapturedTensor attn_output = capture_tensor(gb, attn_trace.output);
        CapturedTensor block_input = capture_tensor(gb, layer_trace.block_input);
        CapturedTensor attention_output = capture_tensor(gb, layer_trace.attention_output);
        CapturedTensor residual_pre_norm1 = capture_tensor(gb, layer_trace.residual_pre_norm1);
        CapturedTensor norm1_output = capture_tensor(gb, layer_trace.norm1_output);
        CapturedTensor mlp_output = capture_tensor(gb, layer_trace.mlp_output);
        CapturedTensor residual_post_mlp = capture_tensor(gb, layer_trace.residual_post_mlp);
        CapturedTensor norm2_output = capture_tensor(gb, layer_trace.norm2_output);

        CapturedTensor layer0_block_input;
        CapturedTensor layer0_attention_output;
        CapturedTensor layer0_residual_pre_norm1;
        CapturedTensor layer0_norm1_output;
        CapturedTensor layer0_mlp_output;
        CapturedTensor layer0_residual_post_mlp;
        CapturedTensor layer0_norm2_output;
        CapturedTensor layer0_attn_q;
        CapturedTensor layer0_attn_k;
        CapturedTensor layer0_attn_v;
        CapturedTensor layer0_attn_q_rope;
        CapturedTensor layer0_attn_k_rope;
        CapturedTensor layer0_attn_context;
        bool has_layer0 = false;

        if (auto lt0_it = model.layer_traces.find(0); lt0_it != model.layer_traces.end()) {
            const auto& lt0 = lt0_it->second;
            layer0_block_input = capture_tensor(gb, lt0.block_input);
            layer0_attention_output = capture_tensor(gb, lt0.attention_output);
            layer0_residual_pre_norm1 = capture_tensor(gb, lt0.residual_pre_norm1);
            layer0_norm1_output = capture_tensor(gb, lt0.norm1_output);
            layer0_mlp_output = capture_tensor(gb, lt0.mlp_output);
            layer0_residual_post_mlp = capture_tensor(gb, lt0.residual_post_mlp);
            layer0_norm2_output = capture_tensor(gb, lt0.norm2_output);
            has_layer0 = true;
        } else {
            std::cerr << "Did not capture layer trace for layer 0\n";
        }
        if (auto at0_it = model.attention_traces.find(0); at0_it != model.attention_traces.end()) {
            const auto& at0 = at0_it->second;
            layer0_attn_q = capture_tensor(gb, at0.q_proj);
            layer0_attn_k = capture_tensor(gb, at0.k_proj);
            layer0_attn_v = capture_tensor(gb, at0.v_proj);
            layer0_attn_q_rope = capture_tensor(gb, at0.q_rope);
            layer0_attn_k_rope = capture_tensor(gb, at0.k_rope);
            layer0_attn_context = capture_tensor(gb, at0.context);
            has_layer0 = true;
        } else {
            std::cerr << "Did not capture attention trace for layer 0\n";
        }

        auto precision_to_string = [](Precision prec) {
            switch (prec) {
                case Precision::INT8: return "INT8";
                case Precision::FP16: return "FP16";
                case Precision::FP32: return "FP32";
            }
            return "UNKNOWN";
        };

        auto summarize = [&](const char* name, const CapturedTensor& tensor) {
            double max_abs = 0.0;
            double min_val = std::numeric_limits<double>::infinity();
            double max_val = -std::numeric_limits<double>::infinity();
            size_t non_finite = 0;
            for (float v : tensor.data) {
                if (!std::isfinite(v)) {
                    non_finite++;
                    continue;
                }
                double dv = static_cast<double>(v);
                if (dv < min_val) min_val = dv;
                if (dv > max_val) max_val = dv;
                double abs_v = std::abs(dv);
                if (abs_v > max_abs) max_abs = abs_v;
            }
            std::cout << "  " << name << ": max_abs=" << max_abs
                      << " precision=" << precision_to_string(tensor.precision)
                      << " min=" << min_val
                      << " max=" << max_val
                      << " non_finite=" << non_finite << std::endl;
        };

        if (has_layer0) {
            std::cout << "Layer 0 summaries:" << std::endl;
            summarize("layer0_block_input", layer0_block_input);
            summarize("layer0_attention_output", layer0_attention_output);
            summarize("layer0_residual_pre_norm1", layer0_residual_pre_norm1);
            summarize("layer0_norm1_output", layer0_norm1_output);
            summarize("layer0_attention_q_proj", layer0_attn_q);
            summarize("layer0_attention_k_proj", layer0_attn_k);
            summarize("layer0_attention_v_proj", layer0_attn_v);
            summarize("layer0_attention_q_rope", layer0_attn_q_rope);
            summarize("layer0_attention_k_rope", layer0_attn_k_rope);
            summarize("layer0_attention_context", layer0_attn_context);
            summarize("layer0_mlp_output", layer0_mlp_output);
            summarize("layer0_residual_post_mlp", layer0_residual_post_mlp);
            summarize("layer0_norm2_output", layer0_norm2_output);
        }

        std::cout << "Layer 1 summaries:" << std::endl;
        summarize("norm1", norm);
        summarize("router_logits", router_logits);
        summarize("router_probs", router_probs);
        summarize("top_weights", top_weights);
        summarize("attention_q_proj", attn_q);
        summarize("attention_k_proj", attn_k);
        summarize("attention_v_proj", attn_v);
        summarize("attention_q_rope", attn_q_rope);
        summarize("attention_k_rope", attn_k_rope);
        summarize("attention_context", attn_context);
        summarize("attention_output_internal", attn_output);
        summarize("block_input", block_input);
        summarize("attention_output_block", attention_output);
        summarize("residual_pre_norm1", residual_pre_norm1);
        summarize("norm1_output", norm1_output);
        summarize("expert_outputs", expert_outputs);
        summarize("selected_experts", selected);
        summarize("weights", weights);
        summarize("weighted", weighted);
        summarize("mlp_pre_output", mlp_output);
        summarize("residual_post_mlp", residual_post_mlp);
        summarize("norm2_output", norm2_output);
        summarize("final_output", moe_output);

        auto sample_expert = [&](const char* label, const CapturedTensor& tensor, size_t token, size_t expert, size_t count) {
            if (tensor.shape.size() != 3) return;
            size_t hidden = tensor.shape[2];
            size_t experts = tensor.shape[1];
            if (token >= tensor.shape[0] || expert >= experts) return;
            size_t base = token * experts * hidden + expert * hidden;
            std::cout << "  " << label << " token " << token << " expert " << expert << ":";
            for (size_t i = 0; i < count && i < hidden; ++i) {
                std::cout << " " << tensor.data[base + i];
            }
            std::cout << std::endl;
        };

        sample_expert("expert_outputs", expert_outputs, 0, 0, 5);
        sample_expert("expert_outputs", expert_outputs, 9, 0, 5);
        sample_expert("expert_outputs", expert_outputs, 9, 3, 5);
        sample_expert("selected_experts", selected, 9, 0, 5);
        sample_expert("selected_experts", selected, 9, 1, 5);
        sample_expert("weighted", weighted, 9, 0, 5);
        sample_expert("weighted", weighted, 9, 1, 5);

        std::ofstream out(project_root / "tests" / "cactus_moe_layer1_activations.json");
        if (!out) {
            std::cerr << "Failed to open cactus activations output file\n";
            return false;
        }
        out << std::setprecision(10);
        out << "{\n";
        std::vector<std::pair<std::string, CapturedTensor>> tensors;
        if (has_layer0) {
            tensors.emplace_back("layer0_block_input", std::move(layer0_block_input));
            tensors.emplace_back("layer0_attention_output", std::move(layer0_attention_output));
            tensors.emplace_back("layer0_residual_pre_norm1", std::move(layer0_residual_pre_norm1));
            tensors.emplace_back("layer0_norm1_output", std::move(layer0_norm1_output));
            tensors.emplace_back("layer0_attention_q_proj", std::move(layer0_attn_q));
            tensors.emplace_back("layer0_attention_k_proj", std::move(layer0_attn_k));
            tensors.emplace_back("layer0_attention_v_proj", std::move(layer0_attn_v));
            tensors.emplace_back("layer0_attention_q_rope", std::move(layer0_attn_q_rope));
            tensors.emplace_back("layer0_attention_k_rope", std::move(layer0_attn_k_rope));
            tensors.emplace_back("layer0_attention_context", std::move(layer0_attn_context));
            tensors.emplace_back("layer0_mlp_pre_output", std::move(layer0_mlp_output));
            tensors.emplace_back("layer0_residual_post_mlp", std::move(layer0_residual_post_mlp));
            tensors.emplace_back("layer0_norm2_output", std::move(layer0_norm2_output));
        }
        tensors.emplace_back("norm1", std::move(norm));
        tensors.emplace_back("router_logits", std::move(router_logits));
        tensors.emplace_back("router_probs", std::move(router_probs));
        tensors.emplace_back("top_indices", std::move(top_indices));
        tensors.emplace_back("top_weights", std::move(top_weights));
        tensors.emplace_back("expert_outputs", std::move(expert_outputs));
        tensors.emplace_back("selected_experts", std::move(selected));
        tensors.emplace_back("weights", std::move(weights));
        tensors.emplace_back("weighted", std::move(weighted));
        tensors.emplace_back("attention_q_proj", std::move(attn_q));
        tensors.emplace_back("attention_k_proj", std::move(attn_k));
        tensors.emplace_back("attention_v_proj", std::move(attn_v));
        tensors.emplace_back("attention_q_rope", std::move(attn_q_rope));
        tensors.emplace_back("attention_k_rope", std::move(attn_k_rope));
        tensors.emplace_back("attention_context", std::move(attn_context));
        tensors.emplace_back("attention_output_internal", std::move(attn_output));
        tensors.emplace_back("block_input", std::move(block_input));
        tensors.emplace_back("attention_output", std::move(attention_output));
        tensors.emplace_back("residual_pre_norm1", std::move(residual_pre_norm1));
        tensors.emplace_back("norm1_output", std::move(norm1_output));
        tensors.emplace_back("mlp_pre_output", std::move(mlp_output));
        tensors.emplace_back("residual_post_mlp", std::move(residual_post_mlp));
        tensors.emplace_back("norm2_output", std::move(norm2_output));
        tensors.emplace_back("mlp_output", std::move(moe_output));

        for (size_t i = 0; i < tensors.size(); ++i) {
            write_tensor_json(out, tensors[i].first, tensors[i].second, i + 1 == tensors.size());
        }
        out << "}\n";
        std::cout << "Saved cactus activations to tests/cactus_moe_layer1_activations.json" << std::endl;
        return true;
    } catch (const std::exception& err) {
        std::cerr << "capture_moe_layer1_activations failed: " << err.what() << "\n";
        return false;
    }
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
        for (const auto& node_ptr : gb->nodes_) {
            if (node_ptr->id == 153 || node_ptr->id == 150 || node_ptr->id == 156) {
                const auto& node = *node_ptr;
                std::cout << "DEBUG node " << node.id << " info: op=" << op_type_to_string(node.op_type)
                          << " inputs=[";
                for (size_t i = 0; i < node.input_ids.size(); ++i) {
                    if (i) std::cout << ",";
                    std::cout << node.input_ids[i];
                }
                std::cout << "] shape=[";
                for (size_t i = 0; i < node.output_buffer.shape.size(); ++i) {
                    if (i) std::cout << ",";
                    std::cout << node.output_buffer.shape[i];
                }
                std::cout << "]";
                if (node.op_type == OpType::INPUT && node.output_buffer.precision == Precision::FP32) {
                    const float* data = static_cast<const float*>(gb->get_output(node.id));
                    std::cout << " ptr=" << static_cast<const void*>(data);
                    std::cout << " values=[";
                    size_t sample = std::min<size_t>(10, node.output_buffer.total_size);
                    for (size_t i = 0; i < sample; ++i) {
                        if (i) std::cout << ",";
                        std::cout << data[i];
                    }
                    if (node.output_buffer.total_size > sample) {
                        std::cout << ",...";
                    }
                    std::cout << "]";
                    float min_val = data[0];
                    float max_val = data[0];
                    for (size_t i = 1; i < node.output_buffer.total_size; ++i) {
                        min_val = std::min(min_val, data[i]);
                        max_val = std::max(max_val, data[i]);
                    }
                    std::cout << " min=" << min_val << " max=" << max_val;
                }
                std::cout << "\n";
            }
            bool references_150 = std::find(node_ptr->input_ids.begin(), node_ptr->input_ids.end(), 150) != node_ptr->input_ids.end();
            if (references_150 && node_ptr->id != 153) {
                std::cout << "DEBUG node " << node_ptr->id << " references 150 with op=" << op_type_to_string(node_ptr->op_type) << "\n";
            }
        }
        
        try {
            gb->execute();
            std::cout << "Execution complete" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Execution failed: " << e.what() << std::endl;
            throw;
        }
        if (const char* save_node_env = std::getenv("CACTUS_SAVE_NODE")) {
            size_t node_id = static_cast<size_t>(std::stoul(save_node_env));
            std::string filename = "/tmp/cactus_node_" + std::string(save_node_env) + ".bin";
            try {
                GraphFile::save_node(*gb, node_id, filename);
                std::cout << "Saved node " << node_id << " activations to " << filename << "\n";
            } catch (const std::exception& e) {
                std::cerr << "Failed to save node " << node_id << ": " << e.what() << "\n";
            }
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
        if (const char* save_path = std::getenv("CACTUS_SAVE_LAST_HIDDEN")) {
            std::ofstream out(save_path);
            if (!out) {
                std::cerr << "Failed to open " << save_path << " for writing\n";
            } else {
                out << std::setprecision(10);
                for (size_t i = 0; i < output_buffer.total_size; ++i) {
                    if (i) {
                        out << '\n';
                    }
                    out << data[i];
                }
                out.close();
                std::cout << "Saved final hidden activations to " << save_path << "\n";
            }
        }
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
    runner.run_test("Capture MoE layer1 activations", capture_moe_layer1_activations());
    runner.run_test("Forward cache guard", test_nomic_forward_rejects_cache());
    runner.run_test("Attention cache guard", test_nomic_attention_rejects_cache());
    runner.run_test("Transformer block cache guard", test_nomic_transformer_block_rejects_cache());

    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
