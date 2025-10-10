#include "model.h"
#include "../graph/graph.h"
#include <algorithm>
#include <cstddef>
#include <set>

namespace cactus {
namespace engine {

NomicModel::NomicModel() : Model() {}

NomicModel::NomicModel(const Config& config) : Model(config) {
    weight_nodes_.layers.resize(config.num_layers);
}

void NomicModel::load_weights_to_graph(CactusGraph* gb) {
    embedding_node_id_ = gb->mmap_embeddings(embedding_file_path_);
    weight_nodes_.embedding_layernorm_weight = gb->mmap_weights(model_folder_path_ + "/embedding_layernorm.weight");
    weight_nodes_.embedding_layernorm_bias = gb->mmap_weights(model_folder_path_ + "/embedding_layernorm.bias");

    for (uint32_t i = 0; i < config_.num_layers; i++) {
        auto& layer = weight_nodes_.layers[i];
        std::string layer_prefix = model_folder_path_ + "/layer_" + std::to_string(i) + "_";
        layer.attn_qkv_weight = gb->mmap_weights(layer_prefix + "attn_qkv.weight");
        layer.attn_qkv_bias = gb->mmap_weights(layer_prefix + "attn_qkv.bias");
        layer.attn_output_weight = gb->mmap_weights(layer_prefix + "attn_output.weight");
        layer.attn_output_bias = gb->mmap_weights(layer_prefix + "attn_output.bias");
        layer.ffn_norm_1_weight = gb->mmap_weights(layer_prefix + "norm1.weight");
        layer.ffn_norm_1_bias = gb->mmap_weights(layer_prefix + "norm1.bias");
        layer.ffn_norm_2_weight = gb->mmap_weights(layer_prefix + "norm2.weight");
        layer.ffn_norm_2_bias = gb->mmap_weights(layer_prefix + "norm2.bias");
        if ((i + 1) % config_.moe_every_n_layers != 0) {
            layer.ffn_up_weight = gb->mmap_weights(layer_prefix + "mlp_fc1.weight");
            layer.ffn_up_bias = gb->mmap_weights(layer_prefix + "mlp_fc1.bias");
            layer.ffn_down_weight = gb->mmap_weights(layer_prefix + "mlp_fc2.weight");
            layer.ffn_down_bias = gb->mmap_weights(layer_prefix + "mlp_fc2.bias");
        } else {
            layer.mlp_router_layer_weight = gb->mmap_weights(layer_prefix + "mlp_router.layer.weight");
            layer.mlp_experts_mlp_w1 = gb->mmap_weights(layer_prefix + "mlp_experts.mlp.w1");
            layer.mlp_experts_mlp_w2 = gb->mmap_weights(layer_prefix + "mlp_experts.mlp.w2");
            layer.mlp_experts_bias = gb->mmap_weights(layer_prefix + "mlp_experts.bias");
        }
    }
}

size_t NomicModel::build_attention(CactusGraph* gb, size_t normalized_input, uint32_t layer_idx,
                                  ComputeBackend backend, bool use_cache, size_t position_offset) {
    if (use_cache) {
        throw std::runtime_error("NomicModel does not support cache, it's an encoder model");
    }
    (void)position_offset;

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

    const Precision input_precision = gb->get_output_buffer(normalized_input).precision;
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

    auto q_bias = gb->gather(layer.attn_qkv_bias, q_indices);
    auto k_bias = gb->gather(layer.attn_qkv_bias, k_indices);
    auto v_bias = gb->gather(layer.attn_qkv_bias, v_indices);

    auto q_proj = gb->matmul(normalized_input, q_weight, true, backend);
    q_proj = gb->add(q_proj, q_bias);

    auto k_proj = gb->matmul(normalized_input, k_weight, true, backend);
    k_proj = gb->add(k_proj, k_bias);

    auto v_proj = gb->matmul(normalized_input, v_weight, true, backend);
    v_proj = gb->add(v_proj, v_bias);

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

    auto attn_output_4d = gb->attention(q_proj_4d, k_proj_4d, v_proj_4d, attention_scale_, 0, false);  // is_causal=false for bidirectional attention
    auto attn_output = gb->reshape(attn_output_4d, {seq_len, num_heads * head_dim});

    auto output = gb->matmul(attn_output, layer.attn_output_weight, true, backend);
    output = gb->add(output, layer.attn_output_bias);
    return output;
}

size_t NomicModel::build_mlp(CactusGraph* gb, size_t normalized_h, uint32_t layer_idx,
                           ComputeBackend backend) const {
    if ((layer_idx + 1) % config_.moe_every_n_layers != 0) {
        return build_standard_mlp(gb, normalized_h, layer_idx, backend);
    } else {
        return build_moe_mlp(gb, normalized_h, layer_idx, backend);
    }
}

size_t NomicModel::build_standard_mlp(CactusGraph* gb, size_t normalized_h, uint32_t layer_idx,
                                     ComputeBackend backend) const {
    const auto& layer = weight_nodes_.layers[layer_idx];
    auto hidden = gb->matmul(normalized_h, layer.ffn_up_weight, true, backend);
    hidden = gb->add(hidden, layer.ffn_up_bias);
    hidden = gb->gelu(hidden);
    hidden = gb->matmul(hidden, layer.ffn_down_weight, true, backend);
    hidden = gb->add(hidden, layer.ffn_down_bias);
    return hidden;
}

size_t NomicModel::build_moe_mlp(CactusGraph* gb, size_t normalized_h, uint32_t layer_idx,
                                ComputeBackend backend) const {
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

    /*
    MOE Logic:
    # X: [T_total, H]
    G = X @ W_gate                         # [T_total, E]
    G = softmax(G)
    topk_idx, topk_w = topk(G, k)          # routing map & weights

    # Build per-expert views
    buckets = bucketize_tokens(topk_idx)   # list of token indices per expert

    # === Dispatch ===
    # Option A: all-to-all (experts sharded)
    X_by_expert = all_to_all_pack(X, buckets)  # permute+send; contiguous per-expert

    # Grouped GEMM: for each expert e on this GPU
    Y_by_expert = grouped_mlp(X_by_expert, experts_params_local)

    # === Combine ===
    Y_perm = all_to_all_unpack(Y_by_expert)    # bring results back, inverse permute
    Y = combine_topk(Y_perm, topk_idx, topk_w) # if k=2, weighted sum; else assign
    */
    return normalized_h;
}

size_t NomicModel::build_transformer_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx,
                                          ComputeBackend backend, bool use_cache, size_t position_offset) {
    if (use_cache) {
        throw std::runtime_error("NomicModel does not support cache, it's an encoder model");
    }
    (void)position_offset;
    const auto& layer = weight_nodes_.layers[layer_idx];
    auto attn_output = build_attention(gb, hidden, layer_idx, backend);
    auto residual = gb->add(hidden, attn_output);
    auto normalized_residual = gb->layernorm(residual, layer.ffn_norm_1_weight, layer.ffn_norm_1_bias, config_.layer_norm_eps);
    auto mlp_output = build_mlp(gb, normalized_residual, layer_idx, backend);
    auto final_residual = gb->add(normalized_residual, mlp_output);
    auto normalized_final_residual = gb->layernorm(final_residual, layer.ffn_norm_2_weight, layer.ffn_norm_2_bias, config_.layer_norm_eps);
    return normalized_final_residual;
}

size_t NomicModel::forward(const std::vector<uint32_t>& tokens, bool use_cache) {
    if (use_cache) {
        throw std::runtime_error("NomicModel does not support cache, it's an encoder model");
    }
    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    gb->soft_reset();

    size_t seq_len = static_cast<size_t>(tokens.size());
    auto backend = config_.default_backend == Config::Backend::CPU
        ? ComputeBackend::CPU
        : ComputeBackend::NPU;

    size_t input_node_id = gb->input({seq_len}, Precision::FP32);
    std::vector<float> input_data(seq_len);
    for (size_t i = 0; i < seq_len; i++) {
        input_data[i] = static_cast<float>(tokens[i]);
    }
    gb->set_input(input_node_id, input_data.data(), Precision::FP32);
    
    size_t hidden = gb->embedding(embedding_node_id_, input_node_id);
    
    hidden = gb->layernorm(hidden, weight_nodes_.embedding_layernorm_weight, weight_nodes_.embedding_layernorm_bias, config_.layer_norm_eps);

    static std::set<uint32_t> skip_layers = {};
    for (uint32_t layer_idx = 0; layer_idx < config_.num_layers; layer_idx++) {
        if (skip_layers.count(layer_idx)) {
            continue;
        }
        hidden = build_transformer_block(gb, hidden, layer_idx, backend);
    }

    return hidden;
}

}
}
