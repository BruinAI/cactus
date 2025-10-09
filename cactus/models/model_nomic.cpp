#include "model.h"
#include "../graph/graph.h"
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
        layer.ffn_norm_1_weight = gb->mmap_weights(layer_prefix + "ffn_norm_1.weight");
        layer.ffn_norm_1_bias = gb->mmap_weights(layer_prefix + "ffn_norm_1.bias");
        layer.ffn_norm_2_weight = gb->mmap_weights(layer_prefix + "ffn_norm_2.weight");
        layer.ffn_norm_2_bias = gb->mmap_weights(layer_prefix + "ffn_norm_2.bias");
        if ((i + 1) % config_.moe_every_n_layers != 0) {
            layer.ffn_up_weight = gb->mmap_weights(layer_prefix + "ffn_up.weight");
            layer.ffn_up_bias = gb->mmap_weights(layer_prefix + "ffn_up.bias");
            layer.ffn_down_weight = gb->mmap_weights(layer_prefix + "ffn_down.weight");
            layer.ffn_down_bias = gb->mmap_weights(layer_prefix + "ffn_down.bias");
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
    // TODO: double check dims
    const auto& layer = weight_nodes_.layers[layer_idx];
    auto qkv_proj = gb->matmul(normalized_input, layer.attn_qkv_weight, true, backend);
    qkv_proj = gb->add(qkv_proj, layer.attn_qkv_bias);

    const auto& qkv_shape = gb->get_output_buffer(qkv_proj).shape;
    size_t batch_seq = qkv_shape[0];
    size_t num_heads = config_.attention_heads;
    size_t head_dim = config_.attention_head_dim;

    qkv_proj = gb->reshape(qkv_proj, {batch_seq, num_heads * head_dim});
    size_t seq_len = batch_seq;

    auto qkv_proj_4d = gb->reshape(qkv_proj, {1, seq_len, config_.attention_heads, config_.attention_head_dim});
    if (config_.rope_theta > 0) {
        qkv_proj_4d = gb->rope(qkv_proj_4d, config_.rope_theta, position_offset);
    }

    size_t final_qkv = qkv_proj_4d;
    auto attn_output_4d = gb->attention(qkv_proj_4d, final_qkv, final_qkv, attention_scale_, position_offset);
    auto attn_output = gb->reshape(attn_output_4d, {seq_len, config_.attention_head_dim * config_.attention_heads});
    return gb->matmul(attn_output, layer.attn_output_weight, true, backend);
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
    
    // Router: compute weights and select top 2 experts
    auto weights = gb->matmul(normalized_h, layer.mlp_router_layer_weight, true, backend);
    weights = gb->softmax(weights);
    auto [top_experts, top_weights] = gb->topk(weights, 2);
    
    // Implementing Experts
    // 1. Gather expert weights for all selected experts: [seq_len, 2, hidden, intermediate]
    // 2. Batch process expert MLPs
    // 3. Weight by router weights and sum
    
    auto gathered_w1 = gb->gather(layer.mlp_experts_mlp_w1, top_experts);
    auto gathered_w2 = gb->gather(layer.mlp_experts_mlp_w2, top_experts);
    
    // Process through expert MLP
    // This is a simplified version - in practice, we'd need to handle the expert dimension
    auto expert_output1 = gb->matmul(normalized_h, gathered_w1, true, backend);
    auto expert_output2 = gb->matmul(normalized_h, gathered_w2, true, backend);
    
    // Weight by router weights and sum
    expert_output1 = gb->multiply(expert_output1, top_weights);
    expert_output2 = gb->multiply(expert_output2, top_weights);
    
    // Add bias
    auto output = gb->add(expert_output1, expert_output2);
    output = gb->add(output, layer.mlp_experts_bias);
    
    return output;
}

size_t NomicModel::build_transformer_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx,
                                          ComputeBackend backend, bool use_cache, size_t position_offset) {
    if (use_cache) {
        throw std::runtime_error("NomicModel does not support cache, it's an encoder model");
    }
    const auto& layer = weight_nodes_.layers[layer_idx];
    auto attn_output = build_attention(gb, hidden, layer_idx, backend);
    auto residual = gb->add(hidden, attn_output);
    auto normalized_residual = gb->layernorm(residual, layer.ffn_norm_1_weight, layer.ffn_norm_1_bias, config_.layer_norm_eps);
    auto mlp_output = build_mlp(gb, normalized_residual, layer_idx, backend);
    auto final_residual = gb->add(residual, mlp_output);
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
