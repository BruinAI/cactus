#include "model.h"
#include "../graph/graph.h"
#include <cstddef>
#include <stdexcept>

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
        layer.attn_q_weight = gb->mmap_weights(layer_prefix + "attn_q.weights");
        layer.attn_k_weight = gb->mmap_weights(layer_prefix + "attn_k.weights");
        layer.attn_v_weight = gb->mmap_weights(layer_prefix + "attn_v.weights");
        layer.attn_q_bias = gb->mmap_weights(layer_prefix + "attn_q.bias");
        layer.attn_k_bias = gb->mmap_weights(layer_prefix + "attn_k.bias");
        layer.attn_v_bias = gb->mmap_weights(layer_prefix + "attn_v.bias");
        layer.attn_output_weight = gb->mmap_weights(layer_prefix + "attn_output.weights");
        layer.attn_output_bias = gb->mmap_weights(layer_prefix + "attn_output.bias");
        layer.ffn_norm_1_weight = gb->mmap_weights(layer_prefix + "norm1.weights");
        layer.ffn_norm_1_bias = gb->mmap_weights(layer_prefix + "norm1.bias");
        layer.ffn_norm_2_weight = gb->mmap_weights(layer_prefix + "norm2.weights");
        layer.ffn_norm_2_bias = gb->mmap_weights(layer_prefix + "norm2.bias");
        if ((i + 1) % config_.moe_every_n_layers != 0) {
            layer.ffn_up_weight = gb->mmap_weights(layer_prefix + "mlp_fc1.weights");
            layer.ffn_up_bias = gb->mmap_weights(layer_prefix + "mlp_fc1.bias");
            layer.ffn_down_weight = gb->mmap_weights(layer_prefix + "mlp_fc2.weights");
            layer.ffn_down_bias = gb->mmap_weights(layer_prefix + "mlp_fc2.bias");
        } else {
            layer.mlp_router_layer_weight = gb->mmap_weights(layer_prefix + "mlp_router.layer.weights");
            layer.mlp_experts_bias = gb->mmap_weights(layer_prefix + "mlp_experts.bias");
            for (uint32_t j = 0; j < config_.num_experts; j++) {
                layer.mlp_experts_mlp1_weight.push_back(gb->mmap_weights(layer_prefix + "mlp_expert_" + std::to_string(j) + ".mlp1.weights"));
                layer.mlp_experts_mlp2_weight.push_back(gb->mmap_weights(layer_prefix + "mlp_expert_" + std::to_string(j) + ".mlp2.weights"));
            }
        }
    }
}

size_t NomicModel::build_attention(CactusGraph* gb, size_t normalized_input, uint32_t layer_idx,
                                 ComputeBackend backend, bool use_cache, size_t position_offset) {
    throw std::runtime_error("NomicModel::build_attention not implemented yet");
}

size_t NomicModel::build_mlp(CactusGraph* gb, size_t normalized_h, uint32_t layer_idx,
                           ComputeBackend backend) const {
    throw std::runtime_error("NomicModel::build_mlp not implemented yet");
}

size_t NomicModel::build_standard_mlp(CactusGraph* gb, size_t normalized_h, uint32_t layer_idx,
                                    ComputeBackend backend) const {
throw std::runtime_error("NomicModel::build_standard_mlp not implemented yet");
}

size_t NomicModel::build_moe_mlp(CactusGraph* gb, size_t normalized_h, uint32_t layer_idx,
                               ComputeBackend backend) const {
throw std::runtime_error("NomicModel::build_moe_mlp not implemented yet");
}

size_t NomicModel::build_transformer_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx,
                                         ComputeBackend backend, bool use_cache, size_t position_offset) {
    throw std::runtime_error("NomicModel::build_transformer_block not implemented yet");
}

size_t NomicModel::forward(const std::vector<uint32_t>& tokens, bool use_cache) {
    throw std::runtime_error("NomicModel::forward not implemented yet");
}

}
}
