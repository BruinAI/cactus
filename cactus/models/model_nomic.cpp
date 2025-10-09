#include "model.h"
#include "../graph/graph.h"

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

}
}