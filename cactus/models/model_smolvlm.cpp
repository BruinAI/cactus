#include "model.h"
#include "../graph/graph.h"
#include <cmath>
#include <stdexcept>
#include <set>

namespace cactus {
namespace engine {

SmolVLMModel::SmolVLMModel() : SmolModel() {}
SmolVLMModel::SmolVLMModel(const Config& cfg) : SmolModel(cfg) {}

void SmolVLMModel::load_weights_to_graph(CactusGraph* gb) {
    SmolModel::load_weights_to_graph(gb);

    weight_nodes_.vision_layers.resize(config_.vision_num_layers);

    std::string base = model_folder_path_ + "/";
    weight_nodes_.vision_proj_weight = gb->mmap_weights(base + "vision_patch_embedding.weights");
    weight_nodes_.vision_position_embedding = gb->mmap_weights(base + "vision_position_embedding.weights");

    for (uint32_t i = 0; i < weight_nodes_.vision_layers.size(); ++i) {
        auto& vw = weight_nodes_.vision_layers[i];
        std::string prefix = base + "vision_layer_" + std::to_string(i) + "_";

        auto try_mmap = [&](const std::vector<std::string>& candidates) -> size_t {
            for (const auto& c : candidates) {
                return gb->mmap_weights(c);
            }
            return 0;
        };

        vw.attn_q_weight = try_mmap({prefix + "self_attn_q.weights", prefix + "self_attn_q.weight"});
        vw.attn_k_weight = try_mmap({prefix + "self_attn_k.weights", prefix + "self_attn_k.weight"});
        vw.attn_v_weight = try_mmap({prefix + "self_attn_v.weights", prefix + "self_attn_v.weight"});
        vw.attn_output_weight = try_mmap({prefix + "self_attn_out.weights", prefix + "self_attn_out.weight"});

        vw.layer_norm1_weight = try_mmap({prefix + "layer_norm1.weights", prefix + "layer_norm1.weight.weights", prefix + "layer_norm1.weight"});
        vw.layer_norm1_bias = try_mmap({prefix + "layer_norm1.bias.weights", prefix + "layer_norm1.bias.weight"});
        vw.layer_norm2_weight = try_mmap({prefix + "layer_norm2.weights", prefix + "layer_norm2.weight.weights", prefix + "layer_norm2.weight"});
        vw.layer_norm2_bias = try_mmap({prefix + "layer_norm2.bias.weights", prefix + "layer_norm2.bias.weight"});

        vw.mlp_fc1_weight = try_mmap({prefix + "ffn_fc1.weights", prefix + "ffn_fc1.weight"});
        vw.mlp_fc1_bias = try_mmap({prefix + "ffn_fc1.bias.weights", prefix + "ffn_fc1.bias.weight"});
        vw.mlp_fc2_weight = try_mmap({prefix + "ffn_fc2.weights", prefix + "ffn_fc2.weight"});
        vw.mlp_fc2_bias = try_mmap({prefix + "ffn_fc2.bias.weights", prefix + "ffn_fc2.bias.weight"});
    }
}

size_t build_vision_embeddings(CactusGraph* gb, const std::vector<ImageBatch>& images,
                                   ComputeBackend backend) {

}
size_t build_combined_input(CactusGraph* gb, size_t vision_embeds, const std::vector<uint32_t>& tokens,
                                ComputeBackend backend, uint32_t& prefix_len) {

}

size_t forward_mm(const std::vector<uint32_t>& tokens,const std::vector<ImageBatch>& images, bool use_cache) {

}

}
}