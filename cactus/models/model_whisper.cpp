#include "model.h"
#include "../graph/graph.h"
#include <cmath>
#include <stdexcept>
#include <set>

namespace cactus {
namespace engine {

WhisperModel::WhisperModel() : Model() {}

WhisperModel::WhisperModel(const Config& config) : Model(config) {
    weight_nodes_.layers.resize(config.num_layers);
}

void WhisperModel::load_weights_to_graph(CactusGraph* gb) {
    embedding_node_id_ = gb->mmap_embeddings(embedding_file_path_);
    weight_nodes_.output_norm_weight = gb->mmap_weights(model_folder_path_ + "/output_norm.weights");

    if (config_.tie_word_embeddings) {
        weight_nodes_.output_weight = embedding_node_id_;
        output_weight_node_id_ = embedding_node_id_;
    } else {
        weight_nodes_.output_weight = gb->mmap_weights(model_folder_path_ + "/output_weight.weights");
        output_weight_node_id_ = weight_nodes_.output_weight;
    }

    for (uint32_t i = 0; i < config_.num_layers; i++) {
        auto& layer = weight_nodes_.layers[i];
        std::string layer_prefix = model_folder_path_ + "/layer_" + std::to_string(i) + "_";
        layer.encoder_attn_q_weight = gb->mmap_weights(layer_prefix + "q_proj.weights");
        layer.encoder_attn_v_weight = gb->mmap_weights(layer_prefix + "v_proj.weights");
        layer.encoder_attn_q_bias = gb->mmap_weights(layer_prefix + "q_proj.bias");
        layer.encoder_attn_v_bias = gb->mmap_weights(layer_prefix + "q_proj.bias");
        layer.encoder_attn_output_weight = gb->mmap_weights(layer_prefix + "")
        
        // layer.encoder_attn_k_weight = gb->mmap_weights(layer_prefix + "attn_k.weights");
        // layer.attn_v_weight = gb->mmap_weights(layer_prefix + "attn_v.weights");
        // layer.attn_output_weight = gb->mmap_weights(layer_prefix + "attn_output.weights");
        // layer.input_layernorm_weight = gb->mmap_weights(layer_prefix + "input_norm.weights");
        // layer.attn_q_norm_weight = gb->mmap_weights(layer_prefix + "attn_q_norm.weights");
        // layer.attn_k_norm_weight = gb->mmap_weights(layer_prefix + "attn_k_norm.weights");
        // layer.fc_weight = gb->mmap_weights(layer_prefix + "fc.weights");
        // layer.post_attention_layernorm_weight = gb->mmap_weights(layer_prefix + "post_attn_norm.weights");
        // layer.pre_feedforward_layernorm_weight = gb->mmap_weights(layer_prefix + "pre_ffn_norm.weights");
        // layer.post_feedforward_layernorm_weight = gb->mmap_weights(layer_prefix + "post_ffn_norm.weights");
    }
}

size_t WhisperModel::build_attention(CactusGraph* gb, size_t normalized_input, uint32_t layer_idx,ComputeBackend backend, bool use_cache, size_t position_offset) {
    const auto& layer = weight_nodes_.layers[layer_idx];

    // TODO
}


size_t WhisperModel::build_mlp(CactusGraph* gb, size_t input, uint32_t layer_idx, ComputeBackend backend) const {
    const auto& layer = weight_nodes_.layers[layer_idx];

    // TODO
}

size_t WhisperModel::build_transformer_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx,
                                         ComputeBackend backend, bool use_cache, size_t position_offset) {
    const auto& layer = weight_nodes_.layers[layer_idx];

    // TODO
}


size_t WhisperModel::forward(const std::vector<uint32_t>& tokens, bool use_cache) {
    if (!initialized_ || !graph_handle_) {
        throw std::runtime_error("Model not initialized - call init() first");
    }

    if (tokens.empty()) {
        throw std::runtime_error("Token sequence cannot be empty");
    }

    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    gb->soft_reset();

    auto seq_len = static_cast<size_t>(tokens.size());

    size_t position_offset = use_cache ? kv_cache_.get_total_seq_len() : 0;

    auto backend = config_.default_backend == Config::Backend::CPU
        ? ComputeBackend::CPU
        : ComputeBackend::NPU;

    auto input_node_id = gb->input({seq_len}, Precision::FP32);
    auto hidden = gb->embedding(embedding_node_id_, input_node_id);

    float embed_scale = std::sqrt(static_cast<float>(config_.hidden_dim));
    hidden = gb->scalar_multiply(hidden, embed_scale);

    static std::set<uint32_t> skip_layers = {};
    for (uint32_t layer_idx = 0; layer_idx < config_.num_layers; layer_idx++) {
        if (skip_layers.count(layer_idx)) {
            continue;
        }
    
    // TODO
}

}
}