#include "model.h"
#include "../graph/graph.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace cactus {
namespace engine {

Siglip2VisionModelTileAblation::Siglip2VisionModelTileAblation(const Config& cfg)
    : Siglip2VisionModel(cfg) {}

size_t Siglip2VisionModelTileAblation::forward_tiles_independently(
    CactusGraph* gb,
    const Lfm2VlPreprocessor::PreprocessedImage& preprocessed_image,
    ComputeBackend backend) {
    const int num_tiles = preprocessed_image.num_tiles;
    const int max_patches = preprocessed_image.max_patches_per_tile;
    const int patch_dim = preprocessed_image.patch_dim;

    if (num_tiles <= 0) {
        throw std::runtime_error("Per-tile ablation received zero tiles");
    }
    if (patch_dim <= 0 || max_patches <= 0) {
        throw std::runtime_error("Invalid tile metadata for per-tile ablation");
    }

    const size_t expected_size = static_cast<size_t>(num_tiles) *
                                 static_cast<size_t>(max_patches) *
                                 static_cast<size_t>(patch_dim);
    if (preprocessed_image.pixel_values.size() < expected_size) {
        throw std::runtime_error("Pixel values buffer smaller than expected for per-tile ablation");
    }

    size_t reshaped_weight = gb->reshape(
        vision_weight_nodes_.patch_embedding_weight,
        {static_cast<size_t>(config_.vision_embed_dim), static_cast<size_t>(patch_dim)});
    capture_debug_node(0, "vision_tile_ablation_patch_weight", reshaped_weight);

    std::vector<size_t> tile_outputs;
    tile_outputs.reserve(static_cast<size_t>(num_tiles));
    std::vector<size_t> tile_position_nodes;
    tile_position_nodes.reserve(static_cast<size_t>(num_tiles));

    for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        const auto& shape = preprocessed_image.spatial_shapes[tile_idx];
        const int tile_h = shape.first;
        const int tile_w = shape.second;
        const int actual_patches = tile_h * tile_w;

        if (actual_patches <= 0) {
            continue;
        }

        const float* tile_data = preprocessed_image.pixel_values.data() +
                                 static_cast<size_t>(tile_idx) *
                                     static_cast<size_t>(max_patches) *
                                     static_cast<size_t>(patch_dim);

        size_t tile_input_fp32 = gb->input(
            {static_cast<size_t>(actual_patches), static_cast<size_t>(patch_dim)},
            Precision::FP32);
        gb->set_input(tile_input_fp32, tile_data, Precision::FP32);
        capture_debug_node(tile_idx, "vision_tile_" + std::to_string(tile_idx) + "_patches_fp32", tile_input_fp32);

        size_t tile_input = gb->precision_cast(tile_input_fp32, Precision::FP16);
        capture_debug_node(tile_idx, "vision_tile_" + std::to_string(tile_idx) + "_patches", tile_input);

        size_t tile_patch_embeds = gb->matmul(tile_input, reshaped_weight, true, backend);
        size_t tile_patch_bias = gb->add(tile_patch_embeds, vision_weight_nodes_.patch_embedding_bias);
        capture_debug_node(tile_idx, "vision_tile_" + std::to_string(tile_idx) + "_patch_embeds", tile_patch_bias);

        size_t tile_pos = gb->bilinear_interpolation(
            vision_weight_nodes_.position_embedding,
            static_cast<size_t>(tile_h),
            static_cast<size_t>(tile_w));
        capture_debug_node(tile_idx, "vision_tile_" + std::to_string(tile_idx) + "_pos", tile_pos);
    tile_position_nodes.push_back(tile_pos);

        size_t tile_pos_cast = gb->precision_cast(tile_pos, Precision::FP16);
        size_t tile_embeddings = gb->add(tile_patch_bias, tile_pos_cast);
        capture_debug_node(tile_idx, "vision_tile_" + std::to_string(tile_idx) + "_embeddings", tile_embeddings);

        size_t hidden_states = tile_embeddings;
        for (uint32_t layer_idx = 0; layer_idx < config_.vision_num_layers; ++layer_idx) {
            hidden_states = build_vision_transformer_layer(gb, hidden_states, layer_idx, backend);
        }

        hidden_states = gb->layer_norm(hidden_states,
                                       vision_weight_nodes_.post_layernorm_weight,
                                       vision_weight_nodes_.post_layernorm_bias,
                                       config_.layer_norm_eps);
        capture_debug_node(config_.vision_num_layers,
                           "vision_tile_" + std::to_string(tile_idx) + "_post_norm",
                           hidden_states);

        tile_outputs.push_back(hidden_states);
    }

    if (tile_outputs.empty()) {
        throw std::runtime_error("Per-tile ablation produced no outputs");
    }

    size_t combined = tile_outputs.front();
    for (size_t i = 1; i < tile_outputs.size(); ++i) {
        combined = gb->concat(combined, tile_outputs[i], /*axis=*/0);
    }

    capture_debug_node(config_.vision_num_layers,
                       "vision_tile_ablation_final",
                       combined);

    if (!tile_position_nodes.empty()) {
        size_t combined_pos = tile_position_nodes.front();
        for (size_t i = 1; i < tile_position_nodes.size(); ++i) {
            combined_pos = gb->concat(combined_pos, tile_position_nodes[i], /*axis=*/0);
        }
        capture_debug_node(0, "vision_all_pos_embeddings", combined_pos);
    }

    return combined;
}

size_t Siglip2VisionModelTileAblation::forward_vision(
    CactusGraph* gb,
    const Lfm2VlPreprocessor::PreprocessedImage& preprocessed_image,
    ComputeBackend backend) {
    return forward_tiles_independently(gb, preprocessed_image, backend);
}

size_t Siglip2VisionModelTileAblation::forward_vision(
    const Lfm2VlPreprocessor::PreprocessedImage& preprocessed_image) {
    if (!initialized_ || !graph_handle_) {
        throw std::runtime_error("Model not initialized - call init() first");
    }

    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    gb->soft_reset();

    auto backend = config_.default_backend == Config::Backend::CPU ? ComputeBackend::CPU
                                                                   : ComputeBackend::NPU;
    return forward_tiles_independently(gb, preprocessed_image, backend);
}

} // namespace engine
} // namespace cactus
