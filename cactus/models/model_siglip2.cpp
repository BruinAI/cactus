#include "model.h"
#include "../graph/graph.h"
#include <cmath>
#include <stdexcept>
#include <utility>
#include <iostream>

namespace cactus {
namespace engine {

Siglip2VisionModel::Siglip2VisionModel() : Model() {
    config_.model_type = Config::ModelType::SIGLIP2;
    std::cout << "[Siglip2VisionModel::ctor] default constructor called" << std::endl;
}

Siglip2VisionModel::Siglip2VisionModel(const Config& cfg) : Model(cfg) {
    std::cout << "[Siglip2VisionModel::ctor] config-based constructor called" << std::endl;
    // Initialize LFM2-VL preprocessor with config values from model config
    Lfm2VlPreprocessor::Config preprocessor_config;
    preprocessor_config.patch_size = static_cast<int>(config_.vision_patch_size);
    preprocessor_config.downsample_factor = static_cast<int>(config_.downsample_factor);
    preprocessor_config.min_tiles = static_cast<int>(config_.min_tiles);
    preprocessor_config.max_tiles = static_cast<int>(config_.max_tiles);
    preprocessor_config.use_thumbnail = config_.use_thumbnail;
    preprocessor_config.min_image_tokens = static_cast<int>(config_.min_image_tokens);
    preprocessor_config.max_image_tokens = static_cast<int>(config_.max_image_tokens);
    preprocessor_config.max_num_patches = static_cast<int>(config_.max_num_patches);
    preprocessor_config.tile_size = static_cast<int>(config_.tile_size);
    preprocessor_config.max_pixels_tolerance = config_.max_pixels_tolerance;
    preprocessor_config.do_resize = true;
    preprocessor_config.do_rescale = true;
    preprocessor_config.do_normalize = true;
    preprocessor_config.do_convert_rgb = true;
    preprocessor_config.do_image_splitting = config_.do_image_splitting;
    preprocessor_config.rescale_factor = config_.rescale_factor;
    preprocessor_config.image_mean[0] = config_.image_mean;
    preprocessor_config.image_mean[1] = config_.image_mean;
    preprocessor_config.image_mean[2] = config_.image_mean;
    preprocessor_config.image_std[0] = config_.image_std;
    preprocessor_config.image_std[1] = config_.image_std;
    preprocessor_config.image_std[2] = config_.image_std;
    
    preprocessor_ = Lfm2VlPreprocessor(preprocessor_config);
    std::cout << "[Siglip2VisionModel::ctor] preprocessor initialized" << std::endl;
}

void Siglip2VisionModel::load_weights_to_graph(CactusGraph* gb) {
    auto precision_to_string = [](Precision p) -> const char* {
        switch (p) {
            case Precision::FP32: return "FP32";
            case Precision::FP16: return "FP16";
            case Precision::INT8: return "INT8";
        }
        return "UNKNOWN";
    };
    auto log_buffer = [&](const std::string& label, size_t node_id) {
        const auto& buf = gb->get_output_buffer(node_id);
        std::cout << "[Siglip2VisionModel::build_vision_embeddings] " << label
                  << " node=" << node_id
                  << " precision=" << precision_to_string(buf.precision)
                  << " shape=[";
        for (size_t dim_idx = 0; dim_idx < buf.shape.size(); ++dim_idx) {
            std::cout << buf.shape[dim_idx];
            if (dim_idx + 1 < buf.shape.size()) {
                std::cout << ", ";
            }
        }
        std::cout << "] total_size=" << buf.total_size << std::endl;
    };
    vision_weight_nodes_.vision_layers.resize(config_.vision_num_layers);
    std::cout << "[Siglip2VisionModel::load_weights_to_graph] vision layers count=" << vision_weight_nodes_.vision_layers.size() << std::endl;

    std::string base = model_folder_path_ + "/";
    
    // Patch embedding weights (using short names from convert_hf.py)
    vision_weight_nodes_.patch_embedding_weight = gb->mmap_weights(base + "vision_patch_embedding.weights");
    vision_weight_nodes_.patch_embedding_bias = gb->mmap_weights(base + "vision_patch_embedding.bias.weights");
    std::cout << "[Siglip2VisionModel::load_weights_to_graph] patch embedding nodes weight=" << vision_weight_nodes_.patch_embedding_weight << " bias=" << vision_weight_nodes_.patch_embedding_bias << std::endl;
    // get shapes of loaded weights for verification
    std::cout << "[Siglip2VisionModel::load_weights_to_graph] patch embedding weight shape: ";
    log_buffer("patch embedding weight", vision_weight_nodes_.patch_embedding_weight);
    std::cout << "[Siglip2VisionModel::load_weights_to_graph] patch embedding bias shape: ";
    log_buffer("patch embedding bias", vision_weight_nodes_.patch_embedding_bias);
    
    // Position embedding
    vision_weight_nodes_.position_embedding = gb->mmap_weights(base + "vision_position_embedding.weights");
    std::cout << "[Siglip2VisionModel::load_weights_to_graph] position_embedding node=" << vision_weight_nodes_.position_embedding << std::endl;
    
    // Post layer norm
    vision_weight_nodes_.post_layernorm_weight = gb->mmap_weights(base + "vision_post_layernorm.weights");
    vision_weight_nodes_.post_layernorm_bias = gb->mmap_weights(base + "vision_post_layernorm.bias.weights");
    std::cout << "[Siglip2VisionModel::load_weights_to_graph] post layernorm nodes weight=" << vision_weight_nodes_.post_layernorm_weight << " bias=" << vision_weight_nodes_.post_layernorm_bias << std::endl;
    
    // Load encoder layers
    for (uint32_t i = 0; i < vision_weight_nodes_.vision_layers.size(); ++i) {
        auto& layer = vision_weight_nodes_.vision_layers[i];
        std::string prefix = base + "vision_layer_" + std::to_string(i) + "_";
    std::cout << "[Siglip2VisionModel::load_weights_to_graph] loading vision layer=" << i << std::endl;

        // Self attention weights
        layer.attn_q_weight = gb->mmap_weights(prefix + "self_attn_q.weights");
        layer.attn_q_bias = gb->mmap_weights(prefix + "self_attn_q.bias.weights");
        layer.attn_k_weight = gb->mmap_weights(prefix + "self_attn_k.weights");
        layer.attn_k_bias = gb->mmap_weights(prefix + "self_attn_k.bias.weights");
        layer.attn_v_weight = gb->mmap_weights(prefix + "self_attn_v.weights");
        layer.attn_v_bias = gb->mmap_weights(prefix + "self_attn_v.bias.weights");
        layer.attn_output_weight = gb->mmap_weights(prefix + "self_attn_out.weights");
        layer.attn_output_bias = gb->mmap_weights(prefix + "self_attn_out.bias.weights");
    std::cout << "[Siglip2VisionModel::load_weights_to_graph] attention weights loaded for layer=" << i << std::endl;

        // Layer norms
        layer.layer_norm1_weight = gb->mmap_weights(prefix + "layer_norm1.weights");
        layer.layer_norm1_bias = gb->mmap_weights(prefix + "layer_norm1.bias.weights");
        layer.layer_norm2_weight = gb->mmap_weights(prefix + "layer_norm2.weights");
        layer.layer_norm2_bias = gb->mmap_weights(prefix + "layer_norm2.bias.weights");
    std::cout << "[Siglip2VisionModel::load_weights_to_graph] layer norms loaded for layer=" << i << std::endl;

        // MLP weights
        layer.mlp_fc1_weight = gb->mmap_weights(prefix + "ffn_fc1.weights");
        layer.mlp_fc1_bias = gb->mmap_weights(prefix + "ffn_fc1.bias.weights");
        layer.mlp_fc2_weight = gb->mmap_weights(prefix + "ffn_fc2.weights");
        layer.mlp_fc2_bias = gb->mmap_weights(prefix + "ffn_fc2.bias.weights");
        std::cout << "[Siglip2VisionModel::load_weights_to_graph] MLP weights loaded for layer=" << i << std::endl;
    }
    
    // Note: Pooling head weights not loaded - vision_use_head is false in LFM2-VL
    // The model outputs last_hidden_state directly without pooling
}

Siglip2VisionModel::VisionEmbeddingResult Siglip2VisionModel::build_vision_embeddings(
    CactusGraph* gb,
    const Lfm2VlPreprocessor::PreprocessedImage& preprocessed_image,
    ComputeBackend backend) {

    std::cout << "[Siglip2VisionModel::build_vision_embeddings] Starting to build vision embeddings" << std::endl;
    const int num_tiles = preprocessed_image.num_tiles;
    const int max_patches = preprocessed_image.max_patches_per_tile;
    const int patch_dim = preprocessed_image.patch_dim;

    const size_t expected_size = static_cast<size_t>(num_tiles) * static_cast<size_t>(max_patches) *
                                 static_cast<size_t>(patch_dim);
    if (preprocessed_image.pixel_values.size() != expected_size) {
        throw std::runtime_error(
            "Pixel values size mismatch: expected " + std::to_string(expected_size) +
            " (tiles=" + std::to_string(num_tiles) + " * max_patches=" + std::to_string(max_patches) +
            " * patch_dim=" + std::to_string(patch_dim) + ") but got " +
            std::to_string(preprocessed_image.pixel_values.size()));
    }
    std::cout << "[Siglip2VisionModel::build_vision_embeddings] num_tiles=" << num_tiles << " max_patches=" << max_patches << " patch_dim=" << patch_dim << std::endl;

    for (size_t i = 0; i < std::min<size_t>(100, preprocessed_image.pixel_values.size()); ++i) {
        float val = preprocessed_image.pixel_values[i];
        if (std::isnan(val) || std::isinf(val)) {
            throw std::runtime_error(
                "Invalid value in pixel_values at index " + std::to_string(i) + ": " + std::to_string(val));
        }
    }

    size_t reshaped_weight = gb->reshape(
        vision_weight_nodes_.patch_embedding_weight,
        {static_cast<size_t>(config_.vision_embed_dim), static_cast<size_t>(patch_dim)});
    capture_debug_node(0, "vision_reshaped_weight", reshaped_weight);
    std::cout << "[Siglip2VisionModel::build_vision_embeddings] reshaped_weight node=" << reshaped_weight << std::endl;

    auto precision_to_string = [](Precision p) -> const char* {
        switch (p) {
            case Precision::FP32: return "FP32";
            case Precision::FP16: return "FP16";
            case Precision::INT8: return "INT8";
        }
        return "UNKNOWN";
    };

    auto log_buffer = [&](const std::string& label, size_t node_id) {
        const auto& buf = gb->get_output_buffer(node_id);
        std::cout << "[Siglip2VisionModel::build_vision_embeddings] " << label
                  << " node=" << node_id
                  << " precision=" << precision_to_string(buf.precision)
                  << " shape=[";
        for (size_t dim_idx = 0; dim_idx < buf.shape.size(); ++dim_idx) {
            std::cout << buf.shape[dim_idx];
            if (dim_idx + 1 < buf.shape.size()) {
                std::cout << ", ";
            }
        }
        std::cout << "] total_size=" << buf.total_size << std::endl;
    };

    log_buffer("patch_embedding_weight buffer (raw)", vision_weight_nodes_.patch_embedding_weight);
    log_buffer("patch_embedding_bias buffer (raw)", vision_weight_nodes_.patch_embedding_bias);
    log_buffer("position_embedding buffer (raw)", vision_weight_nodes_.position_embedding);

    log_buffer("reshaped_weight buffer", reshaped_weight);

    size_t patch_bias = vision_weight_nodes_.patch_embedding_bias;
    log_buffer("patch_embedding_bias buffer", patch_bias);

    std::vector<size_t> tile_embeddings;
    tile_embeddings.reserve(static_cast<size_t>(num_tiles));

    for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        const auto& shape = preprocessed_image.spatial_shapes[tile_idx];
        const int tile_h = shape.first;
        const int tile_w = shape.second;
        const int actual_patches = tile_h * tile_w;
        std::cout << "[Siglip2VisionModel::build_vision_embeddings] tile_idx=" << tile_idx << " tile_h=" << tile_h << " tile_w=" << tile_w << " actual_patches=" << actual_patches << std::endl;

        if (actual_patches <= 0) {
            std::cout << "[Siglip2VisionModel::build_vision_embeddings] skipping tile due to non-positive patches" << std::endl;
            continue;
        }

        const float* tile_data = preprocessed_image.pixel_values.data() +
                                 static_cast<size_t>(tile_idx) * static_cast<size_t>(max_patches) *
                                     static_cast<size_t>(patch_dim);

        size_t tile_input_fp32 = gb->input(
            {static_cast<size_t>(actual_patches), static_cast<size_t>(patch_dim)}, Precision::FP32);
        gb->set_input(tile_input_fp32, tile_data, Precision::FP32);
        capture_debug_node(tile_idx, "vision_tile_" + std::to_string(tile_idx) + "_patches_fp32", tile_input_fp32);
        std::cout << "[Siglip2VisionModel::build_vision_embeddings] tile_input_fp32 node=" << tile_input_fp32 << std::endl;
        log_buffer("tile_input_fp32 buffer", tile_input_fp32);

        size_t tile_input = gb->precision_cast(tile_input_fp32, Precision::FP16);
        capture_debug_node(tile_idx, "vision_tile_" + std::to_string(tile_idx) + "_patches", tile_input);
        std::cout << "[Siglip2VisionModel::build_vision_embeddings] tile_input node=" << tile_input << std::endl;
        log_buffer("tile_input_fp16 buffer", tile_input);

        log_buffer("reshaped_weight buffer", reshaped_weight);

        size_t tile_patch = gb->matmul(tile_input, reshaped_weight, true, backend);
    log_buffer("tile_patch buffer", tile_patch);
    log_buffer("patch_embedding_bias_reshaped buffer", patch_bias);
    size_t tile_bias = gb->add(tile_patch, patch_bias);
        log_buffer("tile_bias buffer", tile_bias);
        capture_debug_node(tile_idx, "vision_tile_" + std::to_string(tile_idx) + "_patch_embeds", tile_bias);
        std::cout << "[Siglip2VisionModel::build_vision_embeddings] tile_patch node=" << tile_patch << " tile_bias node=" << tile_bias << std::endl;

        std::cout << "[Siglip2VisionModel::build_vision_embeddings] positional destination sizes" << tile_h << " x " << tile_w << std::endl;

        size_t tile_pos = gb->bilinear_interpolation(
            vision_weight_nodes_.position_embedding,
            static_cast<size_t>(tile_h),
            static_cast<size_t>(tile_w));
        log_buffer("tile_pos buffer", tile_pos);
        capture_debug_node(tile_idx, "vision_tile_pos_" + std::to_string(tile_idx), tile_pos);
        std::cout << "[Siglip2VisionModel::build_vision_embeddings] tile_pos node=" << tile_pos << std::endl;

        size_t tile_pos_cast = gb->precision_cast(tile_pos, Precision::FP16);
        log_buffer("tile_pos_cast buffer", tile_pos_cast);
        log_buffer("tile_bias buffer before add_pos", tile_bias);
        size_t tile_embed = gb->add(tile_bias, tile_pos_cast);
        log_buffer("tile_embed buffer", tile_embed);
        capture_debug_node(tile_idx, "vision_tile_" + std::to_string(tile_idx) + "_embeddings", tile_embed);
        std::cout << "[Siglip2VisionModel::build_vision_embeddings] tile_embed node=" << tile_embed << std::endl;

        tile_embeddings.push_back(tile_embed);
    }

    if (tile_embeddings.empty()) {
        throw std::runtime_error("No valid tiles produced embeddings in build_vision_embeddings");
    }
    std::cout << "[Siglip2VisionModel::build_vision_embeddings] tile_embeddings size=" << tile_embeddings.size() << std::endl;

    auto concat_nodes = [&](const std::vector<size_t>& nodes) {
        if (nodes.empty()) {
            throw std::runtime_error("Attempted to concatenate an empty node list");
        }
        size_t combined = nodes.front();
        for (size_t i = 1; i < nodes.size(); ++i) {
            combined = gb->concat(combined, nodes[i], /*axis=*/0);
        }
        return combined;
    };

    size_t embeddings = concat_nodes(tile_embeddings);
    capture_debug_node(0, "vision_final_embeddings", embeddings);
    std::cout << "[Siglip2VisionModel::build_vision_embeddings] embeddings node=" << embeddings << std::endl;

    return VisionEmbeddingResult{embeddings, std::move(tile_embeddings)};
}

size_t Siglip2VisionModel::build_vision_attention(CactusGraph* gb, size_t hidden_states, 
                                                  uint32_t layer_idx, ComputeBackend backend) {
    const auto& layer = vision_weight_nodes_.vision_layers[layer_idx];
    
    // Project to Q, K, V
    size_t q = gb->matmul(hidden_states, layer.attn_q_weight, true, backend);
    q = gb->add(q, layer.attn_q_bias);
    capture_debug_node(layer_idx, "vision_attn_q", q);
    std::cout << "[Siglip2VisionModel::build_vision_attention] q node=" << q << std::endl;
    
    size_t k = gb->matmul(hidden_states, layer.attn_k_weight, true, backend);
    k = gb->add(k, layer.attn_k_bias);
    capture_debug_node(layer_idx, "vision_attn_k", k);
    std::cout << "[Siglip2VisionModel::build_vision_attention] k node=" << k << std::endl;
    
    size_t v = gb->matmul(hidden_states, layer.attn_v_weight, true, backend);
    v = gb->add(v, layer.attn_v_bias);
    capture_debug_node(layer_idx, "vision_attn_v", v);
    std::cout << "[Siglip2VisionModel::build_vision_attention] v node=" << v << std::endl;

    // Reshape for multi-head attention
    const size_t num_heads = static_cast<size_t>(config_.vision_attention_heads);
    const size_t head_dim = static_cast<size_t>(config_.vision_embed_dim / config_.vision_attention_heads);
    const auto& q_buf = gb->get_output_buffer(q);
    size_t seq_len = q_buf.shape[0];
    
    size_t q_4d = gb->reshape(q, {1, seq_len, num_heads, head_dim});
    size_t k_4d = gb->reshape(k, {1, seq_len, num_heads, head_dim});
    size_t v_4d = gb->reshape(v, {1, seq_len, num_heads, head_dim});
    std::cout << "[Siglip2VisionModel::build_vision_attention] q_4d=" << q_4d << " k_4d=" << k_4d << " v_4d=" << v_4d << std::endl;
    
    // Scaled dot-product attention
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    size_t attn_output = gb->attention(q_4d, k_4d, v_4d, scale, false, backend);
    capture_debug_node(layer_idx, "vision_attn_scores", attn_output);
    std::cout << "[Siglip2VisionModel::build_vision_attention] attn_output node=" << attn_output << std::endl;
    
    // Reshape back
    size_t attn_2d = gb->reshape(attn_output, {seq_len, num_heads * head_dim});
    capture_debug_node(layer_idx, "vision_attn_2d", attn_2d);  // <-- ADD THIS
    std::cout << "[Siglip2VisionModel::build_vision_attention] attn_2d node=" << attn_2d << std::endl;

    // Output projection
    size_t output = gb->matmul(attn_2d, layer.attn_output_weight, true, backend);
    output = gb->add(output, layer.attn_output_bias);
    capture_debug_node(layer_idx, "vision_attn_output", output);
    std::cout << "[Siglip2VisionModel::build_vision_attention] output node=" << output << std::endl;
    
    return output;
}

size_t Siglip2VisionModel::build_vision_mlp(CactusGraph* gb, size_t hidden_states, 
                                           uint32_t layer_idx, ComputeBackend backend) {
    const auto& layer = vision_weight_nodes_.vision_layers[layer_idx];
    
    // FC1
    size_t fc1_output = gb->matmul(hidden_states, layer.mlp_fc1_weight, true, backend);
    fc1_output = gb->add(fc1_output, layer.mlp_fc1_bias);
    capture_debug_node(layer_idx, "vision_mlp_fc1", fc1_output);
    std::cout << "[Siglip2VisionModel::build_vision_mlp] fc1_output node=" << fc1_output << std::endl;
    
    // Activation (GELU for SigLip2)
    size_t activated = gb->gelu(fc1_output);
    capture_debug_node(layer_idx, "vision_mlp_gelu", activated);
    std::cout << "[Siglip2VisionModel::build_vision_mlp] activated node=" << activated << std::endl;
    
    // FC2
    size_t fc2_output = gb->matmul(activated, layer.mlp_fc2_weight, true, backend);
    fc2_output = gb->add(fc2_output, layer.mlp_fc2_bias);
    capture_debug_node(layer_idx, "vision_mlp_fc2", fc2_output);
    std::cout << "[Siglip2VisionModel::build_vision_mlp] fc2_output node=" << fc2_output << std::endl;
    
    return fc2_output;
}

size_t Siglip2VisionModel::build_vision_transformer_layer(CactusGraph* gb, size_t hidden_states, 
                                                          uint32_t layer_idx, ComputeBackend backend) {
    const auto& layer = vision_weight_nodes_.vision_layers[layer_idx];
    
    // Pre-norm architecture: LayerNorm -> Attention -> Residual
    size_t residual = hidden_states;
    size_t normalized = gb->layer_norm(hidden_states, layer.layer_norm1_weight, 
                                      layer.layer_norm1_bias, config_.layer_norm_eps);
    capture_debug_node(layer_idx, "vision_attn_norm", normalized);
    std::cout << "[Siglip2VisionModel::build_vision_transformer_layer] normalized before attn node=" << normalized << std::endl;
    size_t attn_output = build_vision_attention(gb, normalized, layer_idx, backend);
    hidden_states = gb->add(residual, attn_output);
    capture_debug_node(layer_idx, "vision_after_attn", hidden_states);
    std::cout << "[Siglip2VisionModel::build_vision_transformer_layer] hidden_states after attn node=" << hidden_states << std::endl;
    
    // Pre-norm architecture: LayerNorm -> MLP -> Residual
    residual = hidden_states;
    normalized = gb->layer_norm(hidden_states, layer.layer_norm2_weight, 
                               layer.layer_norm2_bias, config_.layer_norm_eps);
    capture_debug_node(layer_idx, "vision_mlp_norm", normalized);
    std::cout << "[Siglip2VisionModel::build_vision_transformer_layer] normalized before mlp node=" << normalized << std::endl;
    size_t mlp_output = build_vision_mlp(gb, normalized, layer_idx, backend);
    hidden_states = gb->add(residual, mlp_output);
    capture_debug_node(layer_idx, "vision_after_mlp", hidden_states);
    std::cout << "[Siglip2VisionModel::build_vision_transformer_layer] hidden_states after mlp node=" << hidden_states << std::endl;
    
    return hidden_states;
}

size_t Siglip2VisionModel::forward_vision(
    CactusGraph* gb,
    const Lfm2VlPreprocessor::PreprocessedImage& preprocessed_image,
    ComputeBackend backend) {
    auto embedding_result = build_vision_embeddings(gb, preprocessed_image, backend);
    std::cout << "[Siglip2VisionModel::forward_vision external] embedding_result.combined=" << embedding_result.combined_embeddings << " tile_embeddings size=" << embedding_result.tile_embeddings.size() << std::endl;

    auto concat_nodes = [&](const std::vector<size_t>& nodes) {
        if (nodes.empty()) {
            throw std::runtime_error("Attempted to concatenate an empty node list in forward_vision");
        }
        size_t combined = nodes.front();
        for (size_t i = 1; i < nodes.size(); ++i) {
            combined = gb->concat(combined, nodes[i], /*axis=*/0);
        }
        return combined;
    };

    std::vector<size_t> tile_outputs;
    tile_outputs.reserve(embedding_result.tile_embeddings.size());

    for (size_t tile_idx = 0; tile_idx < embedding_result.tile_embeddings.size(); ++tile_idx) {
        size_t hidden_states = embedding_result.tile_embeddings[tile_idx];
        std::cout << "[Siglip2VisionModel::forward_vision external] processing tile=" << tile_idx << " node=" << hidden_states << std::endl;
        for (uint32_t layer_idx = 0; layer_idx < config_.vision_num_layers; ++layer_idx) {
            hidden_states = build_vision_transformer_layer(gb, hidden_states, layer_idx, backend);
            std::cout << "[Siglip2VisionModel::forward_vision external] tile=" << tile_idx << " after layer=" << layer_idx << " node=" << hidden_states << std::endl;
        }

        hidden_states = gb->layer_norm(hidden_states,
                                       vision_weight_nodes_.post_layernorm_weight,
                                       vision_weight_nodes_.post_layernorm_bias,
                                       config_.layer_norm_eps);
        capture_debug_node(config_.vision_num_layers,
                           "vision_tile_" + std::to_string(tile_idx) + "_post_norm",
                           hidden_states);
        std::cout << "[Siglip2VisionModel::forward_vision external] tile=" << tile_idx << " post layer norm node=" << hidden_states << std::endl;

        tile_outputs.push_back(hidden_states);
        std::cout << "[Siglip2VisionModel::forward_vision external] tile_outputs size=" << tile_outputs.size() << std::endl;
    }

    if (tile_outputs.empty()) {
        throw std::runtime_error("No tile outputs generated in forward_vision");
    }

    size_t combined_output = concat_nodes(tile_outputs);
    capture_debug_node(config_.vision_num_layers, "vision_tile_ablation_final", combined_output);
    capture_debug_node(config_.vision_num_layers, "vision_post_norm", combined_output);
    std::cout << "[Siglip2VisionModel::forward_vision external] combined_output node=" << combined_output << std::endl;

    return combined_output;
}

size_t Siglip2VisionModel::forward_vision(const Lfm2VlPreprocessor::PreprocessedImage& preprocessed_image) {
    if (!initialized_ || !graph_handle_) {
        throw std::runtime_error("Model not initialized - call init() first");
    }
    std::cout << "[Siglip2VisionModel::forward_vision internal] called" << std::endl;

    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    gb->soft_reset();
    std::cout << "[Siglip2VisionModel::forward_vision internal] graph soft reset" << std::endl;

    auto backend = config_.default_backend == Config::Backend::CPU ? ComputeBackend::CPU : ComputeBackend::NPU;
    std::cout << "[Siglip2VisionModel::forward_vision internal] backend=" << static_cast<int>(backend) << std::endl;

    return forward_vision(gb, preprocessed_image, backend);
}

std::vector<float> Siglip2VisionModel::get_image_features(const std::string& image_path) {
    // Preprocess image
    auto preprocessed = preprocessor_.preprocess_from_file(image_path);
    std::cout << "[Siglip2VisionModel::get_image_features path] preprocessed num_tiles=" << preprocessed.num_tiles << std::endl;
    return get_image_features(preprocessed);
}

size_t Siglip2VisionModel::get_image_features_node(const Lfm2VlPreprocessor::PreprocessedImage& preprocessed_image) {
    // Build the vision forward pass and return the output node ID
    // This allows other models to plug this into their graph
    return forward_vision(preprocessed_image);
}

std::vector<float> Siglip2VisionModel::get_image_features(const Lfm2VlPreprocessor::PreprocessedImage& preprocessed_image) {
    // Forward pass
    size_t last_hidden_state = forward_vision(preprocessed_image);
    std::cout << "[Siglip2VisionModel::get_image_features preprocessed] last_hidden_state node=" << last_hidden_state << std::endl;
    
    // Execute graph
    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    gb->execute();
    std::cout << "[Siglip2VisionModel::get_image_features preprocessed] graph executed" << std::endl;
    
    // Get output
    const auto& output_buf = gb->get_output_buffer(last_hidden_state);
    size_t total_elements = 1;
    for (auto dim : output_buf.shape) {
        total_elements *= dim;
    }
    std::cout << "[Siglip2VisionModel::get_image_features preprocessed] total_elements=" << total_elements << std::endl;
    
    std::vector<float> features(total_elements);
    void* output_data = gb->get_output(last_hidden_state);
    const float* output_ptr = static_cast<const float*>(output_data);
    std::copy(output_ptr, output_ptr + total_elements, features.begin());
    std::cout << "[Siglip2VisionModel::get_image_features preprocessed] copied features" << std::endl;
    
    return features;
}

// Override forward - not used for vision-only model
size_t Siglip2VisionModel::forward(const std::vector<uint32_t>&, bool) {
    // Stub implementation - vision-only model doesn't use token-based forward
    // Returns 0 as a no-op; use forward_vision() for actual vision processing
    return 0;
}

// Stub implementations for Model pure virtual methods (not used for vision-only model)
size_t Siglip2VisionModel::build_attention(CactusGraph*, size_t, uint32_t,
                                           ComputeBackend, bool, size_t) {
    // Stub - vision-only model uses build_vision_attention instead
    return 0;
}

size_t Siglip2VisionModel::build_mlp(CactusGraph*, size_t, uint32_t,
                                     ComputeBackend) const {
    // Stub - vision-only model uses build_vision_mlp instead
    return 0;
}

size_t Siglip2VisionModel::build_transformer_block(CactusGraph*, size_t, uint32_t,
                                                   ComputeBackend, bool, size_t) {
    // Stub - vision-only model uses build_vision_transformer_layer instead
    return 0;
}

}
}
