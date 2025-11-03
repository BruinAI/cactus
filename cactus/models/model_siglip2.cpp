#include "model.h"
#include "../graph/graph.h"
#include <cmath>
#include <stdexcept>

namespace cactus {
namespace engine {

Siglip2VisionModel::Siglip2VisionModel() : Model() {
    config_.model_type = Config::ModelType::SMOLVLM;  // We'll need a SIGLIP2 type
}

Siglip2VisionModel::Siglip2VisionModel(const Config& cfg) : Model(cfg) {
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
}

void Siglip2VisionModel::load_weights_to_graph(CactusGraph* gb) {
    vision_weight_nodes_.vision_layers.resize(config_.vision_num_layers);

    std::string base = model_folder_path_ + "/";
    
    // Patch embedding weights (using short names from convert_hf.py)
    vision_weight_nodes_.patch_embedding_weight = gb->mmap_weights(base + "vision_patch_embedding.weights");
    vision_weight_nodes_.patch_embedding_bias = gb->mmap_weights(base + "vision_patch_embedding.bias.weights");
    
    // Position embedding
    vision_weight_nodes_.position_embedding = gb->mmap_weights(base + "vision_position_embedding.weights");
    
    // Post layer norm
    vision_weight_nodes_.post_layernorm_weight = gb->mmap_weights(base + "vision_post_layernorm.weights");
    vision_weight_nodes_.post_layernorm_bias = gb->mmap_weights(base + "vision_post_layernorm.bias.weights");
    
    // Load encoder layers
    for (uint32_t i = 0; i < vision_weight_nodes_.vision_layers.size(); ++i) {
        auto& layer = vision_weight_nodes_.vision_layers[i];
        std::string prefix = base + "vision_layer_" + std::to_string(i) + "_";

        // Self attention weights
        layer.attn_q_weight = gb->mmap_weights(prefix + "self_attn_q.weights");
        layer.attn_q_bias = gb->mmap_weights(prefix + "self_attn_q.bias.weights");
        layer.attn_k_weight = gb->mmap_weights(prefix + "self_attn_k.weights");
        layer.attn_k_bias = gb->mmap_weights(prefix + "self_attn_k.bias.weights");
        layer.attn_v_weight = gb->mmap_weights(prefix + "self_attn_v.weights");
        layer.attn_v_bias = gb->mmap_weights(prefix + "self_attn_v.bias.weights");
        layer.attn_output_weight = gb->mmap_weights(prefix + "self_attn_out.weights");
        layer.attn_output_bias = gb->mmap_weights(prefix + "self_attn_out.bias.weights");

        // Layer norms
        layer.layer_norm1_weight = gb->mmap_weights(prefix + "layer_norm1.weights");
        layer.layer_norm1_bias = gb->mmap_weights(prefix + "layer_norm1.bias.weights");
        layer.layer_norm2_weight = gb->mmap_weights(prefix + "layer_norm2.weights");
        layer.layer_norm2_bias = gb->mmap_weights(prefix + "layer_norm2.bias.weights");

        // MLP weights
        layer.mlp_fc1_weight = gb->mmap_weights(prefix + "ffn_fc1.weights");
        layer.mlp_fc1_bias = gb->mmap_weights(prefix + "ffn_fc1.bias.weights");
        layer.mlp_fc2_weight = gb->mmap_weights(prefix + "ffn_fc2.weights");
        layer.mlp_fc2_bias = gb->mmap_weights(prefix + "ffn_fc2.bias.weights");
    }
    
    // Note: Pooling head weights not loaded - vision_use_head is false in LFM2-VL
    // The model outputs last_hidden_state directly without pooling
}

size_t Siglip2VisionModel::build_vision_embeddings(CactusGraph* gb, 
                                                   const Lfm2VlPreprocessor::PreprocessedImage& preprocessed_image,
                                   ComputeBackend backend) {
    // Use the preprocessed image structure fully
    const int num_tiles = preprocessed_image.num_tiles;
    const int max_patches = preprocessed_image.max_patches_per_tile;
    const int patch_dim = preprocessed_image.patch_dim;
    const int total_patches = num_tiles * max_patches;
    
    // Validate pixel_values size
    size_t expected_size = static_cast<size_t>(total_patches) * static_cast<size_t>(patch_dim);
    if (preprocessed_image.pixel_values.size() != expected_size) {
        throw std::runtime_error(
            "Pixel values size mismatch: expected " + std::to_string(expected_size) + 
            " (tiles=" + std::to_string(num_tiles) + " * max_patches=" + std::to_string(max_patches) + 
            " * patch_dim=" + std::to_string(patch_dim) + ")" +
            " but got " + std::to_string(preprocessed_image.pixel_values.size())
        );
    }
    
    // Check for nan/inf in input data
    for (size_t i = 0; i < std::min(size_t(100), preprocessed_image.pixel_values.size()); ++i) {
        float val = preprocessed_image.pixel_values[i];
        if (std::isnan(val) || std::isinf(val)) {
            throw std::runtime_error(
                "Invalid value in pixel_values at index " + std::to_string(i) + 
                ": " + std::to_string(val)
            );
        }
    }
    
    // Create input node using the pre-computed shape
    // Shape: (num_tiles * max_patches, patch_dim) - flattened for matmul
    size_t patches_input_fp32 = gb->input(
        {static_cast<size_t>(total_patches), static_cast<size_t>(patch_dim)}, 
        Precision::FP32
    );
    gb->set_input(patches_input_fp32, preprocessed_image.pixel_values.data(), Precision::FP32);
    capture_debug_node(0, "vision_patches_input_fp32", patches_input_fp32);
    
    // Cast to FP16 to match weight precision
    size_t patches_input = gb->precision_cast(patches_input_fp32, Precision::FP16);
    capture_debug_node(0, "vision_patches_input", patches_input);
    
    // Apply patch embedding (linear projection)
    size_t reshaped_weight = gb->reshape(
        vision_weight_nodes_.patch_embedding_weight,
        {static_cast<size_t>(config_.vision_embed_dim), static_cast<size_t>(patch_dim)}
    );
    capture_debug_node(0, "vision_reshaped_weight", reshaped_weight);
    size_t patch_embeds = gb->matmul(patches_input, reshaped_weight, true, backend);
    capture_debug_node(0, "vision_patch_embeds", patch_embeds);
    size_t added_patch_embeds = gb->add(patch_embeds, vision_weight_nodes_.patch_embedding_bias);
    capture_debug_node(0, "vision_added_patch_embeds", added_patch_embeds);

    // Process position embeddings per-tile (matching HF's resize_positional_embeddings)
    std::vector<size_t> tile_pos_embeddings;
    tile_pos_embeddings.reserve(num_tiles);
    
    for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        auto [tile_h, tile_w] = preprocessed_image.spatial_shapes[tile_idx];
        int actual_patches = tile_h * tile_w;
        
        // Interpolate position embedding to this tile's specific size
        // Base position embedding is stored as sqrt(num_patches) x sqrt(num_patches) grid
        size_t tile_pos = gb->bilinear_interpolation(
            vision_weight_nodes_.position_embedding,
            static_cast<size_t>(tile_h),
            static_cast<size_t>(tile_w)
        ); // -> (tile_h * tile_w, embed_dim)
        
        capture_debug_node(tile_idx, "vision_tile_pos_" + std::to_string(tile_idx), tile_pos);
        
        // Pad to max_patches if needed (for thumbnail which is typically smaller)
        size_t padded_pos;
        if (actual_patches < max_patches) {
            // Pad with zeros - these will be masked out by pixel_attention_mask
            int pad_size = max_patches - actual_patches;
            std::vector<float> zero_data(pad_size * config_.vision_embed_dim, 0.0f);
            size_t zero_pad = gb->input(
                {static_cast<size_t>(pad_size), static_cast<size_t>(config_.vision_embed_dim)}, 
                Precision::FP32
            );
            gb->set_input(zero_pad, zero_data.data(), Precision::FP32);
            
            padded_pos = gb->concat(tile_pos, zero_pad, /*axis=*/0);
            capture_debug_node(tile_idx, "vision_padded_pos_" + std::to_string(tile_idx), padded_pos);
        } else {
            padded_pos = tile_pos;  // Already exact size (1024 patches)
        }
        
        tile_pos_embeddings.push_back(padded_pos);
    }
    
    // Concatenate all tile position embeddings
    // Chain concat operations for all tiles
    // Shape: (num_tiles * max_patches, embed_dim)
    size_t all_pos = tile_pos_embeddings[0];
    for (size_t i = 1; i < tile_pos_embeddings.size(); ++i) {
        all_pos = gb->concat(all_pos, tile_pos_embeddings[i], /*axis=*/0);
    }
    capture_debug_node(0, "vision_all_pos_embeddings", all_pos);
    
    // Cast to match patch embeddings precision and add
    size_t pos_cast = gb->precision_cast(all_pos, Precision::FP16);
    size_t embeddings = gb->add(added_patch_embeds, pos_cast);
    capture_debug_node(0, "vision_final_embeddings", embeddings);
    
    int total_valid_patches = 0;
    for (int i = 0; i < num_tiles; ++i) {
        auto [h, w] = preprocessed_image.spatial_shapes[i];
        total_valid_patches += h * w;
    }

    // Slice off padding (if any)
    if (total_valid_patches < total_patches) {
        embeddings = gb->slice(embeddings, /*axis=*/0, /*start=*/0, /*end=*/total_valid_patches);
    }

    return embeddings;
}

size_t Siglip2VisionModel::build_vision_attention(CactusGraph* gb, size_t hidden_states, 
                                                  uint32_t layer_idx, ComputeBackend backend) {
    const auto& layer = vision_weight_nodes_.vision_layers[layer_idx];
    
    // Project to Q, K, V
    size_t q = gb->matmul(hidden_states, layer.attn_q_weight, true, backend);
    q = gb->add(q, layer.attn_q_bias);
    capture_debug_node(layer_idx, "vision_attn_q", q);
    
    size_t k = gb->matmul(hidden_states, layer.attn_k_weight, true, backend);
    k = gb->add(k, layer.attn_k_bias);
    capture_debug_node(layer_idx, "vision_attn_k", k);
    
    size_t v = gb->matmul(hidden_states, layer.attn_v_weight, true, backend);
    v = gb->add(v, layer.attn_v_bias);
    capture_debug_node(layer_idx, "vision_attn_v", v);

    // Reshape for multi-head attention
    const size_t num_heads = static_cast<size_t>(config_.vision_attention_heads);
    const size_t head_dim = static_cast<size_t>(config_.vision_embed_dim / config_.vision_attention_heads);
    const auto& q_buf = gb->get_output_buffer(q);
    size_t seq_len = q_buf.shape[0];
    
    size_t q_4d = gb->reshape(q, {1, seq_len, num_heads, head_dim});
    size_t k_4d = gb->reshape(k, {1, seq_len, num_heads, head_dim});
    size_t v_4d = gb->reshape(v, {1, seq_len, num_heads, head_dim});
    
    // Scaled dot-product attention
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    size_t attn_output = gb->attention(q_4d, k_4d, v_4d, scale, false, backend);
    capture_debug_node(layer_idx, "vision_attn_scores", attn_output);
    
    // Reshape back
    size_t attn_2d = gb->reshape(attn_output, {seq_len, num_heads * head_dim});
    capture_debug_node(layer_idx, "vision_attn_2d", attn_2d);  // <-- ADD THIS

    // Output projection
    size_t output = gb->matmul(attn_2d, layer.attn_output_weight, true, backend);
    output = gb->add(output, layer.attn_output_bias);
    capture_debug_node(layer_idx, "vision_attn_output", output);
    
    return output;
}

size_t Siglip2VisionModel::build_vision_mlp(CactusGraph* gb, size_t hidden_states, 
                                           uint32_t layer_idx, ComputeBackend backend) {
    const auto& layer = vision_weight_nodes_.vision_layers[layer_idx];
    
    // FC1
    size_t fc1_output = gb->matmul(hidden_states, layer.mlp_fc1_weight, true, backend);
    fc1_output = gb->add(fc1_output, layer.mlp_fc1_bias);
    capture_debug_node(layer_idx, "vision_mlp_fc1", fc1_output);
    
    // Activation (GELU for SigLip2)
    size_t activated = gb->gelu(fc1_output);
    capture_debug_node(layer_idx, "vision_mlp_gelu", activated);
    
    // FC2
    size_t fc2_output = gb->matmul(activated, layer.mlp_fc2_weight, true, backend);
    fc2_output = gb->add(fc2_output, layer.mlp_fc2_bias);
    capture_debug_node(layer_idx, "vision_mlp_fc2", fc2_output);
    
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
    size_t attn_output = build_vision_attention(gb, normalized, layer_idx, backend);
    hidden_states = gb->add(residual, attn_output);
    capture_debug_node(layer_idx, "vision_after_attn", hidden_states);
    
    // Pre-norm architecture: LayerNorm -> MLP -> Residual
    residual = hidden_states;
    normalized = gb->layer_norm(hidden_states, layer.layer_norm2_weight, 
                               layer.layer_norm2_bias, config_.layer_norm_eps);
    capture_debug_node(layer_idx, "vision_mlp_norm", normalized);
    size_t mlp_output = build_vision_mlp(gb, normalized, layer_idx, backend);
    hidden_states = gb->add(residual, mlp_output);
    capture_debug_node(layer_idx, "vision_after_mlp", hidden_states);
    
    return hidden_states;
}

size_t Siglip2VisionModel::forward_vision(CactusGraph* gb, 
                                          const Lfm2VlPreprocessor::PreprocessedImage& preprocessed_image,
                                          ComputeBackend backend) {
    // Build vision embeddings from preprocessed patches
    size_t embeddings = build_vision_embeddings(gb, preprocessed_image, backend);
    
    // Pass through transformer layers
    size_t hidden_states = embeddings;
    for (uint32_t layer_idx = 0; layer_idx < config_.vision_num_layers; ++layer_idx) {
        hidden_states = build_vision_transformer_layer(gb, hidden_states, layer_idx, backend);
    }
    
    // Post layer norm
    hidden_states = gb->layer_norm(hidden_states, 
                                   vision_weight_nodes_.post_layernorm_weight,
                                   vision_weight_nodes_.post_layernorm_bias,
                                   config_.layer_norm_eps);
    capture_debug_node(config_.vision_num_layers, "vision_post_norm", hidden_states);
    
    // Return last_hidden_state (no pooling head - vision_use_head: false)
    return hidden_states;
}

size_t Siglip2VisionModel::forward_vision(const Lfm2VlPreprocessor::PreprocessedImage& preprocessed_image) {
    if (!initialized_ || !graph_handle_) {
        throw std::runtime_error("Model not initialized - call init() first");
    }

    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    gb->soft_reset();

    auto backend = config_.default_backend == Config::Backend::CPU ? ComputeBackend::CPU : ComputeBackend::NPU;

    return forward_vision(gb, preprocessed_image, backend);
}

std::vector<float> Siglip2VisionModel::get_image_features(const std::string& image_path) {
    // Preprocess image
    auto preprocessed = preprocessor_.preprocess_from_file(image_path);
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
    
    // Execute graph
    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    gb->execute();
    
    // Get output
    const auto& output_buf = gb->get_output_buffer(last_hidden_state);
    size_t total_elements = 1;
    for (auto dim : output_buf.shape) {
        total_elements *= dim;
    }
    
    std::vector<float> features(total_elements);
    void* output_data = gb->get_output(last_hidden_state);
    const float* output_ptr = static_cast<const float*>(output_data);
    std::copy(output_ptr, output_ptr + total_elements, features.begin());
    
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
