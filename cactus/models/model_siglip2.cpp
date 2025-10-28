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
    // Get preprocessed patches
    // Note: pixel_values are already downsampled, so use tokens_per_tile * num_tiles
    int num_tiles = preprocessed_image.image_rows * preprocessed_image.image_cols;
    int num_tokens = preprocessed_image.tokens_per_tile * num_tiles + preprocessed_image.thumbnail_tokens;
    int patch_dim = config_.vision_patch_size * config_.vision_patch_size * 3;
    
    // Validate pixel_values size
    size_t expected_size = static_cast<size_t>(num_tokens) * static_cast<size_t>(patch_dim);
    if (preprocessed_image.pixel_values.size() != expected_size) {
        throw std::runtime_error(
            "Pixel values size mismatch: expected " + std::to_string(expected_size) + 
            " (tokens=" + std::to_string(num_tokens) + " * patch_dim=" + std::to_string(patch_dim) + ")" +
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
    
    // Create input node for pixel values (already in patch format from preprocessor)
    size_t patches_input = gb->input({static_cast<size_t>(num_tokens), static_cast<size_t>(patch_dim)}, Precision::FP32);
    gb->set_input(patches_input, preprocessed_image.pixel_values.data(), Precision::FP32);
    
    // Apply patch embedding (linear projection)
    size_t reshaped_weight = gb->reshape(
        vision_weight_nodes_.patch_embedding_weight,
        {static_cast<size_t>(config_.vision_embed_dim), static_cast<size_t>(patch_dim)}
    );
    size_t patch_embeds = gb->matmul(patches_input, reshaped_weight, true, backend);
    patch_embeds = gb->add(patch_embeds, vision_weight_nodes_.patch_embedding_bias);
    
    // Bilinear interpolation of positional embeddings
    // Position embeddings are stored as (num_positions, embed_dim) where num_positions = src_height * src_width
    // For LFM2-VL with tiling: interpolate for a single tile, then broadcast to all tiles
    
    // Get source position embeddings data
    const auto& pos_embed_buf = gb->get_output_buffer(vision_weight_nodes_.position_embedding);
    int total_pos_embeds = static_cast<int>(pos_embed_buf.shape[0]);
    int embed_dim = static_cast<int>(pos_embed_buf.shape[1]);
    const float* pos_embed_data = static_cast<const float*>(gb->get_output(vision_weight_nodes_.position_embedding));
    
    // Check for nan/inf in position embeddings
    for (int i = 0; i < std::min(100, total_pos_embeds * embed_dim); ++i) {
        if (std::isnan(pos_embed_data[i]) || std::isinf(pos_embed_data[i])) {
            throw std::runtime_error(
                "Invalid value in position embeddings at index " + std::to_string(i) + 
                ": " + std::to_string(pos_embed_data[i])
            );
        }
    }
    
    // Source grid is square (standard for ViT-style position embeddings)
    int src_height = static_cast<int>(std::sqrt(total_pos_embeds));
    int src_width = src_height;
    
    // Target dimensions: tokens per tile after downsampling
    // Each tile has the same positional embeddings applied
    int tokens_per_tile = preprocessed_image.tokens_per_tile;
    int dst_height = static_cast<int>(std::sqrt(tokens_per_tile));
    int dst_width = dst_height;
    
    // Perform bilinear interpolation for a single tile using vectors
    std::vector<float> tile_pos_embeds(tokens_per_tile * embed_dim);
    
    // Compute scale factors for coordinate mapping
    float scale_h = (src_height > 1 && dst_height > 1) 
                    ? static_cast<float>(src_height - 1) / static_cast<float>(dst_height - 1)
                    : 0.0f;
    float scale_w = (src_width > 1 && dst_width > 1)
                    ? static_cast<float>(src_width - 1) / static_cast<float>(dst_width - 1)
                    : 0.0f;
    
    // Interpolate position embeddings for a single tile
    for (int dst_y = 0; dst_y < dst_height; ++dst_y) {
        for (int dst_x = 0; dst_x < dst_width; ++dst_x) {
            // Map to source coordinates (fractional)
            float src_y_float = dst_y * scale_h;
            float src_x_float = dst_x * scale_w;
            
            // Get 4 nearest neighbors (integer coordinates)
            int y0 = static_cast<int>(std::floor(src_y_float));
            int x0 = static_cast<int>(std::floor(src_x_float));
            int y1 = std::min(y0 + 1, src_height - 1);
            int x1 = std::min(x0 + 1, src_width - 1);
            
            // Compute fractional parts (how far between integer coordinates)
            float dy = src_y_float - y0;
            float dx = src_x_float - x0;
            
            // Compute bilinear weights
            float w00 = (1.0f - dx) * (1.0f - dy);  // Top-left
            float w01 = dx * (1.0f - dy);            // Top-right
            float w10 = (1.0f - dx) * dy;            // Bottom-left
            float w11 = dx * dy;                     // Bottom-right
            
            // Source indices in flat array (row-major: idx = y * width + x)
            int idx00 = (y0 * src_width + x0) * embed_dim;
            int idx01 = (y0 * src_width + x1) * embed_dim;
            int idx10 = (y1 * src_width + x0) * embed_dim;
            int idx11 = (y1 * src_width + x1) * embed_dim;
            
            // Output index
            int out_idx = (dst_y * dst_width + dst_x) * embed_dim;
            
            // Interpolate all embedding dimensions
            for (int d = 0; d < embed_dim; ++d) {
                tile_pos_embeds[out_idx + d] = 
                    pos_embed_data[idx00 + d] * w00 +
                    pos_embed_data[idx01 + d] * w01 +
                    pos_embed_data[idx10 + d] * w10 +
                    pos_embed_data[idx11 + d] * w11;
            }
        }
    }
    
    // Create graph node for single tile position embeddings
    size_t tile_pos_embeds_node = gb->input(
        {static_cast<size_t>(tokens_per_tile), static_cast<size_t>(embed_dim)}, 
        Precision::FP32
    );
    gb->set_input(tile_pos_embeds_node, tile_pos_embeds.data(), Precision::FP32);
    
    // Reshape patch embeddings to separate tiles: (num_tokens, embed_dim) -> (num_tiles, tokens_per_tile, embed_dim)
    size_t reshaped_patches = gb->reshape(
        patch_embeds,
        {static_cast<size_t>(num_tiles), static_cast<size_t>(tokens_per_tile), static_cast<size_t>(embed_dim)}
    );
    
    // Add position embeddings (broadcasts across tile dimension)
    // (num_tiles, tokens_per_tile, embed_dim) + (tokens_per_tile, embed_dim) -> (num_tiles, tokens_per_tile, embed_dim)
    size_t embeddings_with_pos = gb->add(reshaped_patches, tile_pos_embeds_node);
    
    // Reshape back to (num_tokens, embed_dim)
    size_t embeddings = gb->reshape(
        embeddings_with_pos,
        {static_cast<size_t>(num_tokens), static_cast<size_t>(embed_dim)}
    );
    
    return embeddings;
}

size_t Siglip2VisionModel::build_vision_attention(CactusGraph* gb, size_t hidden_states, 
                                                  uint32_t layer_idx, ComputeBackend backend) {
    const auto& layer = vision_weight_nodes_.vision_layers[layer_idx];
    
    // Project to Q, K, V
    size_t q = gb->matmul(hidden_states, layer.attn_q_weight, true, backend);
    q = gb->add(q, layer.attn_q_bias);
    
    size_t k = gb->matmul(hidden_states, layer.attn_k_weight, true, backend);
    k = gb->add(k, layer.attn_k_bias);
    
    size_t v = gb->matmul(hidden_states, layer.attn_v_weight, true, backend);
    v = gb->add(v, layer.attn_v_bias);

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
    
    // Reshape back
    size_t attn_2d = gb->reshape(attn_output, {seq_len, num_heads * head_dim});
    
    // Output projection
    size_t output = gb->matmul(attn_2d, layer.attn_output_weight, true, backend);
    output = gb->add(output, layer.attn_output_bias);
    
    return output;
}

size_t Siglip2VisionModel::build_vision_mlp(CactusGraph* gb, size_t hidden_states, 
                                           uint32_t layer_idx, ComputeBackend backend) {
    const auto& layer = vision_weight_nodes_.vision_layers[layer_idx];
    
    // FC1
    size_t fc1_output = gb->matmul(hidden_states, layer.mlp_fc1_weight, true, backend);
    fc1_output = gb->add(fc1_output, layer.mlp_fc1_bias);
    
    // Activation (GELU for SigLip2)
    size_t activated = gb->gelu(fc1_output);
    
    // FC2
    size_t fc2_output = gb->matmul(activated, layer.mlp_fc2_weight, true, backend);
    fc2_output = gb->add(fc2_output, layer.mlp_fc2_bias);
    
    return fc2_output;
}

size_t Siglip2VisionModel::build_vision_transformer_layer(CactusGraph* gb, size_t hidden_states, 
                                                          uint32_t layer_idx, ComputeBackend backend) {
    const auto& layer = vision_weight_nodes_.vision_layers[layer_idx];
    
    // Pre-norm architecture: LayerNorm -> Attention -> Residual
    size_t residual = hidden_states;
    size_t normalized = gb->layer_norm(hidden_states, layer.layer_norm1_weight, 
                                      layer.layer_norm1_bias, config_.layer_norm_eps);
    size_t attn_output = build_vision_attention(gb, normalized, layer_idx, backend);
    hidden_states = gb->add(residual, attn_output);
    
    // Pre-norm architecture: LayerNorm -> MLP -> Residual
    residual = hidden_states;
    normalized = gb->layer_norm(hidden_states, layer.layer_norm2_weight, 
                               layer.layer_norm2_bias, config_.layer_norm_eps);
    size_t mlp_output = build_vision_mlp(gb, normalized, layer_idx, backend);
    hidden_states = gb->add(residual, mlp_output);
    
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
