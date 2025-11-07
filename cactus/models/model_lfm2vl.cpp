#include "model.h"
#include "../graph/graph.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <filesystem>
#include <iostream>

namespace cactus {
namespace engine {

Lfm2VlModel::Lfm2VlModel() : Model() {
    config_.model_type = Config::ModelType::LFM2;
}

Lfm2VlModel::Lfm2VlModel(const Config& config)
        : Model(config),
            vision_tower_(config),
            language_model_(config) {
    // Initialize preprocessor
    Lfm2VlPreprocessor::Config preprocessor_config;
    preprocessor_config.patch_size = static_cast<int>(config.vision_patch_size);
    preprocessor_config.downsample_factor = static_cast<int>(config.downsample_factor);
    preprocessor_config.min_tiles = static_cast<int>(config.min_tiles);
    preprocessor_config.max_tiles = static_cast<int>(config.max_tiles);
    preprocessor_config.use_thumbnail = config.use_thumbnail;
    preprocessor_config.min_image_tokens = static_cast<int>(config.min_image_tokens);
    preprocessor_config.max_image_tokens = static_cast<int>(config.max_image_tokens);
    preprocessor_config.max_num_patches = static_cast<int>(config.max_num_patches);
    preprocessor_config.tile_size = static_cast<int>(config.tile_size);
    preprocessor_config.max_pixels_tolerance = config.max_pixels_tolerance;
    preprocessor_config.do_resize = true;
    preprocessor_config.do_rescale = true;
    preprocessor_config.do_normalize = true;
    preprocessor_config.do_convert_rgb = true;
    preprocessor_config.do_image_splitting = config.do_image_splitting;
    preprocessor_config.rescale_factor = config.rescale_factor;
    preprocessor_config.image_mean[0] = config.image_mean;
    preprocessor_config.image_mean[1] = config.image_mean;
    preprocessor_config.image_mean[2] = config.image_mean;
    preprocessor_config.image_std[0] = config.image_std;
    preprocessor_config.image_std[1] = config.image_std;
    preprocessor_config.image_std[2] = config.image_std;
    
    preprocessor_ = Lfm2VlPreprocessor(preprocessor_config);
}

bool Lfm2VlModel::init(const std::string& model_folder, size_t context_size, const std::string& system_prompt) {
    // Initialize base model
    if (!Model::init(model_folder, context_size, system_prompt, false)) {
        return false;
    }
    std::cout << "[Lfm2VlModel::init] Base model initialized for folder=" << model_folder << std::endl;

    auto* shared_graph = static_cast<CactusGraph*>(graph_handle_);
    if (!shared_graph) {
        throw std::runtime_error("Shared graph was not initialized for Lfm2VlModel");
    }
    
    // Initialize vision tower (loads vision weights from model_folder/vision/)
    std::string vision_folder = model_folder;  // Vision weights should be in same folder with vision_ prefix
    if (!vision_tower_.init(shared_graph, vision_folder, context_size, "", false)) {
        throw std::runtime_error("Failed to initialize vision tower");
    }
    std::cout << "[Lfm2VlModel::init] Vision tower initialized with folder=" << vision_folder << std::endl;
    vision_weights_loaded_ = true;
    std::cout << "[Lfm2VlModel::init] vision_weights_loaded_ set to true" << std::endl;
    
    // Initialize language model (loads text weights from model_folder)
    if (!language_model_.init(shared_graph, model_folder, context_size, system_prompt, false)) {
        throw std::runtime_error("Failed to initialize language model");
    }
    std::cout << "[Lfm2VlModel::init] Language model initialized" << std::endl;
    language_weights_loaded_ = true;
    std::cout << "[Lfm2VlModel::init] language_weights_loaded_ set to true" << std::endl;
    
    return true;
}

void Lfm2VlModel::reset_cache() {
    Model::reset_cache();
    language_model_.reset_cache();
    image_prefill_completed_ = false;
    last_token_count_ = 0;
}

void Lfm2VlModel::load_weights_to_graph(CactusGraph* gb) {
    // Load multimodal projector weights (handle both underscore and condensed filenames)
    namespace fs = std::filesystem;
    fs::path base(model_folder_path_);

    auto resolve_weight = [&](const std::string& primary, const std::string& fallback = "") -> std::string {
        fs::path primary_path = base / primary;
        if (fs::exists(primary_path)) {
            return primary_path.string();
        }
        if (!fallback.empty()) {
            fs::path fallback_path = base / fallback;
            if (fs::exists(fallback_path)) {
                return fallback_path.string();
            }
        }
        return primary_path.string();
    };

    projector_weights_.layer_norm_weight = gb->mmap_weights(resolve_weight("projector_layer_norm.weights"));
    projector_weights_.layer_norm_bias = gb->mmap_weights(resolve_weight("projector_layer_norm.bias.weights"));
    projector_weights_.linear_1_weight = gb->mmap_weights(resolve_weight("projector_linear_1.weights", "projector_linear1.weights"));
    std::cout << "[Lfm2VlModel::load_weights_to_graph] projector_linear_1_weight mapped" << std::endl;
    projector_weights_.linear_1_bias = gb->mmap_weights(resolve_weight("projector_linear_1.bias.weights", "projector_linear1.bias.weights"));
    std::cout << "[Lfm2VlModel::load_weights_to_graph] projector_linear_1_bias mapped" << std::endl;
    projector_weights_.linear_2_weight = gb->mmap_weights(resolve_weight("projector_linear_2.weights", "projector_linear2.weights"));
    std::cout << "[Lfm2VlModel::load_weights_to_graph] projector_linear_2_weight mapped" << std::endl;
    projector_weights_.linear_2_bias = gb->mmap_weights(resolve_weight("projector_linear_2.bias.weights", "projector_linear2.bias.weights"));
    std::cout << "[Lfm2VlModel::load_weights_to_graph] projector_linear_2_bias mapped" << std::endl;
    
    // Note: Vision and language model weights are loaded through their own init() calls
}

size_t Lfm2VlModel::pixel_unshuffle(CactusGraph* gb, size_t hidden_states, 
                                     size_t height, size_t width, size_t channels) {
    // Input shape: [batch=1, height, width, channels]
    // Output shape: [batch=1, height/factor, width/factor, channels*factor^2]
    
    const size_t factor = config_.downsample_factor;
    const size_t new_height = height / factor;
    const size_t new_width = width / factor;
    std::cout << "[Lfm2VlModel::pixel_unshuffle] factor=" << factor << " new_height=" << new_height << " new_width=" << new_width << std::endl;
    
    // Step 1: Reshape [1, height, width, channels] -> [1, height, width/factor, channels*factor]
    size_t step1 = gb->reshape(hidden_states, {1, height, new_width, channels * factor});
    std::cout << "[Lfm2VlModel::pixel_unshuffle] step1 node=" << step1 << std::endl;
    
    // Step 2: Transpose [1, height, width/factor, channels*factor] -> [1, width/factor, height, channels*factor]
    // transposeN with perm [0, 2, 1, 3]
    step1 = gb->transposeN(step1, {0, 2, 1, 3});
    std::cout << "[Lfm2VlModel::pixel_unshuffle] step1 transposed node=" << step1 << std::endl;
    
    // Step 3: Reshape [1, width/factor, height, channels*factor] -> [1, width/factor, height/factor, channels*factor^2]
    size_t step2 = gb->reshape(step1, {1, new_width, new_height, channels * factor * factor});
    std::cout << "[Lfm2VlModel::pixel_unshuffle] step2 node=" << step2 << std::endl;
    
    // Step 4: Transpose [1, width/factor, height/factor, channels*factor^2] -> [1, height/factor, width/factor, channels*factor^2]
    // transpose_n with perm [0, 2, 1, 3]
    size_t result = gb->transposeN(step2, {0, 2, 1, 3});
    std::cout << "[Lfm2VlModel::pixel_unshuffle] result node=" << result << std::endl;
    
    return result;
}

size_t Lfm2VlModel::build_multimodal_projector(CactusGraph* gb, size_t image_features,
                                               size_t tile_h, size_t tile_w, ComputeBackend backend) {
    // image_features shape: [1, tile_h, tile_w, vision_hidden_size]
    const size_t vision_hidden = config_.vision_embed_dim;
    std::cout << "[Lfm2VlModel::build_multimodal_projector] vision_hidden=" << vision_hidden << " tile_h=" << tile_h << " tile_w=" << tile_w << std::endl;
    
    // Ensure features are in FP32 since downstream reshape/transpose ops may not support FP16
    size_t image_features_fp32 = gb->precision_cast(image_features, Precision::FP32);
    // Apply pixel unshuffle
    size_t unshuffled = pixel_unshuffle(gb, image_features_fp32, tile_h, tile_w, vision_hidden);
    std::cout << "[Lfm2VlModel::build_multimodal_projector] unshuffled node=" << unshuffled << std::endl;
    
    // Reshape to 2D for layer operations: [1, new_h, new_w, in_channels] -> [new_h * new_w, in_channels]
    const size_t factor = config_.downsample_factor;
    const size_t new_h = tile_h / factor;
    const size_t new_w = tile_w / factor;
    const size_t in_channels = vision_hidden * factor * factor;
    const size_t seq_len = new_h * new_w;
    std::cout << "[Lfm2VlModel::build_multimodal_projector] factor=" << factor << " new_h=" << new_h << " new_w=" << new_w << " in_channels=" << in_channels << " seq_len=" << seq_len << std::endl;
    
    size_t flattened = gb->reshape(unshuffled, {seq_len, in_channels});
    std::cout << "[Lfm2VlModel::build_multimodal_projector] flattened node=" << flattened << std::endl;
    
    // Layer norm
    size_t normalized = gb->layer_norm(flattened, projector_weights_.layer_norm_weight,
                                      projector_weights_.layer_norm_bias, config_.layer_norm_eps);
    std::cout << "[Lfm2VlModel::build_multimodal_projector] normalized node=" << normalized << std::endl;
    
    // Linear 1
    size_t hidden = gb->matmul(normalized, projector_weights_.linear_1_weight, true, backend);
    std::cout << "[Lfm2VlModel::build_multimodal_projector] hidden node after linear1=" << hidden << std::endl;
    hidden = gb->add(hidden, projector_weights_.linear_1_bias);
    std::cout << "[Lfm2VlModel::build_multimodal_projector] hidden node after bias1=" << hidden << std::endl;
    
    // GELU activation
    hidden = gb->gelu(hidden);
    std::cout << "[Lfm2VlModel::build_multimodal_projector] hidden node after GELU=" << hidden << std::endl;
    
    // Linear 2
    size_t output = gb->matmul(hidden, projector_weights_.linear_2_weight, true, backend);
    std::cout << "[Lfm2VlModel::build_multimodal_projector] output node after linear2=" << output << std::endl;
    output = gb->add(output, projector_weights_.linear_2_bias);
    std::cout << "[Lfm2VlModel::build_multimodal_projector] output node after bias2=" << output << std::endl;
    
    return output;
}

std::vector<Lfm2VlModel::ProjectedTileFeature> Lfm2VlModel::get_image_features(
    CactusGraph* gb,
    const Lfm2VlPreprocessor::PreprocessedImage& preprocessed_image,
    ComputeBackend backend) {
    
    // Use vision tower's forward_vision with our graph - this builds all vision ops into our graph
    size_t vision_output = vision_tower_.forward_vision(gb, preprocessed_image, backend);
    std::cout << "[Lfm2VlModel::get_image_features] vision_output node=" << vision_output << std::endl;
    
    // The vision output is concatenated tiles, we need to split them and process each
    std::vector<ProjectedTileFeature> projected_features;
    projected_features.reserve(preprocessed_image.spatial_shapes.size());
    
    size_t offset = 0;
    for (size_t tile_idx = 0; tile_idx < preprocessed_image.spatial_shapes.size(); ++tile_idx) {
        const auto& shape = preprocessed_image.spatial_shapes[tile_idx];
        const size_t tile_h = shape.first;
        const size_t tile_w = shape.second;
        const size_t tile_tokens = tile_h * tile_w;
        const size_t factor = config_.downsample_factor;
        std::cout << "[Lfm2VlModel::get_image_features] tile_idx=" << tile_idx << " tile_h=" << tile_h << " tile_w=" << tile_w << " tile_tokens=" << tile_tokens << " factor=" << factor << std::endl;

        if (factor == 0) {
            throw std::runtime_error("Downsample factor must be greater than zero");
        }
        if (tile_h % factor != 0 || tile_w % factor != 0) {
            throw std::runtime_error("Tile dimensions must be divisible by downsample factor");
        }
        const size_t new_h = tile_h / factor;
        const size_t new_w = tile_w / factor;
        const size_t projected_tokens = new_h * new_w;
        std::cout << "[Lfm2VlModel::get_image_features] new_h=" << new_h << " new_w=" << new_w << " projected_tokens=" << projected_tokens << std::endl;
        
        // Slice out this tile's features from the concatenated output
        size_t tile_features = gb->slice(vision_output, 0, offset, tile_tokens);
        std::cout << "[Lfm2VlModel::get_image_features] tile_features node=" << tile_features << " offset=" << offset << std::endl;
        offset += tile_tokens;
        std::cout << "[Lfm2VlModel::get_image_features] updated offset=" << offset << std::endl;
        
        // Reshape to [1, tile_h, tile_w, vision_hidden]
        size_t reshaped = gb->reshape(tile_features, {1, tile_h, tile_w, config_.vision_embed_dim});
        std::cout << "[Lfm2VlModel::get_image_features] reshaped node=" << reshaped << std::endl;
        
        // Apply multimodal projector
        size_t projected = build_multimodal_projector(gb, reshaped, tile_h, tile_w, backend);
        std::cout << "[Lfm2VlModel::get_image_features] projected node=" << projected << std::endl;
        
        ProjectedTileFeature feature{};
        feature.node_id = projected;
        feature.token_count = projected_tokens;
        projected_features.push_back(feature);
        std::cout << "[Lfm2VlModel::get_image_features] pushed ProjectedTileFeature with node_id=" << feature.node_id << " token_count=" << feature.token_count << std::endl;
    }
    
    return projected_features;
}

Lfm2VlModel::MergedEmbeddingResult Lfm2VlModel::merge_image_text_embeddings(
    CactusGraph* gb,
    const std::vector<uint32_t>& tokens,
    const std::vector<std::vector<ProjectedTileFeature>>& image_embedding_nodes,
    std::vector<TextEmbeddingInput>& text_embedding_inputs) {

    text_embedding_inputs.clear();

    Tokenizer* tokenizer = language_model_.get_tokenizer();
    if (!tokenizer) {
        throw std::runtime_error("Tokenizer must be initialized before merging embeddings");
    }

    const uint32_t image_token_id = tokenizer->get_image_token_id();

    auto get_token_id = [tokenizer](const std::string& token_text) -> uint32_t {
        auto encoded = tokenizer->encode(token_text);
        if (encoded.size() != 1) {
            std::cout << "[Lfm2VlModel::merge_image_text_embeddings] get_token_id for " << token_text << " encoded size=" << encoded.size() << std::endl;
            throw std::runtime_error("Expected single token encoding for " + token_text);
        }
        return encoded[0];
    };

    const uint32_t image_start_id = get_token_id("<|image_start|>");
    const uint32_t image_end_id = get_token_id("<|image_end|>");

    std::cout << "[Lfm2VlModel::merge_image_text_embeddings_special_tokens] image_start_id=" << image_start_id << " image_end_id=" << image_end_id << " image_token_id=" << image_token_id << std::endl;

    std::vector<size_t> sequence_nodes;
    sequence_nodes.reserve(tokens.size() + image_embedding_nodes.size());

    std::vector<uint32_t> current_segment;
    current_segment.reserve(tokens.size());

    size_t total_seq_len = 0;
    std::cout << "[Lfm2VlModel::merge_image_text_embeddings] starting merge with tokens size=" << tokens.size() << std::endl;
    std::cout << "[LFM2_DEBUG] merge_start tokens=" << tokens.size() << " image_embeddings=" << image_embedding_nodes.size() << std::endl;

    auto flush_segment = [&](void) {
        if (current_segment.empty()) {
            std::cout << "[Lfm2VlModel::merge_image_text_embeddings] flush_segment skipped empty segment" << std::endl;
            return;
        }

        const size_t segment_len = current_segment.size();
        std::cout << "[Lfm2VlModel::merge_image_text_embeddings] flush_segment segment_len=" << segment_len << std::endl;

        TextEmbeddingInput segment;
        segment.tokens.swap(current_segment);
        segment.input_node = gb->input({segment.tokens.size()}, Precision::FP32);
        std::cout << "[Lfm2VlModel::merge_image_text_embeddings] created input node=" << segment.input_node << std::endl;

        const auto& embedding_buffer = gb->get_output_buffer(language_model_.embedding_node_id_);
        std::cout << "[LFM2-VL] embedding tensor shape: [";
        for (size_t i = 0; i < embedding_buffer.shape.size(); ++i) {
            std::cout << embedding_buffer.shape[i];
            if (i + 1 < embedding_buffer.shape.size()) {
                std::cout << ", ";
            }
        }
        std::cout << "] segment_len=" << segment.tokens.size() << std::endl;

        size_t embedding_node = gb->embedding(language_model_.embedding_node_id_, segment.input_node);
        std::cout << "[Lfm2VlModel::merge_image_text_embeddings] embedding_node=" << embedding_node << std::endl;

        text_embedding_inputs.push_back(std::move(segment));
        std::cout << "[Lfm2VlModel::merge_image_text_embeddings] text_embedding_inputs size=" << text_embedding_inputs.size() << std::endl;
        sequence_nodes.push_back(embedding_node);
        std::cout << "[Lfm2VlModel::merge_image_text_embeddings] sequence_nodes size=" << sequence_nodes.size() << std::endl;
        total_seq_len += segment_len;
        std::cout << "[Lfm2VlModel::merge_image_text_embeddings] total_seq_len updated=" << total_seq_len << std::endl;

        current_segment.clear();
        std::cout << "[Lfm2VlModel::merge_image_text_embeddings] current_segment cleared" << std::endl;
    };

    size_t token_index = 0;
    size_t image_index = 0;

    while (token_index < tokens.size()) {
        uint32_t token_id = tokens[token_index];
        std::cout << "[Lfm2VlModel::merge_image_text_embeddings] processing token_index=" << token_index << " token_id=" << token_id << std::endl;

        if (token_id == image_start_id) {
            flush_segment();

            if (image_index >= image_embedding_nodes.size()) {
                std::cout << "[LFM2_DEBUG] missing_image_features image_index=" << image_index << " available=" << image_embedding_nodes.size() << std::endl;
                throw std::runtime_error("Encountered <|image_start|> without corresponding image features");
            }

            current_segment.push_back(token_id);
            ++token_index;
            std::cout << "[Lfm2VlModel::merge_image_text_embeddings] advanced past image_start, token_index=" << token_index << " image_index=" << image_index << std::endl;

            const auto& tiles = image_embedding_nodes[image_index];
            size_t tile_index = 0;
            std::cout << "[Lfm2VlModel::merge_image_text_embeddings] tiles size=" << tiles.size() << std::endl;
            std::cout << "[LFM2_DEBUG] image_block_start image_index=" << image_index << " tiles=" << tiles.size() << std::endl;

            while (token_index < tokens.size()) {
                
                uint32_t inner_token = tokens[token_index];
                std::cout << "[Lfm2VlModel::merge_image_text_embeddings] inner_token at index=" << token_index << " value=" << inner_token << std::endl;

                if (inner_token == image_token_id) {
                    
                    flush_segment();
                    std::cout << "[LFM2_DEBUG] image_placeholder token_index=" << token_index << " tile_index=" << tile_index << " tiles=" << tiles.size() << std::endl;

                    if (tile_index >= tiles.size()) {
                        std::cout << "[LFM2_DEBUG] tile_overrun tile_index=" << tile_index << " tiles_size=" << tiles.size() << std::endl;
                        throw std::runtime_error("More <image> placeholders than projected tile features");
                    }

                    const auto& tile = tiles[tile_index++];
                    std::cout << "[Lfm2VlModel::merge_image_text_embeddings] inserting tile node=" << tile.node_id << " token_count=" << tile.token_count << std::endl;
                    std::cout << "[LFM2_DEBUG] insert_tile image_index=" << image_index << " tile_index=" << (tile_index - 1) << " token_count=" << tile.token_count << std::endl;
                    sequence_nodes.push_back(tile.node_id);
                    std::cout << "[Lfm2VlModel::merge_image_text_embeddings] sequence_nodes size after tile=" << sequence_nodes.size() << std::endl;
                    total_seq_len += tile.token_count;
                    std::cout << "[Lfm2VlModel::merge_image_text_embeddings] total_seq_len after tile=" << total_seq_len << std::endl;

                    for (size_t count = 0; count < tile.token_count; ++count) {
                        if (token_index >= tokens.size()) {
                            throw std::runtime_error("Insufficient <image> tokens for projected features");
                        }
                        if (tokens[token_index] != image_token_id) {
                            throw std::runtime_error("Unexpected token encountered within image feature span");
                        }
                        ++token_index;
                        std::cout << "[Lfm2VlModel::merge_image_text_embeddings] consumed image token, token_index=" << token_index << std::endl;
                    }

                    continue;
                }

                current_segment.push_back(inner_token);
                std::cout << "[Lfm2VlModel::merge_image_text_embeddings] appended inner token to current_segment, size=" << current_segment.size() << std::endl;
                ++token_index;
                std::cout << "[Lfm2VlModel::merge_image_text_embeddings] incremented token_index to " << token_index << std::endl;

                if (inner_token == image_end_id) {
                    flush_segment();
                    break;
                }
            }

            if (tile_index != tiles.size()) {
                std::cout << "[LFM2_DEBUG] tile_mismatch image_index=" << image_index
                          << " consumed_tiles=" << tile_index
                          << " available_tiles=" << tiles.size() << std::endl;
                if (tile_index < tiles.size()) {
                    for (size_t remaining = tile_index; remaining < tiles.size(); ++remaining) {
                        const auto& remaining_tile = tiles[remaining];
                        std::cout << "[LFM2_DEBUG] tile_unused image_index=" << image_index
                                  << " tile_index=" << remaining
                                  << " node_id=" << remaining_tile.node_id
                                  << " token_count=" << remaining_tile.token_count << std::endl;
                    }
                }
                throw std::runtime_error("Unused projected tile features remain after processing image block");
            }

            ++image_index;
            std::cout << "[Lfm2VlModel::merge_image_text_embeddings] completed image block, image_index=" << image_index << std::endl;
        } else {
            current_segment.push_back(token_id);
            std::cout << "[Lfm2VlModel::merge_image_text_embeddings] appended token to current_segment, size=" << current_segment.size() << std::endl;
            ++token_index;
            std::cout << "[Lfm2VlModel::merge_image_text_embeddings] token_index incremented to " << token_index << std::endl;
        }
    }

    flush_segment();
    std::cout << "[Lfm2VlModel::merge_image_text_embeddings] finished main loop, total_seq_len=" << total_seq_len << std::endl;

    if (image_index != image_embedding_nodes.size()) {
        throw std::runtime_error("Not all image features were consumed while merging embeddings");
    }

    if (sequence_nodes.empty()) {
        throw std::runtime_error("Failed to build embedding sequence from provided tokens");
    }

    size_t merged = sequence_nodes[0];
    std::cout << "[Lfm2VlModel::merge_image_text_embeddings] initial merged node=" << merged << std::endl;
    for (size_t idx = 1; idx < sequence_nodes.size(); ++idx) {
        merged = gb->concat(merged, sequence_nodes[idx], 0);
        std::cout << "[Lfm2VlModel::merge_image_text_embeddings] concatenated node idx=" << idx << " merged node now=" << merged << std::endl;
    }

    return MergedEmbeddingResult{merged, total_seq_len};
}

// Stub implementations for pure virtual methods
size_t Lfm2VlModel::build_attention(CactusGraph*, size_t, uint32_t, ComputeBackend, bool, size_t) {
    throw std::runtime_error("build_attention should not be called directly on Lfm2VlModel");
}

size_t Lfm2VlModel::build_mlp(CactusGraph*, size_t, uint32_t, ComputeBackend) const {
    throw std::runtime_error("build_mlp should not be called directly on Lfm2VlModel");
}

size_t Lfm2VlModel::build_transformer_block(CactusGraph*, size_t, uint32_t, ComputeBackend, bool, size_t) {
    throw std::runtime_error("build_transformer_block should not be called directly on Lfm2VlModel");
}

size_t Lfm2VlModel::forward(const std::vector<uint32_t>& tokens, bool use_cache) {
    // Text-only forward (no images)
    if (!initialized_ || !graph_handle_) {
        throw std::runtime_error("Model not initialized - call init() first");
    }
    std::cout << "[Lfm2VlModel::forward] text-only forward called, tokens size=" << tokens.size() << " use_cache=" << use_cache << std::endl;

    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    gb->soft_reset();
    std::cout << "[Lfm2VlModel::forward] graph soft reset" << std::endl;

    auto backend = config_.default_backend == Config::Backend::CPU
        ? ComputeBackend::CPU
        : ComputeBackend::NPU;
    std::cout << "[Lfm2VlModel::forward] backend selected=" << static_cast<int>(backend) << std::endl;

    // Use language model's forward with our graph
    return language_model_.forward(gb, tokens, backend, use_cache);
}

Lfm2VlModel::ForwardImageResult Lfm2VlModel::forward_images(
    CactusGraph* gb,
    const std::vector<uint32_t>& tokens,
    const std::vector<std::string>& image_paths,
    ComputeBackend backend,
    bool use_cache) {
    if (!gb) {
        throw std::runtime_error("Graph must be initialized before forwarding");
    }
    if (tokens.empty()) {
        throw std::runtime_error("Token sequence cannot be empty");
    }

    // Process images through vision tower and projector (all in our single graph)
    std::vector<std::vector<ProjectedTileFeature>> all_image_embeddings;
    all_image_embeddings.reserve(image_paths.size());
    for (const auto& image_path : image_paths) {
        // Preprocess image
        auto preprocessed = preprocessor_.preprocess_from_file(image_path);
        
        // Get image features (vision tower + projector) - builds into our graph
        auto image_features = get_image_features(gb, preprocessed, backend);
        
        all_image_embeddings.push_back(std::move(image_features));
    }

    std::vector<TextEmbeddingInput> text_embedding_inputs;
    text_embedding_inputs.reserve(tokens.size() / 4 + 1); // heuristic to avoid frequent realloc
    std::cout << "[Lfm2VlModel::forward_images] text_embedding_inputs reserved size" << std::endl;

    // Merge image and text embeddings
    auto merged_embeddings = merge_image_text_embeddings(gb, tokens, all_image_embeddings, text_embedding_inputs);
    std::cout << "[Lfm2VlModel::forward_images] merged_embeddings node=" << merged_embeddings.node_id << " seq_len=" << merged_embeddings.seq_len << std::endl;

    if (merged_embeddings.seq_len == 0) {
        throw std::runtime_error("Merged embedding sequence length cannot be zero");
    }

    for (const auto& embedding_input : text_embedding_inputs) {
        if (embedding_input.tokens.empty()) {
            continue;
        }

        std::vector<float> segment_data(embedding_input.tokens.size());
        for (size_t i = 0; i < embedding_input.tokens.size(); ++i) {
            segment_data[i] = static_cast<float>(embedding_input.tokens[i]);
        }
        gb->set_input(embedding_input.input_node, segment_data.data(), Precision::FP32);
    }
    
    // TODO: Pass inputs_embeds through language model instead of tokens
    // For now, we'll use the language model's forward with tokens
    // This needs to be updated to accept embeddings directly or we need to build
    // the language model layers manually with inputs_embeds as the starting point

    size_t final_hidden = language_model_.forward(gb, merged_embeddings.node_id, merged_embeddings.seq_len, backend, use_cache);
    std::cout << "[Lfm2VlModel::forward_images] final_hidden node=" << final_hidden << std::endl;

    return ForwardImageResult{final_hidden, merged_embeddings.seq_len};
}

uint32_t Lfm2VlModel::generate_with_images(
    const std::vector<uint32_t>& tokens,
    const std::vector<std::string>& image_paths,
    float temperature,
    float top_p,
    size_t top_k,
    const std::string& profile_file) {

    if (!initialized_ || !graph_handle_) {
        throw std::runtime_error("Model not initialized - call init() first");
    }
    std::cout << "[Lfm2VlModel::generate_with_images] called with tokens size=" << tokens.size() << " image_paths size=" << image_paths.size() << std::endl;

    if (image_paths.empty()) {
        std::cout << "[Lfm2VlModel::generate_with_images] no images provided, delegating to language model" << std::endl;
        image_prefill_completed_ = false;
        last_token_count_ = tokens.size();
        return language_model_.generate(tokens, temperature, top_p, top_k, profile_file);
    }

    if (temperature < 0) {
        temperature = config_.default_temperature;
    }
    if (top_p < 0) {
        top_p = config_.default_top_p;
    }
    if (top_k == 0) {
        top_k = config_.default_top_k;
    }

    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    gb->soft_reset();

    auto backend = config_.default_backend == Config::Backend::CPU
        ? ComputeBackend::CPU
        : ComputeBackend::NPU;
    std::cout << "[Lfm2VlModel::generate_with_images] backend=" << static_cast<int>(backend) << std::endl;

    bool cache_empty = language_model_.is_cache_empty();
    bool need_prefill = cache_empty || !image_prefill_completed_;

    if (!need_prefill && tokens.size() <= last_token_count_) {
        std::cout << "[Lfm2VlModel::generate_with_images] token sequence rewind detected, resetting caches" << std::endl;
        reset_cache();
        need_prefill = true;
    }

    size_t seq_len_for_updates = 0;
    size_t final_hidden_node = 0;

    if (need_prefill) {
        auto forward_result = forward_images(gb, tokens, image_paths, backend, true);
        std::cout << "[Lfm2VlModel::generate_with_images] performed image prefill final_hidden=" << forward_result.final_hidden_node << " seq_len=" << forward_result.seq_len << std::endl;
        final_hidden_node = forward_result.final_hidden_node;
        seq_len_for_updates = forward_result.seq_len;
        image_prefill_completed_ = true;
        last_token_count_ = tokens.size();
    } else {
        size_t delta = tokens.size() - last_token_count_;
        if (delta > tokens.size()) {
            delta = tokens.size();
        }
        if (delta == 0) {
            if (tokens.empty()) {
                throw std::runtime_error("Token sequence cannot be empty for cached decode step");
            }
            delta = 1;
            std::cout << "[Lfm2VlModel::generate_with_images] delta tokens was zero, reusing last token" << std::endl;
        }
        std::vector<uint32_t> incremental_tokens(tokens.end() - delta, tokens.end());
        std::cout << "[Lfm2VlModel::generate_with_images] incremental decode tokens=" << incremental_tokens.size() << std::endl;
        final_hidden_node = language_model_.forward(gb, incremental_tokens, backend, true);
        seq_len_for_updates = incremental_tokens.size();
        last_token_count_ = tokens.size();
    }

    // Use language model's output head for sampling
    auto logits_node_id = gb->matmul(final_hidden_node, language_model_.output_weight_node_id_, true, backend);
    std::cout << "[Lfm2VlModel::generate_with_images] logits_node_id=" << logits_node_id << std::endl;
    auto sampled_token_id = gb->sample(logits_node_id, temperature, top_p, top_k);
    std::cout << "[Lfm2VlModel::generate_with_images] sampled_token_id node=" << sampled_token_id << std::endl;

    if (!profile_file.empty()) {
        gb->execute(profile_file);
        std::cout << "[Lfm2VlModel::generate_with_images] graph executed with profile file" << std::endl;
    } else {
        gb->execute();
        std::cout << "[Lfm2VlModel::generate_with_images] graph executed without profile" << std::endl;
    }

    language_model_.post_execute_updates(gb, seq_len_for_updates);
    std::cout << "[Lfm2VlModel::generate_with_images] post_execute_updates called" << std::endl;
    language_model_.update_kv_cache(gb, seq_len_for_updates);
    std::cout << "[Lfm2VlModel::generate_with_images] update_kv_cache called" << std::endl;

    auto* output_ptr = gb->get_output(sampled_token_id);
    return *static_cast<uint32_t*>(output_ptr);
}

}
}