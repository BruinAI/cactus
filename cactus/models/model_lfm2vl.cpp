#include "model.h"
#include "../graph/graph.h"
#include "../ffi/stb_image.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>

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
    if (!Model::init(model_folder, context_size, system_prompt)) {
        return false;
    }
    
    // Initialize vision tower (loads vision weights from model_folder/vision/)
    std::string vision_folder = model_folder;  // Vision weights should be in same folder with vision_ prefix
    if (!vision_tower_.init(vision_folder, context_size, "")) {
        throw std::runtime_error("Failed to initialize vision tower");
    }
    vision_weights_loaded_ = true;
    
    // Initialize language model (loads text weights from model_folder)
    if (!language_model_.init(model_folder, context_size, system_prompt)) {
        throw std::runtime_error("Failed to initialize language model");
    }
    language_weights_loaded_ = true;
    
    return true;
}

void Lfm2VlModel::load_weights_to_graph(CactusGraph* gb) {
    // Load multimodal projector weights
    std::string base = model_folder_path_ + "/";
    
    projector_weights_.layer_norm_weight = gb->mmap_weights(base + "projector_layer_norm.weights");
    projector_weights_.layer_norm_bias = gb->mmap_weights(base + "projector_layer_norm.bias.weights");
    projector_weights_.linear_1_weight = gb->mmap_weights(base + "projector_linear_1.weights");
    projector_weights_.linear_1_bias = gb->mmap_weights(base + "projector_linear_1.bias.weights");
    projector_weights_.linear_2_weight = gb->mmap_weights(base + "projector_linear_2.weights");
    projector_weights_.linear_2_bias = gb->mmap_weights(base + "projector_linear_2.bias.weights");
    
    // Note: Vision and language model weights are loaded through their own init() calls
}

size_t Lfm2VlModel::pixel_unshuffle(CactusGraph* gb, size_t hidden_states, 
                                     size_t height, size_t width, size_t channels) {
    // Input shape: [batch=1, height, width, channels]
    // Output shape: [batch=1, height/factor, width/factor, channels*factor^2]
    
    const size_t factor = config_.downsample_factor;
    const size_t new_height = height / factor;
    const size_t new_width = width / factor;
    
    // Step 1: Reshape [1, height, width, channels] -> [1, height, width/factor, channels*factor]
    size_t step1 = gb->reshape(hidden_states, {1, height, new_width, channels * factor});
    
    // Step 2: Transpose [1, height, width/factor, channels*factor] -> [1, width/factor, height, channels*factor]
    // transposeN with perm [0, 2, 1, 3]
    step1 = gb->transposeN(step1, {0, 2, 1, 3});
    
    // Step 3: Reshape [1, width/factor, height, channels*factor] -> [1, width/factor, height/factor, channels*factor^2]
    size_t step2 = gb->reshape(step1, {1, new_width, new_height, channels * factor * factor});
    
    // Step 4: Transpose [1, width/factor, height/factor, channels*factor^2] -> [1, height/factor, width/factor, channels*factor^2]
    // transpose_n with perm [0, 2, 1, 3]
    size_t result = gb->transposeN(step2, {0, 2, 1, 3});
    
    return result;
}

size_t Lfm2VlModel::build_multimodal_projector(CactusGraph* gb, size_t image_features,
                                               size_t tile_h, size_t tile_w, ComputeBackend backend) {
    // image_features shape: [1, tile_h, tile_w, vision_hidden_size]
    const size_t vision_hidden = config_.vision_embed_dim;
    
    // Apply pixel unshuffle
    size_t unshuffled = pixel_unshuffle(gb, image_features, tile_h, tile_w, vision_hidden);
    
    // Reshape to 2D for layer operations: [1, new_h, new_w, in_channels] -> [new_h * new_w, in_channels]
    const size_t factor = config_.downsample_factor;
    const size_t new_h = tile_h / factor;
    const size_t new_w = tile_w / factor;
    const size_t in_channels = vision_hidden * factor * factor;
    const size_t seq_len = new_h * new_w;
    
    size_t flattened = gb->reshape(unshuffled, {seq_len, in_channels});
    
    // Layer norm
    size_t normalized = gb->layer_norm(flattened, projector_weights_.layer_norm_weight,
                                      projector_weights_.layer_norm_bias, config_.layer_norm_eps);
    
    // Linear 1
    size_t hidden = gb->matmul(normalized, projector_weights_.linear_1_weight, true, backend);
    hidden = gb->add(hidden, projector_weights_.linear_1_bias);
    
    // GELU activation
    hidden = gb->gelu(hidden);
    
    // Linear 2
    size_t output = gb->matmul(hidden, projector_weights_.linear_2_weight, true, backend);
    output = gb->add(output, projector_weights_.linear_2_bias);
    
    return output;
}

std::vector<Lfm2VlModel::ProjectedTileFeature> Lfm2VlModel::get_image_features(
    CactusGraph* gb,
    const Lfm2VlPreprocessor::PreprocessedImage& preprocessed_image,
    ComputeBackend backend) {
    
    // Use vision tower's forward_vision with our graph - this builds all vision ops into our graph
    size_t vision_output = vision_tower_.forward_vision(gb, preprocessed_image, backend);
    
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

        if (factor == 0) {
            throw std::runtime_error("Downsample factor must be greater than zero");
        }
        if (tile_h % factor != 0 || tile_w % factor != 0) {
            throw std::runtime_error("Tile dimensions must be divisible by downsample factor");
        }
        const size_t new_h = tile_h / factor;
        const size_t new_w = tile_w / factor;
        const size_t projected_tokens = new_h * new_w;
        
        // Slice out this tile's features from the concatenated output
        size_t tile_features = gb->slice(vision_output, 0, offset, tile_tokens);
        offset += tile_tokens;
        
        // Reshape to [1, tile_h, tile_w, vision_hidden]
        size_t reshaped = gb->reshape(tile_features, {1, tile_h, tile_w, config_.vision_embed_dim});
        
        // Apply multimodal projector
        size_t projected = build_multimodal_projector(gb, reshaped, tile_h, tile_w, backend);
        
        ProjectedTileFeature feature{};
        feature.node_id = projected;
        feature.token_count = projected_tokens;
        projected_features.push_back(feature);
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
            throw std::runtime_error("Expected single token encoding for " + token_text);
        }
        return encoded[0];
    };

    const uint32_t image_start_id = get_token_id("<|image_start|>");
    const uint32_t image_end_id = get_token_id("<|image_end|>");

    std::vector<size_t> sequence_nodes;
    sequence_nodes.reserve(tokens.size() + image_embedding_nodes.size());

    std::vector<uint32_t> current_segment;
    current_segment.reserve(tokens.size());

    size_t total_seq_len = 0;

    auto flush_segment = [&](void) {
        if (current_segment.empty()) {
            return;
        }

        const size_t segment_len = current_segment.size();

        TextEmbeddingInput segment;
        segment.tokens.swap(current_segment);
        segment.input_node = gb->input({segment.tokens.size()}, Precision::FP32);
        size_t embedding_node = gb->embedding(language_model_.embedding_node_id_, segment.input_node);

        text_embedding_inputs.push_back(std::move(segment));
        sequence_nodes.push_back(embedding_node);
        total_seq_len += segment_len;

        current_segment.clear();
    };

    size_t token_index = 0;
    size_t image_index = 0;

    while (token_index < tokens.size()) {
        uint32_t token_id = tokens[token_index];

        if (token_id == image_start_id) {
            flush_segment();

            if (image_index >= image_embedding_nodes.size()) {
                throw std::runtime_error("Encountered <|image_start|> without corresponding image features");
            }

            current_segment.push_back(token_id);
            ++token_index;

            const auto& tiles = image_embedding_nodes[image_index];
            size_t tile_index = 0;

            while (token_index < tokens.size()) {
                uint32_t inner_token = tokens[token_index];

                if (inner_token == image_token_id) {
                    flush_segment();

                    if (tile_index >= tiles.size()) {
                        throw std::runtime_error("More <image> placeholders than projected tile features");
                    }

                    const auto& tile = tiles[tile_index++];
                    sequence_nodes.push_back(tile.node_id);
                    total_seq_len += tile.token_count;

                    for (size_t count = 0; count < tile.token_count; ++count) {
                        if (token_index >= tokens.size()) {
                            throw std::runtime_error("Insufficient <image> tokens for projected features");
                        }
                        if (tokens[token_index] != image_token_id) {
                            throw std::runtime_error("Unexpected token encountered within image feature span");
                        }
                        ++token_index;
                    }

                    continue;
                }

                current_segment.push_back(inner_token);
                ++token_index;

                if (inner_token == image_end_id) {
                    flush_segment();
                    break;
                }
            }

            if (tile_index != tiles.size()) {
                throw std::runtime_error("Unused projected tile features remain after processing image block");
            }

            ++image_index;
        } else {
            current_segment.push_back(token_id);
            ++token_index;
        }
    }

    flush_segment();

    if (image_index != image_embedding_nodes.size()) {
        throw std::runtime_error("Not all image features were consumed while merging embeddings");
    }

    if (sequence_nodes.empty()) {
        throw std::runtime_error("Failed to build embedding sequence from provided tokens");
    }

    size_t merged = sequence_nodes[0];
    for (size_t idx = 1; idx < sequence_nodes.size(); ++idx) {
        merged = gb->concat(merged, sequence_nodes[idx], 0);
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

    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    gb->soft_reset();

    auto backend = config_.default_backend == Config::Backend::CPU
        ? ComputeBackend::CPU
        : ComputeBackend::NPU;

    // Use language model's forward with our graph
    return language_model_.forward(gb, tokens, backend, use_cache);
}

size_t Lfm2VlModel::generate_with_images(
    const std::vector<uint32_t>& tokens,
    const std::vector<std::string>& image_paths,
    bool use_cache) {
    
    if (!initialized_ || !graph_handle_) {
        throw std::runtime_error("Model not initialized - call init() first");
    }

    if (tokens.empty()) {
        throw std::runtime_error("Token sequence cannot be empty");
    }

    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    gb->soft_reset();

    auto backend = config_.default_backend == Config::Backend::CPU
        ? ComputeBackend::CPU
        : ComputeBackend::NPU;

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

    // Merge image and text embeddings
    auto merged_embeddings = merge_image_text_embeddings(gb, tokens, all_image_embeddings, text_embedding_inputs);

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

    return final_hidden;
}

}
}