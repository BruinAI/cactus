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

    vision_weight_nodes_.vision_layers.resize(config_.vision_num_layers);

    std::string base = model_folder_path_ + "/";
    
    vision_weight_nodes_.vision_proj_weight = gb->mmap_weights(base + "vision_patch_embedding.weights");
    try {
        vision_weight_nodes_.vision_proj_bias = gb->mmap_weights(base + "vision_patch_embedding.bias.weights");
    } catch (...) {
        vision_weight_nodes_.vision_proj_bias = 0;
    }
    
    vision_weight_nodes_.vision_position_embedding = gb->mmap_weights(base + "vision_position_embedding.weights");
    
    auto try_mmap_optional = [&](const std::vector<std::string>& candidates) -> size_t {
        for (const auto& c : candidates) {
            try {
                return gb->mmap_weights(c);
            } catch (...) {
                continue;
            }
        }
        return 0;
    };
    
    vision_weight_nodes_.vision_post_layernorm_weight = try_mmap_optional({
        base + "vision_post_layernorm.weight.weights",
        base + "vision_post_layernorm.weight",
        base + "vision_post_layernorm.weights"
    });
    vision_weight_nodes_.vision_post_layernorm_bias = try_mmap_optional({
        base + "vision_post_layernorm.bias.weights",
        base + "vision_post_layernorm.bias"
    });
    
    vision_weight_nodes_.connector_proj_weight = try_mmap_optional({
        base + "connector_proj.weights",
        base + "connector_proj.weight",
        base + "modality_projection.weights"
    });

    for (uint32_t i = 0; i < vision_weight_nodes_.vision_layers.size(); ++i) {
        auto& vw = vision_weight_nodes_.vision_layers[i];
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
        vw.attn_q_bias = try_mmap({prefix + "self_attn_q.bias.weights", prefix + "self_attn_q.bias"});
        vw.attn_k_bias = try_mmap({prefix + "self_attn_k.bias.weights", prefix + "self_attn_k.bias"});
        vw.attn_v_bias = try_mmap({prefix + "self_attn_v.bias.weights", prefix + "self_attn_v.bias"});
        vw.attn_output_bias = try_mmap({prefix + "self_attn_out.bias.weights", prefix + "self_attn_out.bias"});

        vw.layer_norm1_weight = try_mmap({prefix + "layer_norm1.weights", prefix + "layer_norm1.weight.weights", prefix + "layer_norm1.weight"});
        vw.layer_norm1_bias = try_mmap({prefix + "layer_norm1.bias.weights", prefix + "layer_norm1.bias.weight", prefix + "layer_norm1.bias"});
        vw.layer_norm2_weight = try_mmap({prefix + "layer_norm2.weights", prefix + "layer_norm2.weight.weights", prefix + "layer_norm2.weight"});
        vw.layer_norm2_bias = try_mmap({prefix + "layer_norm2.bias.weights", prefix + "layer_norm2.bias.weight", prefix + "layer_norm2.bias"});

        vw.mlp_fc1_weight = try_mmap({prefix + "ffn_fc1.weights", prefix + "ffn_fc1.weight", prefix + "mlp_fc1.weights"});
        vw.mlp_fc1_bias = try_mmap({prefix + "ffn_fc1.bias.weights", prefix + "ffn_fc1.bias.weight", prefix + "mlp_fc1.bias"});
        vw.mlp_fc2_weight = try_mmap({prefix + "ffn_fc2.weights", prefix + "ffn_fc2.weight", prefix + "mlp_fc2.weights"});
        vw.mlp_fc2_bias = try_mmap({prefix + "ffn_fc2.bias.weights", prefix + "ffn_fc2.bias.weight", prefix + "mlp_fc2.bias"});
    }
}

size_t SmolVLMModel::build_vision_embeddings(CactusGraph* gb, const std::vector<ImageBatch>& images,
                                   ComputeBackend backend) {
    if (images.empty()) return 0;

    std::vector<size_t> per_image_nodes;
    
    for (const auto &ib : images) {
        int patch_size = config_.vision_patch_size;
        int num_patches_h = ib.height / patch_size;
        int num_patches_w = ib.width / patch_size;
        int num_patches = num_patches_h * num_patches_w;

        size_t patch_dim = static_cast<size_t>(ib.channels) * patch_size * patch_size;
        size_t patches_input = gb->input({static_cast<size_t>(num_patches), patch_dim}, Precision::FP32);
        std::vector<float> patch_data(static_cast<size_t>(num_patches) * patch_dim);
        const int H = static_cast<int>(ib.height);
        const int W = static_cast<int>(ib.width);
        const size_t plane = static_cast<size_t>(W) * H;
        size_t out_index = 0;
        for (int ph = 0; ph < num_patches_h; ++ph) {
            for (int pw = 0; pw < num_patches_w; ++pw) {
                for (int c = 0; c < static_cast<int>(ib.channels); ++c) {
                    const size_t c_offset = static_cast<size_t>(c) * plane;
                    for (int dy = 0; dy < patch_size; ++dy) {
                        int y = ph * patch_size + dy;
                        for (int dx = 0; dx < patch_size; ++dx) {
                            int x = pw * patch_size + dx;
                            size_t src_idx = c_offset + static_cast<size_t>(y) * W + static_cast<size_t>(x);
                            patch_data[out_index++] = ib.data[src_idx];
                        }
                    }
                }
            }
        }
        gb->set_input(patches_input, patch_data.data(), Precision::FP32);
        
        size_t reshaped_w = gb->reshape(
            vision_weight_nodes_.vision_proj_weight,
            {static_cast<size_t>(config_.vision_embed_dim), patch_dim}
        );
        size_t patch_embeds = gb->matmul(patches_input, reshaped_w, true, backend);
        if (vision_weight_nodes_.vision_proj_bias != 0) {
            patch_embeds = gb->add(patch_embeds, vision_weight_nodes_.vision_proj_bias);
        }
        
        size_t embeddings = patch_embeds;
        if (vision_weight_nodes_.vision_position_embedding != 0) {
            int patch_size = config_.vision_patch_size;
            int num_patches_h = ib.height / patch_size;
            int num_patches_w = ib.width / patch_size;
            int base_side = static_cast<int>(config_.vision_image_size / config_.vision_patch_size);
            if (base_side <= 0) base_side = std::max(num_patches_h, num_patches_w);

            std::vector<int> bucket_h(num_patches_h);
            std::vector<int> bucket_w(num_patches_w);
            std::vector<float> boundaries(base_side - 1);
            for (int i = 0; i < base_side - 1; ++i) boundaries[i] = static_cast<float>(i + 1) / static_cast<float>(base_side);
            auto bucketize = [&](int n, std::vector<int>& out) {
                float step = 1.0f / static_cast<float>(n);
                for (int i = 0; i < n; ++i) {
                    float coord = std::min(1.0f - 1e-6f, i * step);
                    int b = 0;
                    while (b < base_side - 1 && coord >= boundaries[b]) ++b;
                    out[i] = b;
                }
            };
            bucketize(num_patches_h, bucket_h);
            bucketize(num_patches_w, bucket_w);

            std::vector<float> pos_ids(num_patches_h * num_patches_w);
            for (int h = 0; h < num_patches_h; ++h) {
                for (int w = 0; w < num_patches_w; ++w) {
                    int idx = h * num_patches_w + w;
                    int pos = bucket_h[h] * base_side + bucket_w[w];
                    pos_ids[idx] = static_cast<float>(pos);
                }
            }

            size_t pos_input = gb->input({static_cast<size_t>(num_patches_h * num_patches_w)}, Precision::FP32);
            gb->set_input(pos_input, pos_ids.data(), Precision::FP32);
            size_t pos_embed = gb->gather(vision_weight_nodes_.vision_position_embedding, pos_input);
            embeddings = gb->add(patch_embeds, pos_embed);
        }
        
        size_t hidden_states = embeddings;
        for (uint32_t layer_idx = 0; layer_idx < config_.vision_num_layers; ++layer_idx) {
            hidden_states = build_vision_transformer_layer(gb, hidden_states, layer_idx, backend);
        }
        
        if (vision_weight_nodes_.vision_post_layernorm_weight != 0 && 
            vision_weight_nodes_.vision_post_layernorm_bias != 0) {
            hidden_states = gb->layer_norm(hidden_states, 
                                          vision_weight_nodes_.vision_post_layernorm_weight,
                                          vision_weight_nodes_.vision_post_layernorm_bias,
                                          config_.layer_norm_eps);
        }
        
        int scale_factor = config_.pixel_shuffle_factor;
        size_t shuffled = pixel_shuffle(gb, hidden_states, scale_factor);
        
        size_t projected = shuffled;
        if (vision_weight_nodes_.connector_proj_weight != 0) {
            projected = gb->matmul(shuffled, vision_weight_nodes_.connector_proj_weight, true, backend);
        }
        
        per_image_nodes.push_back(projected);
    }

    if (per_image_nodes.empty()) return 0;

    size_t vision_seq = per_image_nodes[0];
    for (size_t i = 1; i < per_image_nodes.size(); ++i) {
        vision_seq = gb->concat(vision_seq, per_image_nodes[i], 0);
    }

    return vision_seq;
}

size_t SmolVLMModel::build_combined_input(CactusGraph* graph, size_t vision_embeds, const std::vector<uint32_t>& tokens,
                                ComputeBackend backend, uint32_t& prefix_len) {
    if (!initialized_ || !graph_handle_) {
        throw std::runtime_error("Model not initialized - call init() first");
    }

    (void)backend;

    size_t token_count = tokens.size();
    size_t token_input_node = graph->input({token_count}, Precision::FP32);

    std::vector<float> token_data(token_count);
    for (size_t i = 0; i < token_count; ++i) token_data[i] = static_cast<float>(tokens[i]);
    graph->set_input(token_input_node, token_data.data(), Precision::FP32);

    size_t token_embeddings = graph->embedding(embedding_node_id_, token_input_node);

    if (vision_embeds == 0) {
        prefix_len = 0;
        return token_embeddings;
    }

    uint32_t IMAGE_TOKEN_ID = tokenizer_ ? tokenizer_->get_image_token_id() : 49190;
    std::vector<size_t> image_token_positions;
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (tokens[i] == IMAGE_TOKEN_ID) {
            image_token_positions.push_back(i);
        }
    }
    
    if (image_token_positions.empty()) {
        prefix_len = 0;
        return token_embeddings;
    }
    
    uint32_t patch_size = config_.image_seq_len;
    size_t num_image_tokens = image_token_positions.size();
    if (num_image_tokens % patch_size != 0) {
        throw std::runtime_error("Number of <image> tokens (" + std::to_string(num_image_tokens) + 
                               ") must be divisible by image_seq_len (" + std::to_string(patch_size) + ")");
    }
    
    const auto& vision_buf = graph->get_output_buffer(vision_embeds);
    if (vision_buf.shape.size() < 2) {
        throw std::runtime_error("Vision embeddings must be 2D [num_tokens, embed_dim]");
    }
    size_t num_vision_tokens = vision_buf.shape[0];
    
    size_t num_blocks = num_image_tokens / patch_size;
    if (num_vision_tokens != num_blocks * patch_size) {
        throw std::runtime_error("Vision tokens (" + std::to_string(num_vision_tokens) + 
                               ") doesn't match expected (" + std::to_string(num_blocks * patch_size) + 
                               ") for " + std::to_string(num_blocks) + " image blocks");
    }

    size_t result_node = 0;
    bool has_result = false;
    size_t text_cursor = 0;
    size_t image_token_cursor = 0;

    auto append_text_segment = [&](size_t start, size_t end) {
        if (end <= start) return;
        std::vector<uint32_t> seg(tokens.begin() + start, tokens.begin() + end);
        std::vector<float> seg_data(seg.size());
        for (size_t i = 0; i < seg.size(); ++i) seg_data[i] = static_cast<float>(seg[i]);
        size_t seg_input = graph->input({seg.size()}, Precision::FP32);
        graph->set_input(seg_input, seg_data.data(), Precision::FP32);
        size_t seg_embeds = graph->embedding(embedding_node_id_, seg_input);
        if (!has_result) { result_node = seg_embeds; has_result = true; }
        else { result_node = graph->concat(result_node, seg_embeds, 0); }
    };

    for (size_t i = 0; i < token_count; ++i) {
        if (tokens[i] == IMAGE_TOKEN_ID) {
            if (i > text_cursor) {
                append_text_segment(text_cursor, i);
            }
            
            size_t block_idx = image_token_cursor / patch_size;
            size_t local_idx = image_token_cursor % patch_size;
            
            size_t global_idx = block_idx * patch_size + local_idx;
            std::vector<float> idx_data = {static_cast<float>(global_idx)};
            size_t idx_node = graph->input({1}, Precision::FP32);
            graph->set_input(idx_node, idx_data.data(), Precision::FP32);
            size_t vision_embed = graph->gather(vision_embeds, idx_node);
            
            if (!has_result) { result_node = vision_embed; has_result = true; }
            else { result_node = graph->concat(result_node, vision_embed, 0); }
            
            image_token_cursor++;
            text_cursor = i + 1;
        }
    }
    
    if (text_cursor < token_count) {
        append_text_segment(text_cursor, token_count);
    }

    prefix_len = static_cast<uint32_t>(token_count);
    return has_result ? result_node : token_embeddings;
}

size_t SmolVLMModel::pixel_shuffle(CactusGraph* gb, size_t input, int scale_factor) {
    const auto& in = gb->get_output_buffer(input);
    if (in.shape.size() != 2) {
        throw std::runtime_error("pixel_shuffle requires 2D input [num_patches, embed_dim]");
    }

    size_t S = in.shape[0];
    size_t D = in.shape[1];
    int H = static_cast<int>(std::sqrt(static_cast<double>(S)));
    int W = H;
    if (H * W != static_cast<int>(S)) {
        throw std::runtime_error("pixel_shuffle requires square number of patches");
    }
    int s = scale_factor;
    if (H % s != 0 || W % s != 0) {
        throw std::runtime_error("pixel_shuffle scale must divide height/width");
    }
    int Hs = H / s;
    int Ws = W / s;

    size_t x = gb->reshape(input, {static_cast<size_t>(H), static_cast<size_t>(W), D});
    size_t x1 = gb->reshape(x, {static_cast<size_t>(H), static_cast<size_t>(Ws), static_cast<size_t>(D * s)});
    size_t x2 = gb->transposeN(x1, {1, 0, 2});
    size_t x3 = gb->reshape(x2, {static_cast<size_t>(Ws), static_cast<size_t>(Hs), static_cast<size_t>(D * s * s)});
    size_t x4 = gb->transposeN(x3, {1, 0, 2});
    size_t out = gb->reshape(x4, {static_cast<size_t>(Hs * Ws), static_cast<size_t>(D * s * s)});
    return out;
}

size_t SmolVLMModel::build_vision_attention(CactusGraph* gb, size_t hidden_states, uint32_t layer_idx,
                                            ComputeBackend backend) {
    const auto& layer = vision_weight_nodes_.vision_layers[layer_idx];
    
    size_t q = gb->matmul(hidden_states, layer.attn_q_weight, true, backend);
    if (layer.attn_q_bias != 0) q = gb->add(q, layer.attn_q_bias);
    size_t k = gb->matmul(hidden_states, layer.attn_k_weight, true, backend);
    if (layer.attn_k_bias != 0) k = gb->add(k, layer.attn_k_bias);
    size_t v = gb->matmul(hidden_states, layer.attn_v_weight, true, backend);
    if (layer.attn_v_bias != 0) v = gb->add(v, layer.attn_v_bias);

    const size_t Hh = static_cast<size_t>(config_.vision_attention_heads);
    const size_t Dh = static_cast<size_t>(config_.vision_embed_dim / config_.vision_attention_heads);
    const auto& q_buf = gb->get_output_buffer(q);
    size_t Slen = q_buf.shape[0];
    size_t q4 = gb->reshape(q, {1, Slen, Hh, Dh});
    size_t k4 = gb->reshape(k, {1, Slen, Hh, Dh});
    size_t v4 = gb->reshape(v, {1, Slen, Hh, Dh});

    float scale = 1.0f / std::sqrt(static_cast<float>(Dh));
    size_t attn4 = gb->attention(q4, k4, v4, scale, true, backend);

    size_t out2 = gb->reshape(attn4, {Slen, Hh * Dh});
    size_t output = gb->matmul(out2, layer.attn_output_weight, true, backend);
    if (layer.attn_output_bias != 0) output = gb->add(output, layer.attn_output_bias);
    
    return output;
}

size_t SmolVLMModel::build_vision_transformer_layer(CactusGraph* gb, size_t hidden_states, uint32_t layer_idx,
                                                    ComputeBackend backend) {
    const auto& layer = vision_weight_nodes_.vision_layers[layer_idx];
    
    size_t residual = hidden_states;
    size_t normalized = gb->layer_norm(hidden_states, layer.layer_norm1_weight, layer.layer_norm1_bias, 
                                      config_.layer_norm_eps);
    size_t attn_output = build_vision_attention(gb, normalized, layer_idx, backend);
    hidden_states = gb->add(residual, attn_output);
    
    residual = hidden_states;
    normalized = gb->layer_norm(hidden_states, layer.layer_norm2_weight, layer.layer_norm2_bias,
                               config_.layer_norm_eps);
    
    size_t fc1_output = gb->matmul(normalized, layer.mlp_fc1_weight, true, backend);
    if (layer.mlp_fc1_bias != 0) {
        fc1_output = gb->add(fc1_output, layer.mlp_fc1_bias);
    }
    size_t gelu_output = gb->gelu(fc1_output);
    size_t fc2_output = gb->matmul(gelu_output, layer.mlp_fc2_weight, true, backend);
    if (layer.mlp_fc2_bias != 0) {
        fc2_output = gb->add(fc2_output, layer.mlp_fc2_bias);
    }
    
    hidden_states = gb->add(residual, fc2_output);
    
    return hidden_states;
}

size_t SmolVLMModel::forward_mm(const std::vector<uint32_t>& tokens,const std::vector<ImageBatch>& images, bool use_cache) {
    if (!initialized_ || !graph_handle_) {
        throw std::runtime_error("Model not initialized - call init() first");
    }

    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    gb->soft_reset();

    auto backend = config_.default_backend == Config::Backend::CPU ? ComputeBackend::CPU : ComputeBackend::NPU;

    size_t vision_seq = 0;
    if (!images.empty()) {
        vision_seq = build_vision_embeddings(gb, images, backend);
    }

    uint32_t prefix_len = 0;
    size_t combined = build_combined_input(gb, vision_seq, tokens, backend, prefix_len);

    size_t hidden = combined;
    size_t position_offset = use_cache ? kv_cache_.get_total_seq_len() : 0;

    for (uint32_t layer_idx = 0; layer_idx < config_.num_layers; ++layer_idx) {
        hidden = build_transformer_block(gb, hidden, layer_idx, backend, use_cache, position_offset);
    }

    auto final_hidden = gb->rms_norm(hidden, weight_nodes_.output_norm_weight, config_.layer_norm_eps);

    return final_hidden;
}

uint32_t SmolVLMModel::generate_with_images(const std::vector<uint32_t>& tokens, const std::vector<ImageBatch>& images,
                                           float temperature, float top_p, size_t top_k, const std::string& profile_file) {
    (void)temperature; (void)top_p; (void)top_k; (void)profile_file;
    size_t final_hidden_node = forward_mm(tokens, images, true);

    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    auto backend = config_.default_backend == Config::Backend::CPU
        ? ComputeBackend::CPU
        : ComputeBackend::NPU;

    auto logits_node_id = gb->matmul(final_hidden_node, output_weight_node_id_, true, backend);
    auto sampled_token_id = gb->sample(logits_node_id, config_.default_temperature, config_.default_top_p, config_.default_top_k);

    gb->execute();
    update_kv_cache(gb, tokens.size());

    auto* output_ptr = gb->get_output(sampled_token_id);
    return *static_cast<uint32_t*>(output_ptr);
}

}
}