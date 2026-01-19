#include "model.h"
#include "../graph/graph.h"
#include "../npu/npu.h"
#include "../kernel/kernel.h"
#include <cmath>
#include <stdexcept>
#include <set>
#include <iostream>
#include <algorithm>

namespace cactus {
namespace engine {

MoonshineModel::MoonshineModel() : Model() {}

MoonshineModel::MoonshineModel(const Config& config) : Model(config) {
    weight_nodes_.encoder_layers.resize(config.num_encoder_layers);
    weight_nodes_.decoder_layers.resize(config.num_decoder_layers);

    float hd = static_cast<float>(config.attention_head_dim);
    if (hd <= 0.0f) {
        hd = 64.0f;
    }

    attention_scale_ = 1.0f / std::sqrt(hd);

    encoder_block_out_nodes_.resize(config.num_encoder_layers, 0);
    // encoder_k/v_nodes store Cross Attention Cache -> Decoder Layers
    encoder_k_nodes_.assign(config.num_decoder_layers, 0);
    encoder_v_nodes_.assign(config.num_decoder_layers, 0);
    // Persistent nodes for encoder K/V (survive soft_reset)
    encoder_k_persistent_.assign(config.num_decoder_layers, 0);
    encoder_v_persistent_.assign(config.num_decoder_layers, 0);

}

void MoonshineModel::load_weights_to_graph(CactusGraph* gb) {

    embedding_node_id_ = gb->mmap_embeddings(embedding_file_path_);
    
    // Moonshine: decoder_norm -> output_norm.weights (no bias)
    weight_nodes_.decoder_norm_weight = gb->mmap_weights(model_folder_path_ + "/output_norm.weights");
    // weight_nodes_.decoder_norm_bias = gb->mmap_weights(model_folder_path_ + "/output_norm.bias"); // No bias in converted weights

    // weight_nodes_.decoder_position_embeddings_weight = gb->mmap_weights(model_folder_path_ + "/decoder_position_embeddings.weights"); // Not present

    weight_nodes_.output_weight = embedding_node_id_;
    output_weight_node_id_ = embedding_node_id_;

    if (npu::is_npu_available()) {
        std::string npu_encoder_path = model_folder_path_ + "/model.mlpackage";
        npu_encoder_ = npu::create_encoder();
        if (npu_encoder_ && npu_encoder_->load(npu_encoder_path)) {
            use_npu_encoder_ = true;

            std::vector<int> typical_input_shape = {1, 80, 3000};
            npu_encoder_->preallocate(typical_input_shape, "x", "");
        } else {
            use_npu_encoder_ = false;
            npu_encoder_.reset();
        }
    }

    if (!use_npu_encoder_) {
        // Moonshine Encoder: Conv1, Conv2, Conv3 + GroupNorm
        // weight_nodes_.encoder_position_embeddings = gb->mmap_weights(model_folder_path_ + "/encoder_position_embeddings.weights"); // Not present

        weight_nodes_.encoder_conv1_weight = gb->mmap_weights(model_folder_path_ + "/encoder_conv1_weight.weights");
        // weight_nodes_.encoder_conv1_bias = gb->mmap_weights(model_folder_path_ + "/encoder_conv1_bias.bias"); // No bias for conv1

        weight_nodes_.encoder_conv2_weight = gb->mmap_weights(model_folder_path_ + "/encoder_conv2_weight.weights");
        weight_nodes_.encoder_conv2_bias = gb->mmap_weights(model_folder_path_ + "/encoder_conv2_bias.bias");
        
        // Added Conv3 (Need to make sure MoonshineModel struct has these members? Assuming user handles header/globals or struct update if needed. I am only editing .cpp)
        // CHECK: Does weight_nodes_ have conv3? The user prompt implied I should just modify "this" function.
        // If struct lacks members, this will fail compile. But typically I should follow existing pattern.
        // Assuming I might need to reuse existing slots or if mapped differently. 
        // For now, I will assume members exist or I map to what's available. 
        // Wait, MoonshineModel usually has conv1/conv2. If Moonshine has 3, I should check if I can map them.
        // If not definable, I cannot add them here without header change. 
        // User asked to "modify this load weights... to cover the weights we got".
        // I will assume header was updated or I should try to fit it.
        // Actually, I can't see header. I will add the lines and if compile fails, I'll know.
        // Better yet, I'll rely on generic names if possible.
        // Let's assume `encoder_conv3_weight` exists in the struct for Moonshine support.
        weight_nodes_.encoder_conv3_weight = gb->mmap_weights(model_folder_path_ + "/encoder_conv3_weight.weights");
        weight_nodes_.encoder_conv3_bias = gb->mmap_weights(model_folder_path_ + "/encoder_conv3_bias.bias");

        weight_nodes_.encoder_norm_weight = gb->mmap_weights(model_folder_path_ + "/encoder_norm_weight.weights"); // GroupNorm
        weight_nodes_.encoder_norm_bias = gb->mmap_weights(model_folder_path_ + "/encoder_norm_bias.bias");

        weight_nodes_.encoder_layer_norm_weight = gb->mmap_weights(model_folder_path_ + "/encoder_layer_norm_weight.weights"); // Final Encoder LayerNorm
    }

    // Decoder Layers
    for (uint32_t i = 0; i < config_.num_decoder_layers; i++) {
        auto& layer = weight_nodes_.decoder_layers[i];
        std::string layer_prefix = model_folder_path_ + "/layer_" + std::to_string(i) + "_";

        layer.decoder_encoder_attn_k_weight = gb->mmap_weights(layer_prefix + "encoder_attn_k.weights");
        layer.decoder_encoder_attn_q_weight = gb->mmap_weights(layer_prefix + "encoder_attn_q.weights");
        layer.decoder_encoder_attn_v_weight = gb->mmap_weights(layer_prefix + "encoder_attn_v.weights");
        layer.decoder_encoder_attn_output_weight = gb->mmap_weights(layer_prefix + "encoder_attn_output.weights");

        layer.decoder_post_encoder_layernorm_weight = gb->mmap_weights(layer_prefix + "post_attn_norm.weights");

        layer.decoder_ffn1_weight = gb->mmap_weights(layer_prefix + "mlp_fc1.weights");
        layer.decoder_ffn1_bias = gb->mmap_weights(layer_prefix + "mlp_fc1.bias");
        layer.decoder_ffn2_weight = gb->mmap_weights(layer_prefix + "mlp_fc2.weights");
        layer.decoder_ffn2_bias = gb->mmap_weights(layer_prefix + "mlp_fc2.bias");

        layer.decoder_post_ffn_layernorm_weight = gb->mmap_weights(layer_prefix + "final_norm.weights");

        layer.decoder_self_attn_k_weight = gb->mmap_weights(layer_prefix + "attn_k.weights");
        layer.decoder_self_attn_q_weight = gb->mmap_weights(layer_prefix + "attn_q.weights");
        layer.decoder_self_attn_v_weight = gb->mmap_weights(layer_prefix + "attn_v.weights");
        layer.decoder_self_attn_output_weight = gb->mmap_weights(layer_prefix + "attn_output.weights");

        layer.decoder_post_attn_layernorm_weight = gb->mmap_weights(layer_prefix + "input_norm.weights");
    }

    // Encoder Layers
    if (!use_npu_encoder_) {
        for (uint32_t i = 0; i < config_.num_encoder_layers; i++) {
            auto& layer = weight_nodes_.encoder_layers[i];
            std::string layer_prefix = model_folder_path_ + "/encoder_layer_" + std::to_string(i) + "_";

            layer.encoder_ffn1_weight = gb->mmap_weights(layer_prefix + "mlp_fc1.weights");
            layer.encoder_ffn1_bias = gb->mmap_weights(layer_prefix + "mlp_fc1.bias");
            layer.encoder_ffn2_weight = gb->mmap_weights(layer_prefix + "mlp_fc2.weights");
            layer.encoder_ffn2_bias = gb->mmap_weights(layer_prefix + "mlp_fc2.bias");

            layer.encoder_post_ffn_layernorm_weight = gb->mmap_weights(layer_prefix + "post_attn_norm.weights");

            layer.encoder_self_attn_k_weight = gb->mmap_weights(layer_prefix + "attn_k.weights");
            layer.encoder_self_attn_q_weight = gb->mmap_weights(layer_prefix + "attn_q.weights");
            layer.encoder_self_attn_v_weight = gb->mmap_weights(layer_prefix + "attn_v.weights");
            layer.encoder_self_attn_output_weight = gb->mmap_weights(layer_prefix + "attn_output.weights");
            
            layer.encoder_post_attn_layernorm_weight = gb->mmap_weights(layer_prefix + "input_norm.weights");
        }
    }
}   


static size_t build_encoder_mlp_gelu(CactusGraph* gb, size_t input, size_t w1, size_t b1, size_t w2, size_t b2, ComputeBackend backend, uint32_t layer_idx) {
    auto ffn1_weight = gb->matmul(input, w1, true, backend);
    std::cout << "[BUILD DEBUG] enc_mlp_ffn1_weight layer=" << layer_idx << std::endl;
    auto ffn1_bias = gb->add(ffn1_weight, b1);
    std::cout << "[BUILD DEBUG] enc_mlp_ffn1_bias layer=" << layer_idx << std::endl;
    auto ffn1_act = gb->gelu(ffn1_bias);
    std::cout << "[BUILD DEBUG] enc_mlp_ffn1_act layer=" << layer_idx << std::endl;
    auto ffn2_weight = gb->matmul(ffn1_act, w2, true, backend);
    std::cout << "[BUILD DEBUG] enc_mlp_ffn2_weight layer=" << layer_idx << std::endl;
    auto ffn2_bias = gb->add(ffn2_weight, b2);
    std::cout << "[BUILD DEBUG] enc_mlp_ffn2_bias layer=" << layer_idx << std::endl;
    
    gb->capture_debug_node(layer_idx, "enc_mlp_ffn1_weight", ffn1_weight);
    gb->capture_debug_node(layer_idx, "enc_mlp_ffn1_bias", ffn1_bias);
    gb->capture_debug_node(layer_idx, "enc_mlp_ffn1_act", ffn1_act);
    gb->capture_debug_node(layer_idx, "enc_mlp_ffn2_weight", ffn2_weight);
    gb->capture_debug_node(layer_idx, "enc_mlp_ffn2_bias", ffn2_bias);
    
    return ffn2_bias;
}

static size_t build_encoder_mlp_silu(CactusGraph* gb, size_t input, size_t w1, size_t b1, size_t w2, size_t b2, ComputeBackend backend, uint32_t layer_idx) {
    auto ffn1_weight = gb->matmul(input, w1, true, backend);
    std::cout << "[BUILD DEBUG] enc_mlp_ffn1_weight_silu layer=" << layer_idx << std::endl;
    auto ffn1_bias = gb->add(ffn1_weight, b1);
    std::cout << "[BUILD DEBUG] enc_mlp_ffn1_bias_silu layer=" << layer_idx << std::endl;
    auto ffn1_act = gb->silu(ffn1_bias);
    std::cout << "[BUILD DEBUG] enc_mlp_ffn1_act_silu layer=" << layer_idx << std::endl;
    auto ffn2_weight = gb->matmul(ffn1_act, w2, true, backend);
    std::cout << "[BUILD DEBUG] enc_mlp_ffn2_weight_silu layer=" << layer_idx << std::endl;
    auto ffn2_bias = gb->add(ffn2_weight, b2);
    std::cout << "[BUILD DEBUG] enc_mlp_ffn2_bias_silu layer=" << layer_idx << std::endl;

    gb->capture_debug_node(layer_idx, "enc_mlp_ffn1_weight", ffn1_weight);
    gb->capture_debug_node(layer_idx, "enc_mlp_ffn1_bias", ffn1_bias);
    gb->capture_debug_node(layer_idx, "enc_mlp_ffn1_act", ffn1_act);
    gb->capture_debug_node(layer_idx, "enc_mlp_ffn2_weight", ffn2_weight);
    gb->capture_debug_node(layer_idx, "enc_mlp_ffn2_bias", ffn2_bias);

    return ffn2_bias;
}

static size_t build_decoder_mlp_silu(CactusGraph* gb, size_t input, size_t w1, size_t b1, size_t w2, size_t b2, ComputeBackend backend, size_t intermediate_size, uint32_t layer_idx) {
    auto ffn1_weight = gb->matmul(input, w1, true, backend);
    std::cout << "[BUILD DEBUG] dec_mlp_ffn1_weight layer=" << layer_idx << std::endl;
    auto ffn1_bias = gb->add(ffn1_weight, b1);
    std::cout << "[BUILD DEBUG] dec_mlp_ffn1_bias layer=" << layer_idx << std::endl;
    
    auto val = gb->slice(ffn1_bias, -1, 0, intermediate_size);
    std::cout << "[BUILD DEBUG] dec_mlp_val layer=" << layer_idx << std::endl;
    auto gate = gb->slice(ffn1_bias, -1, intermediate_size, intermediate_size);
    std::cout << "[BUILD DEBUG] dec_mlp_gate layer=" << layer_idx << std::endl;
    
    auto gate_act = gb->silu(gate);
    std::cout << "[BUILD DEBUG] dec_mlp_gate_act layer=" << layer_idx << std::endl;
    auto post_act = gb->multiply(val, gate_act);
    std::cout << "[BUILD DEBUG] dec_mlp_post_act layer=" << layer_idx << std::endl;
    
    auto ffn2_weight = gb->matmul(post_act, w2, true, backend);
    std::cout << "[BUILD DEBUG] dec_mlp_ffn2_weight layer=" << layer_idx << std::endl;
    auto ffn2_bias = gb->add(ffn2_weight, b2);
    std::cout << "[BUILD DEBUG] dec_mlp_ffn2_bias layer=" << layer_idx << std::endl;

    gb->capture_debug_node(layer_idx, "dec_mlp_ffn1_weight", ffn1_weight);
    gb->capture_debug_node(layer_idx, "dec_mlp_ffn1_bias", ffn1_bias);
    gb->capture_debug_node(layer_idx, "dec_mlp_val", val);
    gb->capture_debug_node(layer_idx, "dec_mlp_gate", gate);
    gb->capture_debug_node(layer_idx, "dec_mlp_gate_act", gate_act);
    gb->capture_debug_node(layer_idx, "dec_mlp_post_act", post_act);
    gb->capture_debug_node(layer_idx, "dec_mlp_ffn2_weight", ffn2_weight);
    gb->capture_debug_node(layer_idx, "dec_mlp_ffn2_bias", ffn2_bias);

    return ffn2_bias;
}

static size_t build_decoder_mlp_gelu(CactusGraph* gb, size_t input, size_t w1, size_t b1, size_t w2, size_t b2, ComputeBackend backend, size_t intermediate_size, uint32_t layer_idx) {

    auto ffn1_weight = gb->matmul(input, w1, true, backend);
    std::cout << "[BUILD DEBUG] dec_mlp_ffn1_weight_gelu layer=" << layer_idx << std::endl;
    auto ffn1_bias = gb->add(ffn1_weight, b1);
    std::cout << "[BUILD DEBUG] dec_mlp_ffn1_bias_gelu layer=" << layer_idx << std::endl;
    
    auto val = gb->slice(ffn1_bias, -1, 0, intermediate_size);
    std::cout << "[BUILD DEBUG] dec_mlp_val_gelu layer=" << layer_idx << std::endl;
    auto gate = gb->slice(ffn1_bias, -1, intermediate_size, intermediate_size);
    std::cout << "[BUILD DEBUG] dec_mlp_gate_gelu layer=" << layer_idx << std::endl;
    
    auto gate_act = gb->gelu(gate);
    std::cout << "[BUILD DEBUG] dec_mlp_gate_act_gelu layer=" << layer_idx << std::endl;
    auto post_act = gb->multiply(val, gate_act);
    std::cout << "[BUILD DEBUG] dec_mlp_post_act_gelu layer=" << layer_idx << std::endl;
    
    auto ffn2_weight = gb->matmul(post_act, w2, true, backend);
    std::cout << "[BUILD DEBUG] dec_mlp_ffn2_weight_gelu layer=" << layer_idx << std::endl;
    auto ffn2_bias = gb->add(ffn2_weight, b2);
    std::cout << "[BUILD DEBUG] dec_mlp_ffn2_bias_gelu layer=" << layer_idx << std::endl;

    gb->capture_debug_node(layer_idx, "dec_mlp_ffn1_weight", ffn1_weight);
    gb->capture_debug_node(layer_idx, "dec_mlp_ffn1_bias", ffn1_bias);
    gb->capture_debug_node(layer_idx, "dec_mlp_val", val);
    gb->capture_debug_node(layer_idx, "dec_mlp_gate", gate);
    gb->capture_debug_node(layer_idx, "dec_mlp_gate_act", gate_act);
    gb->capture_debug_node(layer_idx, "dec_mlp_post_act", post_act);
    gb->capture_debug_node(layer_idx, "dec_mlp_ffn2_weight", ffn2_weight);
    gb->capture_debug_node(layer_idx, "dec_mlp_ffn2_bias", ffn2_bias);

    return ffn2_bias;
}



size_t MoonshineModel::build_encoder_attention(CactusGraph* gb, size_t input, uint32_t layer_idx, ComputeBackend backend, bool use_cache, size_t /*position_offset*/){

    const auto& layer = weight_nodes_.encoder_layers[layer_idx];

    size_t q = gb->matmul(input, layer.decoder_encoder_attn_q_weight, true, backend);
    std::cout << "[BUILD DEBUG] enc_attn_q_matmul layer=" << layer_idx << std::endl;
    q = gb->add(q, layer.decoder_encoder_attn_q_bias);
    std::cout << "[BUILD DEBUG] enc_attn_q_add_bias layer=" << layer_idx << std::endl;

    const auto& q_buf   = gb->get_output_buffer(q);
    if (q_buf.shape.size() != 2) {
        throw std::runtime_error("encoder cross-attn: q must be [T_dec, D]");
    }

    size_t T_dec   = q_buf.shape[0];
    size_t q_heads = config_.attention_heads;
    size_t kv_heads = config_.attention_kv_heads;
    size_t head_dim = config_.attention_head_dim;

    q = gb->reshape(q, {1, T_dec, q_heads, head_dim});

    size_t k_4d = 0;
    size_t v_4d = 0;

    // Check if persistent K/V nodes are populated (from previous execute)
    if (use_cache && encoder_k_persistent_[layer_idx] != 0 && 
        gb->is_populated(encoder_k_persistent_[layer_idx])) {
        // Warm path: use persistent K/V directly
        k_4d = encoder_k_persistent_[layer_idx];
        v_4d = encoder_v_persistent_[layer_idx];
    } else {
        // Cold path: compute K/V from encoder output
        size_t enc_norm = last_encoder_post_norm_node_;

        size_t k = gb->matmul(enc_norm, layer.decoder_encoder_attn_k_weight, true, backend);
        std::cout << "[BUILD DEBUG] enc_attn_k_matmul layer=" << layer_idx << std::endl;
        size_t v = gb->matmul(enc_norm, layer.decoder_encoder_attn_v_weight, true, backend);
        std::cout << "[BUILD DEBUG] enc_attn_v_matmul layer=" << layer_idx << std::endl;
        v = gb->add(v, layer.decoder_encoder_attn_v_bias);
        std::cout << "[BUILD DEBUG] enc_attn_v_add_bias layer=" << layer_idx << std::endl;

        const auto& k_buf = gb->get_output_buffer(k);
        if (k_buf.shape.size() != 2) {
            throw std::runtime_error("encoder cross-attn: k must be [T_enc, D]");
        }
        size_t T_enc = k_buf.shape[0];

        k_4d = gb->reshape(k, {1, T_enc, kv_heads, head_dim});
        v_4d = gb->reshape(v, {1, T_enc, kv_heads, head_dim});

        gb->capture_debug_node(layer_idx, "cross_attn_k_raw", k);
        gb->capture_debug_node(layer_idx, "cross_attn_v_raw", v);
        gb->capture_debug_node(layer_idx, "cross_attn_k_4d", k_4d);
        gb->capture_debug_node(layer_idx, "cross_attn_v_4d", v_4d);

        // Create persistent nodes on first pass (they'll be populated during execute)
        if (encoder_k_persistent_[layer_idx] == 0) {
            encoder_k_persistent_[layer_idx] = gb->persistent(k_4d);
            encoder_v_persistent_[layer_idx] = gb->persistent(v_4d);
        }
        
        // Use the persistent nodes for attention (so they get populated)
        k_4d = encoder_k_persistent_[layer_idx];
        v_4d = encoder_v_persistent_[layer_idx];
    }
    
    gb->capture_debug_node(layer_idx, "cross_attn_k_final", k_4d);
    gb->capture_debug_node(layer_idx, "cross_attn_v_final", v_4d);

    size_t attn = gb->attention(q, k_4d, v_4d, attention_scale_, false);
    std::cout << "[BUILD DEBUG] enc_attn_scores layer=" << layer_idx << std::endl;

    gb->capture_debug_node(layer_idx, "cross_attn_q", q);
    gb->capture_debug_node(layer_idx, "cross_attn_k_persistent", k_4d);
    gb->capture_debug_node(layer_idx, "cross_attn_v_persistent", v_4d);
    gb->capture_debug_node(layer_idx, "cross_attn_scores", attn);

    attn = gb->reshape(attn, {T_dec, q_heads * head_dim});
    std::cout << "[BUILD DEBUG] enc_attn_reshape_out layer=" << layer_idx << std::endl;
    size_t out = gb->matmul(attn, layer.decoder_encoder_attn_output_weight, true, backend);
    std::cout << "[BUILD DEBUG] enc_attn_out_matmul layer=" << layer_idx << std::endl;
    out = gb->add(out, layer.decoder_encoder_attn_output_bias);
    std::cout << "[BUILD DEBUG] enc_attn_out_add_bias layer=" << layer_idx << std::endl;

    gb->capture_debug_node(layer_idx, "cross_attn_out_matmul", out);

    return out;
}

void MoonshineModel::reset_graph_side_cache_nodes() {
    cache_k_output_nodes_.assign(config_.num_decoder_layers, 0);
    cache_v_output_nodes_.assign(config_.num_decoder_layers, 0);
}

void MoonshineModel::reset_cache() {
    Model::reset_cache();
    encoder_ready_ = false;
    encoder_kv_ready_ = false;
    first_decode_step_ = true;
    encoder_output_host_.clear();
    encoder_k_host_.clear();
    encoder_v_host_.clear();
    encoder_k_shape_.clear();
    encoder_v_shape_.clear();
    
    // Invalidate persistent K/V nodes so next audio gets fresh encoder K/V
    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    if (gb) {
        for (size_t i = 0; i < encoder_k_persistent_.size(); ++i) {
            if (encoder_k_persistent_[i] != 0) {
                gb->invalidate_persistent(encoder_k_persistent_[i]);
            }
            if (encoder_v_persistent_[i] != 0) {
                gb->invalidate_persistent(encoder_v_persistent_[i]);
            }
        }
    }
}

size_t MoonshineModel::build_decoder_self_attention(CactusGraph* gb, size_t input, uint32_t layer_idx, ComputeBackend backend, bool use_cache, size_t position_offset){
    const auto& layer = weight_nodes_.decoder_layers[layer_idx];

    size_t q_pre = gb->matmul(input, layer.decoder_self_attn_q_weight, true, backend);
    std::cout << "[BUILD DEBUG] dec_sa_q_matmul layer=" << layer_idx << std::endl;
    size_t q = gb->add(q_pre, layer.decoder_self_attn_q_bias);
    std::cout << "[BUILD DEBUG] dec_sa_q_add_bias layer=" << layer_idx << std::endl;

    size_t k_pre = gb->matmul(input, layer.decoder_self_attn_k_weight, true, backend);
    std::cout << "[BUILD DEBUG] dec_sa_k_matmul layer=" << layer_idx << std::endl;
    size_t k = gb->add(k_pre, layer.decoder_self_attn_k_bias);
    std::cout << "[BUILD DEBUG] dec_sa_k_add_bias layer=" << layer_idx << std::endl;

    size_t v_pre = gb->matmul(input, layer.decoder_self_attn_v_weight, true, backend);
    std::cout << "[BUILD DEBUG] dec_sa_v_matmul layer=" << layer_idx << std::endl;
    size_t v = gb->add(v_pre, layer.decoder_self_attn_v_bias);
    std::cout << "[BUILD DEBUG] dec_sa_v_add_bias layer=" << layer_idx << std::endl;

    gb->capture_debug_node(layer_idx, "dec_sa_q_pre", q_pre);
    gb->capture_debug_node(layer_idx, "dec_sa_q", q);
    gb->capture_debug_node(layer_idx, "dec_sa_k_pre", k_pre);
    gb->capture_debug_node(layer_idx, "dec_sa_k", k);
    gb->capture_debug_node(layer_idx, "dec_sa_v_pre", v_pre);
    gb->capture_debug_node(layer_idx, "dec_sa_v", v);

    const auto& q_shape = gb->get_output_buffer(q).shape;
    if (q_shape.size() != 2) {
        throw std::runtime_error("decoder self-attn: q must be [T_new, D]");
    }

    size_t seq_new = q_shape[0];
    size_t num_heads = config_.attention_heads;
    size_t head_dim = config_.attention_head_dim;
    size_t num_kv_heads = config_.attention_kv_heads;
 
    size_t q_4d = gb->reshape(q, {1, seq_new, num_heads, head_dim});
    std::cout << "[BUILD DEBUG] dec_sa_q_reshape layer=" << layer_idx << std::endl;
    size_t k_4d = gb->reshape(k, {1, seq_new, num_kv_heads, head_dim});
    std::cout << "[BUILD DEBUG] dec_sa_k_reshape layer=" << layer_idx << std::endl;
    size_t v_4d = gb->reshape(v, {1, seq_new, num_kv_heads, head_dim});
    std::cout << "[BUILD DEBUG] dec_sa_v_reshape layer=" << layer_idx << std::endl;

    gb->capture_debug_node(layer_idx, "dec_sa_q4d", q_4d);
    gb->capture_debug_node(layer_idx, "dec_sa_k4d", k_4d);
    gb->capture_debug_node(layer_idx, "dec_sa_v4d", v_4d);

    size_t rot_dim = std::max(head_dim / 2, (size_t)32);
    if (config_.rope_theta > 0) {
        q_4d = gb->rope_gptj(q_4d, config_.rope_theta, position_offset, rot_dim);
        k_4d = gb->rope_gptj(k_4d, config_.rope_theta, position_offset, rot_dim);
        gb->capture_debug_node(layer_idx, "dec_sa_q_rope", q_4d);
        gb->capture_debug_node(layer_idx, "dec_sa_k_rope", k_4d);
    }

    size_t final_k = k_4d;
    size_t final_v = v_4d;

    if (use_cache && !kv_cache_.is_empty()) {
        auto k_view = kv_cache_.get_key_view(layer_idx);
        auto v_view = kv_cache_.get_value_view(layer_idx);

        if (!k_view.ptr1 || !v_view.ptr1) {
            throw std::runtime_error("KV cache view is empty but kv_cache_.is_empty()==false");
        }

        size_t cache_len = kv_cache_.current_seq_len;

        size_t cache_k_node = gb->input(
            {1, cache_len, num_kv_heads, head_dim},
            kv_cache_.precision
        );
        size_t cache_v_node = gb->input(
            {1, cache_len, num_kv_heads, head_dim},
            kv_cache_.precision
        );

        if (k_view.ptr2 == nullptr && v_view.ptr2 == nullptr) {
            gb->set_input(cache_k_node, k_view.ptr1, kv_cache_.precision);
            gb->set_input(cache_v_node, v_view.ptr1, kv_cache_.precision);
        } else {
            gb->set_input(cache_k_node, kv_cache_.get_key_ptr(layer_idx), kv_cache_.precision);
            gb->set_input(cache_v_node, kv_cache_.get_value_ptr(layer_idx), kv_cache_.precision);
        }

        final_k = gb->concat(cache_k_node, k_4d, 1);
        std::cout << "[BUILD DEBUG] dec_sa_k_concat_cache layer=" << layer_idx << std::endl;
        final_v = gb->concat(cache_v_node, v_4d, 1);
        std::cout << "[BUILD DEBUG] dec_sa_v_concat_cache layer=" << layer_idx << std::endl;

        gb->capture_debug_node(layer_idx, "dec_sa_cache_k", cache_k_node);
        gb->capture_debug_node(layer_idx, "dec_sa_cache_v", cache_v_node);
    }

    if (use_cache) {
        cache_k_output_nodes_[layer_idx] = final_k;
        cache_v_output_nodes_[layer_idx] = final_v;
    } else {
        cache_k_output_nodes_[layer_idx] = k_4d;
        cache_v_output_nodes_[layer_idx] = v_4d;
    }

    auto attn_out_4d = gb->attention(q_4d, final_k, final_v, attention_scale_, position_offset);
    
    gb->capture_debug_node(layer_idx, "dec_sa_q_final", q_4d);
    gb->capture_debug_node(layer_idx, "dec_sa_k_final", final_k);
    gb->capture_debug_node(layer_idx, "dec_sa_v_final", final_v);
    gb->capture_debug_node(layer_idx, "dec_sa_attn_scores", attn_out_4d);

    auto attn_out    = gb->reshape(attn_out_4d, {seq_new, num_heads * head_dim});

    auto output = gb->matmul(attn_out, layer.decoder_self_attn_output_weight, true, backend);
    std::cout << "[BUILD DEBUG] dec_sa_out_matmul layer=" << layer_idx << std::endl;
    output = gb->add(output, layer.decoder_self_attn_output_bias);
    std::cout << "[BUILD DEBUG] dec_sa_out_add_bias layer=" << layer_idx << std::endl;

    gb->capture_debug_node(layer_idx, "dec_sa_out_matmul", output);

    return output;
}

size_t MoonshineModel::build_encoder_self_attention(CactusGraph* gb, size_t input, uint32_t layer_idx, ComputeBackend backend, bool use_cache, size_t /*position_offset*/){
    const auto& layer = weight_nodes_.encoder_layers[layer_idx];

    if(use_cache)
        throw std::runtime_error("The encoder attention layers are not auto-regressive, and thus don't use KV caching!");

    auto q = gb->matmul(input, layer.encoder_self_attn_q_weight, false, backend);
    std::cout << "[BUILD DEBUG] enc_sa_q_matmul layer=" << layer_idx << std::endl;
    auto v = gb->matmul(input, layer.encoder_self_attn_v_weight, false, backend);
    std::cout << "[BUILD DEBUG] enc_sa_v_matmul layer=" << layer_idx << std::endl;
    auto k = gb->matmul(input, layer.encoder_self_attn_k_weight, false, backend);
    std::cout << "[BUILD DEBUG] enc_sa_k_matmul layer=" << layer_idx << std::endl;

    size_t seq_len = gb->get_output_buffer(q).shape[0];
    size_t num_heads = config_.attention_heads;
    size_t head_dim  = config_.attention_head_dim;

    q = gb->reshape(q, {1, seq_len, num_heads, head_dim});
    std::cout << "[BUILD DEBUG] enc_sa_q_reshape layer=" << layer_idx << std::endl;
    k = gb->reshape(k, {1, seq_len, num_heads, head_dim});
    std::cout << "[BUILD DEBUG] enc_sa_k_reshape layer=" << layer_idx << std::endl;
    v = gb->reshape(v, {1, seq_len, num_heads, head_dim});
    std::cout << "[BUILD DEBUG] enc_sa_v_reshape layer=" << layer_idx << std::endl;

    gb->capture_debug_node(layer_idx, "enc_sa_q_prebias", q);
    gb->capture_debug_node(layer_idx, "enc_sa_k_prebias", k);
    gb->capture_debug_node(layer_idx, "enc_sa_v_prebias", v);

    size_t rot_dim = std::max(head_dim / 2, (size_t)32);
    if (config_.rope_theta > 0) {
        q = gb->rope_gptj(q, config_.rope_theta, 0, rot_dim);
        k = gb->rope_gptj(k, config_.rope_theta, 0, rot_dim);
    }

    auto attn = gb->attention(q, k, v, attention_scale_, false);
    std::cout << "[BUILD DEBUG] enc_sa_scores layer=" << layer_idx << std::endl;

    attn = gb->reshape(attn, {seq_len, num_heads * head_dim});
    std::cout << "[BUILD DEBUG] enc_sa_reshape_out layer=" << layer_idx << std::endl;

    auto output = gb->matmul(attn, layer.encoder_self_attn_output_weight, false, backend);
    std::cout << "[BUILD DEBUG] enc_sa_out_matmul layer=" << layer_idx << std::endl;

    gb->capture_debug_node(layer_idx, "enc_sa_out_matmul", output);

    return output;
}

size_t MoonshineModel::build_audio_preprocessor(CactusGraph* gb, size_t input)
{
    size_t conv_input = input;
    const auto& xbuf = gb->get_output_buffer(input);

    if (xbuf.precision == Precision::INT8) { 
        conv_input = gb->precision_cast(input, Precision::FP16);
    }

    // Conv1: kernel=127, stride=64, no bias
    size_t conv1 = gb->conv1d(conv_input, weight_nodes_.encoder_conv1_weight, 64);
    std::cout << "[BUILD DEBUG] preprocessor_conv1" << std::endl;

    last_conv1_node_ = conv1;

    conv1 = gb->tanh(conv1);

    // GroupNorm with 1 group (like LayerNorm but per-channel)
    size_t gn = gb->groupnorm(conv1, weight_nodes_.encoder_norm_weight, weight_nodes_.encoder_norm_bias);
    std::cout << "[BUILD DEBUG] preprocessor_gn" << std::endl;

    // Conv2: kernel=7, stride=3, with bias
    size_t conv2 = gb->conv1d(gn, weight_nodes_.encoder_conv2_weight, 3);
    std::cout << "[BUILD DEBUG] preprocessor_conv2_matmul" << std::endl;

    // Add Conv2 bias manually (reshape to broadcast)
    auto bias2_shape = gb->get_output_buffer(weight_nodes_.encoder_conv2_bias).shape;
    size_t C2 = bias2_shape[0];
    size_t bias2 = gb->reshape(weight_nodes_.encoder_conv2_bias, {1, C2, 1});
    std::cout << "[BUILD DEBUG] preprocessor_bias2_reshape" << std::endl;
    size_t conv2_with_bias = gb->add(conv2, bias2);
    std::cout << "[BUILD DEBUG] preprocessor_conv2_add_bias" << std::endl;
    size_t conv2_gelu = gb->gelu(conv2_with_bias);

    // Conv3: kernel=3, stride=2, with bias
    size_t conv3 = gb->conv1d(conv2_gelu, weight_nodes_.encoder_conv3_weight, 2);
    std::cout << "[BUILD DEBUG] preprocessor_conv3_matmul" << std::endl;

    // Add Conv3 bias manually (reshape to broadcast)
    auto bias3_shape = gb->get_output_buffer(weight_nodes_.encoder_conv3_bias).shape;
    size_t C3 = bias3_shape[0];
    size_t bias3 = gb->reshape(weight_nodes_.encoder_conv3_bias, {1, C3, 1});
    std::cout << "[BUILD DEBUG] preprocessor_bias3_reshape" << std::endl;
    size_t conv3_with_bias = gb->add(conv3, bias3);
    std::cout << "[BUILD DEBUG] preprocessor_conv3_add_bias" << std::endl;
    size_t conv3_gelu = gb->gelu(conv3_with_bias);

    gb->capture_debug_node(0, "preprocessor_conv1", conv1);
    gb->capture_debug_node(0, "preprocessor_gn", gn);
    gb->capture_debug_node(0, "preprocessor_conv2_raw", conv2);
    gb->capture_debug_node(0, "preprocessor_conv2_bias", bias2);
    gb->capture_debug_node(0, "preprocessor_conv2_with_bias", conv2_with_bias);
    gb->capture_debug_node(0, "preprocessor_conv2_gelu", conv2_gelu);
    gb->capture_debug_node(0, "preprocessor_conv3_raw", conv3);
    gb->capture_debug_node(0, "preprocessor_conv3_bias", bias3);
    gb->capture_debug_node(0, "preprocessor_conv3_with_bias", conv3_with_bias);
    gb->capture_debug_node(0, "preprocessor_conv3_gelu", conv3_gelu);

    const auto& buf = gb->get_output_buffer(conv3_gelu);

    size_t conv_out_transposed;
    if (buf.precision == Precision::FP16) {
        conv_out_transposed = gb->transpose(conv3_gelu, ComputeBackend::CPU);
    } else {
        size_t conv3_f16 = gb->precision_cast(conv3_gelu, Precision::FP16);
        conv_out_transposed = gb->transpose(conv3_f16, ComputeBackend::CPU);
    }
    
    gb->capture_debug_node(0, "preprocessor_final_transposed", conv_out_transposed);

    return conv_out_transposed;
}

size_t MoonshineModel::build_encoder_transformer_block(
    CactusGraph* gb,
    size_t hidden,
    uint32_t layer_idx,
    ComputeBackend backend,
    bool use_cache,
    size_t position_offset)
{
    const auto& layer = weight_nodes_.encoder_layers[layer_idx];

    size_t ln1 = gb->layernorm(
        hidden,
        layer.encoder_post_attn_layernorm_weight
    );
    std::cout << "[BUILD DEBUG] encoder_ln1 layer=" << layer_idx << std::endl;

    size_t sa = build_encoder_self_attention(
        gb, ln1, layer_idx, backend, use_cache, position_offset
    );
    std::cout << "[BUILD DEBUG] encoder_sa_block layer=" << layer_idx << std::endl;

    size_t x_post_sa = gb->add(hidden, sa);
    std::cout << "[BUILD DEBUG] encoder_x_post_sa layer=" << layer_idx << std::endl;

    size_t ln2 = gb->layernorm(
        x_post_sa,
        layer.encoder_post_ffn_layernorm_weight
    );
    std::cout << "[BUILD DEBUG] encoder_ln2 layer=" << layer_idx << std::endl;

    gb->capture_debug_node(layer_idx, "encoder_ln1", ln1);
    gb->capture_debug_node(layer_idx, "encoder_x_post_sa", x_post_sa);
    gb->capture_debug_node(layer_idx, "encoder_ln2", ln2);

    size_t ffn_out;
    if (config_.encoder_act_gelu) {
        ffn_out = build_encoder_mlp_gelu(
            gb, ln2, layer.encoder_ffn1_weight, layer.encoder_ffn1_bias,
            layer.encoder_ffn2_weight, layer.encoder_ffn2_bias, backend, layer_idx
        );
    } else {
        ffn_out = build_encoder_mlp_silu(
            gb, ln2, layer.encoder_ffn1_weight, layer.encoder_ffn1_bias,
            layer.encoder_ffn2_weight, layer.encoder_ffn2_bias, backend, layer_idx
        );
    }

    size_t out = gb->add(x_post_sa, ffn_out);
    std::cout << "[BUILD DEBUG] encoder_out_residual layer=" << layer_idx << std::endl;

    gb->capture_debug_node(layer_idx, "encoder_sa", sa);
    gb->capture_debug_node(layer_idx, "encoder_ffn", ffn_out);
    gb->capture_debug_node(layer_idx, "encoder_output", out);

    if (layer_idx < encoder_block_out_nodes_.size()) {
        encoder_block_out_nodes_[layer_idx] = out;
    }

    return out;
}

size_t MoonshineModel::build_decoder_transformer_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx, ComputeBackend backend, bool use_cache, size_t position_offset){
    const auto& layer = weight_nodes_.decoder_layers[layer_idx];

    size_t ln1 = gb->layernorm(hidden, layer.decoder_post_attn_layernorm_weight, layer.decoder_post_attn_layernorm_bias);
    std::cout << "[BUILD DEBUG] decoder_ln1 layer=" << layer_idx << std::endl;
    size_t sa = build_decoder_self_attention(gb, ln1, layer_idx, backend, use_cache, position_offset);
    std::cout << "[BUILD DEBUG] decoder_sa_block layer=" << layer_idx << std::endl;
    size_t x_post_sa = gb->add(hidden, sa);
    std::cout << "[BUILD DEBUG] decoder_x_post_sa layer=" << layer_idx << std::endl;

    size_t ln2 = gb->layernorm(x_post_sa, layer.decoder_post_encoder_layernorm_weight, layer.decoder_post_encoder_layernorm_bias);
    std::cout << "[BUILD DEBUG] decoder_ln2 layer=" << layer_idx << std::endl;
    size_t ca = build_encoder_attention(gb, ln2, layer_idx, backend, use_cache, position_offset);
    std::cout << "[BUILD DEBUG] decoder_ca_block layer=" << layer_idx << std::endl;
    size_t x_post_ca = gb->add(x_post_sa, ca);
    std::cout << "[BUILD DEBUG] decoder_x_post_ca layer=" << layer_idx << std::endl;

    size_t ln3 = gb->layernorm(x_post_ca,layer.decoder_post_ffn_layernorm_weight,layer.decoder_post_ffn_layernorm_bias);
    std::cout << "[BUILD DEBUG] decoder_ln3 layer=" << layer_idx << std::endl;
    size_t ffn_out;
    if (config_.decoder_act_gelu) {
        ffn_out = build_decoder_mlp_gelu(
            gb, ln3, layer.decoder_ffn1_weight, layer.decoder_ffn1_bias,
            layer.decoder_ffn2_weight, layer.decoder_ffn2_bias, backend,
            config_.ffn_intermediate_dim, layer_idx
        );
    } else {
        ffn_out = build_decoder_mlp_silu(
            gb, ln3, layer.decoder_ffn1_weight, layer.decoder_ffn1_bias,
            layer.decoder_ffn2_weight, layer.decoder_ffn2_bias, backend,
            config_.ffn_intermediate_dim, layer_idx
        );
    }
    size_t x_post_ffn = gb->add(x_post_ca, ffn_out);
    std::cout << "[BUILD DEBUG] decoder_out_residual layer=" << layer_idx << std::endl;

    gb->capture_debug_node(layer_idx, "decoder_sa", sa);
    gb->capture_debug_node(layer_idx, "decoder_ca", ca);
    gb->capture_debug_node(layer_idx, "decoder_ffn", ffn_out);
    gb->capture_debug_node(layer_idx, "decoder_output", x_post_ffn);

    return x_post_ffn;

}

size_t MoonshineModel::build_encoder(CactusGraph* gb, const std::vector<float>& audio_features)
{
    if (use_npu_encoder_ && npu_encoder_ && npu_encoder_->is_available()) {
        std::vector<int> out_shape = npu_encoder_->get_output_shape();
        size_t T_enc, D_enc;
        if (out_shape.size() == 3) {
            T_enc = static_cast<size_t>(out_shape[1]);
            D_enc = static_cast<size_t>(out_shape[2]);
        } else if (out_shape.size() == 2) {
            T_enc = static_cast<size_t>(out_shape[0]);
            D_enc = static_cast<size_t>(out_shape[1]);
        } else {
            throw std::runtime_error("NPU encoder output has unexpected shape");
        }

        std::vector<__fp16> audio_f16(audio_features.size());
        cactus_fp32_to_fp16(audio_features.data(), audio_f16.data(), audio_features.size());

        std::vector<int> input_shape = {1, 80, static_cast<int>(T_enc)};

        __fp16* output_buffer = npu_encoder_->get_output_buffer();
        if (output_buffer) {
            size_t elements_written = npu_encoder_->encode(
                audio_f16.data(),
                output_buffer,  
                input_shape,
                "x",
                ""
            );

            if (elements_written > 0) {
                size_t enc_output_node = gb->input({T_enc, D_enc}, Precision::FP16);
                gb->set_input(enc_output_node, output_buffer, Precision::FP16);

                last_encoder_post_norm_node_ = enc_output_node;
                return enc_output_node;
            }
        } else {
            std::vector<__fp16> npu_output(T_enc * D_enc);
            size_t elements_written = npu_encoder_->encode(
                audio_f16.data(),
                npu_output.data(),
                input_shape,
                "x",
                ""
            );

            if (elements_written > 0) {
                size_t enc_output_node = gb->input({T_enc, D_enc}, Precision::FP16);
                gb->set_input(enc_output_node, npu_output.data(), Precision::FP16);

                last_encoder_post_norm_node_ = enc_output_node;
                return enc_output_node;
            }
        }
    }

    auto backend =
        (config_.default_backend == Config::Backend::CPU)
        ? ComputeBackend::CPU
        : ComputeBackend::NPU;

    // Moonshine takes raw 16kHz audio, not mel features
    size_t audio_input = 0;
    std::vector<__fp16> audio_f16(audio_features.size());
    cactus_fp32_to_fp16(audio_features.data(), audio_f16.data(), audio_features.size());

    gb->capture_debug_node(0, "audio_input", audio_input);

    size_t audio_length = audio_features.size();  // Raw audio samples at 16kHz
    audio_input = gb->input({1, 1, audio_length}, Precision::FP16);
    std::cout << "[BUILD DEBUG] encoder_audio_input" << std::endl;
    gb->set_input(audio_input, audio_f16.data(), Precision::FP16);

    size_t conv2_transposed = build_audio_preprocessor(gb, audio_input);
    std::cout << "[BUILD DEBUG] encoder_preprocessor_out" << std::endl;

    const auto& conv_shape = gb->get_output_buffer(conv2_transposed).shape;
    if (conv_shape.size() != 3 || conv_shape[0] != 1)
        throw std::runtime_error("Conv2 transpose should be [1, T_enc, D].");

    size_t T_enc = conv_shape[1];
    size_t D_enc = conv_shape[2];

    // Moonshine Encoder does not use absolute positional embeddings (uses RoPE in self-attention)
    // size_t pos_slice = gb->slice(weight_nodes_.encoder_position_embeddings, 0, 0, T_enc);
    //
    // size_t h2d = gb->reshape(conv2_transposed, {T_enc, D_enc});
    //
    // auto& h2d_buf = gb->get_output_buffer(h2d);
    // auto& pos_buf = gb->get_output_buffer(pos_slice);
    //
    // if (pos_buf.precision != h2d_buf.precision) {
    //    pos_slice = gb->precision_cast(pos_slice, h2d_buf.precision);
    // }
    //
    // size_t h_pos = gb->add(h2d, pos_slice);
    // last_enc_plus_pos_node_ = h_pos;

    size_t h = gb->reshape(conv2_transposed, {T_enc, D_enc}); // h is just reshaped conv output, no pos add
    gb->capture_debug_node(0, "encoder_initial_h", h);

    for (uint32_t i = 0; i < config_.num_encoder_layers; ++i){
        h = build_encoder_transformer_block(gb, h, i, backend, false, 0);
        gb->capture_debug_node(i, "encoder_block_out", h);
    }

    size_t h_norm = gb->layernorm(
        h,
        weight_nodes_.encoder_norm_weight
    );
    gb->capture_debug_node(0, "encoder_final_norm", h_norm);

    last_encoder_post_norm_node_ = h_norm;

    // encoder_output is only needed during first decode step to compute K/V
    // After that, persistent K/V nodes are used directly
    return h_norm;
}


size_t MoonshineModel::build_decoder(const std::vector<uint32_t>& tokens, bool use_cache, bool last_token_only) {
    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    
    const size_t full_len = tokens.size();
    if (full_len == 0) {
        throw std::runtime_error("Decoder token list cannot be empty.");
    }

    auto backend = config_.default_backend == Config::Backend::CPU
        ? ComputeBackend::CPU
        : ComputeBackend::NPU;

    size_t start_idx = (use_cache && kv_cache_.current_seq_len > 0) ? full_len - 1 : 0;
    size_t new_tokens = full_len - start_idx;
    size_t position_offset = use_cache ? kv_cache_.current_seq_len : 0;

    size_t tok_input = gb->input({new_tokens}, Precision::FP32);
    std::cout << "[BUILD DEBUG] decoder_tok_input" << std::endl;
    std::vector<float> tok_f(new_tokens);
    for (size_t i = 0; i < new_tokens; i++) {
        tok_f[i] = static_cast<float>(tokens[start_idx + i]);
    }
    gb->set_input(tok_input, tok_f.data(), Precision::FP32);

    size_t dec_hidden = gb->embedding(embedding_node_id_, tok_input);
    std::cout << "[BUILD DEBUG] decoder_embedding_out" << std::endl;
    gb->capture_debug_node(0, "decoder_initial_embedding", dec_hidden);

    for (uint32_t layer_idx = 0; layer_idx < config_.num_decoder_layers; ++layer_idx) {
        dec_hidden = build_decoder_transformer_block(
            gb,
            dec_hidden,
            layer_idx,
            backend,
            use_cache,
            position_offset
        );
        std::cout << "[BUILD DEBUG] decoder_block_done layer=" << layer_idx << std::endl;
    }
    gb->capture_debug_node(0, "decoder_hidden_all_layers", dec_hidden);

    size_t dec_norm = gb->layernorm(
        dec_hidden,
        weight_nodes_.decoder_norm_weight,
        weight_nodes_.decoder_norm_bias
    );
    std::cout << "[BUILD DEBUG] decoder_final_norm" << std::endl;
    gb->capture_debug_node(0, "decoder_final_norm", dec_norm);

    size_t logits_input = dec_norm;
    if (last_token_only) {
        size_t row_index = new_tokens - 1;
        logits_input = gb->slice(logits_input, 0, row_index, 1); 
        std::cout << "[BUILD DEBUG] decoder_logits_slice" << std::endl;
    }

    auto w_shape = gb->get_output_buffer(output_weight_node_id_).shape;

    size_t logits = gb->matmul(logits_input, output_weight_node_id_, true, backend);
    std::cout << "[BUILD DEBUG] decoder_logits_matmul" << std::endl;

    last_new_tokens_ = new_tokens;
    gb->capture_debug_node(0, "logits", logits);

    return logits;
}


size_t MoonshineModel::forward(const std::vector<float>& audio_features, const std::vector<uint32_t>& tokens, bool use_cache)
{
    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    gb->clear_debug_nodes();

    size_t enc_out = build_encoder(gb, audio_features);
    size_t logits = build_decoder(tokens, use_cache, true);

    gb->capture_debug_node(0, "forward_enc_out", enc_out);
    gb->capture_debug_node(0, "forward_logits", logits);

    return logits;
}

std::vector<float> MoonshineModel::get_audio_embeddings(const std::vector<float>& audio_features) {
    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    build_encoder(gb, audio_features); // Call the public build_encoder
    size_t pooled = gb->mean(last_encoder_post_norm_node_, 0);
    gb->execute();

    const auto& output_buf = gb->get_output_buffer(pooled);
    size_t hidden_dim = output_buf.total_size;

    std::vector<float> embedding(hidden_dim);
    void* output_data = gb->get_output(pooled);
    const float* output_ptr = static_cast<const float*>(output_data);
    std::copy(output_ptr, output_ptr + hidden_dim, embedding.begin());

    reset_cache();
    return embedding;
}

uint32_t MoonshineModel::decode_with_audio(
    const std::vector<uint32_t>& tokens,
    const std::vector<float>& audio_features,
    float temperature,
    float top_p,
    size_t top_k,
    const std::string& profile_file,
    float* out_entropy)
{
    if (!initialized_ || !graph_handle_)
        throw std::runtime_error("Model not initialized - call init() first");
    if (tokens.empty())
        throw std::runtime_error("Token sequence cannot be empty");
    if (audio_features.empty())
        throw std::runtime_error("Mel bins cannot be empty in Moonshine decode_with_audio");

    auto* gb = static_cast<CactusGraph*>(graph_handle_);

    bool cold_start = !encoder_ready_;
    size_t logits_node = 0;

    uint32_t bos = static_cast<uint32_t>(get_tokenizer()->get_bos_token());

    std::vector<uint32_t> full_tokens;
    full_tokens.reserve(tokens.size() + 1);
    full_tokens.push_back(bos);
    full_tokens.insert(full_tokens.end(), tokens.begin(), tokens.end());

    if (cold_start)
    {
        gb->soft_reset();
        kv_cache_.reset();
        kv_cache_.current_seq_len = 0;
        reset_graph_side_cache_nodes();

        encoder_kv_ready_ = false;
        encoder_k_nodes_.assign(config_.num_decoder_layers, 0);
        encoder_v_nodes_.assign(config_.num_decoder_layers, 0);
        encoder_k_host_.clear();
        encoder_v_host_.clear();
        encoder_k_shape_.clear();
        encoder_v_shape_.clear();

        first_decode_step_ = true;
        build_encoder(gb, audio_features);
        logits_node = build_decoder(full_tokens, false, false);
    }

    else
    {
        gb->soft_reset();
        reset_graph_side_cache_nodes();

        std::vector<uint32_t> last_token_vec = { tokens.back() };
        logits_node = build_decoder(last_token_vec, true, true);
    }
    
    size_t sampled_token_id = gb->sample(logits_node, temperature, top_p, top_k);
    if (!profile_file.empty()) gb->execute(profile_file);
    else gb->execute();


    if (out_entropy) {
        const auto& logits_buf = gb->get_output_buffer(logits_node);
        void* logits_ptr = gb->get_output(logits_node);
        size_t vocab_size = logits_buf.shape.back();

        std::vector<float> logits(vocab_size);
        if (logits_buf.precision == Precision::FP32) {
            float* src = static_cast<float*>(logits_ptr);
            std::copy(src, src + vocab_size, logits.begin());
        } else if (logits_buf.precision == Precision::FP16) {
            __fp16* src = static_cast<__fp16*>(logits_ptr);
            Quantization::fp16_to_fp32(src, logits.data(), vocab_size);
        } else {
            int8_t* src = static_cast<int8_t*>(logits_ptr);
            Quantization::int8_to_fp32(src, logits.data(), vocab_size, 1.0f);
        }

        float max_logit = *std::max_element(logits.begin(), logits.end());
        double sum_exp = 0.0;
        for (size_t i = 0; i < vocab_size; ++i) {
            sum_exp += std::exp(static_cast<double>(logits[i] - max_logit));
        }
        double log_sum_exp = static_cast<double>(max_logit) + std::log(sum_exp);

        double entropy = 0.0;
        for (size_t i = 0; i < vocab_size; ++i) {
            double log_prob = static_cast<double>(logits[i]) - log_sum_exp;
            double prob = std::exp(log_prob);
            if (prob > 1e-10) {
                entropy -= prob * log_prob;
            }
        }

        double max_entropy = std::log(static_cast<double>(vocab_size));
        *out_entropy = static_cast<float>(entropy / max_entropy);
    }

    post_execute_updates(gb, full_tokens.size());
    update_kv_cache(gb, last_new_tokens_);

    auto* out_ptr = gb->get_output(sampled_token_id);
    uint32_t sampled = *reinterpret_cast<uint32_t*>(out_ptr);

    return sampled;
}


}
}
