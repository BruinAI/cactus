#include "model.h"
#include "../graph/graph.h"
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <set>
#include <limits>
#include <iostream>

namespace cactus {
namespace engine {

LFM2Model::LFM2Model() : Model() {}

LFM2Model::LFM2Model(const Config& config) : Model(config) {
    weight_nodes_.layers.resize(config.num_layers);
}

bool LFM2Model::init(const std::string& model_folder, size_t context_size, const std::string& system_prompt) {
    // Call base class init first
    if (!Model::init(model_folder, context_size, system_prompt)) {
        return false;
    }

    // Initialize conv cache for LFM2
    if (config_.conv_L_cache > 0) {
        Precision cache_precision;
        switch (config_.precision) {
            case Config::Precision::INT8:
                cache_precision = Precision::INT8;
                break;
            case Config::Precision::FP16:
                cache_precision = Precision::FP16;
                break;
            case Config::Precision::FP32:
                cache_precision = Precision::FP32;
                break;
        }
        conv_cache_.init(config_.num_layers, config_.hidden_dim, config_.conv_L_cache, cache_precision);
    }

    return true;
}

void LFM2Model::load_weights_to_graph(CactusGraph* gb) {
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
        auto& layer_entry = weight_nodes_.layers[i];
        auto& layer = layer_entry.weights;
        std::string layer_prefix = model_folder_path_ + "/layer_" + std::to_string(i) + "_";

        // Determine layer type from config
        bool is_conv_layer = false;
        if (i < config_.layer_types.size()) {
            std::string layer_type = config_.layer_types[i];
            is_conv_layer = (layer_type == "conv" || layer_type == "CONV");
            // Anything else (attn, full_attention, attention) is treated as attention layer
        }

        if (is_conv_layer) {
            // Load conv-specific weights
            layer_entry.type = WeightNodeIDs::LayerType::CONV;
            layer.conv_in_proj_weight = gb->mmap_weights(layer_prefix + "conv_in_proj.weights");
            layer.conv_out_proj_weight = gb->mmap_weights(layer_prefix + "conv_out_proj.weights");
            layer.conv_depthwise_weight = gb->mmap_weights(layer_prefix + "conv_depthwise.weights");
        } else {
            // Load attention-specific weights
            layer_entry.type = WeightNodeIDs::LayerType::ATTENTION;
            layer.attn_q_weight = gb->mmap_weights(layer_prefix + "attn_q.weights");
            layer.attn_k_weight = gb->mmap_weights(layer_prefix + "attn_k.weights");
            layer.attn_v_weight = gb->mmap_weights(layer_prefix + "attn_v.weights");
            layer.attn_output_weight = gb->mmap_weights(layer_prefix + "attn_output.weights");
            layer.attn_q_norm_weight = gb->mmap_weights(layer_prefix + "attn_q_norm.weights");
            layer.attn_k_norm_weight = gb->mmap_weights(layer_prefix + "attn_k_norm.weights");
        }

        // Load shared weights (present in all layers)
        layer.input_layernorm_weight = gb->mmap_weights(layer_prefix + "input_norm.weights");
        layer.post_attention_layernorm_weight = gb->mmap_weights(layer_prefix + "post_attn_norm.weights");
        layer.ffn_gate_weight = gb->mmap_weights(layer_prefix + "ffn_gate.weights");
        layer.ffn_up_weight = gb->mmap_weights(layer_prefix + "ffn_up.weights");
        layer.ffn_down_weight = gb->mmap_weights(layer_prefix + "ffn_down.weights");
    }
}

size_t LFM2Model::build_conv1d(CactusGraph* gb, size_t input, uint32_t layer_idx,
                               ComputeBackend backend) {
    const auto& layer_entry = weight_nodes_.layers[layer_idx];
    const auto& layer       = layer_entry.weights;

    // in_proj: [L, 3*C]
    size_t in_proj = gb->matmul(input, layer.conv_in_proj_weight, true, backend);
    capture_debug_node(layer_idx, "conv_in_proj", in_proj);

    const auto& in_proj_buf = gb->get_output_buffer(in_proj);
    if (in_proj_buf.shape.size() != 2 || (in_proj_buf.shape[1] % 3) != 0) {
        throw std::runtime_error("Conv in_proj output must be [L, 3*C]");
    }

    const size_t L = in_proj_buf.shape[0];
    const size_t C = in_proj_buf.shape[1] / 3;

    // Split into (B, C, X) then elementwise B⊙X
    size_t triplet = gb->reshape(in_proj, {L, static_cast<size_t>(3), C});
    size_t B = gb->slice(triplet, /*axis*/1, /*start*/0, /*len*/1);
    size_t Cg = gb->slice(triplet, /*axis*/1, /*start*/1, /*len*/1);
    size_t X = gb->slice(triplet, /*axis*/1, /*start*/2, /*len*/1);
    capture_debug_node(layer_idx, "conv_triplet", triplet);
    capture_debug_node(layer_idx, "conv_B_sliced", B);

    B  = gb->reshape(B,  {L, C});
    Cg = gb->reshape(Cg, {L, C});
    X  = gb->reshape(X,  {L, C});
    capture_debug_node(layer_idx, "conv_B", B);
    capture_debug_node(layer_idx, "conv_C", Cg);
    capture_debug_node(layer_idx, "conv_X", X);

    size_t Bx = gb->multiply(B, X);                // [L, C]
    capture_debug_node(layer_idx, "conv_bx", Bx);

    // Prepare depthwise weights to [C,1,K]
    const auto& wbuf = gb->get_output_buffer(layer.conv_depthwise_weight);
    size_t K = wbuf.shape.back(); // last dim is K in both [C,K] or [C,1,K]

    size_t conv_w = layer.conv_depthwise_weight;
    if (wbuf.shape.size() == 2) {                   // [C, K] -> [C, 1, K]
        K = wbuf.shape[1];
        conv_w = gb->reshape(conv_w, {wbuf.shape[0], static_cast<size_t>(1), K});
    } else if (wbuf.shape.size() == 3) {            // [C, Cin(=1), K]
        if (wbuf.shape[1] != 1) {
            throw std::runtime_error("Depthwise conv expects weight shape [C,1,K] or [C,K]");
        }
        K = wbuf.shape[2];
        conv_w = gb->reshape(conv_w, {wbuf.shape[0], static_cast<size_t>(1), K});
    } else {
        throw std::runtime_error("Unexpected depthwise weight rank");
    }

    // If HF vs kernel tap order differs, uncomment this once and keep it:
    // conv_w = gb->flip(conv_w, /*axis=*/2); // reverse taps along K

    // Debug: weight views
    capture_debug_node(layer_idx, "conv_depthwise_weight_raw", layer.conv_depthwise_weight);
    capture_debug_node(layer_idx, "conv_depthwise_weight_reshaped_(Cout,1,K)", conv_w);
    {
        size_t w_c0 = gb->slice(conv_w, /*axis*/0, /*start*/0, /*len*/1);
        w_c0 = gb->reshape(w_c0, {K});
        capture_debug_node(layer_idx, "conv_depthwise_weight_c0_kernel_[K]", w_c0);
    }

    // Run causal depthwise conv on the whole sequence (no cache)
    // Input to kernel: [N=1, L, C]
    size_t x_nlc = gb->reshape(Bx, {static_cast<size_t>(1), L, C});
    capture_debug_node(layer_idx, "conv_input_Bx_[1,L,C]", x_nlc);

    const size_t dilation = 1;
    size_t y_nlc = gb->conv1d_causal(x_nlc, conv_w, K, dilation); // [1, L, C]
    capture_debug_node(layer_idx, "conv_output_NLC_[1,L,C]", y_nlc);

    // Back to [L, C] then gate and out-proj
    size_t y_lc = gb->reshape(y_nlc, {L, C});
    capture_debug_node(layer_idx, "conv_output_LC_[L,C]", y_lc);

    size_t gated = gb->multiply(Cg, y_lc);          // [L, C]
    capture_debug_node(layer_idx, "conv_gated_[L,C]", gated);

    size_t projected = gb->matmul(gated, layer.conv_out_proj_weight, true, backend); // [L, C]
    capture_debug_node(layer_idx, "conv_out_proj_[L,C]", projected);

    return projected; // [L, C]
}


size_t LFM2Model::build_attention(CactusGraph* gb, size_t normalized_input, uint32_t layer_idx,
                                 ComputeBackend backend, bool use_cache, size_t position_offset) {
    const auto& layer_entry = weight_nodes_.layers[layer_idx];
    const auto& layer = layer_entry.weights;

    auto q_proj_linear = gb->matmul(normalized_input, layer.attn_q_weight, true, backend);
    auto k_proj_linear = gb->matmul(normalized_input, layer.attn_k_weight, true, backend);
    auto v_proj_linear = gb->matmul(normalized_input, layer.attn_v_weight, true, backend);
    capture_debug_node(layer_idx, "attn_q_linear", q_proj_linear);
    capture_debug_node(layer_idx, "attn_k_linear", k_proj_linear);
    capture_debug_node(layer_idx, "attn_v_linear", v_proj_linear);

    const auto& q_shape = gb->get_output_buffer(q_proj_linear).shape;
    size_t batch_seq = q_shape[0];
    size_t num_heads = config_.attention_heads;
    size_t head_dim = config_.attention_head_dim;
    auto q_proj_reshaped = gb->reshape(q_proj_linear, {batch_seq * num_heads, head_dim});
    auto q_proj_norm = gb->rms_norm(q_proj_reshaped, layer.attn_q_norm_weight, config_.layer_norm_eps);
    auto q_proj = gb->reshape(q_proj_norm, {batch_seq, num_heads * head_dim});
    capture_debug_node(layer_idx, "attn_q_norm", q_proj);

    size_t num_kv_heads = config_.attention_kv_heads;
    auto k_proj_reshaped = gb->reshape(k_proj_linear, {batch_seq * num_kv_heads, head_dim});
    auto k_proj_norm = gb->rms_norm(k_proj_reshaped, layer.attn_k_norm_weight, config_.layer_norm_eps);
    auto k_proj = gb->reshape(k_proj_norm, {batch_seq, num_kv_heads * head_dim});
    capture_debug_node(layer_idx, "attn_k_norm", k_proj);

    size_t seq_len = batch_seq;
    auto q_proj_4d = gb->reshape(q_proj, {1, seq_len, config_.attention_heads, config_.attention_head_dim});
    auto k_proj_4d = gb->reshape(k_proj, {1, seq_len, config_.attention_kv_heads, config_.attention_head_dim});
    auto v_proj_4d = gb->reshape(v_proj_linear, {1, seq_len, config_.attention_kv_heads, config_.attention_head_dim});
    capture_debug_node(layer_idx, "attn_q_pre_rope", q_proj_4d);
    capture_debug_node(layer_idx, "attn_k_pre_rope", k_proj_4d);
    capture_debug_node(layer_idx, "attn_v", v_proj_4d);

    if (config_.rope_theta > 0) {
        q_proj_4d = gb->rope(q_proj_4d, config_.rope_theta, position_offset);
        k_proj_4d = gb->rope(k_proj_4d, config_.rope_theta, position_offset);
        capture_debug_node(layer_idx, "attn_q_rope", q_proj_4d);
        capture_debug_node(layer_idx, "attn_k_rope", k_proj_4d);
    }

    size_t final_k = k_proj_4d;
    size_t final_v = v_proj_4d;

    if (use_cache && !kv_cache_.is_empty()) {
        auto k_view = kv_cache_.get_key_view(layer_idx);
        auto v_view = kv_cache_.get_value_view(layer_idx);

        if (k_view.ptr2 == nullptr && v_view.ptr2 == nullptr) {
            size_t cache_k_node = gb->input({1, kv_cache_.current_seq_len, config_.attention_kv_heads, config_.attention_head_dim}, kv_cache_.precision);
            size_t cache_v_node = gb->input({1, kv_cache_.current_seq_len, config_.attention_kv_heads, config_.attention_head_dim}, kv_cache_.precision);

            gb->set_input(cache_k_node, k_view.ptr1, kv_cache_.precision);
            // std::cout << "Cache K ptr1: " << k_view.ptr1 << std::endl;
            gb->set_input(cache_v_node, v_view.ptr1, kv_cache_.precision);

            final_k = gb->concat(cache_k_node, k_proj_4d, 1);
            final_v = gb->concat(cache_v_node, v_proj_4d, 1);
        } else {
            size_t cache_k_node = gb->input({1, kv_cache_.current_seq_len, config_.attention_kv_heads, config_.attention_head_dim}, kv_cache_.precision);
            size_t cache_v_node = gb->input({1, kv_cache_.current_seq_len, config_.attention_kv_heads, config_.attention_head_dim}, kv_cache_.precision);

            gb->set_input(cache_k_node, kv_cache_.get_key_ptr(layer_idx), kv_cache_.precision);
            gb->set_input(cache_v_node, kv_cache_.get_value_ptr(layer_idx), kv_cache_.precision);

            final_k = gb->concat(cache_k_node, k_proj_4d, 1);
            final_v = gb->concat(cache_v_node, v_proj_4d, 1);
        }
    }

    if (use_cache) {
        cache_k_output_nodes_[layer_idx] = final_k;
        cache_v_output_nodes_[layer_idx] = final_v;
    }


    auto attn_output_4d = gb->attention(q_proj_4d, final_k, final_v, attention_scale_, position_offset);
    capture_debug_node(layer_idx, "attn_scores_output", attn_output_4d);
    auto attn_output = gb->reshape(attn_output_4d, {seq_len, config_.attention_head_dim * config_.attention_heads});
    capture_debug_node(layer_idx, "attn_output_flat", attn_output);
    auto projected = gb->matmul(attn_output, layer.attn_output_weight, true, backend);
    capture_debug_node(layer_idx, "attn_out_proj", projected);
    return projected;
}


size_t LFM2Model::build_mlp(CactusGraph* gb, size_t normalized_h, uint32_t layer_idx,
                           ComputeBackend backend) const {
    const auto& layer_entry = weight_nodes_.layers[layer_idx];
    const auto& layer = layer_entry.weights;
    size_t gate_output = gb->matmul(normalized_h, layer.ffn_gate_weight, true, backend);
    capture_debug_node(layer_idx, "mlp_gate_linear", gate_output);
    size_t up_output = gb->matmul(normalized_h, layer.ffn_up_weight, true, backend);
    capture_debug_node(layer_idx, "mlp_up_linear", up_output);
    size_t gate_silu = gb->silu(gate_output);
    capture_debug_node(layer_idx, "mlp_gate_silu", gate_silu);
    size_t gated = gb->multiply(gate_silu, up_output);
    capture_debug_node(layer_idx, "mlp_gated", gated);
    auto down_output = gb->matmul(gated, layer.ffn_down_weight, true, backend);
    capture_debug_node(layer_idx, "mlp_down_linear", down_output);
    return down_output;
}


size_t LFM2Model::build_transformer_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx,
                                         ComputeBackend backend, bool use_cache, size_t position_offset) {
    const auto& layer_entry = weight_nodes_.layers[layer_idx];
    const auto& layer = layer_entry.weights;
    
    auto normalized_input = gb->rms_norm(hidden, layer.input_layernorm_weight, config_.layer_norm_eps);
    capture_debug_node(layer_idx, "input_norm", normalized_input);
    
    size_t block_output;
    if (layer_entry.type == WeightNodeIDs::LayerType::CONV) {
        // Conv layer: use conv1d instead of attention
        block_output = build_conv1d(gb, normalized_input, layer_idx, backend);
    } else {
        // Attention layer: standard attention
        block_output = build_attention(gb, normalized_input, layer_idx, backend, use_cache, position_offset);
    }
    capture_debug_node(layer_idx, "block_main_output", block_output);
    
    auto after_block = gb->add(hidden, block_output);
    capture_debug_node(layer_idx, "post_block_residual", after_block);
    auto normalized_after_block = gb->rms_norm(after_block, layer.post_attention_layernorm_weight, config_.layer_norm_eps);
    capture_debug_node(layer_idx, "post_block_norm", normalized_after_block);
    auto mlp_output = build_mlp(gb, normalized_after_block, layer_idx, backend);
    capture_debug_node(layer_idx, "mlp_output", mlp_output);
    auto block_result = gb->add(after_block, mlp_output);
    capture_debug_node(layer_idx, "block_output", block_result);
    return block_result;
}


size_t LFM2Model::forward(const std::vector<uint32_t>& tokens, bool use_cache) {
    if (!initialized_ || !graph_handle_) {
        throw std::runtime_error("Model not initialized - call init() first");
    }

    if (tokens.empty()) {
        throw std::runtime_error("Token sequence cannot be empty");
    }

    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    gb->soft_reset();

    pending_conv_updates_.clear();

    auto seq_len = static_cast<size_t>(tokens.size());

    size_t position_offset = use_cache ? kv_cache_.get_total_seq_len() : 0;

    auto backend = config_.default_backend == Config::Backend::CPU
        ? ComputeBackend::CPU
        : ComputeBackend::NPU;

    auto input_node_id = gb->input({seq_len}, Precision::FP32);
    capture_debug_node(input_node_id, "input_tokens", input_node_id);
    auto hidden = gb->embedding(embedding_node_id_, input_node_id);
    const uint32_t global_layer = std::numeric_limits<uint32_t>::max();
    capture_debug_node(global_layer, "embedding_output", hidden);

    for (uint32_t layer_idx = 0; layer_idx < config_.num_layers; layer_idx++) {
        hidden = build_transformer_block(gb, hidden, layer_idx, backend, use_cache, position_offset);
    }

    auto final_hidden = gb->rms_norm(hidden, weight_nodes_.output_norm_weight, config_.layer_norm_eps);
    capture_debug_node(global_layer, "final_norm", final_hidden);

    std::vector<float> input_data(seq_len);
    for (size_t i = 0; i < seq_len; i++) {
        input_data[i] = static_cast<float>(tokens[i]);
    }
    gb->set_input(input_node_id, input_data.data(), Precision::FP32);

    return final_hidden;
}

void LFM2Model::enqueue_conv_cache_update(uint32_t layer_idx, size_t node_id, size_t seq_len, size_t hidden_dim) {
    if (config_.conv_L_cache == 0 || seq_len == 0) {
        return;
    }

    pending_conv_updates_.push_back({layer_idx, node_id, seq_len, hidden_dim});
}

void LFM2Model::apply_pending_conv_cache_updates(CactusGraph* gb) {
    if (pending_conv_updates_.empty() || config_.conv_L_cache == 0) {
        pending_conv_updates_.clear();
        return;
    }

    for (const auto& update : pending_conv_updates_) {
        const auto& buffer = gb->get_output_buffer(update.node_id);
        if (buffer.precision != conv_cache_.precision) {
            std::cerr << "[Cactus][LFM2Model] Conv cache precision mismatch for layer "
                      << update.layer_idx << std::endl;
            continue;
        }

        auto* raw = static_cast<uint8_t*>(gb->get_output(update.node_id));
        if (!raw) {
            continue;
        }

        size_t stride = update.hidden_dim * PrecisionTraits::size_of(buffer.precision);
        for (size_t i = 0; i < update.seq_len; ++i) {
            conv_cache_.update(update.layer_idx, raw + i * stride);
        }
    }

    pending_conv_updates_.clear();
}

void LFM2Model::post_execute_updates(CactusGraph* gb, size_t seq_len) {
    apply_pending_conv_cache_updates(gb);
    Model::post_execute_updates(gb, seq_len);
}

}
}