#include "model.h"
#include "../graph/graph.h"
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <set>

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
    const auto& layer = layer_entry.weights;

    auto in_proj = gb->matmul(input, layer.conv_in_proj_weight, true, backend);
    const auto& in_proj_buffer = gb->get_output_buffer(in_proj);

    if (in_proj_buffer.shape.size() != 2 || in_proj_buffer.shape[1] % 3 != 0) {
        throw std::runtime_error("Conv in_proj output must be 2D with channels divisible by 3");
    }

    size_t seq_len = in_proj_buffer.shape[0];
    size_t hidden_dim = in_proj_buffer.shape[1] / 3;

    auto triplet = gb->reshape(in_proj, {seq_len, static_cast<size_t>(3), hidden_dim});
    auto B = gb->slice(triplet, 1, 0, 1);
    auto C = gb->slice(triplet, 1, 1, 1);
    auto X = gb->slice(triplet, 1, 2, 1);

    B = gb->reshape(B, {seq_len, hidden_dim});
    C = gb->reshape(C, {seq_len, hidden_dim});
    X = gb->reshape(X, {seq_len, hidden_dim});

    auto Bx = gb->multiply(B, X);

    auto& depthwise_buffer = gb->get_output_buffer(layer.conv_depthwise_weight);
    size_t kernel_size = depthwise_buffer.shape.back();  // L

    if (seq_len != 1) {
        size_t conv_weight = layer.conv_depthwise_weight;

        if (depthwise_buffer.shape.size() == 2) {
            kernel_size = depthwise_buffer.shape[1];
            conv_weight = gb->reshape(conv_weight, {depthwise_buffer.shape[0], static_cast<size_t>(1), kernel_size});
        } else if (depthwise_buffer.shape.size() == 3 && depthwise_buffer.shape[1] != 1) {
            conv_weight = gb->reshape(conv_weight, {depthwise_buffer.shape[0], depthwise_buffer.shape[1], depthwise_buffer.shape[2]});
            kernel_size = depthwise_buffer.shape[2];
        }

        auto bx_prefill = gb->reshape(Bx, {static_cast<size_t>(1), seq_len, hidden_dim});
        auto conv_prefill = gb->conv1d_causal(bx_prefill, conv_weight, kernel_size, 1);
        conv_prefill = gb->reshape(conv_prefill, {seq_len, hidden_dim});

        auto gated = gb->multiply(C, conv_prefill);

        void* bx_ptr = gb->get_output(Bx);
        if (bx_ptr) {
            auto* bx_bytes = static_cast<uint8_t*>(bx_ptr);
            size_t stride = hidden_dim * conv_cache_.element_size;
            for (size_t i = 0; i < seq_len; ++i) {
                conv_cache_.update(layer_idx, bx_bytes + i * stride);
            }
        }

        return gb->matmul(gated, layer.conv_out_proj_weight, true, backend);
    }

    auto view = conv_cache_.get_window(layer_idx);

    size_t window_node;

    if (view.len2 > 0) {
        size_t L_node = gb->input({view.len2, hidden_dim}, conv_cache_.precision);
        size_t R_node = gb->input({view.len1, hidden_dim}, conv_cache_.precision);

        gb->set_input(L_node, view.ptr2, conv_cache_.precision);
        gb->set_input(R_node, view.ptr1, conv_cache_.precision);

        window_node = gb->concat(L_node, R_node, 0);  // chronological order: [head:L] + [0:head]
    } else {
        size_t cache_node = gb->input({view.total_len, hidden_dim}, conv_cache_.precision);
        gb->set_input(cache_node, view.ptr1, conv_cache_.precision);
        window_node = cache_node;
    }

    window_node = gb->concat(window_node, Bx, 0);

    const auto& window_buffer = gb->get_output_buffer(window_node);
    size_t total_window_len = window_buffer.shape[0];
    size_t slice_start = (total_window_len >= kernel_size) ? (total_window_len - kernel_size) : 0;
    size_t slice_len = std::min(kernel_size, total_window_len);

    auto sliced_window = gb->slice(window_node, 0, slice_start, slice_len);

    sliced_window = gb->reshape(sliced_window, {slice_len, hidden_dim});
    sliced_window = gb->transpose(sliced_window);

    size_t depthwise = layer.conv_depthwise_weight;
    if (depthwise_buffer.shape.size() == 3) {
        depthwise = gb->reshape(depthwise, {depthwise_buffer.shape[0], depthwise_buffer.shape[2]});
    } else if (depthwise_buffer.shape.size() == 2) {
        depthwise = gb->reshape(depthwise, {depthwise_buffer.shape[0], depthwise_buffer.shape[1]});
    }

    auto conv_mul = gb->multiply(sliced_window, depthwise);
    auto conv_out = gb->sum(conv_mul, 1);
    conv_out = gb->reshape(conv_out, {static_cast<size_t>(1), hidden_dim});

    auto gate = gb->slice(C, 0, seq_len - 1, 1);
    gate = gb->reshape(gate, {static_cast<size_t>(1), hidden_dim});

    auto gated = gb->multiply(gate, conv_out);

    void* bx_ptr = gb->get_output(Bx);
    conv_cache_.update(layer_idx, bx_ptr);

    return gb->matmul(gated, layer.conv_out_proj_weight, true, backend);
}

size_t LFM2Model::build_attention(CactusGraph* gb, size_t normalized_input, uint32_t layer_idx,
                                 ComputeBackend backend, bool use_cache, size_t position_offset) {
    const auto& layer_entry = weight_nodes_.layers[layer_idx];
    const auto& layer = layer_entry.weights;

    auto q_proj = gb->matmul(normalized_input, layer.attn_q_weight, true, backend);
    auto k_proj = gb->matmul(normalized_input, layer.attn_k_weight, true, backend);
    auto v_proj = gb->matmul(normalized_input, layer.attn_v_weight, true, backend);

    const auto& q_shape = gb->get_output_buffer(q_proj).shape;
    size_t batch_seq = q_shape[0];
    size_t num_heads = config_.attention_heads;
    size_t head_dim = config_.attention_head_dim;
    q_proj = gb->reshape(q_proj, {batch_seq * num_heads, head_dim});
    q_proj = gb->rms_norm(q_proj, layer.attn_q_norm_weight, config_.layer_norm_eps);
    q_proj = gb->reshape(q_proj, {batch_seq, num_heads * head_dim});

    size_t num_kv_heads = config_.attention_kv_heads;
    k_proj = gb->reshape(k_proj, {batch_seq * num_kv_heads, head_dim});
    k_proj = gb->rms_norm(k_proj, layer.attn_k_norm_weight, config_.layer_norm_eps);
    k_proj = gb->reshape(k_proj, {batch_seq, num_kv_heads * head_dim});

    size_t seq_len = batch_seq;

    auto q_proj_4d = gb->reshape(q_proj, {1, seq_len, config_.attention_heads, config_.attention_head_dim});
    auto k_proj_4d = gb->reshape(k_proj, {1, seq_len, config_.attention_kv_heads, config_.attention_head_dim});
    auto v_proj_4d = gb->reshape(v_proj, {1, seq_len, config_.attention_kv_heads, config_.attention_head_dim});

    if (config_.rope_theta > 0) {
        q_proj_4d = gb->rope(q_proj_4d, config_.rope_theta, position_offset);
        k_proj_4d = gb->rope(k_proj_4d, config_.rope_theta, position_offset);
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
    auto attn_output = gb->reshape(attn_output_4d, {seq_len, config_.attention_head_dim * config_.attention_heads});
    return gb->matmul(attn_output, layer.attn_output_weight, true, backend);
}


size_t LFM2Model::build_mlp(CactusGraph* gb, size_t normalized_h, uint32_t layer_idx,
                           ComputeBackend backend) const {
    const auto& layer_entry = weight_nodes_.layers[layer_idx];
    const auto& layer = layer_entry.weights;
    size_t gate_output = gb->matmul(normalized_h, layer.ffn_gate_weight, true, backend);
    size_t up_output = gb->matmul(normalized_h, layer.ffn_up_weight, true, backend);
    size_t gate_silu = gb->silu(gate_output);
    size_t gated = gb->multiply(gate_silu, up_output);
    return gb->matmul(gated, layer.ffn_down_weight, true, backend);
}


size_t LFM2Model::build_transformer_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx,
                                         ComputeBackend backend, bool use_cache, size_t position_offset) {
    const auto& layer_entry = weight_nodes_.layers[layer_idx];
    const auto& layer = layer_entry.weights;
    
    auto normalized_input = gb->rms_norm(hidden, layer.input_layernorm_weight, config_.layer_norm_eps);
    
    size_t block_output;
    if (layer_entry.type == WeightNodeIDs::LayerType::CONV) {
        // Conv layer: use conv1d instead of attention
        block_output = build_conv1d(gb, normalized_input, layer_idx, backend);
    } else {
        // Attention layer: standard attention
        block_output = build_attention(gb, normalized_input, layer_idx, backend, use_cache, position_offset);
    }
    
    auto after_block = gb->add(hidden, block_output);
    auto normalized_after_block = gb->rms_norm(after_block, layer.post_attention_layernorm_weight, config_.layer_norm_eps);
    auto mlp_output = build_mlp(gb, normalized_after_block, layer_idx, backend);
    return gb->add(after_block, mlp_output);
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

    auto seq_len = static_cast<size_t>(tokens.size());

    size_t position_offset = use_cache ? kv_cache_.get_total_seq_len() : 0;

    auto backend = config_.default_backend == Config::Backend::CPU
        ? ComputeBackend::CPU
        : ComputeBackend::NPU;

    auto input_node_id = gb->input({seq_len}, Precision::FP32);
    auto hidden = gb->embedding(embedding_node_id_, input_node_id);

    for (uint32_t layer_idx = 0; layer_idx < config_.num_layers; layer_idx++) {
        hidden = build_transformer_block(gb, hidden, layer_idx, backend, use_cache, position_offset);
    }

    auto final_hidden = gb->rms_norm(hidden, weight_nodes_.output_norm_weight, config_.layer_norm_eps);

    std::vector<float> input_data(seq_len);
    for (size_t i = 0; i < seq_len; i++) {
        input_data[i] = static_cast<float>(tokens[i]);
    }
    gb->set_input(input_node_id, input_data.data(), Precision::FP32);

    return final_hidden;
}

}
}