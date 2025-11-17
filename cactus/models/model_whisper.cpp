#include "model.h"
#include "../graph/graph.h"
#include <cmath>
#include <stdexcept>
#include <set>
#include <iostream>

namespace cactus {
namespace engine {


struct ConvDebugNodes {
    size_t conv1;
    size_t conv2;
    size_t conv2_transposed;
    size_t output;
};


static void apply_whisper_logits_processors(
    float* logits,
    size_t vocab_size,
    const std::vector<size_t>& suppress_tokens,
    const std::vector<size_t>& begin_suppress_tokens,
    bool is_first_decode_step
) {
    // 1. Always-suppressed tokens
    for (int tid : suppress_tokens) {
        if (tid >= 0 && static_cast<size_t>(tid) < vocab_size) {
            logits[tid] = -1e9f;  // effectively -inf
        }
    }

    // 2. Tokens suppressed only at the *beginning*
    if (is_first_decode_step) {
        for (int tid : begin_suppress_tokens) {
            if (tid >= 0 && static_cast<size_t>(tid) < vocab_size) {
                logits[tid] = -1e9f;
            }
        }
    }
}

static void debug_print_tensor_5x5(CactusGraph* gb, size_t node_id, const char* name) {
    auto& buf = gb->get_output_buffer(node_id);
    const std::vector<size_t>& shape = buf.shape;
    const float* ptr = reinterpret_cast<const float*>(gb->get_output(node_id));

    if (shape.size() < 2) {
        std::cout << "(Tensor rank < 2, cannot print 5×5)" << std::endl;
        return;
    }

    size_t rows = shape[0];
    size_t cols = shape[1];

    size_t rmax = std::min<size_t>(rows, 5);
    size_t cmax = std::min<size_t>(cols, 5);

    for (size_t r = 0; r < rmax; ++r) {
        std::cout << "row " << r << ": ";
        for (size_t c = 0; c < cmax; ++c) {
            float v = ptr[r * cols + c];
            std::cout << v << "  ";
        }
        std::cout << std::endl;
    }
    std::cout << "------------------------------------------" << std::endl;
}

WhisperModel::WhisperModel() : Model() {}

WhisperModel::WhisperModel(const Config& config) : Model(config) {
    weight_nodes_.layers.resize(config.num_layers);

    float hd = static_cast<float>(config.attention_head_dim);
    if (hd <= 0.0f) {
        // Fallback to Whisper medium defaults
        hd = 64.0f;
    }

    // Safe scaling to avoid overflow in softmax inside the ATTENTION op
    attention_scale_ = 1.0f / std::sqrt(hd);

    encoder_block_out_nodes_.resize(config.num_layers, 0);
}

void WhisperModel::load_weights_to_graph(CactusGraph* gb) {

    embedding_node_id_ = gb->mmap_embeddings(embedding_file_path_); //Updated engine_model to use decoder embeddings for whisper
    
    weight_nodes_.decoder_norm_weight = gb->mmap_weights(model_folder_path_ + "/decoder_norm.weights");
    weight_nodes_.decoder_norm_bias = gb->mmap_weights(model_folder_path_ + "/decoder_norm.bias");
    weight_nodes_.decoder_position_embeddings_weight = gb->mmap_weights(model_folder_path_ + "/decoder_position_embeddings.weights");

    weight_nodes_.encoder_position_embeddings = gb->mmap_weights(model_folder_path_ + "/encoder_position_embeddings.weights");
    weight_nodes_.encoder_conv1_weight = gb->mmap_weights(model_folder_path_ + "/encoder_conv1_weight.weights");
    weight_nodes_.encoder_conv1_bias = gb->mmap_weights(model_folder_path_ + "/encoder_conv1_bias.bias");
    weight_nodes_.encoder_conv2_weight = gb->mmap_weights(model_folder_path_ + "/encoder_conv2_weight.weights");
    weight_nodes_.encoder_conv2_bias = gb->mmap_weights(model_folder_path_ + "/encoder_conv2_bias.bias");
    weight_nodes_.encoder_norm_weight = gb->mmap_weights(model_folder_path_ + "/encoder_norm_weight.weights");
    weight_nodes_.encoder_norm_bias = gb->mmap_weights(model_folder_path_ + "/encoder_norm_bias.bias");

    if (config_.tie_word_embeddings) {
        weight_nodes_.output_weight = embedding_node_id_;
        output_weight_node_id_ = embedding_node_id_;
    } else {
        weight_nodes_.output_weight = gb->mmap_weights(model_folder_path_ + "/output_weight.weights");
        output_weight_node_id_ = weight_nodes_.output_weight;
    }

    for (uint32_t i = 0; i < config_.num_layers; i++) {
        auto& layer = weight_nodes_.layers[i];

        //Decoder Layers
        std::string layer_prefix = model_folder_path_ + "/decoder.layer_" + std::to_string(i) + "_";

        layer.decoder_encoder_attn_k_weight = gb->mmap_weights(layer_prefix + "encoder_attn_k.weights");
        layer.decoder_encoder_attn_q_weight = gb->mmap_weights(layer_prefix + "encoder_attn_q.weights");
        layer.decoder_encoder_attn_v_weight = gb->mmap_weights(layer_prefix + "encoder_attn_v.weights");
        layer.decoder_encoder_attn_output_weight = gb->mmap_weights(layer_prefix + "encoder_attn_output.weights");
        layer.decoder_encoder_attn_q_bias = gb->mmap_weights(layer_prefix + "encoder_attn_q.bias");
        layer.decoder_encoder_attn_v_bias = gb->mmap_weights(layer_prefix + "encoder_attn_v.bias");
        layer.decoder_encoder_attn_output_bias = gb->mmap_weights(layer_prefix + "encoder_attn_output.bias");

        layer.decoder_post_encoder_layernorm_weight = gb->mmap_weights(layer_prefix + "encoder_attn_norm.weights");
        layer.decoder_post_encoder_layernorm_bias = gb->mmap_weights(layer_prefix + "encoder_attn_norm.bias");

        layer.decoder_ffn1_weight = gb->mmap_weights(layer_prefix + "mlp_fc1.weights");
        layer.decoder_ffn1_bias = gb->mmap_weights(layer_prefix + "mlp_fc1.bias");
        layer.decoder_ffn2_weight = gb->mmap_weights(layer_prefix + "mlp_fc2.weights");
        layer.decoder_ffn2_bias = gb->mmap_weights(layer_prefix + "mlp_fc2.bias");

        layer.decoder_post_ffn_layernorm_weight = gb->mmap_weights(layer_prefix + "final_norm.weights");
        layer.decoder_post_ffn_layernorm_bias = gb->mmap_weights(layer_prefix + "final_norm.bias");

        layer.decoder_self_attn_k_weight = gb->mmap_weights(layer_prefix + "self_attn_k.weights");
        layer.decoder_self_attn_q_weight = gb->mmap_weights(layer_prefix + "self_attn_q.weights");
        layer.decoder_self_attn_v_weight = gb->mmap_weights(layer_prefix + "self_attn_v.weights");
        layer.decoder_self_attn_output_weight = gb->mmap_weights(layer_prefix + "self_attn_output.weights");
        layer.decoder_self_attn_q_bias = gb->mmap_weights(layer_prefix + "self_attn_q.bias");
        layer.decoder_self_attn_v_bias = gb->mmap_weights(layer_prefix + "self_attn_v.bias");
        layer.decoder_self_attn_output_bias = gb->mmap_weights(layer_prefix + "self_attn_output.bias");

        layer.decoder_post_attn_layernorm_weight = gb->mmap_weights(layer_prefix + "self_attn_norm.weights");
        layer.decoder_post_attn_layernorm_bias = gb->mmap_weights(layer_prefix + "self_attn_norm.bias");

        //Encoder Layers
        layer_prefix = model_folder_path_ + "/encoder.layer_" + std::to_string(i) + "_";

        layer.encoder_ffn1_weight = gb->mmap_weights(layer_prefix + "mlp_fc1.weights");
        layer.encoder_ffn1_bias = gb->mmap_weights(layer_prefix + "mlp_fc1.bias");
        layer.encoder_ffn2_weight = gb->mmap_weights(layer_prefix + "mlp_fc2.weights");
        layer.encoder_ffn2_bias = gb->mmap_weights(layer_prefix + "mlp_fc2.bias");

        layer.encoder_post_ffn_layernorm_weight = gb->mmap_weights(layer_prefix + "final_norm.weights");
        layer.encoder_post_ffn_layernorm_bias = gb->mmap_weights(layer_prefix + "final_norm.bias");

        layer.encoder_self_attn_k_weight = gb->mmap_weights(layer_prefix + "self_attn_k.weights");
        layer.encoder_self_attn_q_weight = gb->mmap_weights(layer_prefix + "self_attn_q.weights");
        layer.encoder_self_attn_v_weight = gb->mmap_weights(layer_prefix + "self_attn_v.weights");
        layer.encoder_self_attn_output_weight = gb->mmap_weights(layer_prefix + "self_attn_output.weights");
        layer.encoder_self_attn_q_bias = gb->mmap_weights(layer_prefix + "self_attn_q.bias");
        layer.encoder_self_attn_v_bias = gb->mmap_weights(layer_prefix + "self_attn_v.bias");
        layer.encoder_self_attn_output_bias = gb->mmap_weights(layer_prefix + "self_attn_output.bias");

        layer.encoder_post_attn_layernorm_weight = gb->mmap_weights(layer_prefix + "self_attn_norm.weights");
        layer.encoder_post_attn_layernorm_bias = gb->mmap_weights(layer_prefix + "self_attn_norm.bias");
    }
}   

size_t WhisperModel::build_encoder_mlp(CactusGraph* gb, size_t input, uint32_t layer_idx, ComputeBackend backend) {
    const auto& layer = weight_nodes_.layers[layer_idx];

    auto ffn1_weight = gb->matmul(input, layer.encoder_ffn1_weight, true, backend);
    auto ffn1_bias = gb->add(ffn1_weight, layer.encoder_ffn1_bias);

    encoder_pre_gelu = ffn1_bias;

    auto ffn1_act = gb->gelu_erf(ffn1_bias);

    encoder_post_gelu = ffn1_act;

    auto ffn2_weight = gb->matmul(ffn1_act, layer.encoder_ffn2_weight, true, backend);
    auto ffn2_bias = gb->add(ffn2_weight, layer.encoder_ffn2_bias);
    return ffn2_bias;
}

size_t WhisperModel::build_decoder_mlp(CactusGraph* gb, size_t input, uint32_t layer_idx, ComputeBackend backend) const {
    const auto& layer = weight_nodes_.layers[layer_idx];
    auto ffn1_weight = gb->matmul(input, layer.decoder_ffn1_weight, true, backend);
    auto ffn1_bias = gb->add(ffn1_weight, layer.decoder_ffn1_bias);
    auto ffn1_act = gb->gelu_erf(ffn1_bias);
    auto ffn2_weight = gb->matmul(ffn1_act, layer.decoder_ffn2_weight, true, backend);
    auto ffn2_bias = gb->add(ffn2_weight, layer.decoder_ffn2_bias);
    return ffn2_bias;
}



size_t WhisperModel::build_encoder_attention(CactusGraph* gb, size_t input, uint32_t layer_idx, ComputeBackend backend, bool use_cache, size_t position_offset){
    const auto& layer = weight_nodes_.layers[layer_idx];

    //
    // 1. Q from *already normalized* decoder hidden
    //
    size_t q = gb->matmul(input, layer.decoder_encoder_attn_q_weight, true, backend);
    q = gb->add(q, layer.decoder_encoder_attn_q_bias);

    //
    // 2. K/V from encoder output (already normalized by encoder stack)
    //
    size_t enc_norm = weight_nodes_.encoder_output;

    // K: no bias
    size_t k = gb->matmul(enc_norm, layer.decoder_encoder_attn_k_weight, true, backend);

    // V: with bias
    size_t v = gb->matmul(enc_norm, layer.decoder_encoder_attn_v_weight, true, backend);
    v = gb->add(v, layer.decoder_encoder_attn_v_bias);

    //
    // 3. Reshape Q/K/V
    //
    size_t T_dec = gb->get_output_buffer(q).shape[0];
    size_t T_enc = gb->get_output_buffer(k).shape[0];

    size_t q_heads  = config_.attention_heads;
    size_t kv_heads = config_.attention_kv_heads;
    size_t head_dim = config_.attention_head_dim;

    q = gb->reshape(q, {1, T_dec, q_heads,  head_dim});
    k = gb->reshape(k, {1, T_enc, kv_heads, head_dim});
    v = gb->reshape(v, {1, T_enc, kv_heads, head_dim});

    //
    // 4. Cross-attention (non-causal)
    //
    size_t attn = gb->attention(q, k, v, attention_scale_, /*is_causal=*/false);

    //
    // 5. Output projection
    //
    attn = gb->reshape(attn, {T_dec, q_heads * head_dim});
    size_t out = gb->matmul(attn, layer.decoder_encoder_attn_output_weight, true, backend);
    out = gb->add(out, layer.decoder_encoder_attn_output_bias);

    return out;
}

void WhisperModel::reset_graph_side_cache_nodes() {
    cache_k_output_nodes_.assign(config_.num_layers, 0);
    cache_v_output_nodes_.assign(config_.num_layers, 0);
}

size_t WhisperModel::build_decoder_self_attention(CactusGraph* gb, size_t input, uint32_t layer_idx, ComputeBackend backend, bool use_cache, size_t position_offset){
    const auto& layer = weight_nodes_.layers[layer_idx];
    auto q = gb->matmul(input, layer.decoder_self_attn_q_weight, true, backend);

    q = gb->add(q, layer.decoder_self_attn_q_bias);
    auto k = gb->matmul(input, layer.decoder_self_attn_k_weight, true, backend);
    auto v = gb->matmul(input, layer.decoder_self_attn_v_weight, true, backend);
    v = gb->add(v, layer.decoder_self_attn_v_bias);

    size_t seq_new   = gb->get_output_buffer(q).shape[0];
    size_t num_heads = config_.attention_heads;
    size_t head_dim  = config_.attention_head_dim;
    q = gb->reshape(q, {1, seq_new, num_heads, head_dim});
    k = gb->reshape(k, {1, seq_new, num_heads, head_dim});
    v = gb->reshape(v, {1, seq_new, num_heads, head_dim});

    size_t final_k = k;
    size_t final_v = v;

    if (use_cache && kv_cache_.current_seq_len > 0 && cache_k_output_nodes_[layer_idx] != 0 && cache_v_output_nodes_[layer_idx] != 0) {
        // These nodes were freshly recreated in generate_with_audio()
        if (cache_k_output_nodes_[layer_idx] != 0 && cache_v_output_nodes_[layer_idx] != 0) {
            final_k = gb->concat(cache_k_output_nodes_[layer_idx], k, 1);
            final_v = gb->concat(cache_v_output_nodes_[layer_idx], v, 1);
        } else {
            std::cerr << "[Warning] Missing cache nodes for layer " << layer_idx << ", skipping concat." << std::endl;
        }
    }

    cache_k_output_nodes_[layer_idx] = final_k;
    cache_v_output_nodes_[layer_idx] = final_v;

    auto attn_out_4d = gb->attention(q, final_k, final_v, attention_scale_, position_offset);
    auto attn_out = gb->reshape(attn_out_4d, {seq_new, num_heads * head_dim});
    auto output = gb->matmul(attn_out, layer.decoder_self_attn_output_weight, true, backend);
    output = gb->add(output, layer.decoder_self_attn_output_bias);
    return output;
}

size_t WhisperModel::build_encoder_self_attention(CactusGraph* gb, size_t input, uint32_t layer_idx, ComputeBackend backend, bool use_cache, size_t position_offset){
    const auto& layer = weight_nodes_.layers[layer_idx];

    if(use_cache)
        throw std::runtime_error("The encoder attention layers are not auto-regressive, and thus don't use KV caching!");

    auto q = gb->matmul(input, layer.encoder_self_attn_q_weight, true, backend);
    q = gb->add(q, layer.encoder_self_attn_q_bias);
    auto v = gb->matmul(input, layer.encoder_self_attn_v_weight, true, backend);
    v = gb->add(v, layer.encoder_self_attn_v_bias);
    auto k = gb->matmul(input, layer.encoder_self_attn_k_weight, true, backend);

    size_t seq_len = gb->get_output_buffer(q).shape[0];
    size_t num_heads = config_.attention_heads;
    size_t head_dim  = config_.attention_head_dim;

    q = gb->reshape(q, {1, seq_len, num_heads, head_dim});
    k = gb->reshape(k, {1, seq_len, num_heads, head_dim});
    v = gb->reshape(v, {1, seq_len, num_heads, head_dim});

    auto attn = gb->attention(q, k, v, attention_scale_, false);

    attn = gb->reshape(attn, {seq_len, num_heads * head_dim});

    auto output = gb->matmul(attn, layer.encoder_self_attn_output_weight, true, backend);
    output = gb->add(output, layer.encoder_self_attn_output_bias);

    return output;
}

size_t WhisperModel::build_conv1d(CactusGraph* gb, size_t input, ComputeBackend backend)
{
    //
    // Conv1 (stride 1) + bias
    //
    size_t conv1 = gb->conv1d_k3(input, weight_nodes_.encoder_conv1_weight, 1);

    auto bias1_shape = gb->get_output_buffer(weight_nodes_.encoder_conv1_bias).shape;
    size_t C1 = bias1_shape[0];
    size_t bias1 = gb->reshape(weight_nodes_.encoder_conv1_bias, {1, C1, 1});
    conv1 = gb->add(conv1, bias1);

    last_conv1_node_ = conv1;

    conv1 = gb->gelu_erf(conv1);

    //
    // Conv2 (stride 2) + bias
    //
    size_t conv2 = gb->conv1d_k3(conv1, weight_nodes_.encoder_conv2_weight, 2);

    auto bias2_shape = gb->get_output_buffer(weight_nodes_.encoder_conv2_bias).shape;
    size_t C2 = bias2_shape[0];
    size_t bias2 = gb->reshape(weight_nodes_.encoder_conv2_bias, {1, C2, 1});
    conv2 = gb->add(conv2, bias2);

    last_conv2_node_ = conv2;

    conv2 = gb->gelu_erf(conv2);

    //
    // Transpose → NLC
    //
    size_t conv2_transposed = gb->transpose(conv2, ComputeBackend::CPU);
    last_conv2_transposed_node_ = conv2_transposed;

    return conv2_transposed;
}




size_t WhisperModel::build_encoder_transformer_block(
    CactusGraph* gb,
    size_t hidden,
    uint32_t layer_idx,
    ComputeBackend backend,
    bool use_cache,
    size_t position_offset)
{
    const auto& layer = weight_nodes_.layers[layer_idx];

    size_t ln1 = gb->layernorm(
        hidden,
        layer.encoder_post_attn_layernorm_weight,
        layer.encoder_post_attn_layernorm_bias
    );

    size_t sa = build_encoder_self_attention(
        gb, ln1, layer_idx, backend, use_cache, position_offset
    );

    size_t x_post_sa = gb->add(hidden, sa);

    size_t ln2 = gb->layernorm(
        x_post_sa,
        layer.encoder_post_ffn_layernorm_weight,
        layer.encoder_post_ffn_layernorm_bias
    );

    size_t ffn_out = build_encoder_mlp(
        gb, ln2, layer_idx, backend
    );

    size_t out = gb->add(x_post_sa, ffn_out);

    // record block output for debugging
    if (layer_idx < encoder_block_out_nodes_.size()) {
        encoder_block_out_nodes_[layer_idx] = out;
    }

    if(layer_idx > encoder_block_out_nodes_.size())
        std::cout<<"WTF"<<std::endl;

    return out;
}

size_t WhisperModel::build_decoder_transformer_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx, ComputeBackend backend, bool use_cache, size_t position_offset){
    const auto& layer = weight_nodes_.layers[layer_idx];

    size_t ln1 = gb->layernorm(hidden, layer.decoder_post_attn_layernorm_weight, layer.decoder_post_attn_layernorm_bias);
    size_t sa = build_decoder_self_attention(gb, ln1, layer_idx, backend, use_cache, position_offset);
    size_t x_post_sa = gb->add(hidden, sa);

    size_t ln2 = gb->layernorm(x_post_sa, layer.decoder_post_encoder_layernorm_weight, layer.decoder_post_encoder_layernorm_bias);
    size_t ca = build_encoder_attention(gb, ln2, layer_idx, backend, use_cache, position_offset);
    size_t x_post_ca = gb->add(x_post_sa, ca);

    size_t ln3 = gb->layernorm(x_post_ca,layer.decoder_post_ffn_layernorm_weight,layer.decoder_post_ffn_layernorm_bias);
    size_t ffn_out = build_decoder_mlp(gb, ln3, layer_idx, backend);
    size_t x_post_ffn = gb->add(x_post_ca, ffn_out);

    return x_post_ffn;

}

void WhisperModel::run_encoder(const std::vector<float>& mel_bins)
{
    if (mel_bins.size() % 80 != 0)
        throw std::runtime_error("Mel bins length must be divisible by 80.");

    size_t T_mel = mel_bins.size() / 80;
    if (T_mel == 0)
        throw std::runtime_error("Mel bins has zero frames.");

    auto backend =
        (config_.default_backend == Config::Backend::CPU)
        ? ComputeBackend::CPU
        : ComputeBackend::NPU;

    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    if (!gb)
        throw std::runtime_error("Graph handle is null in run_encoder.");

    size_t mel_input = gb->input({1, 80, T_mel}, Precision::FP32);
    gb->set_input(mel_input, mel_bins.data(), Precision::FP32);

    size_t conv2_transposed = build_conv1d(gb, mel_input, backend);

    const auto& conv_shape = gb->get_output_buffer(conv2_transposed).shape;
    if (conv_shape.size() != 3 || conv_shape[0] != 1)
        throw std::runtime_error("Conv2 transpose should be [1, T_enc, D].");

    size_t T_enc = conv_shape[1];
    size_t D_enc = conv_shape[2];

    size_t pos_slice = gb->slice(
        weight_nodes_.encoder_position_embeddings,
        /*dim*/0,
        /*start*/0,
        /*length*/T_enc
    );

    size_t h2d = gb->reshape(conv2_transposed, {T_enc, D_enc});

    size_t h_pos = gb->add(h2d, pos_slice);
    last_enc_plus_pos_node_ = h_pos;

    size_t h = h_pos;
    for (uint32_t i = 0; i < config_.num_layers; ++i){
        h = build_encoder_transformer_block(gb, h, i, backend, false, 0);
        if (i == 0) {
            encoder_transformer_block_0 = h;
        }
    }

    size_t h_norm = gb->layernorm(
        h,
        weight_nodes_.encoder_norm_weight,
        weight_nodes_.encoder_norm_bias
    );
    last_encoder_post_norm_node_ = h_norm;
    

    weight_nodes_.encoder_output = h_norm;
}



size_t WhisperModel::run_decoder_step(const std::vector<uint32_t>& tokens, bool use_cache) {
    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    
    const size_t full_len = tokens.size();
    if (full_len == 0) {
        throw std::runtime_error("Decoder token list cannot be empty.");
    }

    auto backend = config_.default_backend == Config::Backend::CPU
        ? ComputeBackend::CPU
        : ComputeBackend::NPU;

    // If we have a cache and we’re doing incremental decode, only feed the last token.
    size_t start_idx = (use_cache && kv_cache_.current_seq_len > 0) ? full_len - 1 : 0;
    size_t new_tokens = full_len - start_idx;

    // Token ids input (kept as FP32 since that’s what the rest of your engine uses here)

    std::cout << "[Debug] full_len=" << full_len 
          << " use_cache=" << use_cache
          << " kv_len=" << kv_cache_.current_seq_len << std::endl;
    std::cout << "[Debug] tokens: ";
    for (auto t : tokens) std::cout << t << " ";
    std::cout << std::endl;
    size_t tok_input = gb->input({new_tokens}, Precision::FP32);
    std::vector<float> tok_f(new_tokens);
    for (size_t i = 0; i < new_tokens; i++) {
        tok_f[i] = static_cast<float>(tokens[start_idx + i]);
    }
    gb->set_input(tok_input, tok_f.data(), Precision::FP32);

    // Embedding + positional encodings
    size_t dec_hidden = gb->embedding(embedding_node_id_, tok_input);

    size_t position_offset = kv_cache_.current_seq_len; // how many tokens already in cache
    size_t dec_pos = gb->slice(
        weight_nodes_.decoder_position_embeddings_weight,
        /*dim*/ 0,
        /*start*/ position_offset,
        /*length*/ new_tokens
    );
    dec_hidden = gb->add(dec_hidden, dec_pos);

    // Decoder stack
    for (uint32_t layer_idx = 0; layer_idx < config_.num_layers; ++layer_idx) {
        dec_hidden = build_decoder_transformer_block(
            gb,
            dec_hidden,
            layer_idx,
            backend,
            use_cache,
            position_offset
        );
    }

    size_t dec_norm = gb->layernorm(
        dec_hidden,
        weight_nodes_.decoder_norm_weight,
        weight_nodes_.decoder_norm_bias
    );

    auto dec_shape = gb->get_output_buffer(dec_norm).shape;
    auto w_shape   = gb->get_output_buffer(output_weight_node_id_).shape;
    std::cout << "[Debug] dec_norm shape: " << dec_shape[0] << " x " << dec_shape[1] << std::endl;
    std::cout << "[Debug] output_weight shape: " << w_shape[0] << " x " << w_shape[1] << std::endl;

    // Project to logits
    size_t logits = gb->matmul(dec_norm, output_weight_node_id_, /*transpose_b=*/true, backend);

    // If we passed multiple tokens (prefill), keep only the last logits row
    if (new_tokens > 1) {
        logits = gb->slice(logits, /*dim*/ 0, /*start*/ new_tokens - 1, /*length*/ 1);
    }

    last_new_tokens_ = new_tokens;
    return logits;
}


size_t WhisperModel::forward(const std::vector<float>& mel_bins, const std::vector<uint32_t>& tokens, bool use_cache)
{

    if (!initialized_ || !graph_handle_) {
        throw std::runtime_error("Model not initialized");
    }

    if (!use_cache) {
        kv_cache_.reset();
        kv_cache_.current_seq_len = 0;
        reset_graph_side_cache_nodes();
        run_encoder(mel_bins);
    }

    return run_decoder_step(tokens, use_cache);
}

uint32_t WhisperModel::generate_with_audio(
    const std::vector<uint32_t>& tokens,
    const std::vector<float>& mel_bins,
    float temperature,
    float top_p,
    size_t top_k,
    const std::string& profile_file)
{
    if (!initialized_ || !graph_handle_)
        throw std::runtime_error("Model not initialized - call init() first");
    if (tokens.empty())
        throw std::runtime_error("Token sequence cannot be empty");
    if (mel_bins.empty())
        throw std::runtime_error("Mel bins cannot be empty in Whisper generate_with_audio");

    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    auto backend = (config_.default_backend == Config::Backend::CPU)
        ? ComputeBackend::CPU
        : ComputeBackend::NPU;

    std::cout << "[Debug] eos=" << get_tokenizer()->get_eos_token()
              << " bos=" << get_tokenizer()->get_bos_token() << std::endl;

    //--------------------------------------------------------------------
    // Detect whether this is the prefill call (encoder hasn't run yet)
    //--------------------------------------------------------------------
    bool cold_start = !encoder_ready_;
    size_t logits_node = 0;

    //--------------------------------------------------------------------
    // Build BOS + user tokens (HF does: decoder_start_token_id=50258)
    //--------------------------------------------------------------------
    uint32_t bos = static_cast<uint32_t>(get_tokenizer()->get_bos_token());

    std::vector<uint32_t> full_tokens;
    full_tokens.reserve(tokens.size() + 1);
    full_tokens.push_back(bos);
    full_tokens.insert(full_tokens.end(), tokens.begin(), tokens.end());

    std::cout << "[Debug] full_tokens (input to decoder): ";
    for (auto t : full_tokens) std::cout << t << " ";
    std::cout << std::endl;

    //--------------------------------------------------------------------
    // COLD START → full encoder run + decoder prefill
    //--------------------------------------------------------------------
    if (cold_start)
    {
        std::cout << "[Debug] Running encoder (prefill)" << std::endl;

        gb->soft_reset();                     // full graph reset
        kv_cache_.reset();                    // KV cache reset
        kv_cache_.current_seq_len = 0;
        reset_graph_side_cache_nodes();
        first_decode_step_ = true;

        // Run encoder ONCE
        run_encoder(mel_bins);

        // PREFILL: pass full BOS + forced prompt
        logits_node = run_decoder_step(full_tokens, /*use_cache=*/false);
    }
    //--------------------------------------------------------------------
    // WARM STEP → reuse encoder + KV cache (fast autoregressive decode)
    //--------------------------------------------------------------------
    else
    {
        std::cout << "[Debug] Warm decoder step, KV len=" 
                  << kv_cache_.current_seq_len << std::endl;

        gb->soft_reset();                 // but DO NOT rerun encoder
        reset_graph_side_cache_nodes();

        //----------------------------------------------------------------
        // Reinject encoder output into fresh graph
        //----------------------------------------------------------------
        if (encoder_output_host_.empty())
            throw std::runtime_error("Missing encoder_output_host_ in warm step!");

        size_t enc_node = gb->input(encoder_output_shape_, Precision::FP32);
        gb->set_input(enc_node, encoder_output_host_.data(), Precision::FP32);
        weight_nodes_.encoder_output = enc_node;

        //----------------------------------------------------------------
        // Reinject KV cache tensors
        //----------------------------------------------------------------
        if (kv_cache_.current_seq_len > 0)
        {
            for (uint32_t i = 0; i < config_.num_layers; i++)
            {
                auto k_view = kv_cache_.get_key_view(i);
                auto v_view = kv_cache_.get_value_view(i);
                if (!k_view.ptr1 || !v_view.ptr1) continue;

                size_t k_node = gb->input(
                    {1, kv_cache_.current_seq_len,
                     config_.attention_kv_heads,
                     config_.attention_head_dim},
                    kv_cache_.precision
                );
                size_t v_node = gb->input(
                    {1, kv_cache_.current_seq_len,
                     config_.attention_kv_heads,
                     config_.attention_head_dim},
                    kv_cache_.precision
                );

                gb->set_input(k_node, k_view.ptr1, kv_cache_.precision);
                gb->set_input(v_node, v_view.ptr1, kv_cache_.precision);

                cache_k_output_nodes_[i] = k_node;
                cache_v_output_nodes_[i] = v_node;
            }
        }

        //----------------------------------------------------------------
        // Only decode the NEWEST user token (autoregressive)
        //----------------------------------------------------------------
        std::vector<uint32_t> last_token_vec = { tokens.back() };
        logits_node = run_decoder_step(last_token_vec, /*use_cache=*/true);
    }

    //--------------------------------------------------------------------
    // EXECUTE graph
    //--------------------------------------------------------------------
    auto logits_shape = gb->get_output_buffer(logits_node).shape;
    std::cout << "[Debug] logits shape before execute: "
              << logits_shape[0] << " x " << logits_shape[1] << std::endl;

    size_t sampled_token_id = gb->sample(logits_node, temperature, top_p, top_k);

    if (!profile_file.empty()) gb->execute(profile_file);
    else gb->execute();

    //--------------------------------------------------------------------
    // DEBUG PRINTS
    //--------------------------------------------------------------------
    std::cout << "==== Encoder output AFTER execute ====\n";
    debug_print_tensor_5x5(gb, weight_nodes_.encoder_output, "encoder_output");

    //--------------------------------------------------------------------
    // If first step: snapshot encoder output for warm reuse
    //--------------------------------------------------------------------
    if (cold_start)
    {
        auto& out_buf = gb->get_output_buffer(weight_nodes_.encoder_output);
        size_t total = 1;
        for (auto s : out_buf.shape) total *= s;

        encoder_output_host_.resize(total);
        std::memcpy(encoder_output_host_.data(),
                    gb->get_output(weight_nodes_.encoder_output),
                    total * sizeof(float));

        encoder_output_shape_ = out_buf.shape;
        encoder_ready_ = true;
    }

    //--------------------------------------------------------------------
    // POST-DECODE: update KV cache for next autoregressive call
    //--------------------------------------------------------------------
    std::cout << "[Debug] Updating KV cache (" << last_new_tokens_ 
              << " new tokens)" << std::endl;
    post_execute_updates(gb, full_tokens.size());
    update_kv_cache(gb, last_new_tokens_);

    //--------------------------------------------------------------------
    // Read sampled token
    //--------------------------------------------------------------------
    auto* out_ptr = gb->get_output(sampled_token_id);
    uint32_t sampled = *reinterpret_cast<uint32_t*>(out_ptr);

    std::cout << "[Debug] Returning sampled token " << sampled
              << " (\"" << get_tokenizer()->decode({sampled}) << "\")\n";
    std::cout << "[Debug] KV len now = " << kv_cache_.current_seq_len << std::endl;

    return sampled;
}


}
}
