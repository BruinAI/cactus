#include "model.h"
#include "../graph/graph.h"
#include <cmath>
#include <stdexcept>
#include <set>

namespace cactus {
namespace engine {

WhisperModel::WhisperModel() : Model() {}

WhisperModel::WhisperModel(const Config& config) : Model(config) {
    weight_nodes_.layers.resize(config.num_layers);
}

void WhisperModel::load_weights_to_graph(CactusGraph* gb) {

    embedding_node_id_ = gb->mmap_embeddings(embedding_file_path_); //Updated engine_model to use decoder embeddings for whisper
    weight_nodes_.output_norm_weight = gb->mmap_weights(model_folder_path_ + "/output_norm.weights"); //UPDATE THIS TO BE NORM FROM LAST DECODER LAYER
    
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

size_t WhisperModel::build_encoder_mlp(CactusGraph* gb, size_t input, uint32_t layer_idx, ComputeBackend backend) const {
    const auto& layer = weight_nodes_.layers[layer_idx];

    auto ffn1_weight = gb->matmul(input, layer.encoder_ffn1_weight, true, backend);
    auto ffn1_bias = gb->add(ffn1_weight, layer.encoder_ffn1_bias);
    auto ffn2_weight = gb->matmul(ffn1_bias, layer.encoder_ffn2_weight, true, backend);
    auto ffn2_bias = gb->add(ffn2_weight, layer.encoder_ffn2_bias);
    return ffn2_bias;
}

size_t WhisperModel::build_decoder_mlp(CactusGraph* gb, size_t input, uint32_t layer_idx,
                    ComputeBackend backend) const {
    const auto& layer = weight_nodes_.layers[layer_idx];

    auto ffn1_weight = gb->matmul(input, layer.decoder_ffn1_weight, true, backend);
    auto ffn1_bias = gb->add(ffn1_weight, layer.decoder_ffn1_bias);
    auto ffn2_weight = gb->matmul(ffn1_bias, layer.decoder_ffn2_weight, true, backend);
    auto ffn2_bias = gb->add(ffn2_weight, layer.decoder_ffn2_bias);
    return ffn2_bias;
}

size_t WhisperModel::build_encoder_attention(CactusGraph* gb, size_t input, uint32_t layer_idx, ComputeBackend backend, bool use_cache, size_t position_offset){
    const auto& layer = weight_nodes_.layers[layer_idx];

    size_t q = gb->matmul(input, layer.decoder_encoder_attn_q_weight, true, backend);
    q = gb->add(q, layer.decoder_encoder_attn_q_bias);

    // K is multiplied by encoder model output
    size_t k = gb->matmul(weight_nodes_.encoder_output, layer.decoder_encoder_attn_k_weight, true, backend);

    // V is also multiplied by encoder model otuput
    size_t v = gb->matmul(weight_nodes_.encoder_output, layer.decoder_encoder_attn_v_weight, true, backend);
    v = gb->add(v, layer.decoder_encoder_attn_v_bias);

    size_t T_dec = gb->get_output_buffer(q).shape[0];
    size_t T_enc = gb->get_output_buffer(k).shape[0];

    size_t num_heads = config_.attention_heads;
    size_t head_dim  = config_.attention_head_dim;

    q = gb->reshape(q, {1, T_dec, num_heads, head_dim});
    k = gb->reshape(k, {1, T_enc, num_heads, head_dim});
    v = gb->reshape(v, {1, T_enc, num_heads, head_dim});

    size_t attn = gb->attention(q, k, v, attention_scale_, position_offset);
    attn = gb->reshape(attn, {T_dec, num_heads * head_dim});

    size_t out = gb->matmul(attn, layer.decoder_encoder_attn_output_weight, true, backend);
    out = gb->add(out, layer.decoder_encoder_attn_output_bias);

    return out;
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

    if (use_cache) {
        auto k_view = kv_cache_.get_key_view(layer_idx);
        auto v_view = kv_cache_.get_value_view(layer_idx);

        if (kv_cache_.current_seq_len > 0) {
            size_t cache_k = gb->input({1, kv_cache_.current_seq_len, num_heads, head_dim}, kv_cache_.precision);
            size_t cache_v = gb->input({1, kv_cache_.current_seq_len, num_heads, head_dim}, kv_cache_.precision);

            gb->set_input(cache_k, k_view.ptr1, kv_cache_.precision);
            gb->set_input(cache_v, v_view.ptr1, kv_cache_.precision);

            final_k = gb->concat(cache_k, k, 1);
            final_v = gb->concat(cache_v, v, 1);
        }

        cache_k_output_nodes_[layer_idx] = final_k;
        cache_v_output_nodes_[layer_idx] = final_v;
    }

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

    auto attn = gb->attention(q, k, v, attention_scale_, position_offset=0);

    attn = gb->reshape(attn, {seq_len, num_heads * head_dim});

    auto output = gb->matmul(attn, layer.encoder_self_attn_output_weight, true, backend);
    output = gb->add(output, layer.encoder_self_attn_output_bias);

    return output;
}

size_t WhisperModel::build_conv1d(CactusGraph* gb, size_t input, ComputeBackend backend) 
{
    size_t ncl = gb->transpose(input, ComputeBackend::CPU);

    size_t conv1 = gb->conv1d_k3(ncl, weight_nodes_.encoder_conv1_weight, 1);
    conv1 = gb->add(conv1, weight_nodes_.encoder_conv1_bias);

    size_t conv2 = gb->conv1d_k3(conv1, weight_nodes_.encoder_conv2_weight, 2);
    conv2 = gb->add(conv2, weight_nodes_.encoder_conv2_bias);

    size_t output_nlc = gb->transpose(conv2, ComputeBackend::CPU);

    return output_nlc;
}

size_t WhisperModel::build_encoder_transformer_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx, ComputeBackend backend, bool use_cache, size_t position_offset){
    const auto& layer = weight_nodes_.layers[layer_idx];

    size_t ln1 = gb->layernorm(hidden,layer.encoder_post_attn_layernorm_weight,layer.encoder_post_attn_layernorm_bias);
    size_t sa = build_encoder_self_attention(gb, ln1, layer_idx, backend, use_cache, position_offset);
    size_t x_post_sa = gb->add(hidden, sa);

    size_t ln2 = gb->layernorm(x_post_sa,layer.encoder_post_ffn_layernorm_weight,layer.encoder_post_ffn_layernorm_bias);
    size_t ffn_out = build_encoder_mlp(gb, ln2, layer_idx, backend);
    size_t ffn_attn = gb->add(x_post_sa, ffn_out);

    return ffn_attn;
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



size_t WhisperModel::forward(const std::vector<uint32_t>& mel_bins, const std::vector<uint32_t>& tokens, bool use_cache)
{
    if (!initialized_ || !graph_handle_) {
        throw std::runtime_error("Model not initialized - call init() first");
    }

    if (mel_bins.empty()) {
        throw std::runtime_error("Mel spectrogram cannot be empty");
    }
    if (tokens.empty()) {
        throw std::runtime_error("Token sequence cannot be empty");
    }

    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    gb->soft_reset();

    auto backend = config_.default_backend == Config::Backend::CPU
        ? ComputeBackend::CPU
        : ComputeBackend::NPU;
    
    
    //Encoder
    const size_t T_mel = mel_bins.size() / 80;
    size_t mel_input = gb->input({T_mel, (size_t)80}, Precision::FP32);
    size_t enc_hidden = build_conv1d(gb, mel_input, backend);

    size_t T_enc = gb->get_output_buffer(enc_hidden).shape[0];
    size_t enc_pos = gb->slice(weight_nodes_.encoder_position_embeddings, 0, 0, T_enc);
    enc_hidden = gb->add(enc_hidden, enc_pos);
    enc_hidden = gb->layernorm(enc_hidden, weight_nodes_.encoder_norm_weight, weight_nodes_.encoder_norm_bias);

    for (uint32_t layer_idx = 0; layer_idx < config_.num_layers; layer_idx++) {
        enc_hidden = build_encoder_transformer_block(gb, enc_hidden, layer_idx, backend, false);
    }

    weight_nodes_.encoder_output = enc_hidden;
    gb->set_input(mel_input, mel_bins.data(), Precision::FP32);


    //Decoder
    const size_t full_len = tokens.size();
    size_t position_offset = use_cache ? kv_cache_.current_seq_len : 0;

    size_t start_idx = 0;
    if (use_cache && kv_cache_.current_seq_len > 0) {
        start_idx = full_len - 1;
    }
    size_t new_tokens = full_len - start_idx;
    size_t tok_input = gb->input({new_tokens}, Precision::FP32);
    size_t dec_hidden = gb->embedding(embedding_node_id_, tok_input);

    size_t dec_pos = gb->slice(weight_nodes_.decoder_position_embeddings_weight, 0, position_offset, new_tokens);
    dec_hidden = gb->add(dec_hidden, dec_pos);

    for (uint32_t layer_idx = 0; layer_idx < config_.num_layers; layer_idx++) {
        dec_hidden = build_decoder_transformer_block(gb, dec_hidden, layer_idx, backend, use_cache, position_offset);
    }

    // Update cache length after processing new tokens
    kv_cache_.current_seq_len += new_tokens;
    size_t dec_norm = gb->layernorm(dec_hidden, weight_nodes_.decoder_norm_weight, weight_nodes_.decoder_norm_bias);
    size_t logits = gb->matmul(dec_norm, output_weight_node_id_, true, backend);

    std::vector<float> tok_f(new_tokens);
    for (size_t i = 0; i < new_tokens; i++)
        tok_f[i] = float(tokens[start_idx + i]);
    gb->set_input(tok_input, tok_f.data(), Precision::FP32);

    if (new_tokens > 1) logits = gb->slice(logits, 0, new_tokens - 1,1); 

    return logits;
}


}
}