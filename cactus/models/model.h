#pragma once

#include "../engine/engine.h"

namespace cactus {
namespace engine {



class QwenModel : public Model {
public:
    QwenModel();
    explicit QwenModel(const Config& config);
    ~QwenModel() override = default;

protected:
    size_t build_attention(CactusGraph* gb, size_t normalized_input, uint32_t layer_idx,
                          ComputeBackend backend, bool use_cache = false, size_t position_offset = 0) override;

    size_t build_mlp(CactusGraph* gb, size_t normalized_h, uint32_t layer_idx,
                    ComputeBackend backend) const override;

    size_t build_transformer_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx,
                                  ComputeBackend backend, bool use_cache = false, size_t position_offset = 0) override;

    size_t forward(const std::vector<uint32_t>& tokens, bool use_cache = false) override;
    void load_weights_to_graph(CactusGraph* gb) override;

private:
    struct WeightNodeIDs {
        size_t output_weight;
        size_t output_norm_weight;

        struct LayerWeights {
            size_t attn_q_weight;
            size_t attn_k_weight;
            size_t attn_v_weight;
            size_t attn_output_weight;
            size_t input_layernorm_weight;
            size_t attn_q_norm_weight;
            size_t attn_k_norm_weight;
            size_t ffn_gate_weight;
            size_t ffn_up_weight;
            size_t ffn_down_weight;
            size_t post_attention_layernorm_weight;
        };

        std::vector<LayerWeights> layers;
    } weight_nodes_;
};



class GemmaModel : public Model {
public:
    GemmaModel();
    explicit GemmaModel(const Config& config);
    ~GemmaModel() override = default;

protected:
    size_t build_attention(CactusGraph* gb, size_t normalized_input, uint32_t layer_idx,
                          ComputeBackend backend, bool use_cache = false, size_t position_offset = 0) override;

    size_t build_mlp(CactusGraph* gb, size_t normalized_h, uint32_t layer_idx,
                    ComputeBackend backend) const override;

    size_t build_transformer_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx,
                                  ComputeBackend backend, bool use_cache = false, size_t position_offset = 0) override;

    size_t forward(const std::vector<uint32_t>& tokens, bool use_cache = false) override;
    void load_weights_to_graph(CactusGraph* gb) override;
    void post_init() override;

private:
    struct WeightNodeIDs {
        size_t output_weight;
        size_t output_norm_weight;

        struct LayerWeights {
            size_t attn_q_weight;
            size_t attn_k_weight;
            size_t attn_v_weight;
            size_t attn_output_weight;
            size_t input_layernorm_weight;
            size_t attn_q_norm_weight;
            size_t attn_k_norm_weight;
            size_t pre_feedforward_layernorm_weight;
            size_t post_feedforward_layernorm_weight;
            size_t ffn_gate_weight;
            size_t ffn_up_weight;
            size_t ffn_down_weight;
            size_t post_attention_layernorm_weight;
        };

        std::vector<LayerWeights> layers;
    } weight_nodes_;
};



class SmolModel : public Model{
public:
    SmolModel();
    explicit SmolModel(const Config& config);
    ~SmolModel() override = default;

protected:
    size_t build_attention(CactusGraph* gb, size_t normalized_input, uint32_t layer_idx,
                          ComputeBackend backend, bool use_cache = false, size_t position_offset = 0) override;

    size_t build_mlp(CactusGraph* gb, size_t normalized_h, uint32_t layer_idx,
                    ComputeBackend backend) const override;

    size_t build_transformer_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx,
                                  ComputeBackend backend, bool use_cache = false, size_t position_offset = 0) override;

    size_t forward(const std::vector<uint32_t>& tokens, bool use_cache = false) override;
    void load_weights_to_graph(CactusGraph* gb) override;
    
private:
    struct WeightNodeIDs {
        size_t output_weight;
        size_t output_norm_weight;

        struct LayerWeights {
            size_t attn_q_weight;
            size_t attn_k_weight;
            size_t attn_v_weight;
            size_t attn_output_weight;
            size_t input_layernorm_weight;
            size_t ffn_gate_weight;
            size_t ffn_up_weight;
            size_t ffn_down_weight;
            size_t post_attention_layernorm_weight;
        };

        std::vector<LayerWeights> layers;
    } weight_nodes_;
};



class LFM2Model : public Model {
public:
    LFM2Model();
    explicit LFM2Model(const Config& config);
    ~LFM2Model() override = default;

    bool init(const std::string& model_folder, size_t context_size, const std::string& system_prompt = "");

protected:
    size_t build_attention(CactusGraph* gb, size_t normalized_input, uint32_t layer_idx,
                          ComputeBackend backend, bool use_cache = false, size_t position_offset = 0) override;

    size_t build_conv1d(CactusGraph* gb, size_t input, uint32_t layer_idx,
                    ComputeBackend backend, bool use_cache);

    size_t build_mlp(CactusGraph* gb, size_t normalized_h, uint32_t layer_idx,
                    ComputeBackend backend) const override;

    size_t build_transformer_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx,
                                  ComputeBackend backend, bool use_cache = false, size_t position_offset = 0) override;

    size_t forward(const std::vector<uint32_t>& tokens, bool use_cache = false) override;
    void post_init() override;
    void post_execute_updates(CactusGraph* gb, size_t seq_len) override;
    void reset_cache() override;
    void load_weights_to_graph(CactusGraph* gb) override;

private:

    struct WeightNodeIDs {
        size_t output_weight;
        size_t output_norm_weight;

        struct LayerWeights {
        size_t attn_q_weight;
        size_t attn_k_weight;
        size_t attn_v_weight;
        size_t attn_output_weight;
        size_t attn_q_norm_weight;   
        size_t attn_k_norm_weight;

        size_t conv_depthwise_weight;
        size_t conv_in_proj_weight;
        size_t conv_out_proj_weight;

        size_t input_layernorm_weight;
        size_t post_attention_layernorm_weight;
        size_t ffn_gate_weight;
        size_t ffn_up_weight;
        size_t ffn_down_weight;
        };

        enum class LayerType : uint8_t { ATTENTION, CONV };

        struct LayerEntry {
            LayerType type;
            LayerWeights weights;
        };

        std::vector<LayerEntry> layers;
    } weight_nodes_;

    ConvCache conv_cache_;
    std::vector<size_t> conv_cache_bx_nodes_;
    bool last_forward_used_cache_ = false;
};


class NomicModel : public Model {
public:
    NomicModel();
    explicit NomicModel(const Config& config);
    ~NomicModel() override = default;

protected:
    size_t build_attention(CactusGraph* gb, size_t normalized_input, uint32_t layer_idx,
                            ComputeBackend backend, bool use_cache = false, size_t position_offset = 0) override;

    size_t build_mlp(CactusGraph* gb, size_t normalized_h, uint32_t layer_idx,
                    ComputeBackend backend) const override;

    size_t build_transformer_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx,
                                    ComputeBackend backend, bool use_cache = false, size_t position_offset = 0) override;

    size_t forward(const std::vector<uint32_t>& tokens, bool use_cache = false) override;

    void load_weights_to_graph(CactusGraph* gb) override;

private:
    size_t build_standard_mlp(CactusGraph* gb, size_t normalized_h, uint32_t layer_idx,
                                ComputeBackend backend) const;

    size_t build_moe_mlp(CactusGraph* gb, size_t normalized_h, uint32_t layer_idx,
                        ComputeBackend backend) const;

    struct WeightNodeIDs {
        size_t embedding_layernorm_weight;
        size_t embedding_layernorm_bias;

        struct LayerWeights {
            size_t attn_q_weight;
            size_t attn_k_weight;
            size_t attn_v_weight;
            size_t attn_q_bias;
            size_t attn_k_bias;
            size_t attn_v_bias;
            size_t attn_output_weight;
            size_t attn_output_bias;
            size_t ffn_up_weight;
            size_t ffn_up_bias;
            size_t ffn_norm_1_weight;
            size_t ffn_norm_1_bias;
            size_t ffn_down_weight;
            size_t ffn_down_bias;
            size_t ffn_norm_2_weight;
            size_t ffn_norm_2_bias;
            size_t mlp_router_layer_weight;
            size_t mlp_experts_bias;
            std::vector<size_t> mlp_experts_mlp1_weight;
            std::vector<size_t> mlp_experts_mlp2_weight;
        };

        std::vector<LayerWeights> layers;
    } weight_nodes_;
};

class WhisperModel : public Model {
public:
    WhisperModel();
    explicit WhisperModel(const Config& config);
    ~WhisperModel() override = default;

protected:
    size_t build_attention(CactusGraph*, size_t, uint32_t,ComputeBackend, bool, size_t) override {
        throw std::runtime_error("Whisper: build_attention unused");
    }

    size_t build_mlp(CactusGraph*, size_t, uint32_t, ComputeBackend) const override {
        throw std::runtime_error("Whisper: build_mlp unused");
    }

    size_t build_transformer_block(CactusGraph*, size_t, uint32_t, ComputeBackend, bool, size_t) override {
        throw std::runtime_error("Whisper: build_transformer_block unused");
    }

    size_t forward(const std::vector<uint32_t>& tokens, bool use_cache = false) override {
        throw std::runtime_error("Whisper requires mel+token forward().");
    }

    size_t forward(const std::vector<float>& mel_bins, const std::vector<uint32_t>& tokens, bool use_cache = false) override;

    void run_encoder(const std::vector<float>& mel_bins);
    void reset_graph_side_cache_nodes();

    size_t run_decoder_step(const std::vector<uint32_t>& tokens, bool use_cache, bool last_token_only);

    void load_weights_to_graph(CactusGraph* gb) override;

    size_t build_encoder_attention(CactusGraph* gb, size_t normalized_input, uint32_t layer_idx,
                          ComputeBackend backend, bool use_cache = false, size_t position_offset = 0);
    
    size_t build_decoder_self_attention(CactusGraph* gb, size_t normalized_input, uint32_t layer_idx,
                          ComputeBackend backend, bool use_cache = false, size_t position_offset = 0);

    size_t build_encoder_self_attention(CactusGraph* gb, size_t normalized_input, uint32_t layer_idx,
                          ComputeBackend backend, bool use_cache = false, size_t position_offset = 0);

    size_t build_encoder_mlp(CactusGraph* gb, size_t normalized_h, uint32_t layer_idx,
                    ComputeBackend backend);
    
    size_t build_decoder_mlp(CactusGraph* gb, size_t normalized_h, uint32_t layer_idx,
                    ComputeBackend backend) const;
    
    size_t build_encoder_transformer_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx,
                                  ComputeBackend backend, bool use_cache = false, size_t position_offset = 0);
    
    size_t build_decoder_transformer_block(CactusGraph* gb, size_t hidden, uint32_t layer_idx,
                                  ComputeBackend backend, bool use_cache = false, size_t position_offset = 0);
    
    size_t build_conv1d(CactusGraph* gb, size_t input, ComputeBackend backend);

    uint32_t generate_with_audio(const std::vector<uint32_t>& tokens, const std::vector<float>& mel_bins, 
                                    float temperature = -1.0f, float top_p = -1.0f, size_t top_k = 0, const std::string& profile_file = "") override;

private:
    struct WeightNodeIDs {
        size_t output_weight;
        size_t output_norm_weight;

        size_t decoder_norm_weight;
        size_t decoder_norm_bias;
        size_t decoder_position_embeddings_weight;

        size_t encoder_position_embeddings;
        size_t encoder_conv1_weight;
        size_t encoder_conv1_bias;
        size_t encoder_conv2_weight;
        size_t encoder_conv2_bias;
        size_t encoder_norm_weight;
        size_t encoder_norm_bias;

        size_t encoder_output;

        struct LayerWeights {
            //Decoder layers
            size_t decoder_output_norm_bias;
            size_t decoder_output_norm_weight;
            size_t decoder_position_embeddings_weight;
            size_t decoder_token_embeddings_weight;

            size_t decoder_encoder_attn_q_weight;
            size_t decoder_encoder_attn_k_weight;
            size_t decoder_encoder_attn_v_weight;
            size_t decoder_encoder_attn_q_bias;
            size_t decoder_encoder_attn_v_bias;
            size_t decoder_encoder_attn_output_weight;
            size_t decoder_encoder_attn_output_bias;

            size_t decoder_post_encoder_layernorm_weight;
            size_t decoder_post_encoder_layernorm_bias;

            size_t decoder_ffn1_weight;
            size_t decoder_ffn1_bias;
            size_t decoder_ffn2_weight;
            size_t decoder_ffn2_bias;

            size_t decoder_post_ffn_layernorm_weight;
            size_t decoder_post_ffn_layernorm_bias;
            
            size_t decoder_self_attn_q_weight;
            size_t decoder_self_attn_k_weight;
            size_t decoder_self_attn_v_weight;
            size_t decoder_self_attn_q_bias;
            size_t decoder_self_attn_v_bias;
            size_t decoder_self_attn_output_weight;
            size_t decoder_self_attn_output_bias;

            size_t decoder_post_attn_layernorm_weight;
            size_t decoder_post_attn_layernorm_bias;

            //Encoder layers
            size_t encoder_ffn1_weight;
            size_t encoder_ffn1_bias;
            size_t encoder_ffn2_weight;
            size_t encoder_ffn2_bias;

            size_t encoder_post_ffn_layernorm_weight;
            size_t encoder_post_ffn_layernorm_bias;
            
            size_t encoder_self_attn_q_weight;
            size_t encoder_self_attn_k_weight;
            size_t encoder_self_attn_v_weight;
            size_t encoder_self_attn_q_bias;
            size_t encoder_self_attn_v_bias;
            size_t encoder_self_attn_output_weight;
            size_t encoder_self_attn_output_bias;

            size_t encoder_post_attn_layernorm_weight;
            size_t encoder_post_attn_layernorm_bias;
        };

        std::vector<LayerWeights> layers;
    } weight_nodes_;

    bool encoder_ready_ = false;
    size_t last_new_tokens_;
    std::vector<float> encoder_output_host_;
    std::vector<size_t> encoder_output_shape_;
    size_t last_conv1_node_ = 0;
    size_t last_conv2_node_ = 0;
    size_t last_conv2_transposed_node_ = 0;
    size_t last_encoder_post_norm_node_ = 0;
    size_t last_enc_plus_pos_node_ = 0;
    size_t encoder_transformer_block_0 = 0;
    size_t encoder_pre_gelu = 0;
    size_t encoder_post_gelu = 0;
    size_t encoder_ln1_node_ = 0;
    size_t encoder_sa_out_node_ = 0;
    size_t encoder_ln2_node_ = 0;

    size_t encoder_block1_out_node_ = 0;

    size_t last_dec_norm_node_ = 0;

    size_t decoder_emb_pos_node_ = 0;
    size_t decoder_block0_out_node_ = 0;

    std::vector<size_t> encoder_block_out_nodes_;
    std::vector<uint8_t> encoder_output_bytes_;
    Precision encoder_output_precision_ = Precision::FP32;

    std::vector<size_t> suppress_tokens_ = {
    1,
    2,
    7,
    8,
    9,
    10,
    14,
    25,
    26,
    27,
    28,
    29,
    31,
    58,
    59,
    60,
    61,
    62,
    63,
    90,
    91,
    92,
    93,
    359,
    503,
    522,
    542,
    873,
    893,
    902,
    918,
    922,
    931,
    1350,
    1853,
    1982,
    2460,
    2627,
    3246,
    3253,
    3268,
    3536,
    3846,
    3961,
    4183,
    4667,
    6585,
    6647,
    7273,
    9061,
    9383,
    10428,
    10929,
    11938,
    12033,
    12331,
    12562,
    13793,
    14157,
    14635,
    15265,
    15618,
    16553,
    16604,
    18362,
    18956,
    20075,
    21675,
    22520,
    26130,
    26161,
    26435,
    28279,
    29464,
    31650,
    32302,
    32470,
    36865,
    42863,
    47425,
    49870,
    50254,
    50258,
    50358,
    50359,
    50360,
    50361,
    50362
    };

    std::vector<size_t> begin_suppress_tokens_ = {
    220,
    50257
    };

    bool first_decode_step_ = true;

    std::vector<size_t> encoder_k_nodes_;
    std::vector<size_t> encoder_v_nodes_;

    std::vector<std::vector<uint8_t>> encoder_k_host_;
    std::vector<std::vector<uint8_t>> encoder_v_host_;
    std::vector<std::vector<size_t>>  encoder_k_shape_;
    std::vector<std::vector<size_t>>  encoder_v_shape_;
    Precision encoder_kv_precision_ = Precision::FP32;
    bool encoder_kv_ready_ = false;
    

};

}
}
