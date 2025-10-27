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
            size_t encoder_attn_q_weight;
            size_t encoder_attn_k_weight;
            size_t encoder_attn_v_weight;
            size_t encoder_attn_q_bias;
            size_t encoder_attn_v_bias;
            size_t encoder_attn_output_weight;
            size_t encoder_attn_otuput_bias;

            size_t post_encoder_layernorm_weight1;
            size_t post_encoder_layernorm_bias1;

            size_t ffn1_weight;
            size_t ffn1_bias;
            size_t ffn2_weight;
            size_t ffn2_bias;

            size_t post_ffn_layernorm_weight1;
            size_t post_ffn_layernorm_bias1;
            
            size_t self_attn_q_weight;
            size_t self_attn_k_weight;
            size_t self_attn_v_weight;
            size_t self_attn_q_bias;
            size_t self_attn_v_bias;
            size_t self_attn_output_weight;
            size_t self_attn_otuput_bias;

            size_t post_attn_layernorm_weight1;
            size_t post_attn_layernorm_bias1;


        };

        std::vector<LayerWeights> layers;
    } weight_nodes_;
};

}
}
