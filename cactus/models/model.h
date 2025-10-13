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

class SmolVLMModel : public SmolModel {
public:
    struct VLMConfig : public Config {
        uint32_t vision_hidden_dim = 0;
        uint32_t vision_num_layers = 0;
        uint32_t vision_attention_heads = 0;
        uint32_t vision_image_size = 0;
        uint32_t vision_patch_size = 0;
        uint32_t num_channels = 3;
        uint32_t vision_embed_dim = 0;
        uint32_t visual_tokens_per_img = 0;
        bool     use_pixel_shuffle = false;
        uint32_t pixel_shuffle_factor = 1;
        bool     use_image_tokens = false;
        bool     use_layout_tags = false;
    };

    SmolVLMModel();
    explicit SmolVLMModel(const VLMConfig& cfg);
    ~SmolVLMModel() override = default;

    size_t forward_mm(const std::vector<uint32_t>& tokens,
                   const std::vector<ImageBatch>& images,
                   bool use_cache = false);

protected:
    size_t build_vision_embeddings(CactusGraph* gb,
                                   const std::vector<ImageBatch>& images,
                                   ComputeBackend backend);

    size_t build_combined_input(CactusGraph* gb,
                                size_t vision_embeds,
                                const std::vector<uint32_t>& tokens,
                                ComputeBackend backend,
                                uint32_t& prefix_len);

    void load_weights_to_graph(CactusGraph* gb) override;

private:
    struct WeightNodeIDs {
        size_t vision_proj_weight;
        size_t vision_proj_bias;
        size_t image_start_embedding;
        size_t image_end_embedding;
        size_t row_tag_embedding;
        size_t col_tag_embedding;

        struct VisionLayerWeights {
            size_t attn_q_weight;
            size_t attn_k_weight;
            size_t attn_v_weight;
            size_t attn_output_weight;
            size_t layer_norm1_weight;
            size_t layer_norm1_bias;
            size_t layer_norm2_weight;
            size_t layer_norm2_bias;
            size_t mlp_fc1_weight;
            size_t mlp_fc1_bias;
            size_t mlp_fc2_weight;
            size_t mlp_fc2_bias;
        };

        std::vector<VisionLayerWeights> vision_layers;
    } weight_nodes_;
};

}
}