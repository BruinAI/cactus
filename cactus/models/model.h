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

class NomicModel : public Model {
public:
    NomicModel();
    explicit NomicModel(const Config& config);
    ~NomicModel() override = default;

protected:
    // use_cache must always be false for Nomic
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
            // standard weights
            size_t attn_qkv_weight;  // attn.Wqkv.weight
            size_t attn_qkv_bias;  // attn.Wqkv.bias
            size_t attn_output_weight;  // attn.out_proj.weight
            size_t attn_output_bias;  // attn.out_proj.bias
            size_t ffn_up_weight;  // mlp.fc1.weight
            size_t ffn_up_bias;  // mlp.fc1.bias
            size_t ffn_norm_1_weight;  // norm1.weight
            size_t ffn_norm_1_bias;  // norm1.bias
            size_t ffn_down_weight;  // mlp.fc2.weight
            size_t ffn_down_bias;  // mlp.fc2.bias
            size_t ffn_norm_2_weight;  // norm2.weight
            size_t ffn_norm_2_bias;  // norm2.bias
            // MoE weights
            size_t mlp_router_layer_weight;  // mlp.router.layer.weight
            size_t mlp_experts_mlp_w1;  // mlp.experts.mlp.w1
            size_t mlp_experts_mlp_w2;  // mlp.experts.mlp.w2
            size_t mlp_experts_bias;  // mlp.experts.bias
        };

        std::vector<LayerWeights> layers;
    } weight_nodes_;
};
    
/*
MODEL ARCH

Layer 0 in Depth
1. attention
    a. rotary
    b. qkv
    c. output proj
2. dropout(attention) = 0
3. residual: dropout + hidden states
4. layernorm 1
5. mlp for even layers, moe-mlp for odd layers
6. dropout(mlp) = 0
7. residual: dropout + hidden states
8. layernorm 2

Moe In Depth
print(inspect.getsource(model.encoder._modules['layers'][1].mlp.__class__))
1. Router: Linear(in_features=768, out_features=8, bias=False) -> weights, top_weights, top_experts
    print(inspect.getsource(model.encoder._modules['layers'][1].mlp.router.__class__))
    a. Linear layer: 8 x 768
    b. Softmax
    c. Take top 2 for each token, return its top weight and all weights (only used for training)
        - only 2 top weights are used as multipliers for the experts though
2. NomicExperts: experts(x, weights, top_weights, top_experts) -> output
    print(inspect.getsource(model.encoder._modules['layers'][1].mlp.experts.__class__))
    a. For each token, run the MLP for each of its top 2 experts and add the result times its weight to the output (with Gelu activation)
    b. add bias

NomicBertEncoder(
    (layers): ModuleList(
    (0): NomicBertBlock(
        (attn): NomicBertAttention(
        (rotary_emb): NomicBertRotaryEmbedding()
        (Wqkv): Linear(in_features=768, out_features=2304, bias=True)
        (out_proj): Linear(in_features=768, out_features=768, bias=True)
        (drop): Dropout(p=0.0, inplace=False)
        )
        (mlp): NomicBertMLP(
        (fc1): Linear(in_features=768, out_features=3072, bias=True)
        (fc2): Linear(in_features=3072, out_features=768, bias=True)
        )
        (dropout1): Dropout(p=0.0, inplace=False)
        (norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (dropout2): Dropout(p=0.0, inplace=False)
    )
    (1): NomicBertBlock(
        (attn): NomicBertAttention(
        (rotary_emb): NomicBertRotaryEmbedding()
        (Wqkv): Linear(in_features=768, out_features=2304, bias=True)
        (out_proj): Linear(in_features=768, out_features=768, bias=True)
        (drop): Dropout(p=0.0, inplace=False)
        )
        (mlp): NomicMoELayer(
        (router): NomicRouter(
            (layer): Linear(in_features=768, out_features=8, bias=False)
        )
        (experts): NomicExperts(
            (mlp): NomicExpertMLP()
        )
        )
        (dropout1): Dropout(p=0.0, inplace=False)
        (norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (dropout2): Dropout(p=0.0, inplace=False)
    )
    (2): NomicBertBlock(
        (attn): NomicBertAttention(
        (rotary_emb): NomicBertRotaryEmbedding()
        (Wqkv): Linear(in_features=768, out_features=2304, bias=True)
        (out_proj): Linear(in_features=768, out_features=768, bias=True)
        (drop): Dropout(p=0.0, inplace=False)
        )
        (mlp): NomicBertMLP(
        (fc1): Linear(in_features=768, out_features=3072, bias=True)
        (fc2): Linear(in_features=3072, out_features=768, bias=True)
        )
        (dropout1): Dropout(p=0.0, inplace=False)
        (norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (dropout2): Dropout(p=0.0, inplace=False)
    )
    (3): NomicBertBlock(
        (attn): NomicBertAttention(
        (rotary_emb): NomicBertRotaryEmbedding()
        (Wqkv): Linear(in_features=768, out_features=2304, bias=True)
        (out_proj): Linear(in_features=768, out_features=768, bias=True)
        (drop): Dropout(p=0.0, inplace=False)
        )
        (mlp): NomicMoELayer(
        (router): NomicRouter(
            (layer): Linear(in_features=768, out_features=8, bias=False)
        )
        (experts): NomicExperts(
            (mlp): NomicExpertMLP()
        )
        )
        (dropout1): Dropout(p=0.0, inplace=False)
        (norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (dropout2): Dropout(p=0.0, inplace=False)
    )
    (4): NomicBertBlock(
        (attn): NomicBertAttention(
        (rotary_emb): NomicBertRotaryEmbedding()
        (Wqkv): Linear(in_features=768, out_features=2304, bias=True)
        (out_proj): Linear(in_features=768, out_features=768, bias=True)
        (drop): Dropout(p=0.0, inplace=False)
        )
        (mlp): NomicBertMLP(
        (fc1): Linear(in_features=768, out_features=3072, bias=True)
        (fc2): Linear(in_features=3072, out_features=768, bias=True)
        )
        (dropout1): Dropout(p=0.0, inplace=False)
        (norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (dropout2): Dropout(p=0.0, inplace=False)
    )
    (5): NomicBertBlock(
        (attn): NomicBertAttention(
        (rotary_emb): NomicBertRotaryEmbedding()
        (Wqkv): Linear(in_features=768, out_features=2304, bias=True)
        (out_proj): Linear(in_features=768, out_features=768, bias=True)
        (drop): Dropout(p=0.0, inplace=False)
        )
        (mlp): NomicMoELayer(
        (router): NomicRouter(
            (layer): Linear(in_features=768, out_features=8, bias=False)
        )
        (experts): NomicExperts(
            (mlp): NomicExpertMLP()
        )
        )
        (dropout1): Dropout(p=0.0, inplace=False)
        (norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (dropout2): Dropout(p=0.0, inplace=False)
    )
    (6): NomicBertBlock(
        (attn): NomicBertAttention(
        (rotary_emb): NomicBertRotaryEmbedding()
        (Wqkv): Linear(in_features=768, out_features=2304, bias=True)
        (out_proj): Linear(in_features=768, out_features=768, bias=True)
        (drop): Dropout(p=0.0, inplace=False)
        )
        (mlp): NomicBertMLP(
        (fc1): Linear(in_features=768, out_features=3072, bias=True)
        (fc2): Linear(in_features=3072, out_features=768, bias=True)
        )
        (dropout1): Dropout(p=0.0, inplace=False)
        (norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (dropout2): Dropout(p=0.0, inplace=False)
    )
    (7): NomicBertBlock(
        (attn): NomicBertAttention(
        (rotary_emb): NomicBertRotaryEmbedding()
        (Wqkv): Linear(in_features=768, out_features=2304, bias=True)
        (out_proj): Linear(in_features=768, out_features=768, bias=True)
        (drop): Dropout(p=0.0, inplace=False)
        )
        (mlp): NomicMoELayer(
        (router): NomicRouter(
            (layer): Linear(in_features=768, out_features=8, bias=False)
        )
        (experts): NomicExperts(
            (mlp): NomicExpertMLP()
        )
        )
        (dropout1): Dropout(p=0.0, inplace=False)
        (norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (dropout2): Dropout(p=0.0, inplace=False)
    )
    (8): NomicBertBlock(
        (attn): NomicBertAttention(
        (rotary_emb): NomicBertRotaryEmbedding()
        (Wqkv): Linear(in_features=768, out_features=2304, bias=True)
        (out_proj): Linear(in_features=768, out_features=768, bias=True)
        (drop): Dropout(p=0.0, inplace=False)
        )
        (mlp): NomicBertMLP(
        (fc1): Linear(in_features=768, out_features=3072, bias=True)
        (fc2): Linear(in_features=3072, out_features=768, bias=True)
        )
        (dropout1): Dropout(p=0.0, inplace=False)
        (norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (dropout2): Dropout(p=0.0, inplace=False)
    )
    (9): NomicBertBlock(
        (attn): NomicBertAttention(
        (rotary_emb): NomicBertRotaryEmbedding()
        (Wqkv): Linear(in_features=768, out_features=2304, bias=True)
        (out_proj): Linear(in_features=768, out_features=768, bias=True)
        (drop): Dropout(p=0.0, inplace=False)
        )
        (mlp): NomicMoELayer(
        (router): NomicRouter(
            (layer): Linear(in_features=768, out_features=8, bias=False)
        )
        (experts): NomicExperts(
            (mlp): NomicExpertMLP()
        )
        )
        (dropout1): Dropout(p=0.0, inplace=False)
        (norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (dropout2): Dropout(p=0.0, inplace=False)
    )
    (10): NomicBertBlock(
        (attn): NomicBertAttention(
        (rotary_emb): NomicBertRotaryEmbedding()
        (Wqkv): Linear(in_features=768, out_features=2304, bias=True)
        (out_proj): Linear(in_features=768, out_features=768, bias=True)
        (drop): Dropout(p=0.0, inplace=False)
        )
        (mlp): NomicBertMLP(
        (fc1): Linear(in_features=768, out_features=3072, bias=True)
        (fc2): Linear(in_features=3072, out_features=768, bias=True)
        )
        (dropout1): Dropout(p=0.0, inplace=False)
        (norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (dropout2): Dropout(p=0.0, inplace=False)
    )
    (11): NomicBertBlock(
        (attn): NomicBertAttention(
        (rotary_emb): NomicBertRotaryEmbedding()
        (Wqkv): Linear(in_features=768, out_features=2304, bias=True)
        (out_proj): Linear(in_features=768, out_features=768, bias=True)
        (drop): Dropout(p=0.0, inplace=False)
        )
        (mlp): NomicMoELayer(
        (router): NomicRouter(
            (layer): Linear(in_features=768, out_features=8, bias=False)
        )
        (experts): NomicExperts(
            (mlp): NomicExpertMLP()
        )
        )
        (dropout1): Dropout(p=0.0, inplace=False)
        (norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (dropout2): Dropout(p=0.0, inplace=False)
    )
    )
)
*/

}
}