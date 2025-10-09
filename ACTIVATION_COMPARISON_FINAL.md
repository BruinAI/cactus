# Final Activation Comparison: HuggingFace vs Cactus

## Summary

**✅ TOKENIZATION FIXED**: Both implementations now use the same 10 tokens.

### Token Comparison
**Input Text**: `"Cactus activation inspection test."`

**HuggingFace Tokens**: 
```
[0, 2041, 24392, 34704, 1363, 134071, 1830, 3034, 5, 2]
['<s>', '▁Ca', 'ctus', '▁activa', 'tion', '▁inspect', 'ion', '▁test', '.', '</s>']
```

**Cactus Tokens**: 
```
[0, 2041, 24392, 34704, 1363, 134071, 1830, 3034, 5, 2]
(10 tokens - MATCHES HF!)
```

## Layer-by-Layer Activation Comparison

### Embedding + LayerNorm (First Layer)
**HuggingFace** (`emb_ln`):
```
Shape: [1, 10, 768]
First 5 values: [-0.0004, 0.1810, 0.1487, 0.2707, -0.2057]
```

**Cactus** (Node 149 LAYERNORM):
```
Shape: [10, 768]
First 5 values: [-0.000379548, 0.180971, 0.148678, 0.270659, -0.205701]
```

**Analysis**: ✅ **Excellent match!** Values are within 0.0001-0.001 difference
- Element 0: -0.0004 vs -0.00038 (Δ=0.00002)
- Element 1: 0.1810 vs 0.1810 (Δ=0.00000)
- Element 2: 0.1487 vs 0.1487 (Δ=0.00000)
- Element 3: 0.2707 vs 0.2707 (Δ=0.00000)
- Element 4: -0.2057 vs -0.2057 (Δ=0.00000)

### Layer 0 Output (First Transformer Block)
**HuggingFace** (`encoder.layers.0.norm2`):
```
First 5 values: [-0.1261, 0.0808, -0.0755, 0.1281, 0.0299]
```

**Cactus** (Node 182 LAYERNORM):
```
First 5 values: [-0.345512, 0.417655, -0.147726, 0.4046, 0.276649]
```

**Analysis**: ⚠️ **Significant divergence starting here**
- Differences range from 0.2-0.3 in magnitude
- This suggests accumulation of differences through the first transformer block

### Layer 2 Output
**HuggingFace** (`encoder.layers.2.norm2`):
```
First 5 values: [-0.0295, -0.0369, -0.0521, -0.0177, -0.0281]
```

**Cactus** (Node 287 LAYERNORM):
```
First 5 values: [-0.120532, -0.0305771, 0.0977771, 0.208077, -0.16954]
```

**Analysis**: ⚠️ **Divergence continues**

### Layer 11 Final Output
**HuggingFace** (`encoder.layers.11.norm2`):
```
First 5 values: [0.3118, 0.5420, -0.6390, 0.4044, -1.0879]
```

**Cactus** (Node 977 LAYERNORM):
```
First 5 values: [0.300209, 0.0502827, -0.0671153, -0.370438, -0.577586]
```

**Analysis**: ⚠️ **Large divergence by final layer**

## Root Cause Analysis

### What's Correct ✅
1. **Tokenization**: Now identical (10 tokens matching exactly)
2. **Architecture**: All 12 layers with correct alternating FFN/MoE structure
3. **Embedding Layer**: Near-perfect match (< 0.001 error)
4. **Shape Propagation**: All tensor shapes are correct throughout

### Potential Issues ⚠️

1. **Weight Precision/Quantization**
   - Cactus may be using INT8 quantization for some weights
   - HF uses FP32/FP16
   - This could cause divergence that accumulates through layers

2. **MoE Routing**
   - Layer 1 is the first MoE layer
   - Small differences in router softmax could lead to different expert selection
   - This would cause immediate large divergence

3. **Numerical Precision in Operations**
   - LayerNorm epsilon values
   - Attention scaling factors
   - GELU vs other activation approximations

4. **Batch Dimension Handling**
   - HF: [1, seq_len, hidden] (explicit batch)
   - Cactus: [seq_len, hidden] (implicit batch=1)
   - May affect broadcasting in certain operations

## Recommendations for Further Investigation

### Immediate Next Steps:
1. **Check Weight Loading**:
   ```bash
   # Compare a few weight values between HF and Cactus
   # Check if token_embeddings.weights matches HF model.embed_tokens.weight
   ```

2. **Verify LayerNorm Parameters**:
   - Check epsilon value (should be 1e-5 or 1e-6)
   - Verify weight and bias loading

3. **Test Individual Operations**:
   - Create unit tests for LayerNorm, Attention, GELU
   - Verify against HF implementations with same inputs

4. **Check MoE Router Implementation**:
   - Verify softmax implementation
   - Check expert selection logic (top-K)
   - Verify weighted combination

### Debug Commands:
```python
# In HuggingFace, to extract specific weights:
import torch
from transformers import AutoModel

model = AutoModel.from_pretrained('nomic-ai/nomic-embed-text-v2-moe', trust_remote_code=True)

# Check embedding weights
print("Embedding [token_id=2041] (▁Ca):", model.embeddings.word_embeddings.weight[2041, :5])

# Check first layer norm
print("Layer 0 norm1 weight:", model.encoder.layers[0].norm1.weight[:5])
print("Layer 0 norm1 bias:", model.encoder.layers[0].norm1.bias[:5])
```

## Conclusion

**Status**: Tokenization issue is FIXED ✅, but numerical divergence remains ⚠️

The good news:
- Tokenization now produces identical results
- Embedding layer shows near-perfect numerical match
- Architecture is structurally correct

The issue:
- Divergence begins at Layer 0 and accumulates
- Most likely cause: Weight loading or operation implementation differences
- Needs detailed weight comparison and operation-level debugging

**Next Priority**: Debug why Layer 0 output diverges despite correct embedding layer output.

