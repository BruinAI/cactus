#!/usr/bin/env python3
"""
Test script to compare LoRA model vs saved/reloaded merged Gemma model.
Tests that the LoRA merging logic produces identical outputs.
"""

import os
import gc

import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer
from tunix.models.gemma3 import params_safetensors as params_safetensors_lib
from tunix.models.gemma3 import model as gemma_lib
from tunix.generate import tokenizer_adapter as tokenizer_lib
import qwix

from gemma_utils import download_and_setup_model, create_lora_model, save_lora_weights

# Configuration
MODEL_ID = "google/gemma-3-1b-it"
RANK = 32
ALPHA = 64.0
OUTPUT_DIR = "./test_output/gemma3-1b-lora-merged"

# Mesh configuration
MESH_SHAPE = len(jax.devices()), 1
MESH_AXIS_NAMES = "fsdp", "tp"


def load_tokenizer(model_path: str):
    """Load tokenizer from model directory."""
    print(f"Loading tokenizer from {model_path}")
    hf_tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer = tokenizer_lib.TokenizerAdapter(hf_tokenizer)
    return tokenizer


def run_forward_pass(model, tokenizer, input_text: str, mesh):
    """
    Run a forward pass through the model and return final hidden states.

    Args:
        model: Gemma3 model
        tokenizer: Tokenizer
        input_text: Input text (e.g., "<bos>")
        mesh: JAX mesh

    Returns:
        logits: Output logits from the model [batch, seq_len, vocab_size]
    """
    # Tokenize input
    if input_text == "<bos>":
        token_ids = [tokenizer.bos_id()]
    else:
        token_ids = tokenizer.encode(input_text)

    # Prepare inputs
    batch_size = 1
    seq_len = len(token_ids)
    last_tokens = jnp.array([token_ids], dtype=jnp.int32)  # [1, seq_len]
    positions = jnp.arange(seq_len, dtype=jnp.int32)[None, :]  # [1, seq_len]
    attention_mask = jnp.ones((batch_size, 1, seq_len), dtype=jnp.bool_)

    print(f"  Input tokens: {token_ids}")
    print(f"  Input shape: {last_tokens.shape}")

    # Run forward pass
    with mesh:
        logits, _ = model(
            last_tokens=last_tokens,
            positions=positions,
            cache=None,
            attention_mask=attention_mask,
            output_hidden_states=False
        )

    return logits


def compare_logits(lora_logits, merged_logits, tolerance=1e-4):
    """
    Compare logits from LoRA model and merged model.

    Args:
        lora_logits: Logits from LoRA model
        merged_logits: Logits from merged model
        tolerance: Maximum allowed difference

    Returns:
        bool: True if logits match within tolerance
    """
    print(f"\n{'='*60}")
    print("Comparing Model Outputs")
    print(f"{'='*60}")

    # Convert to numpy for easier analysis
    lora_logits_np = np.array(lora_logits)
    merged_logits_np = np.array(merged_logits)

    print(f"LoRA logits shape: {lora_logits_np.shape}")
    print(f"Merged logits shape: {merged_logits_np.shape}")

    # Compute differences
    abs_diff = np.abs(lora_logits_np - merged_logits_np)
    max_abs_diff = np.max(abs_diff)
    mean_abs_diff = np.mean(abs_diff)

    # Compute relative differences (where values are non-zero)
    mask = np.abs(lora_logits_np) > 1e-8
    rel_diff = np.zeros_like(abs_diff)
    rel_diff[mask] = abs_diff[mask] / np.abs(lora_logits_np[mask])
    max_rel_diff = np.max(rel_diff)
    mean_rel_diff = np.mean(rel_diff[mask]) if np.any(mask) else 0.0

    print("\nAbsolute Difference Statistics:")
    print(f"  Max:  {max_abs_diff:.2e}")
    print(f"  Mean: {mean_abs_diff:.2e}")
    print("\nRelative Difference Statistics:")
    print(f"  Max:  {max_rel_diff:.2e}")
    print(f"  Mean: {mean_rel_diff:.2e}")

    # Check final layer activations (last token)
    print("\nFinal Token Logits (first 10 values):")
    print(f"  LoRA:   {lora_logits_np[0, -1, :10]}")
    print(f"  Merged: {merged_logits_np[0, -1, :10]}")

    # Top-5 predictions for last token
    lora_top5 = np.argsort(lora_logits_np[0, -1])[-5:][::-1]
    merged_top5 = np.argsort(merged_logits_np[0, -1])[-5:][::-1]

    print("\nTop-5 Token Predictions (last position):")
    print(f"  LoRA:   {lora_top5.tolist()}")
    print(f"  Merged: {merged_top5.tolist()}")

    # Determine if models match
    matches = max_abs_diff < tolerance

    print(f"\n{'='*60}")
    if matches:
        print(f"✓ Models MATCH within tolerance ({tolerance:.2e})")
    else:
        print(f"✗ Models DIFFER (max diff: {max_abs_diff:.2e} > tolerance: {tolerance:.2e})")
    print(f"{'='*60}")

    return matches


def main():
    print("="*60)
    print("Testing LoRA Model vs Merged Model")
    print("="*60)
    print(f"Model: {MODEL_ID}")
    print(f"LoRA rank: {RANK}, alpha: {ALPHA}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Devices: {len(jax.devices())} x {jax.devices()[0].platform}")

    # Download model
    local_model_path, eos_tokens = download_and_setup_model(MODEL_ID)

    # Load tokenizer
    tokenizer = load_tokenizer(local_model_path)
    print(f"Tokenizer loaded (vocab size: {tokenizer._tokenizer.vocab_size})")

    # Get model config
    if "gemma-3-270m" in MODEL_ID:
        model_config = gemma_lib.ModelConfig.gemma3_270m()
    elif "gemma-3-1b" in MODEL_ID:
        model_config = gemma_lib.ModelConfig.gemma3_1b()
    else:
        raise ValueError(f"Unsupported model ID: {MODEL_ID}")

    # Create mesh and load base model
    mesh = jax.make_mesh(MESH_SHAPE, MESH_AXIS_NAMES)

    print(f"\n{'='*60}")
    print("Step 1: Loading base model and applying LoRA")
    print(f"{'='*60}")

    with mesh:
        base_model = params_safetensors_lib.create_model_from_safe_tensors(
            local_model_path, model_config, mesh
        )
        print("Base model loaded successfully")

    # Apply LoRA
    lora_model = create_lora_model(base_model, mesh=mesh, rank=RANK, alpha=ALPHA)
    print("LoRA model created successfully")

    # Run forward pass on LoRA model
    print(f"\n{'='*60}")
    print("Step 2: Running forward pass on LoRA model")
    print(f"{'='*60}")

    test_input = "<bos>"
    print(f"Test input: {test_input}")

    lora_logits = run_forward_pass(lora_model, tokenizer, test_input, mesh)
    print(f"LoRA model output shape: {lora_logits.shape}")

    # Save LoRA model
    print(f"\n{'='*60}")
    print("Step 3: Saving merged weights")
    print(f"{'='*60}")

    saved_path = save_lora_weights(lora_model, local_model_path, OUTPUT_DIR)
    print(f"Model saved to: {saved_path}")

    # Clean up LoRA model to free memory
    del lora_model
    del base_model
    gc.collect()

    # Load the merged model
    print(f"\n{'='*60}")
    print("Step 4: Loading merged model from safetensors")
    print(f"{'='*60}")

    with mesh:
        merged_model = params_safetensors_lib.create_model_from_safe_tensors(
            saved_path, model_config, mesh
        )
        print("Merged model loaded successfully")

    # Run forward pass on merged model
    print(f"\n{'='*60}")
    print("Step 5: Running forward pass on merged model")
    print(f"{'='*60}")

    merged_logits = run_forward_pass(merged_model, tokenizer, test_input, mesh)
    print(f"Merged model output shape: {merged_logits.shape}")

    # Compare outputs
    print(f"\n{'='*60}")
    print("Step 6: Comparing outputs")
    print(f"{'='*60}")

    matches = compare_logits(lora_logits, merged_logits, tolerance=1e-4)

    # Final summary
    print(f"\n{'='*60}")
    print("Test Complete!")
    print(f"{'='*60}")
    print(f"Model saved to: {saved_path}")
    print(f"Models match: {matches}")

    if matches:
        print("\n✓ SUCCESS: LoRA merging is working correctly!")
        print("  The merged model produces identical outputs to the LoRA model.")
    else:
        print("\n✗ FAILURE: LoRA merging has issues!")
        print("  The merged model outputs differ from the LoRA model.")

    print(f"\nYou can now test this model with:")
    print(f"  python3 tools/convert_hf.py {saved_path} weights/gemma3-1b-test/ --precision INT8")

    return 0 if matches else 1

if __name__ == "__main__":
    main()
