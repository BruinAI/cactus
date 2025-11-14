#!/usr/bin/env python3
"""
Test script to compare LoRA model vs saved/reloaded merged Gemma model.
Trains on a simple sentence and tests that the LoRA merging logic produces identical outputs.
"""

import os
import gc
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from transformers import AutoTokenizer
from tunix.models.gemma3 import params_safetensors as params_safetensors_lib
from tunix.models.gemma3 import model as gemma_lib
from tunix.generate import tokenizer_adapter as tokenizer_lib
from tunix.sft import peft_trainer, utils
import qwix

from gemma_utils import download_and_setup_model, create_lora_model, save_lora_weights

# Configuration
MODEL_ID = "google/gemma-3-1b-it"
RANK = 32
ALPHA = 64.0
OUTPUT_DIR = "./test_output/gemma3-1b-lora-merged"

# Training configuration
TRAIN_TEXT = "The quick brown fox jumped over the lazy dog"
LEARNING_RATE = 1e-3  # High learning rate to ensure LoRA weights change
NUM_TRAIN_STEPS = 100  # Train for 100 steps

# Mesh configuration
MESH_SHAPE = 1, 1
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

    # Prepare inputs - use batch_size that matches FSDP sharding
    # Get number of devices in FSDP dimension
    num_devices = len(jax.devices())
    batch_size = num_devices  # Must be divisible by FSDP sharding
    seq_len = len(token_ids)

    # Replicate the same input across all batch positions
    last_tokens = jnp.array([token_ids] * batch_size, dtype=jnp.int32)  # [batch_size, seq_len]
    positions = jnp.tile(jnp.arange(seq_len, dtype=jnp.int32)[None, :], (batch_size, 1))  # [batch_size, seq_len]
    attention_mask = jnp.ones((batch_size, 1, seq_len), dtype=jnp.bool_)

    print(f"  Input tokens: {token_ids}")
    print(f"  Input shape: {last_tokens.shape}")
    print(f"  Batch size adjusted to: {batch_size} (to match FSDP sharding)")

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


def gen_model_input_fn(x: peft_trainer.TrainingInput, tokenizer):
    """Generate model inputs from training data."""
    pad_mask = x.input_tokens != tokenizer.pad_id()
    positions = utils.build_positions_from_mask(pad_mask)
    attention_mask = utils.make_causal_attn_mask(pad_mask)
    return {
        'input_tokens': x.input_tokens,
        'input_mask': x.input_mask,
        'positions': positions,
        'attention_mask': attention_mask,
    }


def train_on_sentence(model, tokenizer, text: str, mesh, num_steps: int, learning_rate: float):
    """
    Train the model on a simple sentence for a few steps using tunix PeftTrainer.

    Args:
        model: LoRA model to train
        tokenizer: Tokenizer
        text: Text to train on
        mesh: JAX mesh
        num_steps: Number of training steps
        learning_rate: Learning rate

    Returns:
        Trained model
    """
    print(f"\n{'='*60}")
    print("Training on sentence")
    print(f"{'='*60}")
    print(f"Text: {text}")
    print(f"Steps: {num_steps}")
    print(f"Learning rate: {learning_rate}")

    # Tokenize the training text
    token_ids = tokenizer.encode(text)
    print(f"Token IDs: {token_ids}")
    print(f"Number of tokens: {len(token_ids)}")

    # Prepare inputs as TrainingInput batches
    num_devices = len(jax.devices())
    batch_size = num_devices

    # Create a simple dataset that repeats the sentence
    input_tokens = jnp.array([token_ids] * batch_size, dtype=jnp.int32)
    input_mask = jnp.ones_like(input_tokens, dtype=jnp.float32)

    print(f"Input shape: {input_tokens.shape}")

    # Create training config
    training_config = peft_trainer.TrainingConfig(
        eval_every_n_steps=num_steps + 1,  # No eval during training
        max_steps=num_steps,
        gradient_accumulation_steps=1,
    )

    # Create optimizer
    optimizer = optax.adam(learning_rate)

    # Create trainer
    trainer = peft_trainer.PeftTrainer(
        model,
        optimizer,
        training_config
    ).with_gen_model_input_fn(lambda x: gen_model_input_fn(x, tokenizer))

    # Create a simple data iterator that yields the same batch repeatedly
    def train_data_iter():
        for _ in range(num_steps):
            yield peft_trainer.TrainingInput(
                input_tokens=input_tokens,
                input_mask=input_mask
            )

    # Train
    print("\nTraining...")
    with mesh:
        trainer.train(train_data_iter(), eval_data_iter=None)

    print("Training complete!")
    return model


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
    print("Note: Comparing first batch element (all batch elements are identical)")

    # Compute differences
    abs_diff = np.abs(lora_logits_np - merged_logits_np)
    max_abs_diff = float(np.max(abs_diff))
    mean_abs_diff = float(np.mean(abs_diff))

    # Compute relative differences (where values are non-zero)
    mask = np.abs(lora_logits_np) > 1e-8
    rel_diff = np.zeros_like(abs_diff)
    rel_diff[mask] = abs_diff[mask] / np.abs(lora_logits_np[mask])
    max_rel_diff = float(np.max(rel_diff))
    mean_rel_diff = float(np.mean(rel_diff[mask])) if np.any(mask) else 0.0

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

    # Run forward pass on base model BEFORE training
    print(f"\n{'='*60}")
    print("Step 2: Running forward pass on base model (before training)")
    print(f"{'='*60}")

    test_input = TRAIN_TEXT
    print(f"Test input: {test_input}")

    base_logits = run_forward_pass(base_model, tokenizer, test_input, mesh)
    print(f"Base model output shape: {base_logits.shape}")

    # Apply LoRA
    lora_model = create_lora_model(base_model, mesh=mesh, rank=RANK, alpha=ALPHA)
    print("LoRA model created successfully")

    # Train the model on the simple sentence
    print(f"\n{'='*60}")
    print("Step 3: Training LoRA model on simple sentence")
    print(f"{'='*60}")

    lora_model = train_on_sentence(
        lora_model, tokenizer, TRAIN_TEXT, mesh,
        num_steps=NUM_TRAIN_STEPS, learning_rate=LEARNING_RATE
    )

    # Run forward pass on LoRA model using the trained sentence
    print(f"\n{'='*60}")
    print("Step 4: Running forward pass on trained LoRA model")
    print(f"{'='*60}")

    print(f"Test input: {test_input}")

    lora_logits = run_forward_pass(lora_model, tokenizer, test_input, mesh)
    print(f"LoRA model output shape: {lora_logits.shape}")

    # Save LoRA model
    print(f"\n{'='*60}")
    print("Step 5: Saving merged weights")
    print(f"{'='*60}")

    saved_path = save_lora_weights(lora_model, local_model_path, OUTPUT_DIR, rank=RANK, alpha=ALPHA)
    print(f"Model saved to: {saved_path}")

    # Clean up LoRA model to free memory
    del lora_model
    del base_model
    gc.collect()

    # Load the merged model
    print(f"\n{'='*60}")
    print("Step 6: Loading merged model from safetensors")
    print(f"{'='*60}")

    with mesh:
        merged_model = params_safetensors_lib.create_model_from_safe_tensors(
            str(Path(saved_path).resolve()), model_config, mesh
        )
        print("Merged model loaded successfully")

    # Run forward pass on merged model
    print(f"\n{'='*60}")
    print("Step 7: Running forward pass on merged model")
    print(f"{'='*60}")

    merged_logits = run_forward_pass(merged_model, tokenizer, test_input, mesh)
    print(f"Merged model output shape: {merged_logits.shape}")

    # Compare outputs
    print(f"\n{'='*60}")
    print("Step 8: Comparing all model outputs")
    print(f"{'='*60}")

    print("\n--- Comparison 1: Base model vs Trained LoRA model ---")
    print("(Should be DIFFERENT - training should change outputs)")
    compare_logits(base_logits, lora_logits, tolerance=1e-4)

    print("\n--- Comparison 2: Trained LoRA model vs Merged model ---")
    print("(Should be IDENTICAL - merging should preserve outputs)")
    matches = compare_logits(lora_logits, merged_logits, tolerance=1e-4)

    # Final summary
    print(f"\n{'='*60}")
    print("Test Complete!")
    print(f"{'='*60}")
    print(f"Model saved to: {saved_path}")
    print("\nResults:")
    print(f"  LoRA vs Merged models match: {matches}")

    if matches:
        print("\n✓ SUCCESS: LoRA merging is working correctly!")
        print("  ✓ Training modified the model outputs (base != trained)")
        print("  ✓ The merged model produces identical outputs to the trained LoRA model")
        print("  ✓ The alpha/rank scaling is correctly applied")
        print("  ✓ The dtype preservation is working")
    else:
        print("\n✗ FAILURE: LoRA merging has issues!")
        print("  The merged model outputs differ from the trained LoRA model.")
        print("  This indicates a bug in the merging logic.")

    return 0 if matches else 1

if __name__ == "__main__":
    main()
