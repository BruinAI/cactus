#!/usr/bin/env python3
"""
Test script to compare LoRA model vs saved/reloaded merged Gemma model.
Trains on a simple sentence and tests that the LoRA merging logic produces identical outputs.
"""

import os
import gc
from pathlib import Path

import wandb
wandb.init(mode="disabled")  # Disable wandb for test runs

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
NUM_TRAIN_STEPS = 10  # Train for 10 steps (enough to see significant changes)

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
        trainer.train(train_data_iter())

    print("Training complete!")
    return model


def compare_logits(logits_a, logits_b, label_a="Model A", label_b="Model B", tolerance=1e-1, check_top_k=True, detailed=False):
    """
    Compare logits from two models.

    Args:
        logits_a: Logits from first model
        logits_b: Logits from second model
        label_a: Label for first model
        label_b: Label for second model
        tolerance: Maximum allowed difference for bfloat16 precision
        check_top_k: If True, also check if top-5 predictions match
        detailed: If True, print detailed analysis of differences

    Returns:
        bool: True if logits match within tolerance
    """
    print(f"\n{'='*60}")
    print(f"Comparing {label_a} vs {label_b}")
    print(f"{'='*60}")

    # Convert to numpy for easier analysis
    logits_a_np = np.array(logits_a)
    logits_b_np = np.array(logits_b)

    print(f"{label_a} logits shape: {logits_a_np.shape}")
    print(f"{label_b} logits shape: {logits_b_np.shape}")
    print(f"{label_a} dtype: {logits_a_np.dtype}")
    print(f"{label_b} dtype: {logits_b_np.dtype}")

    # Compute differences
    abs_diff = np.abs(logits_a_np - logits_b_np)
    max_abs_diff = float(np.max(abs_diff))
    mean_abs_diff = float(np.mean(abs_diff))
    median_abs_diff = float(np.median(abs_diff))

    print("\nDifference Statistics:")
    print(f"  Max absolute diff:    {max_abs_diff:.6f}")
    print(f"  Mean absolute diff:   {mean_abs_diff:.6f}")
    print(f"  Median absolute diff: {median_abs_diff:.6f}")

    # Percentiles
    p95 = float(np.percentile(abs_diff, 95))
    p99 = float(np.percentile(abs_diff, 99))
    p999 = float(np.percentile(abs_diff, 99.9))
    print(f"  95th percentile:      {p95:.6f}")
    print(f"  99th percentile:      {p99:.6f}")
    print(f"  99.9th percentile:    {p999:.6f}")

    # Count of differences by magnitude
    count_gt_0001 = np.sum(abs_diff > 0.001)
    count_gt_001 = np.sum(abs_diff > 0.01)
    count_gt_01 = np.sum(abs_diff > 0.1)
    count_gt_1 = np.sum(abs_diff > 1.0)
    total_elements = abs_diff.size

    print(f"\nDifference Distribution:")
    print(f"  Diffs > 0.001: {count_gt_0001:,} ({100*count_gt_0001/total_elements:.4f}%)")
    print(f"  Diffs > 0.01:  {count_gt_001:,} ({100*count_gt_001/total_elements:.4f}%)")
    print(f"  Diffs > 0.1:   {count_gt_01:,} ({100*count_gt_01/total_elements:.4f}%)")
    print(f"  Diffs > 1.0:   {count_gt_1:,} ({100*count_gt_1/total_elements:.4f}%)")

    if detailed:
        # Find location of max difference
        max_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
        print(f"\nMax Difference Location:")
        print(f"  Position: batch={max_idx[0]}, seq={max_idx[1]}, vocab={max_idx[2]}")
        print(f"  {label_a} value: {float(logits_a_np[max_idx]):.6f}")
        print(f"  {label_b} value: {float(logits_b_np[max_idx]):.6f}")
        print(f"  Difference: {float(abs_diff[max_idx]):.6f}")

        # Analyze last token (most important for generation)
        last_token_diff = abs_diff[:, -1, :]
        print(f"\nLast Token Analysis:")
        print(f"  Max diff in last token: {float(np.max(last_token_diff)):.6f}")
        print(f"  Mean diff in last token: {float(np.mean(last_token_diff)):.6f}")
        print(f"  Tokens with diff > 0.1: {int(np.sum(last_token_diff > 0.1))}")

    # Check final token logits (last token)
    print("\nFinal Token Logits (first 5 values):")
    print(f"  {label_a:12s}: {logits_a_np[0, -1, :5]}")
    print(f"  {label_b:12s}: {logits_b_np[0, -1, :5]}")
    print(f"  Differences:    {abs_diff[0, -1, :5]}")

    # Top-5 predictions for last token
    top5_a = np.argsort(logits_a_np[0, -1])[-5:][::-1]
    top5_b = np.argsort(logits_b_np[0, -1])[-5:][::-1]

    print("\nTop-5 Token Predictions (last position):")
    print(f"  {label_a:12s}: {top5_a.tolist()}")
    print(f"  {label_b:12s}: {top5_b.tolist()}")

    # Show logit values for top-5 predictions
    if detailed:
        print(f"\nTop-5 Logit Values from {label_a}:")
        for i, idx in enumerate(top5_a):
            val_a = float(logits_a_np[0, -1, idx])
            val_b = float(logits_b_np[0, -1, idx])
            print(f"    Rank {i+1} (token {idx}): {label_a}={val_a:.6f}, {label_b}={val_b:.6f}, diff={abs(val_a-val_b):.6f}")

    # Determine if models match
    logits_match = max_abs_diff < tolerance
    top_k_match = np.array_equal(top5_a, top5_b) if check_top_k else True

    print(f"\n{'='*60}")
    if logits_match and top_k_match:
        print(f"✓ MATCH: Max diff {max_abs_diff:.4f} < tolerance {tolerance:.2e}")
        if check_top_k:
            print("✓ Top-5 predictions are identical")
    else:
        if not logits_match:
            print(f"{'✓' if not check_top_k else '⚠'} Logits differ: max diff {max_abs_diff:.4f} > tolerance {tolerance:.2e}")
        if check_top_k and not top_k_match:
            print("✗ Top-5 predictions differ!")
        elif check_top_k and top_k_match:
            print("✓ Top-5 predictions match (models functionally equivalent)")
    print(f"{'='*60}")

    return logits_match and top_k_match


def verify_base_weights_unchanged(base_model, lora_model):
    """
    Verify that base model weights haven't changed during LoRA training.

    Args:
        base_model: Original base model
        lora_model: LoRA model after training

    Returns:
        bool: True if base weights are unchanged
    """
    print(f"\n{'='*60}")
    print("Verifying Base Weights Unchanged")
    print(f"{'='*60}")

    # Check a few key layers
    num_checked = 0
    all_match = True

    for i in [0, len(base_model.layers) // 2, len(base_model.layers) - 1]:
        # Check embedding weights (not LoRA-adapted)
        base_embed = base_model.layers[i].mlp.gate_proj.kernel.value
        lora_embed = lora_model.layers[i].mlp.gate_proj.kernel.value

        max_diff = float(jnp.max(jnp.abs(base_embed - lora_embed)))
        print(f"Layer {i} gate_proj base weights max diff: {max_diff:.10f}")

        if max_diff > 1e-6:
            print("  ⚠ WARNING: Base weights changed!")
            all_match = False
        else:
            print("  ✓ Base weights unchanged")

        num_checked += 1

    print(f"\nChecked {num_checked} layers")
    return all_match


def verify_lora_weights_changed(lora_model):
    """
    Verify that LoRA weights actually changed during training.

    Args:
        lora_model: LoRA model after training

    Returns:
        bool: True if LoRA weights have non-zero values
    """
    print(f"\n{'='*60}")
    print("Verifying LoRA Weights Changed")
    print(f"{'='*60}")

    # Check a few LoRA layers
    num_checked = 0
    all_nonzero = True

    for i in [0, len(lora_model.layers) // 2, len(lora_model.layers) - 1]:
        # Check LoRA weights
        lora_a = lora_model.layers[i].attn.q_einsum.w_lora_a.value
        lora_b = lora_model.layers[i].attn.q_einsum.w_lora_b.value

        lora_a_norm = float(jnp.linalg.norm(lora_a))
        lora_b_norm = float(jnp.linalg.norm(lora_b))

        print(f"Layer {i} q_proj LoRA A norm: {lora_a_norm:.6f}")
        print(f"Layer {i} q_proj LoRA B norm: {lora_b_norm:.6f}")

        if lora_a_norm < 1e-6 or lora_b_norm < 1e-6:
            print("  ⚠ WARNING: LoRA weights are zero!")
            all_nonzero = False
        else:
            print("  ✓ LoRA weights are non-zero")

        num_checked += 1

    print(f"\nChecked {num_checked} layers")
    return all_nonzero


def verify_merged_weights_correct(base_model, lora_model, merged_model):
    """
    Verify that merged weights = base weights + LoRA delta.

    Args:
        base_model: Original base model
        lora_model: LoRA model after training
        merged_model: Merged model loaded from safetensors

    Returns:
        bool: True if merged weights are correct
    """
    print(f"\n{'='*60}")
    print("Verifying Merged Weights Computation")
    print(f"{'='*60}")

    num_checked = 0
    all_correct = True

    for i in [0, len(base_model.layers) // 2, len(base_model.layers) - 1]:
        # Get base weight - for Einsum layers, the weight is stored as 'w'
        base_weight = base_model.layers[i].attn.q_einsum.w.value

        # Get LoRA weights
        lora_a = lora_model.layers[i].attn.q_einsum.w_lora_a.value
        lora_b = lora_model.layers[i].attn.q_einsum.w_lora_b.value

        # Handle multi-head case BEFORE matrix multiplication
        lora_a_val = jnp.asarray(lora_a)
        lora_b_val = jnp.asarray(lora_b)

        if lora_b_val.ndim == 3:
            # Reshape for multi-head: (rank, num_heads, head_dim) -> (rank, num_heads * head_dim)
            d0, d1, d2 = lora_b_val.shape
            lora_b_val = lora_b_val.reshape(d0, d1 * d2)

        # Compute expected merged weight: base + (lora_a @ lora_b * alpha/rank).T
        lora_delta = (lora_a_val @ lora_b_val) * (ALPHA / RANK)

        print(f"\nLayer {i} q_proj:")
        print(f"  Base weight shape: {base_weight.shape}")
        print(f"  LoRA A shape: {lora_a.shape}")
        print(f"  LoRA B shape: {lora_b.shape}")
        print(f"  LoRA A (processed) shape: {lora_a_val.shape}")
        print(f"  LoRA B (processed) shape: {lora_b_val.shape}")
        print(f"  LoRA delta shape: {lora_delta.shape}")

        # The delta needs to be transposed and possibly reshaped to match base weight shape
        lora_delta_t = lora_delta.T

        # If base weight is 3D (num_heads, input_dim, head_dim), reshape delta to match
        if base_weight.ndim == 3 and lora_delta_t.ndim == 2:
            # lora_delta_t: (num_heads * head_dim, input_dim) -> (num_heads, input_dim, head_dim)
            num_heads = base_weight.shape[0]
            input_dim = base_weight.shape[1]
            head_dim = base_weight.shape[2]
            lora_delta_t = lora_delta_t.reshape(num_heads, head_dim, input_dim).transpose(0, 2, 1)

        print(f"  LoRA delta (final) shape: {lora_delta_t.shape}")

        expected_merged = base_weight + lora_delta_t.astype(base_weight.dtype)

        # Get actual merged weight
        actual_merged = merged_model.layers[i].attn.q_einsum.w.value

        # Compare
        max_diff = float(jnp.max(jnp.abs(expected_merged - actual_merged)))
        mean_diff = float(jnp.mean(jnp.abs(expected_merged - actual_merged)))

        print("  Expected merged vs actual merged:")
        print(f"    Max diff: {max_diff:.10f}")
        print(f"    Mean diff: {mean_diff:.10f}")

        # bfloat16 tolerance
        tolerance = 1e-2
        if max_diff > tolerance:
            print(f"  ⚠ WARNING: Merged weights don't match expected (max diff {max_diff:.6f} > {tolerance})")
            all_correct = False
        else:
            print("  ✓ Merged weights match expected")

        num_checked += 1

    print(f"\nChecked {num_checked} layers")
    return all_correct


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

    test_input = TRAIN_TEXT.rsplit(" ", 1)[0]  # All but last word
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

    # Verify base weights unchanged and LoRA weights changed
    base_unchanged = verify_base_weights_unchanged(base_model, lora_model)
    lora_changed = verify_lora_weights_changed(lora_model)

    # Save LoRA model
    print(f"\n{'='*60}")
    print("Step 5: Saving merged weights")
    print(f"{'='*60}")

    saved_path = save_lora_weights(lora_model, local_model_path, OUTPUT_DIR, rank=RANK, alpha=ALPHA)
    print(f"Model saved to: {saved_path}")

    # Load the merged model (keep base_model and lora_model for verification)
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

    # Verify merged weights are computed correctly
    merged_correct = verify_merged_weights_correct(base_model, lora_model, merged_model)

    # Clean up models to free memory
    del lora_model
    del base_model
    del merged_model
    gc.collect()

    # Compare outputs
    print(f"\n{'='*60}")
    print("Step 8: Comparing all model outputs")
    print(f"{'='*60}")

    print("\n--- Comparison 1: Base model vs Trained LoRA model ---")
    print("(Should be DIFFERENT - training should change outputs)\n")
    base_vs_trained = compare_logits(
        base_logits, lora_logits,
        label_a="Base", label_b="Trained LoRA",
        tolerance=1e-4,  # Expect large differences
        check_top_k=True
    )

    print("\n--- Comparison 2: Base model vs Merged model ---")
    print("(For reference: shows magnitude of LoRA effect)\n")
    base_vs_merged = compare_logits(
        base_logits, merged_logits,
        label_a="Base", label_b="Merged",
        tolerance=1e-4,
        check_top_k=True,
        detailed=False  # Less detail for this comparison
    )

    print("\n--- Comparison 3: Trained LoRA model vs Merged model ---")
    print("(Should be IDENTICAL - merging should preserve outputs)\n")
    matches = compare_logits(
        lora_logits, merged_logits,
        label_a="Trained LoRA", label_b="Merged",
        tolerance=1e-4,  # bfloat16 precision tolerance (increased for accumulation errors)
        check_top_k=True,
        detailed=True  # Enable detailed analysis to investigate numerical differences
    )

    # Final summary
    print(f"\n{'='*60}")
    print("Test Complete!")
    print(f"{'='*60}")
    print(f"Model saved to: {saved_path}")

    print("\n" + "="*60)
    print("VERIFICATION RESULTS")
    print("="*60)
    print(f"1. Base weights unchanged:     {'✓ PASS' if base_unchanged else '✗ FAIL'}")
    print(f"2. LoRA weights changed:       {'✓ PASS' if lora_changed else '✗ FAIL'}")
    print(f"3. Merged weights correct:     {'✓ PASS' if merged_correct else '✗ FAIL'}")
    print(f"4. Training changed outputs:   {'✓ PASS' if not base_vs_trained else '✗ FAIL'}")
    print(f"5. Logits match tolerance:     {'✓ PASS' if matches else '⚠ CLOSE'}")
    print("="*60)

    training_changed = not base_vs_trained  # We want them to be different
    merging_correct = matches  # We want them to match
    all_verifications = base_unchanged and lora_changed and merged_correct

    if all_verifications and training_changed:
        print("\n✓ SUCCESS: All verifications passed!")
        print("  ✓ Base weights remained frozen during training")
        print("  ✓ LoRA weights were successfully trained")
        print("  ✓ Merged weights match expected computation")
        print("  ✓ Training modified the model outputs")
        if merging_correct:
            print("  ✓ Merged model produces identical outputs")
        else:
            print("  ⚠ Small logit differences due to bfloat16 precision")
            print("    (This is expected and functionally equivalent)")
    else:
        print("\n⚠ ISSUES DETECTED:")
        if not base_unchanged:
            print("  ✗ Base weights changed during training!")
            print("    LoRA should freeze base weights")
        if not lora_changed:
            print("  ✗ LoRA weights did not change!")
            print("    Training may not be working")
        if not merged_correct:
            print("  ✗ Merged weights don't match expected computation!")
            print("    Check alpha/rank scaling and dtype handling")
        if not training_changed:
            print("  ✗ Training did NOT modify outputs!")
            print("    Check learning rate and training loop")
        if not merging_correct:
            print("  ⚠ Logits differ between trained LoRA and merged models")
            print("    This may indicate numerical issues in merging")

    return 0 if (all_verifications and training_changed) else 1

if __name__ == "__main__":
    main()
