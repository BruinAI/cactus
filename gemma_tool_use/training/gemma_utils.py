#!/usr/bin/env python3
"""
Gemma 3 model utilities for LoRA fine-tuning.

This module provides utilities for:
- Downloading Gemma models from HuggingFace
- Applying LoRA (Low-Rank Adaptation) to Gemma models
- Saving LoRA weights merged with base models
"""

import os
import json
import shutil

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from huggingface_hub import snapshot_download
import qwix
from safetensors import numpy as safe_np


def download_and_setup_model(model_id: str):
    """
    Download model from HuggingFace and setup tokenizer.

    Args:
        model_id: HuggingFace model ID (e.g., "google/gemma-3-270m-it")

    Returns:
        Tuple of (local_model_path, eos_tokens)
    """
    print(f"\n{'='*60}")
    print(f"Downloading {model_id} from HuggingFace")
    print(f"{'='*60}")

    ignore_patterns = ["*.pth"]  # Ignore PyTorch weights
    local_model_path = snapshot_download(
        repo_id=model_id,
        ignore_patterns=ignore_patterns
    )

    print(f"Model downloaded to: {local_model_path}")

    # Load generation config for EOS tokens
    generation_config_path = os.path.join(local_model_path, "generation_config.json")
    eos_tokens = []
    if os.path.exists(generation_config_path):
        with open(generation_config_path, "r") as f:
            generation_configs = json.load(f)
        eos_tokens = generation_configs.get("eos_token_id", [])
        print(f"Using EOS token IDs: {eos_tokens}")

    return local_model_path, eos_tokens


def create_lora_model(base_model, mesh, rank: int, alpha: float):
    """
    Apply LoRA to the base model.

    Args:
        base_model: Base Gemma model
        mesh: JAX mesh for sharding
        rank: LoRA rank (dimensionality of low-rank matrices)
        alpha: LoRA alpha (scaling factor)

    Returns:
        LoRA model
    """
    print(f"\n{'='*60}")
    print("Applying LoRA to model")
    print(f"{'='*60}")
    print(f"  Rank: {rank}")
    print(f"  Alpha: {alpha}")

    lora_provider = qwix.LoraProvider(
        module_path=".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj|.*attn_vec_einsum",
        rank=rank,
        alpha=alpha,
    )

    model_input = base_model.get_model_input()
    lora_model = qwix.apply_lora_to_model(
        base_model, lora_provider, **model_input
    )

    with mesh:
        state = nnx.state(lora_model)
        pspecs = nnx.get_partition_spec(state)
        sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
        nnx.update(lora_model, sharded_state)

    return lora_model


def save_lora_weights(lora_model, local_model_path: str, output_dir: str, rank: int, alpha: float):
    """
    Save LoRA weights merged with base model as safetensors.

    Args:
        lora_model: Trained LoRA model
        local_model_path: Path to base model (for loading base weights)
        output_dir: Directory to save merged weights
        rank: LoRA rank
        alpha: LoRA alpha

    Returns:
        Path to saved weights directory
    """
    print(f"\n{'='*60}")
    print("Saving model with merged LoRA weights")
    print(f"{'='*60}")

    # Remove and recreate output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    print(f"Saving to: {output_dir}")

    print("\nStep 1: Extracting LoRA weights from lora_model...")

    def path_to_str(qwix_path):
        """Convert qwix path to string."""
        return '.'.join([str(field) for field in qwix_path])

    # Extract LoRA layers and infer rank/alpha if not provided
    lora_layers = {}
    for layer in lora_model.layers:
        proj = layer.attn.q_einsum
        path = path_to_str(proj.qwix_path)
        lora_layers[path] = (proj.w_lora_a, proj.w_lora_b)

        proj = layer.attn.kv_einsum
        path = path_to_str(proj.qwix_path)
        lora_layers[path.replace('kv_einsum', 'k_einsum')] = (
            proj.w_lora_a, proj.w_lora_b[:, 0]
        )
        lora_layers[path.replace('kv_einsum', 'v_einsum')] = (
            proj.w_lora_a, proj.w_lora_b[:, 1]
        )

        for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
            proj = getattr(layer.mlp, proj_name)
            path = path_to_str(proj.qwix_path)
            lora_layers[path] = (proj.kernel_lora_a, proj.kernel_lora_b)

    print(f"Found {len(lora_layers)} LoRA layers")
    print(f"LoRA layer names: {list(lora_layers.keys())[:3]}...")

    # Load base model state
    print("\nStep 2: Loading base model weights...")
    base_state = safe_np.load_file(local_model_path + "/model.safetensors")
    print(f"Loaded {len(base_state)} base model parameters")

    # Step 3: Apply LoRA deltas to base weights
    print("\nStep 3: Merging LoRA deltas with base weights...")

    for lora_name, (lora_a, lora_b) in lora_layers.items():
        state_key = (
            f'model.{lora_name}.weight'
            .replace('.attn.', '.self_attn.')
            .replace('q_einsum', 'q_proj')
            .replace('k_einsum', 'k_proj')
            .replace('v_einsum', 'v_proj')
        )
        assert state_key in base_state

        lora_a_val = jnp.asarray(getattr(lora_a, 'value', lora_a))
        lora_b_val = jnp.asarray(getattr(lora_b, 'value', lora_b))

        # Reshape 3D LoRA matrices to 2D for matrix multiplication
        # LoRA A: (d0, d1, d2) -> (d0*d1, d2)  |  LoRA B: (d0, d1, d2) -> (d0, d1*d2)
        if lora_a_val.ndim == 3:
            print(f"Merging LoRA layer: {lora_name} -> {state_key}")
            print("  Base weight shape:", base_state[state_key].shape)
            print("  LoRA A shape:", lora_a_val.shape)
            d0, d1, d2 = lora_a_val.shape
            lora_a_val = lora_a_val.reshape(d0 * d1, d2)
            print("    Reshaped LoRA A to:", lora_a_val.shape)
            print("  LoRA B shape:", lora_b_val.shape)
        if lora_b_val.ndim == 3:
            print(f"Merging LoRA layer: {lora_name} -> {state_key}")
            print("  Base weight shape:", base_state[state_key].shape)
            print("  LoRA A shape:", lora_a_val.shape)
            print("  LoRA B shape:", lora_b_val.shape)
            d0, d1, d2 = lora_b_val.shape
            lora_b_val = lora_b_val.reshape(d0, d1 * d2)
            print("    Reshaped LoRA B to:", lora_b_val.shape)

        # Compute LoRA delta: A @ B and apply alpha/rank scaling
        combined_lora = (lora_a_val @ lora_b_val) * (alpha / rank)
        base_state[state_key] += combined_lora.T.astype(base_state[state_key].dtype)

    print(f"\nMerged {len(lora_layers)} LoRA layers into base weights")

    # Step 4: Save merged weights as safetensors
    print("\nStep 4: Saving as safetensors...")
    safetensors_path = os.path.join(output_dir, "model.safetensors")
    safe_np.save_file(base_state, safetensors_path)
    print("Model weights saved")

    # Step 5: Copy other model files
    print("\nStep 5: Copying other model files...")
    for filename in os.listdir(local_model_path):
        # Check if the file is NOT a safetensors file
        if not filename.endswith(".safetensors"):
            src = os.path.join(local_model_path, filename)
            dst = os.path.join(output_dir, filename)

            # Check if it's a file (and not a directory) before copying
            if os.path.isfile(src):
                shutil.copy(src, dst)
                print(f"  Copied {filename} from base model")

    print("\n" + "="*60)
    print("Model saved successfully!")
    print(f"Output directory: {output_dir}")
    print("="*60)

    print("\nSaved files:")
    for f in os.listdir(output_dir):
        size = os.path.getsize(os.path.join(output_dir, f)) / (1024 * 1024)
        print(f"  {f:<30} {size:>10.2f} MB")

    return output_dir
