#!/usr/bin/env python3
"""
Qwen 3 model utilities for LoRA fine-tuning.

This module provides utilities for:
- Downloading Qwen models from HuggingFace
- Applying LoRA (Low-Rank Adaptation) to Qwen models
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
        model_id: HuggingFace model ID (e.g., "Qwen/Qwen3-0.6B-Chat")

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
        # Handle both single value and list
        if not isinstance(eos_tokens, list):
            eos_tokens = [eos_tokens]
        print(f"Using EOS token IDs: {eos_tokens}")

    return local_model_path, eos_tokens


def create_lora_model(base_model, mesh, rank: int, alpha: float):
    """
    Apply LoRA to the base model.

    Args:
        base_model: Base Qwen model
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

    # Apply LoRA to attention and MLP layers
    lora_provider = qwix.LoraProvider(
        module_path=".*q_proj|.*k_proj|.*v_proj|.*o_proj|.*gate_proj|.*down_proj|.*up_proj",
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


def save_lora_weights(lora_model, local_model_path: str, output_dir: str):
    """
    Save LoRA weights merged with base model as safetensors.

    Args:
        lora_model: Trained LoRA model
        local_model_path: Path to base model (for loading base weights)
        output_dir: Directory to save merged weights

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

    # Extract LoRA layers from both attention and MLP
    lora_layers = {}
    for layer in lora_model.layers:
        # Attention projections
        if hasattr(layer.self_attn, 'q_proj') and hasattr(layer.self_attn.q_proj, 'kernel_lora_a'):
            q_proj_path = path_to_str(layer.self_attn.q_proj.qwix_path)
            lora_layers[q_proj_path] = (
                layer.self_attn.q_proj.kernel_lora_a,
                layer.self_attn.q_proj.kernel_lora_b
            )

        if hasattr(layer.self_attn, 'k_proj') and hasattr(layer.self_attn.k_proj, 'kernel_lora_a'):
            k_proj_path = path_to_str(layer.self_attn.k_proj.qwix_path)
            lora_layers[k_proj_path] = (
                layer.self_attn.k_proj.kernel_lora_a,
                layer.self_attn.k_proj.kernel_lora_b
            )

        if hasattr(layer.self_attn, 'v_proj') and hasattr(layer.self_attn.v_proj, 'kernel_lora_a'):
            v_proj_path = path_to_str(layer.self_attn.v_proj.qwix_path)
            lora_layers[v_proj_path] = (
                layer.self_attn.v_proj.kernel_lora_a,
                layer.self_attn.v_proj.kernel_lora_b
            )

        if hasattr(layer.self_attn, 'o_proj') and hasattr(layer.self_attn.o_proj, 'kernel_lora_a'):
            o_proj_path = path_to_str(layer.self_attn.o_proj.qwix_path)
            lora_layers[o_proj_path] = (
                layer.self_attn.o_proj.kernel_lora_a,
                layer.self_attn.o_proj.kernel_lora_b
            )

        # MLP projections
        if hasattr(layer.mlp, 'down_proj') and hasattr(layer.mlp.down_proj, 'kernel_lora_a'):
            down_proj_path = path_to_str(layer.mlp.down_proj.qwix_path)
            lora_layers[down_proj_path] = (
                layer.mlp.down_proj.kernel_lora_a,
                layer.mlp.down_proj.kernel_lora_b
            )

        if hasattr(layer.mlp, 'up_proj') and hasattr(layer.mlp.up_proj, 'kernel_lora_a'):
            up_proj_path = path_to_str(layer.mlp.up_proj.qwix_path)
            lora_layers[up_proj_path] = (
                layer.mlp.up_proj.kernel_lora_a,
                layer.mlp.up_proj.kernel_lora_b
            )

        if hasattr(layer.mlp, 'gate_proj') and hasattr(layer.mlp.gate_proj, 'kernel_lora_a'):
            gate_proj_path = path_to_str(layer.mlp.gate_proj.qwix_path)
            lora_layers[gate_proj_path] = (
                layer.mlp.gate_proj.kernel_lora_a,
                layer.mlp.gate_proj.kernel_lora_b
            )

    print(f"Found {len(lora_layers)} LoRA layers")
    print(f"LoRA layer names: {list(lora_layers.keys())[:3]}...")

    # Load base model state
    print("\nStep 2: Loading base model weights...")
    base_state = safe_np.load_file(local_model_path + "/model.safetensors")
    print(f"Loaded {len(base_state)} base model parameters")

    # Step 3: Apply LoRA deltas to base weights
    print("\nStep 3: Merging LoRA deltas with base weights...")
    for lora_name, (lora_a, lora_b) in lora_layers.items():
        state_key = f'model.{lora_name}.weight'
        assert state_key in base_state, \
               f"LoRA layer {lora_name} not found in base model state dict"

        lora_a_val = jnp.asarray(lora_a.value).astype(np.float32)
        lora_b_val = jnp.asarray(lora_b.value).astype(np.float32)

        combined_lora = lora_a_val @ lora_b_val
        base_state[state_key] = base_state[state_key] + combined_lora.T

    print(f"Merged {len(lora_layers)} LoRA layers into base weights")

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
