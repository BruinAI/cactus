#!/usr/bin/env python3
"""
Simple script to load a Gemma 3 1B LoRA model and immediately save it.
Used for testing and updating LoRA merging logic.
"""

import os
import jax
from tunix.models.gemma3 import params_safetensors as params_safetensors_lib
from tunix.models.gemma3 import model as gemma_lib
from gemma_utils import download_and_setup_model, create_lora_model, save_lora_weights
import qwix

# Configuration
MODEL_ID = "google/gemma-3-1b-it"
RANK = 32
ALPHA = 64.0
OUTPUT_DIR = "./test_output/gemma3-1b-lora-merged"

# Mesh configuration
MESH_SHAPE = len(jax.devices()), 1
MESH_AXIS_NAMES = "fsdp", "tp"

def main():
    print("="*60)
    print("Testing LoRA Load and Save")
    print("="*60)
    print(f"Model: {MODEL_ID}")
    print(f"LoRA rank: {RANK}, alpha: {ALPHA}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Devices: {len(jax.devices())} x {jax.devices()[0].platform}")

    # Download model
    local_model_path, eos_tokens = download_and_setup_model(MODEL_ID)

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
    print("Loading base model")
    print(f"{'='*60}")

    with mesh:
        base_model = params_safetensors_lib.create_model_from_safe_tensors(
            local_model_path, model_config, mesh
        )
        print("Base model loaded successfully")

    # Apply LoRA
    lora_model = create_lora_model(base_model, mesh=mesh, rank=RANK, alpha=ALPHA)
    print("LoRA model created successfully")

    # Save immediately (this tests the save logic)
    saved_path = save_lora_weights(lora_model, local_model_path, OUTPUT_DIR)

    print(f"\n{'='*60}")
    print("Test Complete!")
    print(f"{'='*60}")
    print(f"Model saved to: {saved_path}")
    print(f"\nYou can now test this model with:")
    print(f"  python3 tools/convert_hf.py {saved_path} weights/gemma3-1b-test/ --precision INT8")

if __name__ == "__main__":
    main()
