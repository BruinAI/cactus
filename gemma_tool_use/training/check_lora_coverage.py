#!/usr/bin/env python3
"""
Diagnostic script to check if all LoRA-adapted modules are being extracted.
"""

import jax
import re
from flax import nnx
from tunix.models.gemma3 import params_safetensors as params_safetensors_lib
from tunix.models.gemma3 import model as gemma_lib
import qwix

from gemma_utils import download_and_setup_model, create_lora_model

MODEL_ID = "google/gemma-3-270m-it"
RANK = 32
ALPHA = 64.0
MESH_SHAPE = 1, 1
MESH_AXIS_NAMES = "fsdp", "tp"

print("="*60)
print("Checking LoRA Coverage")
print("="*60)

# Download model
local_model_path, _ = download_and_setup_model(MODEL_ID)

# Get model config
model_config = gemma_lib.ModelConfig.gemma3_270m()

# Create mesh and load base model
mesh = jax.make_mesh(MESH_SHAPE, MESH_AXIS_NAMES)

with mesh:
    base_model = params_safetensors_lib.create_model_from_safe_tensors(
        local_model_path, model_config, mesh
    )
    print("Base model loaded")

# Apply LoRA
lora_model = create_lora_model(base_model, mesh=mesh, rank=RANK, alpha=ALPHA)
print("LoRA applied")

print("\n" + "="*60)
print("Step 1: Recursively enumerate ALL fields in the lora_model")
print("="*60)

# The LoRA pattern that was used
lora_pattern = ".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj"
pattern_re = re.compile(lora_pattern)

def recursively_get_all_paths(obj, current_path="", visited=None, max_depth=4):
    """
    Recursively get all paths to all objects in the model tree.
    Returns a list of (path, obj) tuples.
    """
    if visited is None:
        visited = set()

    # Avoid infinite recursion
    obj_id = id(obj)
    if obj_id in visited or max_depth <= 0:
        return []
    visited.add(obj_id)

    paths = []

    # Add current path
    if current_path:
        paths.append(current_path)

    # Try to iterate through object's attributes
    try:
        if hasattr(obj, '__dict__'):
            for attr_name in dir(obj):
                # Skip private/special attributes
                if attr_name.startswith('_'):
                    continue

                # Skip certain non-data attributes
                if attr_name in ['parent', 'training', 'name', 'T', 'mT', 'real', 'imag', 'raw_value']:
                    continue

                try:
                    attr_value = getattr(obj, attr_name)

                    # Skip methods/functions
                    if callable(attr_value) and not hasattr(attr_value, '__dict__'):
                        continue

                    new_path = f"{current_path}.{attr_name}" if current_path else attr_name

                    # Recursively explore
                    child_paths = recursively_get_all_paths(attr_value, new_path, visited, max_depth - 1)
                    paths.extend(child_paths)

                except Exception:
                    continue
    except Exception:
        pass

    return paths

print("Enumerating all paths in model (this may take a moment)...")
all_paths = []
for layer_idx, layer in enumerate(lora_model.layers):
    layer_paths = recursively_get_all_paths(layer, f"layers.{layer_idx}")
    if not all_paths:
        print(layer_paths)
    all_paths.extend(layer_paths)

print(f"Found {len(all_paths)} total paths in model")

# Show a sample of paths
print("\nSample of paths found:")
for path in sorted(all_paths)[:20]:
    print(f"  {path}")
if len(all_paths) > 20:
    print(f"  ... and {len(all_paths) - 20} more")

# Now test which paths match the LoRA pattern
matching_paths = []
for path in all_paths:
    if pattern_re.fullmatch(path):
        matching_paths.append(path)

print(f"\nPaths matching LoRA pattern: {len(matching_paths)}")
if matching_paths:
    print("\nFirst 10 matching paths:")
    for path in sorted(matching_paths)[:10]:
        print(f"  {path}")
    if len(matching_paths) > 10:
        print(f"  ... and {len(matching_paths) - 10} more")

# Now check which of these matching paths actually have LoRA weights
print("\nChecking which matching paths have LoRA weights...")
lora_modules_found = []

for path in matching_paths:
    # Navigate to the object at this path
    parts = path.split('.')
    obj = lora_model
    try:
        for part in parts:
            # Handle list indexing for "layers"
            if isinstance(obj, (list, tuple)) and part.isdigit():
                obj = obj[int(part)]
            else:
                obj = getattr(obj, part)

        # Check for LoRA attributes (different attributes for different layer types)
        has_lora_a = (hasattr(obj, 'w_lora_a') or hasattr(obj, 'kernel_lora_a'))
        has_lora_b = (hasattr(obj, 'w_lora_b') or hasattr(obj, 'kernel_lora_b'))

        # Debug first few
        if len(lora_modules_found) < 3:
            attrs = [a for a in dir(obj) if 'lora' in a.lower() or 'kernel' in a.lower() or a == 'w']
            print(f"\n  Checking {path}:")
            print(f"    Relevant attributes: {attrs[:10]}")
            print(f"    has_lora_a: {has_lora_a}, has_lora_b: {has_lora_b}")

        if has_lora_a and has_lora_b:
            lora_modules_found.append(path)
    except Exception as e:
        if len(lora_modules_found) < 3:
            print(f"  Error navigating to {path}: {e}")
        continue

print(f"\nModules with actual LoRA weights: {len(lora_modules_found)}")
if lora_modules_found:
    print("\nFirst 10 modules with LoRA:")
    for path in sorted(lora_modules_found)[:10]:
        print(f"  ✓ {path}")
    if len(lora_modules_found) > 10:
        print(f"  ... and {len(lora_modules_found) - 10} more")

print("\n" + "="*60)
print("Step 2: Checking what save_lora_weights extracts")
print("="*60)

def path_to_str(qwix_path):
    """Convert qwix path to string."""
    return '.'.join([str(field) for field in qwix_path])

# Replicate the extraction logic from save_lora_weights
extracted_paths = set()

for layer_idx, layer in enumerate(lora_model.layers):
    # q_einsum
    proj = layer.attn.q_einsum
    path = path_to_str(proj.qwix_path)
    extracted_paths.add(path)

    # kv_einsum (extracted as k and v separately)
    proj = layer.attn.kv_einsum
    path = path_to_str(proj.qwix_path)
    extracted_paths.add(path)

    # MLP projections
    for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
        proj = getattr(layer.mlp, proj_name)
        path = path_to_str(proj.qwix_path)
        extracted_paths.add(path)

print(f"\nExtracted {len(extracted_paths)} unique LoRA layer paths")
print("\nExtracted paths:")
for i, path in enumerate(sorted(extracted_paths), 1):
    print(f"{i}. {path}")

print("\n" + "="*60)
print("Step 3: Comparing found LoRA modules vs extracted paths")
print("="*60)

# Convert lora_modules_found to a set for comparison
all_lora_paths = set(lora_modules_found)
missing_from_extraction = all_lora_paths - extracted_paths
extra_in_extraction = extracted_paths - all_lora_paths

print("\n" + "="*60)
print("RESULTS")
print("="*60)

print(f"\nTotal paths in model: {len(all_paths)}")
print(f"Paths matching LoRA pattern: {len(matching_paths)}")
print(f"Modules with actual LoRA weights: {len(lora_modules_found)}")
print(f"Paths extracted by save_lora_weights: {len(extracted_paths)}")

if missing_from_extraction:
    print(f"\n⚠ WARNING: {len(missing_from_extraction)} LoRA modules NOT being extracted:")
    for path in sorted(missing_from_extraction):
        print(f"  ✗ {path}")
else:
    print("\n✓ All LoRA modules are being extracted")

if extra_in_extraction:
    print(f"\n⚠ WARNING: {len(extra_in_extraction)} paths extracted but no LoRA found:")
    for path in sorted(extra_in_extraction):
        print(f"  ✗ {path}")
else:
    print("✓ No extra paths being extracted")

if not missing_from_extraction and not extra_in_extraction:
    print("\n✓✓✓ Perfect match! All LoRA weights are being extracted correctly.")
else:
    print("\n✗✗✗ Mismatch detected! This could explain output differences.")
