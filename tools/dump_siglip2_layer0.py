#!/usr/bin/env python3
"""Dump selected LFM2-VL layer-0 tensors from Hugging Face into S2VISPEM binaries.

This helper loads ``LiquidAI/LFM2-VL-450M`` (or another compatible checkpoint),
executes its SigLIP-2 vision tower on a single image, captures the outputs of
the following modules for encoder layer 0, and writes each tensor to its own
binary file so they can be compared against the C++ debug runner:

- model.vision_tower.vision_model.embeddings.position_embedding *(resized/padded)*
- model.vision_tower.vision_model.encoder.layers.0.self_attn.k_proj
- model.vision_tower.vision_model.encoder.layers.0.self_attn.v_proj
- model.vision_tower.vision_model.encoder.layers.0.self_attn.out_proj
- model.vision_tower.vision_model.encoder.layers.0.mlp

Outputs use the same custom format as the C++ runner (magic "S2VISPEM" followed
by little-endian header metadata and float32 payloads).
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, Lfm2VlForConditionalGeneration

MAGIC = b"S2VISPEM"
DEFAULT_CHECKPOINT = "LiquidAI/LFM2-VL-450M"
DEFAULT_IMAGE = Path("tests/istockphoto-184978580-2048x2048.jpg")


def write_s2vis_tensor(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(array, dtype="<f4", order="C")
    with path.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", 1))  # version
        f.write(struct.pack("<I", data.ndim))
        for dim in data.shape:
            f.write(struct.pack("<I", dim))
        f.write(data.tobytes())


def to_host_tensor(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to(torch.float32).cpu()


def capture_module_output(module: torch.nn.Module, store: Dict[str, torch.Tensor], key: str):
    def hook(_module, _inputs, output):  # pragma: no cover - runtime hook
        tensor = output[0] if isinstance(output, (tuple, list)) else output
        store[key] = to_host_tensor(tensor)

    return module.register_forward_hook(hook)


def squeeze_batch(t: torch.Tensor) -> torch.Tensor:
    if t.dim() >= 3 and t.shape[0] == 1:
        return t.squeeze(0)
    return t


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump LFM2-VL layer-0 tensors to S2VISPEM bins")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="Hugging Face checkpoint to load (default: %(default)s)")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE,
                        help="Path to the image file (default: %(default)s)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory where the tensor dumps will be written")
    parser.add_argument("--device", default=None,
                        help="Torch device to use (default: cuda if available else cpu)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if not args.image.exists():
        raise FileNotFoundError(f"Image file not found: {args.image}")

    processor = AutoImageProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = Lfm2VlForConditionalGeneration.from_pretrained(
        args.checkpoint, trust_remote_code=True
    )
    model.to(device)
    model.eval()

    if not hasattr(model, "vision_tower"):
        raise AttributeError("Checkpoint does not expose a vision_tower module")
    if not hasattr(model.vision_tower, "vision_model"):
        raise AttributeError("vision_tower is missing a vision_model attribute")

    vm = model.vision_tower.vision_model

    image = Image.open(args.image).convert("RGB")
    batch = processor(images=image, return_tensors="pt")
    batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

    captures: Dict[str, torch.Tensor] = {}
    handles = []

    layer0 = vm.encoder.layers[0]
    handles.append(capture_module_output(layer0.self_attn.k_proj, captures, "layer0_self_attn_k_proj"))
    handles.append(capture_module_output(layer0.self_attn.v_proj, captures, "layer0_self_attn_v_proj"))
    handles.append(capture_module_output(layer0.self_attn.out_proj, captures, "layer0_self_attn_out_proj"))
    handles.append(capture_module_output(layer0.mlp, captures, "layer0_mlp"))

    with torch.no_grad():
        model.vision_tower(
            pixel_values=batch["pixel_values"],
            pixel_attention_mask=batch.get("pixel_attention_mask"),
            spatial_shapes=batch.get("spatial_shapes"),
        )

    for handle in handles:
        handle.remove()

    # Position embedding (resized + padded) computed manually to match the C++ implementation
    with torch.no_grad():
        embeddings = vm.embeddings
        spatial_shapes = batch.get("spatial_shapes")
        if spatial_shapes is None:
            raise RuntimeError("Processor did not return spatial_shapes; ensure the checkpoint exposes them.")
        positional = embeddings.position_embedding.weight
        side = embeddings.position_embedding_size
        positional_grid = positional.reshape(side, side, -1)
        resized = embeddings.resize_positional_embeddings(
            positional_grid,
            spatial_shapes.to(positional_grid.device),
            max_length=batch["pixel_values"].shape[1],
        )
        captures["layer0_position_embedding"] = to_host_tensor(resized.squeeze(0))

    outputs: Dict[str, str] = {
        "layer0_position_embedding": "layer0_position_embedding.bin",
        "layer0_self_attn_k_proj": "layer0_self_attn_k_proj.bin",
        "layer0_self_attn_v_proj": "layer0_self_attn_v_proj.bin",
        "layer0_self_attn_out_proj": "layer0_self_attn_out_proj.bin",
        "layer0_mlp": "layer0_mlp.bin",
    }

    pixel_mask = batch.get("pixel_attention_mask")
    spatial_shapes = batch.get("spatial_shapes")
    if spatial_shapes is None:
        raise RuntimeError("Processor did not return spatial_shapes; ensure the checkpoint exposes them.")

    with torch.no_grad():
        final_embeddings = vm(
            pixel_values=batch["pixel_values"],
            attention_mask=batch.get("pixel_attention_mask"),  # <-- name change here
            spatial_shapes=batch["spatial_shapes"],
        )
    captures["final_hidden_states"] = to_host_tensor(
        squeeze_batch(final_embeddings.last_hidden_state)
    )
    outputs["final_hidden_states"] = "final_hidden_states.bin"


    for key, filename in outputs.items():
        if key not in captures:
            raise RuntimeError(f"Failed to capture tensor '{key}'")
        host = squeeze_batch(captures[key])
        write_s2vis_tensor(args.output_dir / filename, host.numpy())
        print(f"[dump] {key} -> {args.output_dir / filename} shape={tuple(host.shape)}")


if __name__ == "__main__":
    main()
