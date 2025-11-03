#!/usr/bin/env python3
"""Compare C++ layer dumps against Hugging Face SigLIP-2 vision tower outputs.

This script loads an image, runs the selected SigLIP-2 checkpoint through the
vision tower, captures the linear projection (e.g. q_proj) for a specific
layer, loads the corresponding tensor dumped from the C++ debug runner, and
computes absolute/relative error statistics. A scatter plot of relative error
vs. Hugging Face values is also produced.
"""
from __future__ import annotations

import argparse
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

MAGIC = b"S2VISPEM"


def load_s2vis_tensor(path: Path) -> Tuple[np.ndarray, Dict[str, int]]:
    """Read a tensor written by the siglip2_debug_runner binary dump helper."""

    with path.open("rb") as f:
        magic = f.read(8)
        if magic != MAGIC:
            raise ValueError(f"Unexpected magic {magic!r} in {path}")

        version, = struct.unpack("<I", f.read(4))
        rank, = struct.unpack("<I", f.read(4))
        shape = struct.unpack("<" + "I" * rank, f.read(4 * rank))
        tensor = np.frombuffer(f.read(), dtype="<f4")

    expected_size = int(np.prod(shape))
    if tensor.size != expected_size:
        raise ValueError(
            f"Tensor size mismatch for {path}: expected {expected_size} values, got {tensor.size}"
        )
    return tensor.reshape(shape), {"version": version, "shape": shape}


@dataclass
class ComparisonResult:
    hf: np.ndarray
    cactus: np.ndarray
    abs_error: np.ndarray
    rel_error: np.ndarray


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q))


def summarize(values: np.ndarray) -> Dict[str, float]:
    return {
        "min": float(values.min()),
        "q1": percentile(values, 25.0),
        "mean": float(values.mean()),
        "q3": percentile(values, 75.0),
        "max": float(values.max()),
    }


def run_model_and_capture(
    checkpoint: str,
    image_path: Path,
    device: torch.device,
    layer_idx: int,
    proj_name: str,
) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")

    processor = AutoImageProcessor.from_pretrained(checkpoint)
    batch = processor(images=image, return_tensors="pt")

    # Move tensors to device
    batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

    model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True)
    model.to(device)
    model.eval()

    # Locate the desired projection module (q_proj, k_proj, etc.)
    try:
        attn_module = model.vision_model.encoder.layers[layer_idx].self_attn
    except AttributeError as exc:  # pragma: no cover - defensive
        raise RuntimeError("Unexpected SigLIP-2 model structure; vision tower not found") from exc

    if not hasattr(attn_module, proj_name):
        raise ValueError(f"Attention module has no attribute '{proj_name}'")

    target_linear = getattr(attn_module, proj_name)
    captured: Dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        captured["tensor"] = output.detach().to("cpu")

    handle = target_linear.register_forward_hook(hook)
    with torch.no_grad():
        model.vision_model(
            pixel_values=batch["pixel_values"],
            spatial_shapes=batch.get("spatial_shapes"),
            pixel_attention_mask=batch.get("pixel_attention_mask"),
        )
    handle.remove()

    if "tensor" not in captured:
        raise RuntimeError("Failed to capture tensor from forward hook")

    tensor = captured["tensor"].squeeze(0).to(torch.float32).numpy()
    return tensor


def compute_comparison(
    hf_tensor: np.ndarray,
    cactus_tensor: np.ndarray,
    eps: float,
) -> ComparisonResult:
    if hf_tensor.shape != cactus_tensor.shape:
        raise ValueError(
            f"Shape mismatch: HF tensor {hf_tensor.shape} vs cactus tensor {cactus_tensor.shape}"
        )

    abs_err = np.abs(hf_tensor - cactus_tensor)
    denom = np.maximum(np.abs(hf_tensor), eps)
    rel_err = abs_err / denom
    return ComparisonResult(hf_tensor, cactus_tensor, abs_err, rel_err)


def plot_relative_error(hf_values: np.ndarray, rel_error: np.ndarray, path: Path) -> None:
    plt.figure(figsize=(12, 5))
    plt.scatter(hf_values, rel_error, s=3, alpha=0.4)
    plt.xlabel("HF output value")
    plt.ylabel("Relative error (|Δ| / max(|HF|, eps))")
    plt.title("Relative error vs HF output")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def print_summary(label: str, values: np.ndarray) -> None:
    stats = summarize(values)
    print(f"{label} stats:")
    print(
        "  min={min:.6g}  q1={q1:.6g}  mean={mean:.6g}  q3={q3:.6g}  max={max:.6g}".format(
            **stats
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare SigLIP-2 vision layer outputs with C++ dumps")
    parser.add_argument("image", type=Path, help="Path to the input image")
    parser.add_argument("dump", type=Path, help="Path to the C++ dump (S2VISPEM format)")
    parser.add_argument("--checkpoint", default="google/siglip2-base-patch16-naflex",
                        help="Hugging Face checkpoint to load (default: %(default)s)")
    parser.add_argument("--layer", type=int, default=0, help="Vision encoder layer index (default: %(default)s)")
    parser.add_argument("--projection", default="q_proj",
                        help="Projection name inside self-attn (q_proj, k_proj, v_proj, etc.)")
    parser.add_argument("--device", default=None,
                        help="Torch device to use (default: cuda if available else cpu)")
    parser.add_argument("--eps", type=float, default=1e-8,
                        help="Epsilon to stabilize relative error (default: %(default)g)")
    parser.add_argument("--plot", type=Path, default=Path("relative_error_vs_hf.png"),
                        help="Path to save the relative error scatter plot")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    print("Loading C++ dump...")
    cactus_tensor, meta = load_s2vis_tensor(args.dump)
    print(f"  Dump version={meta['version']} shape={meta['shape']}")

    print("Running Hugging Face model and capturing layer output...")
    hf_tensor = run_model_and_capture(args.checkpoint, args.image, device, args.layer, args.projection)
    print(f"  Captured HF tensor shape={hf_tensor.shape}")

    print("Computing error metrics...")
    result = compute_comparison(hf_tensor, cactus_tensor, args.eps)

    flat_abs = result.abs_error.reshape(-1)
    flat_rel = result.rel_error.reshape(-1)
    flat_hf = result.hf.reshape(-1)

    print(f"Compared {flat_abs.size} elements")
    print_summary("Absolute error", flat_abs)
    print_summary("Relative error", flat_rel)

    print(f"Saving scatter plot to {args.plot}...")
    plot_relative_error(flat_hf, flat_rel, args.plot)

    print("Done.")


if __name__ == "__main__":
    main()
