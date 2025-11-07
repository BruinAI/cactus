#!/usr/bin/env python3
"""Utility to verify tokenizer embedding shapes between a Hugging Face checkpoint and local cactus weights."""
from __future__ import annotations

import argparse
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, cast

try:
    import torch  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover - tooling environment
    raise SystemExit("This tool requires PyTorch. Install with `pip install torch`.") from exc

try:
    from transformers import (  # type: ignore[import-not-found]
        AutoConfig,
        AutoModel,
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoTokenizer,
        Lfm2VlForConditionalGeneration,
    )
except ImportError as exc:  # pragma: no cover - tooling environment
    raise SystemExit(
        "This tool requires Hugging Face Transformers. Install with `pip install transformers`."
    ) from exc

# Same candidate list used in convert_hf.py
EMBED_CANDIDATES: Sequence[str] = (
    "model.language_model.embed_tokens.weight",
    "model.text_model.embed_tokens.weight",
    "model.embed_tokens.weight",
    "embed_tokens.weight",
    "embeddings.weight",
    "transformer.wte.weight",
)


@dataclass
class EmbeddingInfo:
    name: str
    shape: Sequence[int]
    dtype: torch.dtype
    num_params: int
    byte_count: int


def _pick_model_class(model_name: str, cfg: AutoConfig):
    model_name_lower = model_name.lower()
    model_type = getattr(cfg, "model_type", "") or ""
    if "lfm2-vl" in model_name_lower or model_type == "lfm2-vl":
        return Lfm2VlForConditionalGeneration
    if model_type in {"lfm2", "smol", "gemma", "qwen"}:
        return AutoModelForCausalLM
    # Fallbacks: try causal LM first, then generic AutoModel
    return AutoModelForCausalLM


def load_embedding(model_name: str, cache_dir: Optional[str] = None) -> EmbeddingInfo:
    cfg = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
    model_cls = _pick_model_class(model_name, cfg)

    try:
        model = model_cls.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
    except ValueError:
        # Some checkpoints (e.g. pure encoders) are not causal LMs
        model = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )

    state_dict = model.state_dict()
    for candidate in EMBED_CANDIDATES:
        if candidate in state_dict:
            tensor = state_dict[candidate]
            num_params = tensor.numel()
            byte_count = num_params * torch.finfo(tensor.dtype).bits // 8
            return EmbeddingInfo(
                name=candidate,
                shape=tuple(tensor.shape),
                dtype=tensor.dtype,
                num_params=num_params,
                byte_count=byte_count,
            )

    raise RuntimeError("Failed to find token embedding tensor in state_dict")


def inspect_cactus_file(path: Path) -> dict[str, int | str | list[int]]:
    with path.open("rb") as f:
        header = f.read(32)
        ndim = struct.unpack("<I", header[:4])[0]
        offset = 4
        dims = []
        for _ in range(ndim):
            dim = struct.unpack("<Q", header[offset:offset + 8])[0]
            dims.append(dim)
            offset += 8
        precision_val = struct.unpack("<I", header[offset:offset + 4])[0]
        offset += 4
        byte_size = struct.unpack("<Q", header[offset:offset + 8])[0]
        offset += 8

    precision_map = {0: "INT8", 1: "FP16", 2: "FP32"}
    precision = precision_map.get(precision_val, f"unknown({precision_val})")

    file_size = path.stat().st_size
    payload_bytes = file_size - offset

    return {
        "ndim": ndim,
        "shape": dims,
        "precision_code": precision_val,
        "precision": precision,
        "declared_bytes": byte_size,
        "payload_bytes": payload_bytes,
        "file_size": file_size,
        "header_bytes": offset,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check token embedding shape/size consistency")
    parser.add_argument("model_name", help="Hugging Face model identifier")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Optional cache directory for HF downloads",
    )
    parser.add_argument(
        "--cactus-weights",
        type=Path,
        default=None,
        help="Path to cactus token_embeddings.weights for byte-level comparison",
    )
    args = parser.parse_args()

    info = load_embedding(args.model_name, cache_dir=args.cache_dir)
    print("HF checkpoint token embedding:")
    print(f"  tensor name : {info.name}")
    print(f"  shape       : {info.shape}")
    print(f"  dtype       : {info.dtype}")
    print(f"  parameters  : {info.num_params:,}")
    print(f"  bytes (raw) : {info.byte_count:,}")

    if args.cactus_weights:
        cactus_report = inspect_cactus_file(args.cactus_weights)
        print("\nCactus weight file header:")
        for key, value in cactus_report.items():
            print(f"  {key:>14}: {value}")

        expected_bytes = int(info.num_params * torch.finfo(info.dtype).bits // 8)
        header_bytes = cast(int, cactus_report["header_bytes"])
        payload_bytes = cast(int, cactus_report["payload_bytes"])

        print("\nConsistency check:")
        print(f"  HF payload bytes : {expected_bytes:,}")
        print(f"  Cactus payload   : {payload_bytes:,}")
        print(f"  Header bytes     : {header_bytes}")
        mismatch = expected_bytes != payload_bytes
        print(f"  Byte mismatch    : {'YES' if mismatch else 'no'}")

        if mismatch:
            diff = payload_bytes - expected_bytes
            print(f"    (payload difference: {diff:+,} bytes)")


if __name__ == "__main__":
    main()
