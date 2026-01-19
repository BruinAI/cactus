#!/usr/bin/env python3
"""
Dump HuggingFace Moonshine model intermediate layer outputs for comparison with Cactus.
Writes in a Cactus-like debug_dump.log style.

Usage:
    pip install transformers torch soundfile numpy
    python3 tools/dump_hf_moonshine.py /path/to/audio.wav -o hf_moonshine_dump.log
"""

import argparse
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import soundfile as sf
from transformers import MoonshineForConditionalGeneration, AutoProcessor


# ----------------------------
# Formatting helpers
# ----------------------------

def _to_numpy_f32(t: torch.Tensor) -> np.ndarray:
    # stats/preview in float32 for consistent debugging
    return t.detach().to(device="cpu").float().numpy()

def format_stats(arr: np.ndarray, max_preview: int = 1000) -> Dict[str, Any]:
    flat = arr.reshape(-1)
    n_total = int(flat.shape[0])
    n = min(n_total, max_preview)
    sample = flat[:n] if n_total > 0 else flat
    if n_total == 0:
        return {
            "min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0,
            "n_sampled": 0, "n_total": 0, "truncated": False
        }
    return {
        "min": float(np.min(sample)),
        "max": float(np.max(sample)),
        "mean": float(np.mean(sample)),
        "std": float(np.std(sample)),
        "n_sampled": int(n),
        "n_total": int(n_total),
        "truncated": bool(n < n_total),
    }

def format_preview(arr: np.ndarray, num_preview: int = 16) -> str:
    flat = arr.reshape(-1)
    n_total = int(flat.shape[0])
    n = min(n_total, num_preview)
    vals = flat[:n]
    preview = ", ".join([f"{float(v):.6f}" for v in vals])
    if n_total > num_preview:
        preview += ", ..."
    return f"[{preview}]"

def write_tensor_record(
    f,
    node_id: int,
    call_idx: int,
    module_name: str,
    module_type: str,
    tensor_name: str,
    arr: np.ndarray,
    orig_dtype: str,
    stats_max: int,
    preview_n: int,
):
    f.write("-" * 60 + "\n")
    f.write(f"Node {node_id} (call {call_idx}) - {module_name} :: {module_type} :: {tensor_name}\n")
    shape_str = "[" + ",".join(str(s) for s in arr.shape) + "]"
    f.write(f"  Shape: {shape_str}  Precision: {orig_dtype}\n")

    stats = format_stats(arr, max_preview=stats_max)
    f.write(
        f"  Stats: min={stats['min']:.6f} max={stats['max']:.6f} "
        f"mean={stats['mean']:.6f} std={stats['std']:.6f}\n"
    )
    if stats["truncated"]:
        f.write(f"  Note: stats computed on first {stats['n_sampled']} of {stats['n_total']} values\n")
    f.write(f"  Preview: {format_preview(arr, num_preview=preview_n)}\n")


# ----------------------------
# Activation Dumper
# ----------------------------

@dataclass
class CapturedTensor:
    node_id: int
    call_idx: int
    module_name: str
    module_type: str
    tensor_name: str
    arr: np.ndarray
    orig_dtype: str

class ActivationDumper:
    """
    Captures outputs of every named module (and optionally inputs) in execution order.
    Supports Tensor / tuple / list / dict / HF ModelOutput.
    """

    def __init__(
        self,
        capture_inputs: bool = False,
        max_stats_elems: int = 1000,
        preview_elems: int = 16,
        dump_dir: Optional[str] = None,
        skip_module_name_prefixes: Optional[List[str]] = None,
    ):
        self.capture_inputs = capture_inputs
        self.max_stats_elems = max_stats_elems
        self.preview_elems = preview_elems
        self.dump_dir = dump_dir
        self.skip_prefixes = tuple(skip_module_name_prefixes or [])

        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._node_id: int = 0
        self._call_idx: int = 0
        self.records: List[CapturedTensor] = []

    def _next_node(self) -> int:
        nid = self._node_id
        self._node_id += 1
        return nid

    def _should_skip(self, module_name: str) -> bool:
        if module_name == "":
            return True  # root
        return any(module_name.startswith(p) for p in self.skip_prefixes)

    def _capture_tensor(self, module_name: str, module_type: str, tensor_name: str, t: torch.Tensor):
        # NOTE: keep original dtype string, but compute stats on float32 CPU
        orig_dtype = str(t.dtype).replace("torch.", "").upper()
        arr = _to_numpy_f32(t)
        self.records.append(
            CapturedTensor(
                node_id=self._next_node(),
                call_idx=self._call_idx,
                module_name=module_name,
                module_type=module_type,
                tensor_name=tensor_name,
                arr=arr,
                orig_dtype=orig_dtype,
            )
        )

        if self.dump_dir:
            # Save raw bytes (float32)
            import os
            fname = os.path.join(self.dump_dir, f"{module_name}.bin")
            # Ensure directory exists? dump_dir created in main. 
            # Subdirectories for modules? No, flatten names with dots.
            with open(fname, "wb") as f:
                f.write(arr.tobytes())

    def _flatten_outputs(self, output: Any) -> List[Tuple[str, torch.Tensor]]:
        """
        Returns list of (name, tensor) extracted from output.
        """
        out: List[Tuple[str, torch.Tensor]] = []

        if isinstance(output, torch.Tensor):
            out.append(("output", output))
            return out

        if isinstance(output, (tuple, list)):
            for i, v in enumerate(output):
                if isinstance(v, torch.Tensor):
                    out.append((f"output[{i}]", v))
                elif hasattr(v, "last_hidden_state") and isinstance(v.last_hidden_state, torch.Tensor):
                    out.append((f"output[{i}].last_hidden_state", v.last_hidden_state))
            return out

        # HF ModelOutput acts like dict + attributes
        if isinstance(output, dict) or hasattr(output, "to_tuple"):
            # Try dict-like iteration first
            if isinstance(output, dict):
                items = list(output.items())
            else:
                # Best-effort: ModelOutput is mapping-like
                try:
                    items = list(output.items())
                except Exception:
                    items = []

            for k, v in items:
                if isinstance(v, torch.Tensor):
                    out.append((f"output.{k}", v))
                elif isinstance(v, (tuple, list)):
                    for i, vv in enumerate(v):
                        if isinstance(vv, torch.Tensor):
                            out.append((f"output.{k}[{i}]", vv))

            # If nothing found, try common attribute
            if not out and hasattr(output, "last_hidden_state") and isinstance(output.last_hidden_state, torch.Tensor):
                out.append(("output.last_hidden_state", output.last_hidden_state))

            return out

        # Unknown output type
        return out

    def _flatten_inputs(self, inputs: Tuple[Any, ...], kwargs: Dict[str, Any]) -> List[Tuple[str, torch.Tensor]]:
        """
        Capture first few tensors from inputs/kwargs.
        """
        found: List[Tuple[str, torch.Tensor]] = []

        for i, v in enumerate(inputs):
            if isinstance(v, torch.Tensor):
                found.append((f"input[{i}]", v))

        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                found.append((f"input.{k}", v))

        return found

    def _make_hook(self, module_name: str, module: torch.nn.Module):
        module_type = module.__class__.__name__

        def hook(mod, inp, out):
            # Each module call increments call_idx; everything recorded during that call shares same call_idx
            self._call_idx += 1

            if self.capture_inputs:
                # forward hooks don't provide kwargs; PyTorch only gives *inputs
                in_tensors = self._flatten_inputs(inp, {})
                for tn, t in in_tensors:
                    self._capture_tensor(module_name, module_type, tn, t)

            out_tensors = self._flatten_outputs(out)
            for tn, t in out_tensors:
                self._capture_tensor(module_name, module_type, tn, t)

        return hook

    def register_all_named_modules(self, model: torch.nn.Module):
        for name, module in model.named_modules():
            if self._should_skip(name):
                continue
            # Register hooks on everything (including containers). Containers will fire too,
            # which is often useful; if you want leaf-only, add a condition here.
            h = module.register_forward_hook(self._make_hook(name, module))
            self._handles.append(h)

    def remove(self):
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()

    def write_log(self, output_path: str, header: str):
        with open(output_path, "w") as f:
            f.write(header)
            f.write("\n")
            f.write("=" * 60 + "\n\n")

            for rec in self.records:
                write_tensor_record(
                    f=f,
                    node_id=rec.node_id,
                    call_idx=rec.call_idx,
                    module_name=rec.module_name,
                    module_type=rec.module_type,
                    tensor_name=rec.tensor_name,
                    arr=rec.arr,
                    orig_dtype=rec.orig_dtype,
                    stats_max=self.max_stats_elems,
                    preview_n=self.preview_elems,
                )


# ----------------------------
# Main dumping logic
# ----------------------------

def pick_decoder_start_id(model, processor) -> int:
    for attr in ["decoder_start_token_id", "bos_token_id"]:
        v = getattr(model.config, attr, None)
        if isinstance(v, int) and v >= 0:
            return v
    tok = getattr(processor, "tokenizer", None)
    if tok is not None:
        for attr in ["bos_token_id", "cls_token_id"]:
            v = getattr(tok, attr, None)
            if isinstance(v, int) and v >= 0:
                return v
    return 0

def dump_moonshine_outputs(
    audio_path: str,
    output_path: str,
    model_name: str = "UsefulSensors/moonshine-tiny",
    capture_inputs: bool = False,
    max_stats_elems: int = 1000,
    preview_elems: int = 16,
    dump_dir: Optional[str] = None,
):
    print(f"Loading model: {model_name}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # Keep model in float32 for reproducible stats/debugging
    torch_dtype = torch.float32

    model = MoonshineForConditionalGeneration.from_pretrained(model_name, trust_remote_code=True)
    model = model.to(device=device, dtype=torch_dtype)
    model.eval()

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    audio, sample_rate = sf.read(audio_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    print(f"Audio shape: {audio.shape}, sample_rate: {sample_rate}, duration: {len(audio)/sample_rate:.2f}s")

    inputs = processor(audio, return_tensors="pt", sampling_rate=sample_rate)
    inputs = inputs.to(device)

    # Some processors use input_values, others input_features
    input_shape = None
    if hasattr(inputs, "input_values"):
        input_shape = tuple(inputs.input_values.shape)
    elif hasattr(inputs, "input_features"):
        input_shape = tuple(inputs.input_features.shape)
    print(f"Processed input shape: {input_shape}")

    if dump_dir:
        Path(dump_dir).mkdir(parents=True, exist_ok=True)
        print(f"Dumping binary tensors to: {dump_dir}")

    # 1) Capture a SINGLE forward pass with hooks ON (prevents generate() from recording 1000s of steps)
    dumper = ActivationDumper(
        capture_inputs=capture_inputs,
        max_stats_elems=max_stats_elems,
        preview_elems=preview_elems,
        dump_dir=dump_dir,
        # skip extremely spammy/less useful modules if desired:
        # skip_module_name_prefixes=["model.decoder.embed_positions"]  # example
        skip_module_name_prefixes=[],
    )
    dumper.register_all_named_modules(model)

    bos_id = pick_decoder_start_id(model, processor)
    decoder_input_ids = torch.tensor([[bos_id]], device=device, dtype=torch.long)

    print("Running single forward pass (hooks ON)...")
    with torch.no_grad():
        # Force one-step decoder forward so decoder layers/cross-attn run once.
        _ = model(
            **inputs,
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
        )

    dumper.remove()

    # 2) Run generate() with hooks OFF, just for transcription
    decoded_text = None
    try:
        print("Running generate() for transcription (hooks OFF)...")
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_length=128)
        decoded_text = processor.decode(generated_ids[0], skip_special_tokens=True)
        print(f"Transcription: {decoded_text}")
    except Exception as e:
        print(f"Generation failed: {e}")

    header = (
        "=== HuggingFace Moonshine Named-Module Intermediate Outputs ===\n"
        f"Model: {model_name}\n"
        f"Audio: {audio_path}\n"
        f"Device: {device}\n"
        f"Captured records: {len(dumper.records)}\n"
        f"Capture inputs: {capture_inputs}\n"
        + (f"Transcription: {decoded_text}\n" if decoded_text is not None else "")
    )

    print(f"Writing {len(dumper.records)} records to {output_path}")
    dumper.write_log(output_path, header)
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Dump Moonshine model intermediate outputs for every named module")
    parser.add_argument("audio_path", help="Path to input audio file (WAV)")
    parser.add_argument("-o", "--output", default="hf_moonshine_dump.log", help="Output log file")
    parser.add_argument("-m", "--model", default="UsefulSensors/moonshine-tiny", help="Model name")
    parser.add_argument("--capture-inputs", action="store_true", help="Also capture tensor inputs to each module")
    parser.add_argument("--max-stats-elems", type=int, default=1000, help="Max elements used for stats")
    parser.add_argument("--preview-elems", type=int, default=16, help="How many values to show in preview")
    parser.add_argument("--dump-dir", help="Directory to dump binary tensors")

    args = parser.parse_args()

    if not Path(args.audio_path).exists():
        print(f"Error: Audio file not found: {args.audio_path}")
        sys.exit(1)

    dump_moonshine_outputs(
        audio_path=args.audio_path,
        output_path=args.output,
        model_name=args.model,
        capture_inputs=args.capture_inputs,
        max_stats_elems=args.max_stats_elems,
        preview_elems=args.preview_elems,
        dump_dir=args.dump_dir,
    )


if __name__ == "__main__":
    main()
