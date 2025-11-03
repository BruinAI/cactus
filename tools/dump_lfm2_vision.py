#!/usr/bin/env python3
"""Selective activation and parameter dump utility for the LFM2-VL vision tower.

This script loads a Hugging Face LFM2-VL checkpoint, prepares an input image,
optionally registers forward (pre/post) hooks on specific modules inside the
vision tower, and saves or prints summaries of the captured tensors. It can also
export static parameters (weights, biases, etc.) without running a forward pass.

Example usage:

  # Dump layer 0 attention q projection output to ./out/q_proj.pt
  python tools/dump_lfm2_vision.py \
      --image tests/istockphoto-184978580-2048x2048.jpg \
      --target encoder.layers.0.self_attn.q_proj:out:as=layer0_q \
      --save-dir out

  # Inspect the input of the patch embedding conv and dump its weights
  python tools/dump_lfm2_vision.py \
      --target embeddings.patch_embedding:pre:as=patch_in \
      --parameter embeddings.patch_embedding.weight:as=patch_weight \
      --synthetic --save-dir out

Targets are specified relative to ``vision_tower.vision_model``.  Each
``--target`` token accepts optional modifiers separated by ``:``:

  * ``pre`` / ``in``   – capture the forward *input* (register_pre_hook)
  * ``out`` / ``post`` – capture the forward *output* (register_forward_hook)
  * ``both``           – capture both input and output
  * ``as=<label>``     – override the label used for logging / filenames

Likewise, ``--parameter`` entries accept an optional ``as=<label>`` suffix.

Use ``--list-modules`` or ``--list-parameters`` to enumerate available paths.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast

import torch
import torch.nn as nn

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - optional dependency
    Image = None  # type: ignore

from transformers import AutoImageProcessor, Lfm2VlForConditionalGeneration

DEFAULT_CKPT = "LiquidAI/LFM2-VL-450M"
DEFAULT_IMAGE = Path("tests/istockphoto-184978580-2048x2048.jpg")


@dataclass
class TargetSpec:
    path: str
    hook_kind: str  # pre | out | both
    label: str


@dataclass
class ParamSpec:
    path: str
    label: str


@dataclass
class Capture:
    label: str
    kind: str
    tensor: torch.Tensor


def parse_dtype(token: str) -> torch.dtype:
    token = token.lower()
    if token in {"auto"}:
        return torch.float32
    if token in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if token in {"fp16", "float16", "half"}:
        return torch.float16
    if token in {"fp32", "float32", "single"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype token: {token}")


def parse_target(token: str) -> TargetSpec:
    if not token:
        raise argparse.ArgumentTypeError("Empty target token")

    parts = token.split(":")
    path = parts[0]
    hook_kind = "out"
    label = path.replace(".", "_")

    for extra in parts[1:]:
        if extra in {"pre", "in"}:
            hook_kind = "pre"
        elif extra in {"out", "post"}:
            hook_kind = "out"
        elif extra == "both":
            hook_kind = "both"
        elif extra.startswith("as="):
            label = extra.split("=", 1)[1]
        else:
            raise argparse.ArgumentTypeError(
                f"Unknown target modifier '{extra}' in token '{token}'"
            )
    return TargetSpec(path=path, hook_kind=hook_kind, label=label)


def parse_parameter(token: str) -> ParamSpec:
    if not token:
        raise argparse.ArgumentTypeError("Empty parameter token")

    parts = token.split(":")
    path = parts[0]
    label = path.replace(".", "_")

    for extra in parts[1:]:
        if extra.startswith("as="):
            label = extra.split("=", 1)[1]
        else:
            raise argparse.ArgumentTypeError(
                f"Unknown parameter modifier '{extra}' in token '{token}'"
            )
    return ParamSpec(path=path, label=label)


def synth_image(size: int = 224):
    if Image is None:
        return None
    img = Image.new("RGB", (size, size), (40, 60, 200))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:  # pragma: no cover - very unlikely
        font = None
    draw.text((10, 10), "VisionTower Test", fill=(255, 255, 255), font=font)
    draw.rectangle([150, 40, 210, 100], outline=(255, 255, 0), width=3)
    draw.ellipse([40, 140, 110, 210], outline=(0, 255, 120), width=3)
    return img


def digest(t: torch.Tensor) -> str:
    t32 = t.detach().to(torch.float32)
    mn = float(t32.min())
    mx = float(t32.max())
    mean = float(t32.mean())
    std = float(t32.std(unbiased=False)) if t32.numel() > 1 else 0.0
    chk = float(t32.flatten()[: min(1_000_000, t32.numel())].sum())
    return f"min={mn:.6g} max={mx:.6g} mean={mean:.6g} std={std:.6g} sum1e6={chk:.6g}"


def head_values(t: torch.Tensor, count: int) -> str:
    flat = t.detach().to(torch.float32).reshape(-1)
    count = min(count, flat.numel())
    return ", ".join(f"{float(flat[i]):.6g}" for i in range(count))


def resolve_module(root: nn.Module, dotted: str) -> nn.Module:
    current: nn.Module = root
    if not dotted:
        raise ValueError("Empty module path")

    pieces = dotted.split(".")
    for part in pieces:
        if part.isdigit():
            idx = int(part)
            if isinstance(current, (nn.ModuleList, nn.Sequential, list, tuple)):
                current = current[idx]
            else:
                raise ValueError(f"Cannot index into non-sequence module at '{part}' in '{dotted}'")
        else:
            if not hasattr(current, part):
                raise ValueError(f"Module '{current.__class__.__name__}' has no attribute '{part}' (path '{dotted}')")
            attr = getattr(current, part)
            if not isinstance(attr, nn.Module):
                raise ValueError(
                    f"Resolved attribute '{part}' in path '{dotted}' is not a module (got {type(attr).__name__})"
                )
            current = attr
    return current


def resolve_parameter(root: nn.Module, dotted: str) -> torch.nn.Parameter:
    try:
        return dict(root.named_parameters())[dotted]
    except KeyError as exc:
        raise ValueError(f"Parameter '{dotted}' not found under vision_model") from exc

def register_hooks(
    vm: nn.Module,
    targets: List[TargetSpec],
    captures: List[Capture],
    print_head: Optional[int],
    quiet: bool,
) -> List[torch.utils.hooks.RemovableHandle]:
    handles: List[torch.utils.hooks.RemovableHandle] = []

    def make_hook(spec: TargetSpec, kind: str):
        tag = f"{spec.label}.{kind}"

        def hook_fn(_module, args, output=None):
            raw = args[0] if kind == "in" else output
            if isinstance(raw, (tuple, list)) and raw:
                raw = raw[0]
            if not isinstance(raw, torch.Tensor):
                typename = type(raw).__name__ if raw is not None else "None"
                print(f"[warn] Hook '{tag}' saw non-tensor ({typename}); skipping")
                return
            tensor_raw = cast(torch.Tensor, raw)
            t_cpu = tensor_raw.detach().to(torch.float32).cpu()
            captures.append(Capture(label=spec.label, kind=kind, tensor=t_cpu))
            if not quiet:
                print(f"[{tag}] shape={tuple(t_cpu.shape)} {digest(t_cpu)}")
                if print_head and t_cpu.numel() > 0:
                    print(f"  head: {head_values(t_cpu, print_head)}")

        return hook_fn

    for spec in targets:
        module = resolve_module(vm, spec.path)
        if spec.hook_kind in {"pre", "both"}:
            handles.append(module.register_forward_pre_hook(make_hook(spec, "in")))
        if spec.hook_kind in {"out", "both"}:
            handles.append(module.register_forward_hook(make_hook(spec, "out")))

    return handles


def save_capture(capture: Capture, save_dir: Path) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{capture.label}.{capture.kind}.pt"
    path = save_dir / fname
    torch.save(capture.tensor, path)
    return path


def run_forward(
    processor: AutoImageProcessor,
    model: Lfm2VlForConditionalGeneration,
    vm: nn.Module,
    dtype: torch.dtype,
    device: torch.device,
    image_path: Optional[Path],
    synthetic: bool,
    targets: List[TargetSpec],
    captures: List[Capture],
    print_head: Optional[int],
    quiet: bool,
):
    model.eval()
    vt = model.vision_tower

    if image_path is not None and image_path.exists():
        if Image is None:
            raise RuntimeError("Pillow is required to load images but is not installed")
        pil = Image.open(image_path).convert("RGB")
    else:
        pil = synth_image(processor.size.get("shortest_edge", 224)) if synthetic else None
        if pil is None:
            raise FileNotFoundError(
                f"Image path '{image_path}' not found and synthetic image disabled"
            )

    inputs = processor(images=pil, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    feeds = {
        "pixel_values": pixel_values.to(dtype=dtype, device=device),
    }
    if "pixel_attention_mask" in inputs:
        feeds["pixel_attention_mask"] = inputs["pixel_attention_mask"].to(device)
    if "spatial_shapes" in inputs:
        feeds["spatial_shapes"] = inputs["spatial_shapes"].to(device)

    if "pixel_attention_mask" not in feeds:
        B, C, H, W = feeds["pixel_values"].shape
        feeds["pixel_attention_mask"] = torch.ones((B, H * W), dtype=torch.long, device=device)
    if "spatial_shapes" not in feeds:
        _, _, H, W = feeds["pixel_values"].shape
        feeds["spatial_shapes"] = torch.tensor([[H, W]], device=device, dtype=torch.long)

    if not quiet:
        print(
            f"[input.pixel_values] shape={tuple(feeds['pixel_values'].shape)} {digest(feeds['pixel_values'])}"
        )

    handles = register_hooks(vm, targets, captures, print_head, quiet)

    with torch.no_grad():
        try:
            vt(
                feeds["pixel_values"],
                feeds["pixel_attention_mask"],
                feeds["spatial_shapes"],
            )
        except TypeError:
            vt(feeds["pixel_values"])

    for handle in handles:
        handle.remove()

    return vm


def list_only(vm: nn.Module, list_modules: bool, list_parameters: bool) -> None:
    if list_modules:
        print("Available modules (relative to vision_model):")
        for name, module in vm.named_modules():
            if name == "":
                continue
            print(f"  {name} ({module.__class__.__name__})")
    if list_parameters:
        print("Available parameters (relative to vision_model):")
        for name, param in vm.named_parameters():
            print(f"  {name} {tuple(param.shape)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Selective hook utility for LFM2-VL vision tower")
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT,
                        help="Hugging Face checkpoint to load (default: %(default)s)")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE,
                        help="Path to the input image (default: %(default)s)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use a procedurally generated synthetic image instead of disk input")
    parser.add_argument("--device", default=None,
                        help="Torch device to use (default: cuda if available else cpu)")
    parser.add_argument("--dtype", default="bfloat16",
                        help="Torch dtype to load the model (bf16, fp16, fp32, auto)")
    parser.add_argument("--target", action="append", default=[], type=parse_target,
                        help="Module path to capture (relative to vision_model); e.g. 'encoder.layers.0.self_attn.q_proj:out'" )
    parser.add_argument("--parameter", action="append", default=[], type=parse_parameter,
                        help="Parameter path to export (relative to vision_model); e.g. 'embeddings.patch_embedding.weight'" )
    parser.add_argument("--save-dir", type=Path, default=None,
                        help="Directory to store captured tensors as .pt files")
    parser.add_argument("--print-head", type=int, default=8,
                        help="Number of leading elements to print from captured tensors (default: %(default)s; 0 to disable)")
    parser.add_argument("--quiet", action="store_true",
                        help="Reduce logging to only essential messages")
    parser.add_argument("--list-modules", action="store_true",
                        help="List available module paths and exit")
    parser.add_argument("--list-parameters", action="store_true",
                        help="List available parameter paths and exit")

    args = parser.parse_args()

    dtype = parse_dtype(args.dtype)
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if args.print_head <= 0:
        print_head: Optional[int] = None
    else:
        print_head = args.print_head

    captures: List[Capture] = []

    processor = AutoImageProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)

    # Load model (may be reused for listings and forward runs).
    model = Lfm2VlForConditionalGeneration.from_pretrained(
        args.checkpoint, torch_dtype=dtype, trust_remote_code=True
    )
    if not hasattr(model, "vision_tower"):
        raise AttributeError("Model checkpoint does not expose a vision_tower module")
    vt = model.vision_tower
    if not hasattr(vt, "vision_model"):
        raise AttributeError("vision_tower module is missing a vision_model attribute")
    vm = vt.vision_model

    if args.list_modules or args.list_parameters:
        list_only(vm, args.list_modules, args.list_parameters)
        return

    if not args.target and not args.parameter:
        parser.error("No targets or parameters specified. Use --target/--parameter or listing options.")

    # Export parameters without running a forward pass.
    for spec in args.parameter:
        param = resolve_parameter(vm, spec.path).detach().to(torch.float32).cpu()
        captures.append(Capture(label=spec.label, kind="param", tensor=param))
        if not args.quiet:
            print(f"[param:{spec.label}] shape={tuple(param.shape)} {digest(param)}")
        if print_head and param.numel() > 0 and not args.quiet:
            print(f"  head: {head_values(param, print_head)}")

    # Only run forward if there are targets to hook.
    if args.target:
        # move model to device for forward run
        model.to(device)
        model.eval()
        run_forward(
            processor,
            model,
            vm,
            dtype,
            device,
            args.image if not args.synthetic else None,
            args.synthetic,
            args.target,
            captures,
            print_head,
            args.quiet,
        )

    if args.save_dir:
        save_dir = args.save_dir
        for capture in captures:
            path = save_capture(capture, save_dir)
            if not args.quiet:
                print(f"[save] {capture.label}.{capture.kind} -> {path}")

    if not args.quiet and not captures:
        print("No tensors captured (did you intend to use --target or --parameter?)")


if __name__ == "__main__":
    main()
