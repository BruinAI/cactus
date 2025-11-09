#!/usr/bin/env python3
"""Log LFM2-VL vision tower activations for debugging against the C++ pipeline.

This utility loads a Hugging Face LFM2-VL checkpoint, parses the list of
vision-layer parameters from ``lfm2_vlm_state_dict.txt``, registers forward
hooks on the corresponding PyTorch modules, and runs the vision tower on a
single image. Instead of dumping tensors to disk, it prints concise statistics
(min/max/mean/std and the leading elements) so the values can be compared to the
instrumentation emitted by the C++ runtime.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor, Lfm2VlForConditionalGeneration

DEFAULT_CHECKPOINT = "LiquidAI/LFM2-VL-450M"
DEFAULT_IMAGE = Path("tests/monkey-nose-muzzle-wallpaper.png")
DEFAULT_STATE_DICT = Path("lfm2_vlm_state_dict.txt")
DEFAULT_PROMPT = "Describe the image."
VISION_PREFIX = "model.vision_tower.vision_model."


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


def digest(t: torch.Tensor) -> str:
    t32 = t.detach().to(torch.float32)
    mn = float(t32.min())
    mx = float(t32.max())
    mean = float(t32.mean())
    std = float(t32.std(unbiased=False)) if t32.numel() > 1 else 0.0
    chk = float(t32.flatten()[: min(1_000_000, t32.numel())].sum())
    return f"min={mn:.6g} max={mx:.6g} mean={mean:.6g} std={std:.6g} sum1e6={chk:.6g}"


def flatten_head_values(t: torch.Tensor, count: int) -> str:
    flat = t.detach().to(torch.float32).reshape(-1)
    count = min(count, flat.numel())
    return ", ".join(f"{float(flat[i]):.6g}" for i in range(count))


def final_token_head_values(t: torch.Tensor, count: int) -> str:
    t32 = t.detach().to(torch.float32)

    if t32.ndim == 0:
        flat = t32.reshape(-1)
    elif t32.ndim == 1:
        flat = t32
    else:
        batch_dim = 0
        batch_count = t32.shape[batch_dim]
        sample = t32.select(batch_dim, 0) if batch_count > 0 else t32.reshape(-1)

        if isinstance(sample, torch.Tensor):
            if sample.ndim == 0:
                flat = sample.reshape(-1)
            elif sample.ndim == 1:
                flat = sample
            else:
                token_dim = 0
                token_count = sample.shape[token_dim]
                if token_count > 0:
                    token_slice = sample.select(token_dim, token_count - 1)
                    flat = token_slice.reshape(-1)
                else:
                    flat = sample.reshape(-1)
        else:
            flat = torch.as_tensor(sample, dtype=torch.float32).reshape(-1)

    count = min(count, flat.numel())
    return ", ".join(f"{float(flat[i]):.6g}" for i in range(count))


def build_dump_file_path(base_dir: Path, dotted_path: str, call_idx: Optional[int]) -> Path:
    parts = dotted_path.split(".") if dotted_path else ["tensor"]
    if len(parts) > 1:
        dir_path = base_dir.joinpath(*parts[:-1])
    else:
        dir_path = base_dir
    dir_path.mkdir(parents=True, exist_ok=True)
    leaf = parts[-1]
    if call_idx is None:
        filename = f"{leaf}.pt"
    else:
        filename = f"{leaf}__{call_idx:03d}.pt"
    return dir_path / filename


def dump_tensor(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, path)


def extract_state_dict_paths(state_dict_path: Path, prefix: Optional[str] = VISION_PREFIX) -> Tuple[List[str], List[str]]:
    text = state_dict_path.read_text().strip()
    if not text:
        raise ValueError(f"State dict file is empty: {state_dict_path}")

    if text.startswith("odict_keys"):
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end < start:
            raise ValueError(f"Failed to locate bracketed key list in {state_dict_path}")
        payload = text[start : end + 1]
    else:
        payload = text

    try:
        raw_list = ast.literal_eval(payload)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Failed to parse state dict keys from {state_dict_path}: {exc}") from exc

    module_paths = set()
    parameter_paths = set()
    for key in raw_list:
        if not isinstance(key, str):
            continue
        if prefix:
            if not key.startswith(prefix):
                continue
            suffix = key[len(prefix) :]
        else:
            suffix = key
        if suffix.endswith((".weight", ".bias")):
            module_paths.add(suffix.rsplit(".", 1)[0])
        else:
            parameter_paths.add(suffix)
    return sorted(module_paths), sorted(parameter_paths)


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
                raise ValueError(
                    f"Cannot index into non-sequence module at '{part}' in '{dotted}'"
                )
        else:
            if not hasattr(current, part):
                raise ValueError(
                    f"Module '{current.__class__.__name__}' has no attribute '{part}' (path '{dotted}')"
                )
            attr = getattr(current, part)
            if not isinstance(attr, nn.Module):
                raise ValueError(
                    f"Resolved attribute '{part}' in path '{dotted}' is not a module (got {type(attr).__name__})"
                )
            current = attr
    return current


def resolve_parameter(root: nn.Module, dotted: str) -> Optional[nn.Parameter]:
    params = dict(root.named_parameters())
    return params.get(dotted)


def filter_paths(paths: Sequence[str], filters: Sequence[str]) -> List[str]:
    if not filters:
        return list(paths)
    lowered = [f.lower() for f in filters]
    result = []
    for path in paths:
        path_l = path.lower()
        if any(f in path_l for f in lowered):
            result.append(path)
    return result


def register_hooks(
    root: nn.Module,
    module_paths: Iterable[str],
    head: int,
    activation_dump_dir: Optional[Path] = None,
) -> Tuple[List[torch.utils.hooks.RemovableHandle], Dict[str, int]]:
    call_counts: Dict[str, int] = Counter()
    handles: List[torch.utils.hooks.RemovableHandle] = []

    def make_hook(path: str):
        def hook_fn(_module, _inputs, output):  # pragma: no cover - runtime hook
            tensors: List[Tuple[str, torch.Tensor]] = []
            if isinstance(output, torch.Tensor):
                tensors.append((path, output))
            elif isinstance(output, (tuple, list)):
                for out_idx, item in enumerate(output):
                    if isinstance(item, torch.Tensor):
                        tensors.append((f"{path}.out{out_idx}", item))
                if not tensors:
                    print(f"[warn] Hook '{path}' saw non-tensor sequence output; skipping")
                    return
            else:
                print(f"[warn] Hook '{path}' saw non-tensor output; skipping")
                return

            idx = call_counts[path]
            call_counts[path] += 1

            for identifier, tensor in tensors:
                tensor_cpu = tensor.detach().cpu()
                host = tensor_cpu.to(torch.float32)
                tag = f"{identifier}#{idx}"
                print(f"[{tag}] shape={tuple(host.shape)} {digest(host)}")
                if head > 0 and host.numel() > 0:
                    print(f"  final_token: {final_token_head_values(host, head)}")
                if activation_dump_dir is not None:
                    dump_path = build_dump_file_path(activation_dump_dir, identifier, idx)
                    dump_tensor(tensor_cpu, dump_path)

        return hook_fn

    for module_path in module_paths:
        try:
            module = resolve_module(root, module_path)
        except ValueError as exc:
            print(f"[warn] Unable to resolve module '{module_path}': {exc}")
            continue
        handles.append(module.register_forward_hook(make_hook(module_path)))
    return handles, call_counts


def log_parameters(
    root: nn.Module,
    parameter_paths: Iterable[str],
    head: int,
    parameter_dump_dir: Optional[Path] = None,
) -> None:
    for param_path in parameter_paths:
        param = resolve_parameter(root, param_path)
        if param is None:
            print(f"[warn] Parameter '{param_path}' not found under vision_model")
            continue
        param_cpu = param.detach().cpu()
        host = param_cpu.to(torch.float32)
        print(f"[param:{param_path}] shape={tuple(host.shape)} {digest(host)}")
        if head > 0 and host.numel() > 0:
            print(f"  head: {flatten_head_values(host, head)}")
        if parameter_dump_dir is not None:
            dump_path = build_dump_file_path(parameter_dump_dir, param_path, None)
            dump_tensor(param_cpu, dump_path)


def load_image(image_path: Path) -> Image.Image:
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    return Image.open(image_path).convert("RGB")


def build_inputs(processor: AutoProcessor, image_path: Path, prompt: str) -> Dict[str, torch.Tensor]:
    image = load_image(image_path)
    conversation = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }]

    batch = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        tokenize=True,
    )
    return batch


def summarize_inputs(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
    head: int,
    input_dump_dir: Optional[Path] = None,
) -> Dict[str, torch.Tensor]:
    moved: Dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            tensor = value.to(device)
            if torch.is_floating_point(tensor):
                tensor = tensor.to(dtype)
            moved[key] = tensor
            summary = digest(tensor)
            print(f"[input.{key}] shape={tuple(tensor.shape)} {summary}")
            if head > 0 and tensor.numel() > 0:
                preview = final_token_head_values(tensor, head)
                print(f"  final_token: {preview}")
            if input_dump_dir is not None:
                dump_path = build_dump_file_path(input_dump_dir, key, None)
                dump_tensor(tensor.detach().cpu(), dump_path)
        else:
            moved[key] = value
    return moved


def main() -> None:
    parser = argparse.ArgumentParser(description="Log LFM2-VL vision tower activations")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="Hugging Face checkpoint to load (default: %(default)s)")
    parser.add_argument("--state-dict", type=Path, default=DEFAULT_STATE_DICT,
                        help="Path to lfm2_vlm state dict key list (default: %(default)s)")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE,
                        help="Image file to process (default: %(default)s)")
    parser.add_argument("--device", default=None,
                        help="Torch device to use (default: cuda if available else cpu)")
    parser.add_argument("--dtype", default="auto",
                        help="Computation dtype (auto, bf16, fp16, fp32)")
    parser.add_argument("--head", type=int, default=32,
                        help="Number of leading tensor elements to print (default: %(default)s)")
    parser.add_argument("--filter", action="append", default=[],
                        help="Substring filter applied to module paths (can be repeated)")
    parser.add_argument("--list-modules", action="store_true",
                        help="List discovered module paths and exit")
    parser.add_argument("--log-parameters", action="store_true",
                        help="Also log static parameter tensors")
    parser.add_argument("--dump-dir", type=Path, default=None,
                        help="Directory to dump full tensors for inputs, activations, and parameters")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT,
                        help="User prompt text to pair with the image (default: %(default)s)")

    args = parser.parse_args()

    all_module_paths, all_parameter_paths = extract_state_dict_paths(args.state_dict, prefix=None)
    # Deduplicate while preserving order
    seen_modules = set()
    module_paths_unique: List[str] = []
    for path in all_module_paths:
        if path not in seen_modules:
            seen_modules.add(path)
            module_paths_unique.append(path)

    seen_params = set()
    parameter_paths_unique: List[str] = []
    for path in all_parameter_paths:
        if path not in seen_params:
            seen_params.add(path)
            parameter_paths_unique.append(path)

    if args.list_modules:
        print("Discovered modules:")
        for path in module_paths_unique:
            print(f"  {path}")
        return

    module_paths = filter_paths(module_paths_unique, args.filter)
    parameter_paths = filter_paths(parameter_paths_unique, args.filter)

    dtype = parse_dtype(args.dtype)
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = Lfm2VlForConditionalGeneration.from_pretrained(
        args.checkpoint, torch_dtype=dtype, trust_remote_code=True
    )
    model.to(device)
    model.eval()

    raw_inputs = build_inputs(processor, args.image, args.prompt)

    dump_dir = args.dump_dir
    if dump_dir is not None:
        dump_dir.mkdir(parents=True, exist_ok=True)
        activation_dump_dir = dump_dir / "activations"
        parameter_dump_dir = dump_dir / "parameters"
        input_dump_dir = dump_dir / "inputs"
    else:
        activation_dump_dir = None
        parameter_dump_dir = None
        input_dump_dir = None

    batch = summarize_inputs(raw_inputs, device, dtype, args.head, input_dump_dir)

    handles, call_counts = register_hooks(model, module_paths, args.head, activation_dump_dir)

    try:
        with torch.no_grad():
            model(**batch)
    finally:
        for handle in handles:
            handle.remove()

    missing = [path for path in module_paths if call_counts[path] == 0]
    if missing:
        print("\n[warn] Modules without activations:")
        for path in missing:
            print(f"  {path}")

    if args.log_parameters:
        print("\n[parameters]")
    log_parameters(model, parameter_paths, args.head, parameter_dump_dir)


if __name__ == "__main__":
    main()
