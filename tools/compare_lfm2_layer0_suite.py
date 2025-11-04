#!/usr/bin/env python3
"""Compare layer-0 S2VISPEM dumps between Hugging Face (Python) and C++ outputs.

Given the per-tensor dump directories produced by ``dump_siglip2_layer0.py`` and
``siglip2_debug_runner --dump-layer0-suite``, this script loads every expected
binary, aligns tensor shapes (handling the C++ truncation of padded positions),
computes summary statistics of the element-wise differences, and prints a
compact report.

Example:

    python tools/compare_lfm2_layer0_suite.py \
        --hf-dir dumps/hf_layer0 \
        --cpp-dir dumps/cpp_layer0
"""

from __future__ import annotations

import argparse
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import numpy as np

MAGIC = b"S2VISPEM"
EXPECTED_FILES = {
    # "layer0_position_embedding.bin",
    "layer0_self_attn_k_proj.bin",
    "layer0_self_attn_v_proj.bin",
    "layer0_self_attn_out_proj.bin",
    "layer0_mlp.bin",
    "final_hidden_states.bin",
}


@dataclass
class TensorInfo:
    path: Path
    data: np.ndarray


@dataclass
class Stats:
    min: float
    q1: float
    median: float
    mean: float
    q3: float
    max: float


@dataclass
class DiffStats:
    name: str
    hf_shape: tuple[int, ...]
    cpp_shape: tuple[int, ...]
    compared: int
    signed: Stats
    absolute: Stats
    relative: Stats


def load_s2vis_tensor(path: Path) -> TensorInfo:
    with path.open("rb") as f:
        magic = f.read(8)
        if magic != MAGIC:
            raise ValueError(f"Unexpected magic {magic!r} in {path}")
        version = struct.unpack("<I", f.read(4))[0]
        if version != 1:
            raise ValueError(f"Unsupported S2VISPEM version {version} in {path}")
        rank = struct.unpack("<I", f.read(4))[0]
        shape = struct.unpack("<" + "I" * rank, f.read(4 * rank))
        data = np.frombuffer(f.read(), dtype="<f4")
    expected_size = int(np.prod(shape))
    if data.size != expected_size:
        raise ValueError(
            f"Tensor size mismatch for {path}: expected {expected_size} values, got {data.size}"
        )
    return TensorInfo(path=path, data=data.reshape(shape))


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q))


def summarize(values: np.ndarray) -> Stats:
    return Stats(
        min=float(values.min()),
        q1=percentile(values, 25.0),
        median=percentile(values, 50.0),
        mean=float(values.mean()),
        q3=percentile(values, 75.0),
        max=float(values.max()),
    )


def summarize_differences(name: str, hf: np.ndarray, cpp: np.ndarray) -> DiffStats:
    hf_flat = hf.reshape(-1)
    cpp_flat = cpp.reshape(-1)

    if hf_flat.size < cpp_flat.size:
        raise ValueError(
            f"HF tensor for {name} has only {hf_flat.size} elements, fewer than C++ {cpp_flat.size}"
        )

    if hf_flat.size != cpp_flat.size:
        print(
            f"[info] Trimming HF tensor '{name}' from {hf_flat.size} to {cpp_flat.size} elements to match C++",
            flush=True,
        )
    hf_aligned = hf_flat[: cpp_flat.size]
    diff = hf_aligned - cpp_flat
    abs_diff = np.abs(diff)
    rel_diff = abs_diff / (np.abs(hf_aligned) + 1e-6)

    return DiffStats(
        name=name,
        hf_shape=tuple(hf.shape),
        cpp_shape=tuple(cpp.shape),
        compared=cpp_flat.size,
        signed=summarize(diff),
        absolute=summarize(abs_diff),
        relative=summarize(rel_diff),
    )


def compare_directories(hf_dir: Path, cpp_dir: Path) -> Iterable[DiffStats]:
    hf_files = {p.name: p for p in hf_dir.iterdir() if p.is_file() and p.suffix == ".bin"}
    cpp_files = {p.name: p for p in cpp_dir.iterdir() if p.is_file() and p.suffix == ".bin"}

    missing_hf = EXPECTED_FILES - hf_files.keys()
    missing_cpp = EXPECTED_FILES - cpp_files.keys()
    if missing_hf:
        raise FileNotFoundError(f"Missing HF dumps: {sorted(missing_hf)}")
    if missing_cpp:
        raise FileNotFoundError(f"Missing C++ dumps: {sorted(missing_cpp)}")

    stats: list[DiffStats] = []
    for name in sorted(EXPECTED_FILES):
        hf_tensor = load_s2vis_tensor(hf_files[name])
        cpp_tensor = load_s2vis_tensor(cpp_files[name])
        stats.append(summarize_differences(name, hf_tensor.data, cpp_tensor.data))
    return stats


def format_stats_row(label: str, stats: Stats) -> str:
    return (
        f"    {label:>8} |"
        f" {stats.min:12.6g} {stats.q1:12.6g} {stats.median:12.6g}"
        f" {stats.mean:12.6g} {stats.q3:12.6g} {stats.max:12.6g}"
    )


def print_report(results: Iterable[DiffStats]) -> None:
    header = (
        f"{'tensor':40} | {'HF shape':>18} | {'CPP shape':>18} | {'N':>8}\n"
        f"{'':40} | {'stat':>8} | {'min':>12} {'q1':>12} {'median':>12}"
        f" {'mean':>12} {'q3':>12} {'max':>12}"
    )
    separator = "-" * 120
    print(header)
    print(separator)
    for stat in results:
        print(
            f"{stat.name:40} | {str(stat.hf_shape):>18} | {str(stat.cpp_shape):>18} | {stat.compared:8d}"
        )
        print(format_stats_row("Δ", stat.signed))
        print(format_stats_row("|Δ|", stat.absolute))
        print(format_stats_row("rel", stat.relative))
        print(separator)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare HF vs C++ layer-0 dumps (LFM2-VL)")
    parser.add_argument("--hf-dir", type=Path, required=True, help="Directory with HF S2VISPEM dumps")
    parser.add_argument("--cpp-dir", type=Path, required=True, help="Directory with C++ S2VISPEM dumps")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hf_dir = args.hf_dir
    cpp_dir = args.cpp_dir

    if not hf_dir.is_dir():
        raise NotADirectoryError(f"HF directory not found: {hf_dir}")
    if not cpp_dir.is_dir():
        raise NotADirectoryError(f"C++ directory not found: {cpp_dir}")

    stats = list(compare_directories(hf_dir, cpp_dir))
    print_report(stats)


if __name__ == "__main__":
    main()
