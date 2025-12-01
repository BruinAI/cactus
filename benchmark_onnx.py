#!/usr/bin/env python3
import argparse
import time
import statistics

import cv2
import numpy as np
import onnxruntime as ort


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark YOLO ONNX model on a single image."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to YOLO ONNX model (e.g. yolov8n.onnx)",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image (e.g. test.jpg)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=50,
        help="Number of timed runs for each benchmark (default: 50)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup runs (default: 10)",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=None,
        help=(
            "Square input size (e.g. 640). If not set, "
            "will use size from model's first input (H,W)."
        ),
    )
    return parser.parse_args()


def print_stats(times_ms: list, label: str):
    """Print detailed statistics for a list of timing measurements."""
    if not times_ms:
        print(f"{label}: No measurements")
        return
    
    times_sorted = sorted(times_ms)
    n = len(times_sorted)
    
    mean = statistics.mean(times_ms)
    std = statistics.stdev(times_ms) if n > 1 else 0.0
    min_t = min(times_ms)
    max_t = max(times_ms)
    median = statistics.median(times_ms)
    p95 = times_sorted[int(n * 0.95)] if n >= 20 else max_t
    p99 = times_sorted[int(n * 0.99)] if n >= 100 else max_t
    
    print(f"\n{label}")
    print("-" * 40)
    print(f"  Runs:      {n}")
    print(f"  Mean:      {mean:.3f} ms")
    print(f"  Std Dev:   {std:.3f} ms")
    print(f"  Min:       {min_t:.3f} ms")
    print(f"  Max:       {max_t:.3f} ms")
    print(f"  Median:    {median:.3f} ms")
    print(f"  P95:       {p95:.3f} ms")
    print(f"  P99:       {p99:.3f} ms")
    print(f"  FPS:       {1000.0 / mean:.2f}")
    print("-" * 40)


def load_session(model_path: str) -> ort.InferenceSession:
    sess_options = ort.SessionOptions()
    # You can tweak these if you want:
    # sess_options.intra_op_num_threads = 4
    # sess_options.inter_op_num_threads = 1

    session = ort.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )
    return session


def get_input_shape(session: ort.InferenceSession, override_size: int = None):
    """
    Returns (N, C, H, W) for the first input.
    If override_size is given, H=W=override_size.
    """
    input_meta = session.get_inputs()[0]
    shape = list(input_meta.shape)  # [N, C, H, W] or [1,3,640,640], etc.

    # Replace dynamic dims with 1 or override_size if needed
    # shape entries can be int or str (for dynamic)
    def _dim_to_int(d, default):
        if isinstance(d, int):
            return d
        return default

    N = _dim_to_int(shape[0], 1)
    C = _dim_to_int(shape[1], 3)

    if override_size is not None:
        H = W = override_size
    else:
        H = _dim_to_int(shape[2], 640)
        W = _dim_to_int(shape[3], 640)

    return N, C, H, W


def preprocess_image(img_bgr: np.ndarray, H: int, W: int) -> np.ndarray:
    """
    Minimal YOLO-style preprocessing:
    - resize to (W, H)
    - BGR -> RGB
    - [0,255] -> [0,1] float32
    - HWC -> CHW
    - add batch dimension
    """
    # Resize
    img = cv2.resize(img_bgr, (W, H))

    # BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize to [0,1]
    img = img.astype(np.float32) / 255.0

    # HWC -> CHW
    img = np.transpose(img, (2, 0, 1))

    # Add batch dimension
    img = np.expand_dims(img, axis=0)  # [1, 3, H, W]

    return img


def benchmark_inference_only(session, input_name, sample_input, warmup, runs):
    """
    Benchmark inference only, measuring each run individually.
    Returns list of times in milliseconds.
    """
    # Warmup - important for JIT compilation in ONNX Runtime
    print(f"  Warming up ({warmup} runs)...", end="", flush=True)
    for _ in range(warmup):
        _ = session.run(None, {input_name: sample_input})
    print(" done")

    # Timed runs - measure EACH run individually for proper statistics
    print(f"  Benchmarking ({runs} runs)...")
    times_ms = []
    for i in range(runs):
        start = time.perf_counter()
        outputs = session.run(None, {input_name: sample_input})
        end = time.perf_counter()
        times_ms.append((end - start) * 1000.0)
        
        # Print first few output values for each run
        out = outputs[0].flatten()
        print(f"    Run {i+1}: {times_ms[-1]:.2f} ms, output[0:5] = {out[:5]}")
    print("  done")

    return times_ms


def benchmark_with_preprocess(session, input_name, img_bgr, H, W, warmup, runs):
    """
    Benchmark preprocessing + inference, measuring each run individually.
    Returns list of times in milliseconds.
    """
    # Warmup
    print(f"  Warming up ({warmup} runs)...", end="", flush=True)
    for _ in range(warmup):
        inp = preprocess_image(img_bgr, H, W)
        _ = session.run(None, {input_name: inp})
    print(" done")

    # Timed runs - measure EACH run individually
    print(f"  Benchmarking ({runs} runs)...")
    times_ms = []
    for i in range(runs):
        start = time.perf_counter()
        inp = preprocess_image(img_bgr, H, W)
        outputs = session.run(None, {input_name: inp})
        end = time.perf_counter()
        times_ms.append((end - start) * 1000.0)
        
        # Print first few output values for each run
        out = outputs[0].flatten()
        print(f"    Run {i+1}: {times_ms[-1]:.2f} ms, output[0:5] = {out[:5]}")
    print("  done")

    return times_ms


def main():
    args = parse_args()

    print("=" * 50)
    print("       ONNX Runtime Benchmark")
    print("=" * 50)
    
    print(f"\nLoading model: {args.model}")
    session = load_session(args.model)
    input_name = session.get_inputs()[0].name
    
    # Print ONNX Runtime info
    print(f"ONNX Runtime version: {ort.__version__}")
    providers = session.get_providers()
    print(f"Execution providers: {providers}")

    print(f"\nUsing image: {args.image}")
    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        raise RuntimeError(f"Failed to load image: {args.image}")
    print(f"Original image size: {img_bgr.shape[1]}x{img_bgr.shape[0]}")

    N, C, H, W = get_input_shape(session, args.input_size)
    print(f"Model input shape: N={N}, C={C}, H={H}, W={W}")

    # Prepare a single preprocessed input for "inference-only" benchmark
    sample_input = preprocess_image(img_bgr, H, W)
    print(f"Preprocessed input shape: {sample_input.shape}, dtype: {sample_input.dtype}")

    print("\n" + "=" * 50)
    print("Benchmark 1: Inference only (no preprocessing in loop)")
    print("=" * 50)
    times_inf = benchmark_inference_only(
        session, input_name, sample_input, args.warmup, args.runs
    )
    print_stats(times_inf, "Inference Only Results")

    print("\n" + "=" * 50)
    print("Benchmark 2: Preprocessing + Inference")
    print("=" * 50)
    times_full = benchmark_with_preprocess(
        session, input_name, img_bgr, H, W, args.warmup, args.runs
    )
    print_stats(times_full, "Preprocessing + Inference Results")
    
    # Calculate preprocessing overhead
    avg_inf = statistics.mean(times_inf)
    avg_full = statistics.mean(times_full)
    preprocess_overhead = avg_full - avg_inf
    print(f"\nEstimated preprocessing overhead: {preprocess_overhead:.3f} ms")

    print("\n" + "=" * 50)
    print("Done.")
    print("=" * 50)


if __name__ == "__main__":
    main()
