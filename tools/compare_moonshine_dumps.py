#!/usr/bin/env python3
import os
import glob
import argparse
import numpy as np
import struct

def read_binary_file(filepath, dtype=np.float32):
    params = np.fromfile(filepath, dtype=dtype)
    return params

def auto_detect_dtype(filepath, reference_size_bytes):
    """
    Try to guess dtype based on file size vs reference file size?
    No, we don't know reference size if we don't know dtype.
    
    If we assume the Python dump is always Float32 (which is typical for HF model unless .half() is called),
    we can use that to infer the count.
    
    Python dump: likely Float32.
    C++ dump: could be Float16 or Float32.
    """
    # For now, let's try reading as float32 first.
    return np.float32

def compare_files(cpp_file, py_file, tolerance=1e-3):
    # Python dump is likely float32
    py_data = read_binary_file(py_file, np.float32)
    
    # C++ dump might be float16 or float32. 
    # Let's check file sizes.
    cpp_size = os.path.getsize(cpp_file)
    py_size = os.path.getsize(py_file)
    
    cpp_dtype = np.float32
    if cpp_size == py_size // 2:
        cpp_dtype = np.float16
        print(f"  [Info] Detected FP16 in C++ dump based on size (C++: {cpp_size}, Py: {py_size})")
    elif cpp_size == py_size:
        pass # match
    else:
        # Size mismatch not 2x. Could be shape mismatch or different precision.
        # Just create a warning
        pass

    cpp_data = read_binary_file(cpp_file, cpp_dtype)
    
    # If C++ was FP16, convert to FP32 for comparison
    if cpp_dtype == np.float16:
        cpp_data = cpp_data.astype(np.float32)

    # Flatten both
    py_flat = py_data.flatten()
    cpp_flat = cpp_data.flatten()
    
    min_len = min(len(py_flat), len(cpp_flat))
    if len(py_flat) != len(cpp_flat):
        print(f"  [Warn] Size mismatch: Py={len(py_flat)}, C++={len(cpp_flat)}. Comparing first {min_len} elements.")
    
    py_flat = py_flat[:min_len]
    cpp_flat = cpp_flat[:min_len]
    
    diff = np.abs(py_flat - cpp_flat)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    mse = np.mean(diff ** 2)
    
    # Relative error (avoid division by zero)
    epsi = 1e-7
    rel_diff = diff / (np.abs(py_flat) + epsi)
    max_rel_diff = np.max(rel_diff)
    
    # Cosine Similarity
    dot = np.dot(py_flat, cpp_flat)
    norm_py = np.linalg.norm(py_flat)
    norm_cpp = np.linalg.norm(cpp_flat)
    cosine_sim = dot / (norm_py * norm_cpp + 1e-9)

    passed = mse < tolerance
    status = "PASS" if passed else "FAIL"
    
    # Color code
    GREEN = "\033[92m"
    RED = "\033[91m"
    RESET = "\033[0m"
    
    color = GREEN if passed else RED
    print(f"{color}[{status}]{RESET} MaxDiff: {max_diff:.6f}, MSE: {mse:.8f}, CosSim: {cosine_sim:.6f}, RelDiff: {max_rel_diff:.6f}")
    
    if not passed:
        # Print first few mismatches
        print("    Top 5 mismatches:")
        indices = np.argsort(diff)[::-1][:5]
        for idx in indices:
            print(f"      Use idx {idx}: Py={py_flat[idx]:.6f}, C++={cpp_flat[idx]:.6f}, Diff={diff[idx]:.6f}")

def main():
    parser = argparse.ArgumentParser(description="Compare C++ and Python binary dumps")
    parser.add_argument("--cpp-dir", required=True, help="Directory containing C++ binary dumps")
    parser.add_argument("--py-dir", required=True, help="Directory containing Python binary dumps")
    parser.add_argument("--tolerance", type=float, default=1e-3, help="MSE tolerance for pass/fail")
    parser.add_argument("--filter", type=str, default="", help="Filter filenames (substring)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.cpp_dir):
        print(f"Error: C++ directory not found: {args.cpp_dir}")
        return
    if not os.path.exists(args.py_dir):
        print(f"Error: Python directory not found: {args.py_dir}")
        return
        
    cpp_files = sorted(glob.glob(os.path.join(args.cpp_dir, "*.bin")))
    
    print(f"Found {len(cpp_files)} files in {args.cpp_dir}")
    
    for cpp_path in cpp_files:
        basename = os.path.basename(cpp_path)
        if args.filter and args.filter not in basename:
            continue
            
        py_path = os.path.join(args.py_dir, basename)
        
        # Helper: handle potential naming differences if any?
        # For now assume exact match. 
        
        print(f"\nComparing {basename}...")
        if not os.path.exists(py_path):
            print(f"  [Skip] No matching file in Python dir: {py_path}")
            continue
            
        try:
            compare_files(cpp_path, py_path, args.tolerance)
        except Exception as e:
            print(f"  [Error] Failed to compare: {e}")

if __name__ == "__main__":
    main()
