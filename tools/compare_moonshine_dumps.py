#!/usr/bin/env python3
import os
import glob
import argparse
import math # Added for erf
import numpy as np
import struct

def read_binary_file(filepath, dtype=np.float32):
    params = np.fromfile(filepath, dtype=dtype)
    return params



def compare_arrays(name, py_data, cpp_data, tolerance=1e-3, logger=print):
    # Flatten both
    py_flat = py_data.flatten()
    cpp_flat = cpp_data.flatten()
    
    min_len = min(len(py_flat), len(cpp_flat))
    if len(py_flat) != len(cpp_flat):
        logger(f"  [Warn] Size mismatch in {name}: Py={len(py_flat)}, C++={len(cpp_flat)}. Comparing first {min_len} elements.")
    
    py_flat = py_flat[:min_len]
    cpp_flat = cpp_flat[:min_len]
    
    diff = np.abs(py_flat - cpp_flat)
    max_diff = np.max(diff) if len(diff) > 0 else 0
    mean_diff = np.mean(diff) if len(diff) > 0 else 0
    mse = np.mean(diff ** 2) if len(diff) > 0 else 0
    
    # Relative error (avoid division by zero)
    epsi = 1e-7
    rel_diff = diff / (np.abs(py_flat) + epsi)
    max_rel_diff = np.max(rel_diff) if len(rel_diff) > 0 else 0
    
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
    logger(f"{color}[{status}]{RESET} MaxDiff: {max_diff:.6f}, MSE: {mse:.8f}, CosSim: {cosine_sim:.6f}, RelDiff: {max_rel_diff:.6f}")
    
    if not passed:
        # Print first few mismatches
        logger("    Top 5 mismatches:")
        indices = np.argsort(diff)[::-1][:5]
        for idx in indices:
            logger(f"      Use idx {idx}: Py={py_flat[idx]:.6f}, C++={cpp_flat[idx]:.6f}, Diff={diff[idx]:.6f}")

def compare_files(cpp_file, py_file, tolerance=1e-3, logger=print):
    # Python dump is FP16
    py_data = read_binary_file(py_file, np.float16).astype(np.float32)
    
    # C++ dump is always FP16
    cpp_data = read_binary_file(cpp_file, np.float16).astype(np.float32)

    # Special handling for attn_out padding (Py=40, C++=36)
    basename = os.path.basename(cpp_file)
    if "attn_out" in basename:
        n_py = len(py_data)
        n_cpp = len(cpp_data)
        
        # Check if Python has 40/36 ratio
        if n_py > n_cpp and n_py % 40 == 0 and n_cpp % 36 == 0:
            count_py = n_py // 40
            count_cpp = n_cpp // 36
            
            if count_py == count_cpp:
                logger(f"  [Info] Detected padded attn_out (Py=40, C++=36). Unpadding Python data...")
                py_reshaped = py_data.reshape(-1, 40)
                py_unpadded = py_reshaped[:, :36]
                py_data = py_unpadded.flatten()

    compare_arrays(basename, py_data, cpp_data, tolerance, logger)

def main():
    parser = argparse.ArgumentParser(description="Compare C++ and Python binary dumps")
    parser.add_argument("--cpp-dir", required=True, help="Directory containing C++ binary dumps")
    parser.add_argument("--py-dir", required=True, help="Directory containing Python binary dumps")
    parser.add_argument("--tolerance", type=float, default=1e-3, help="MSE tolerance for pass/fail")
    parser.add_argument("--filter", type=str, default="", help="Filter filenames (substring)")
    parser.add_argument("--output-file", help="File to write comparison results to")
    
    args = parser.parse_args()

    log_file = None
    if args.output_file:
        log_file = open(args.output_file, "w")

    def log(msg):
        print(msg)
        if log_file:
            # Strip color codes for file
            import re
            clean_msg = re.sub(r'\x1b\[[0-9;]*m', '', msg)
            log_file.write(clean_msg + "\n")
    
    if not os.path.exists(args.cpp_dir):
        log(f"Error: C++ directory not found: {args.cpp_dir}")
        return
    if not os.path.exists(args.py_dir):
        log(f"Error: Python directory not found: {args.py_dir}")
        return
        
    # Define order regexes or exact matches
    # Keys with lower index are processed first
    import re
    def get_sort_key(filepath):
        basename = os.path.basename(filepath)
        
        # 0. Inputs
        if "audio_input" in basename: return (0, -1, basename)
        
        # 1. Encoder Prefixes (convs)
        if "model.encoder.conv1" in basename: return (1, 0, basename)
        if "model.encoder.groupnorm" in basename: return (1, 1, basename)
        if "model.encoder.conv2" in basename: return (1, 2, basename)
        if "model.encoder.conv3" in basename: return (1, 3, basename)
        if "encoder_initial_h" in basename: return (1, 4, basename)

        # 2. Encoder Layers: model.encoder.layers.{i}.{name}
        match = re.search(r"model\.encoder\.layers\.(\d+)\.(.+)", basename)
        if match:
            layer_idx = int(match.group(1))
            subname = match.group(2)
            # define sub-order within layer if desired
            sub_priority = 0
            if "input_layernorm" in subname: sub_priority = 0
            elif "self_attn" in subname: sub_priority = 1
            elif "fc1" in subname or "mlp" in subname: sub_priority = 2
            elif "fc2" in subname: sub_priority = 3
            return (2, layer_idx, sub_priority, subname)
            
        # 3. Encoder Final
        if "model.encoder.layer_norm" in basename: return (3, 0, basename)
        if "encoder_final_norm" in basename: return (3, 1, basename) # legacy/alias

        # 4. Decoder Init
        if "decoder_initial_embedding" in basename: return (4, 0, basename)

        # 5. Decoder Layers: model.decoder.layers.{i}.{name}
        match = re.search(r"model\.decoder\.layers\.(\d+)\.(.+)", basename)
        if match:
            layer_idx = int(match.group(1))
            subname = match.group(2)
            sub_priority = 0
            if "input_layernorm" in subname: sub_priority = 0
            elif "self_attn" in subname: sub_priority = 1
            elif "encoder_attn" in subname: sub_priority = 2
            elif "final_layernorm" in subname: sub_priority = 3
            elif "mlp" in subname: sub_priority = 4
            return (5, layer_idx, sub_priority, subname)

        # 6. Decoder Final
        if "model.decoder.norm" in basename: return (6, 0, basename)
        if "decoder_final_norm" in basename: return (6, 1, basename)

        # 7. Output
        if "logits" in basename: return (7, 0, basename)

        # Otherwise at end, alphabetical
        return (100, 0, basename)

    cpp_files = glob.glob(os.path.join(args.cpp_dir, "*.bin"))
    cpp_files.sort(key=get_sort_key)
    
    log(f"Found {len(cpp_files)} files in {args.cpp_dir}")

    py_files = glob.glob(os.path.join(args.py_dir, "*.bin"))
    
    # Special Comparison: Preprocessor Final
    # Python: model.encoder.conv3 output -> GELU -> Permute(0, 2, 1)
    # C++: preprocessor_final_transposed
    
    # Locate Python conv3 output
    py_conv3_path = None
    for f_path in py_files:
        basename = os.path.basename(f_path)
        if "model.model.encoder.conv3." in basename and basename.endswith(".bin") and "weight" not in basename and "bias" not in basename:
             py_conv3_path = f_path
             break
    # Or maybe it's named model.encoder.conv3 depending on structure?
    # Actually dump script registers named modules. model.model.encoder.conv3 is likely.
    
    if not py_conv3_path:
        # Try finding exact match in dictionary if possible, but we don't have dict here easily.
        # Let's look for "conv3" in python list manually
        for f_path in py_files:
             basename = os.path.basename(f_path)
             if basename == "model.model.encoder.conv3.bin" or basename == "model.encoder.conv3.bin":
                 py_conv3_path = f_path
                 break
    
    if py_conv3_path:
        log(f"Found Python conv3 dump: {py_conv3_path}")
    else:
        log("Could NOT find Python conv3 dump (looked for model.encoder.conv3.bin)")

    cpp_final_path = os.path.join(args.cpp_dir, "preprocessor_final_transposed.bin")
    if py_conv3_path and os.path.exists(cpp_final_path):
        log(f"\nComparing Special: Preprocessor Final (Synthesized from {os.path.basename(py_conv3_path)})")
        
        # Read Python data (FP16)
        conv3_data = read_binary_file(py_conv3_path, np.float16).astype(np.float32)
        
        # C++ is always FP16
        cpp_data = read_binary_file(cpp_final_path, np.float16).astype(np.float32)
        
        # Assume 1 batch. Shape [1, D, L] or [D, L]?
        # Python dump is usually flattened. We need to know shape.
        # D=model.config.hidden_size (e.g. 288 or similar? No, conv3 out is embed_dim).
        # We can infer from size if we know embed_dim.
        # But for comparison we might not strictly need shape if we assume flattened order changes.
        # Transpose CHANGES order. So we DO need shape.
        # C++ preprocessor_final_transposed is "L, D" (row major?).
        # Python conv3 is "D, L" (or 1, D, L). Permute(0, 2, 1) -> "L, D".
        # So essentially Python is Column Major relative to the C++ desired layout?
        # NO. Python tensor is stored Row Major in memory, but logical shape is D, L.
        # If we read it as float32 array, it is D*L elements in logical order (d0, l0), (d0, l1)...
        # C++ output is L*D elements in logical order (l0, d0), (l0, d1)...
        # We need to reshape the Python data to [D, L], then transpose to [L, D], then flatten.
        
        # Hardcode moonshine-tiny dim if possible, or try to deduce.
        # Tiny: embed_dim = hidden_size? 
        # model_moonshine.cpp says "preprocessor_final_transposed".
        
        # Let's try to guess D.
        # C++ input length L is roughly audio_len / ?
        # Strides: 64, 3, 2. Total stride 384?
        # If we have python data size N. N = D * L.
        # Wait, comparison script doesn't know D.
        # Let's try standard tiny D=256? Or whatever config says (288?).
        # From dump script output: Audio shape 320512.
        # Let's guess D based on common factors or brute force? 
        # Better: Assume D is typically config.hidden_size.
        # Tiny hidden_size is often 288? (From Moonshine specs?)
        # Let's try D=288.
        D = 288 # Tiny has 288? user mentioned tiny.
        # Check if size % D == 0.
        if len(conv3_data) % D == 0:
            L = len(conv3_data) // D
            # Python: [D, L] in memory.
            # Convert to numpy
            py_arr = conv3_data.reshape(D, L)
            
            # Apply GELU
            # GeLU approximation or exact? Torch uses strict standard usually.
            # exact gelu: 0.5 * x * (1 + erf(x / sqrt(2)))
            py_arr = 0.5 * py_arr * (1 + np.vectorize(math.erf)(py_arr / np.sqrt(2)))
            
            # Transpose to [L, D]
            py_arr_T = py_arr.transpose() # [L, D]
            
            # Compare with C++ data
            # C++ data should match flattened py_arr_T
            py_flat = py_arr_T.flatten()
            
            compare_arrays("MANUAL: preprocessor_final_transposed", py_flat, cpp_data, args.tolerance, log)
        else:
             log(f"Skipping specialized preprocessor comparison: size {len(conv3_data)} not divisible by presumed D={D}")

    for cpp_path in cpp_files:
        basename = os.path.basename(cpp_path)
        if args.filter and args.filter not in basename:
            continue
            
        py_path = os.path.join(args.py_dir, basename)
        
        log(f"\nComparing {basename}...")
        if not os.path.exists(py_path):
            log(f"  [Skip] No matching file in Python dir: {py_path}")
            continue
            
        try:
            compare_files(cpp_path, py_path, args.tolerance, logger=log)
        except Exception as e:
            log(f"  [Error] Failed to compare: {e}")

    if log_file:
        log_file.close()

if __name__ == "__main__":
    main()
