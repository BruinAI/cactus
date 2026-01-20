import numpy as np
import os

def read_bin(path, dtype):
    return np.fromfile(path, dtype=dtype)

cpp_path = "tests/build/dump_cpp/model.encoder.layers.0.self_attn.attn_out.bin"
py_path = "tests/build/dump_python/model.encoder.layers.0.self_attn.attn_out.bin"

if not os.path.exists(cpp_path):
    print("Missing C++ dump")
    exit()

cpp = read_bin(cpp_path, np.float16).astype(np.float32)
py = read_bin(py_path, np.float32)

# Unpad Python
py = py.reshape(-1, 40)
py = py[:, :36]
py_flat = py.flatten()

# Compare
diff = np.abs(cpp - py_flat)
print(f"Max Diff: {np.max(diff)}")
print(f"Mean Diff: {np.mean(diff)}")

# Check per-dimension error
n_tokens = len(cpp) // 36
cpp_reshaped = cpp.reshape(n_tokens, 36)
py_reshaped = py # already [N, 36]

dim_diffs = np.mean(np.abs(cpp_reshaped - py_reshaped), axis=0)
print("\nMean Diff per Dimension (0-35):")
for i, d in enumerate(dim_diffs):
    print(f"Dim {i:2d}: {d:.4f}")

