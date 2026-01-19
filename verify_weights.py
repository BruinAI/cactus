import torch
import numpy as np
import os
from transformers import AutoModel

def load_weights_file(path, shape):
    # Load raw bytes
    with open(path, "rb") as f:
        # Skip 96-byte header (80 bytes + padding to 32-byte alignment)
        f.seek(96)
        data = f.read()
    
    # Convert to FP16 (numpy float16)
    weights = np.frombuffer(data, dtype=np.float16)
    
    # Reshape
    return weights.reshape(shape)

def main():
    model_id = "UsefulSensors/moonshine-tiny"
    weights_dir = "weights/moonshine-tiny"
    
    print(f"Loading HF model {model_id}...")
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    
    # Check Conv1
    print("\nChecking Conv1 Weights...")
    hf_conv1 = model.encoder.conv1.weight.detach().cpu().numpy()
    conv1_path = os.path.join(weights_dir, "encoder_conv1_weight.weights")
    if os.path.exists(conv1_path):
        try:
            with open(conv1_path, "rb") as f:
                f.seek(96)
                data = f.read()
            bin_conv1 = np.frombuffer(data, dtype=np.float16).astype(np.float32).reshape(hf_conv1.shape)
            diff = np.abs(hf_conv1 - bin_conv1)
            if diff.max() > 1e-3:
                print(f"MISMATCH in Conv1! Max diff: {diff.max():.6f}")
            else:
                print("Conv1 MATCH.")
        except Exception as e:
            print(f"Error checking Conv1: {e}")

    # Check GroupNorm Weights
    print("\nChecking GroupNorm Weights...")
    hf_gn_w = model.encoder.groupnorm.weight.detach().cpu().numpy()
    hf_gn_b = model.encoder.groupnorm.bias.detach().cpu().numpy()
    
    gn_w_path = os.path.join(weights_dir, "encoder_norm_weight.weights")
    gn_b_path = os.path.join(weights_dir, "encoder_norm_bias.weights")
    
    for name, path, hf_val in [("GN Weight", gn_w_path, hf_gn_w), ("GN Bias", gn_b_path, hf_gn_b)]:
        if not os.path.exists(path):
            print(f"Error: {path} not found")
            continue
        try:    
            with open(path, "rb") as f:
                f.seek(96) # Skip header
                data = f.read()
            
            bin_val = np.frombuffer(data, dtype=np.float16).astype(np.float32)
            if bin_val.shape != hf_val.shape:
                print(f"Shape mismatch for {name}: HF {hf_val.shape} vs Bin {bin_val.shape}")
                continue

            diff = np.abs(hf_val - bin_val)
            if diff.max() > 1e-3:
                 print(f"MISMATCH in {name}! Max diff: {diff.max()}")
            else:
                 print(f"{name} MATCH.")
        except Exception as e:
            print(f"Error checking {name}: {e}")

if __name__ == "__main__":
    main()
