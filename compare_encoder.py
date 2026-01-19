import torch
import torchaudio
from transformers import AutoModel, AutoConfig
import numpy as np
import os

import soundfile as sf
import torch

def load_audio(path):
    wav, sr = sf.read(path)
    if sr != 16000:
        print(f"Warning: Sample rate is {sr}, expected 16000")
    # Convert to float32 and shape [1, T]
    wav = wav.astype(np.float32)
    t = torch.from_numpy(wav).unsqueeze(0)
    print(f"[Input Audio] Shape: {t.shape}")
    print(f"  Min: {t.min():.6f}")
    print(f"  Max: {t.max():.6f}")
    print(f"  Mean: {t.mean():.6f}")
    print(f"  Std: {t.std():.6f}")
    return t

def analyze_tensor(name, tensor):
    t = tensor.detach().cpu().numpy()
    print(f"[{name}] Shape: {t.shape}")
    print(f"  Min: {t.min():.6f}")
    print(f"  Max: {t.max():.6f}")
    print(f"  Mean: {t.mean():.6f}")
    print(f"  Std: {t.std():.6f}")
    # Print first few features of first token
    flat = t.flatten()
    print(f"  First 10: {flat[:10]}")

def main():
    model_id = "UsefulSensors/moonshine-tiny"
    wav_path = "tests/assets/test.wav"
    
    print(f"Loading {model_id}...")
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    model.eval()
    
    print(f"Loading {wav_path}...")
    audio = load_audio(wav_path)
    
    # Run encoder
    print("\nRunning Encoder Forward Pass...")
    with torch.no_grad():
        # Preprocess audio (Moonshine specific)
        # Note: Moonshine model class usually handles raw audio if passed correctly, 
        # but let's see how the HF implementation expects it.
        # Based on previous files, it seems to take raw audio.
        
        # Verify input structure from model config/code if needed, 
        # but standard usage is usually input_values
        inputs = audio
        
        # Hook specific layers
        # Moonshine encoder is usually model.encoder
        # Layers are model.encoder.layers (MoonshineEncoderLayer)
        
        activations = {}
        def get_activation(name):
            def hook(model, input, output):
                activations[name] = output[0] if isinstance(output, tuple) else output
            return hook

        for i, layer in enumerate(model.encoder.layers):
            layer.register_forward_hook(get_activation(f"layer_{i}"))
            
        # Hook convolution/pre-processing
        model.encoder.conv1.register_forward_hook(get_activation("conv1"))
        model.encoder.groupnorm.register_forward_hook(get_activation("groupnorm"))
        model.encoder.conv2.register_forward_hook(get_activation("conv2"))
        
        # Analyze weights
        analyze_tensor("Conv1 Weights", model.encoder.conv1.weight)
        
        # Forward
        outputs = model.encoder(inputs)
        
        # Analyze
        if "conv1" in activations:
            c1 = activations["conv1"]
            analyze_tensor("Encoder Conv1 (Raw)", c1)
            analyze_tensor("Encoder Conv1 (Tanh)", torch.tanh(c1))
            
        if "groupnorm" in activations: analyze_tensor("Encoder GroupNorm", activations["groupnorm"])
        if "conv2" in activations: analyze_tensor("Encoder Conv2", activations["conv2"])
        
        for i in range(len(model.encoder.layers)):
            name = f"layer_{i}"
            if name in activations:
                analyze_tensor(f"Encoder Layer {i} Output", activations[name])

if __name__ == "__main__":
    main()
