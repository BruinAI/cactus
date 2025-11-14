#!/usr/bin/env python3
"""
Analyze tokenized lengths of Noah's dataset to find maximum sequence length.
"""

import os
import json
import numpy as np
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from format_noah_dataset import (
    format_qwen3_noah_dataset,
    load_noah_tools,
    load_noah_dataset
)

# Dataset paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DATASET_PATH = os.path.join(DATA_DIR, "noah_finetune_dataset.json")
TOOLS_PATH = os.path.join(DATA_DIR, "noah_tools.json")

# Model for tokenizer (same as training script)
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

def main():
    print("="*60)
    print("Analyzing Dataset Tokenized Lengths")
    print("="*60)

    # Download and load tokenizer (same as training script does)
    print(f"\nDownloading tokenizer from {MODEL_ID}...")
    local_model_path = snapshot_download(
        repo_id=MODEL_ID,
        ignore_patterns=["*.pth", "*.safetensors"]  # Only need tokenizer
    )
    print(f"Model downloaded to: {local_model_path}")

    tokenizer = AutoTokenizer.from_pretrained(local_model_path)

    # Load tools and dataset
    print(f"Loading dataset from {DATASET_PATH}...")
    tools = load_noah_tools(TOOLS_PATH)
    dataset_samples = load_noah_dataset(DATASET_PATH)
    print(f"Loaded {len(tools)} tools")
    print(f"Loaded {len(dataset_samples)} dataset samples")

    # Format and tokenize each sample
    print("\nTokenizing samples...")
    lengths = []
    sample_details = []

    for idx, sample in enumerate(dataset_samples):
        formatted = format_qwen3_noah_dataset(sample, tools, tokenizer)
        if formatted:
            # Concatenate all turns to get full sequence
            full_text = "".join([turn['text'] for turn in formatted])
            tokens = tokenizer.encode(full_text)
            length = len(tokens)
            lengths.append(length)
            sample_details.append({
                'idx': idx,
                'length': length,
                'num_turns': len(formatted),
                'sample': sample
            })

    # Calculate statistics
    lengths = np.array(lengths)
    print(f"\n{'='*60}")
    print("Tokenized Length Statistics")
    print(f"{'='*60}")
    print(f"Total samples: {len(lengths)}")
    print(f"Mean length: {lengths.mean():.2f} tokens")
    print(f"Median length: {np.median(lengths):.2f} tokens")
    print(f"Min length: {lengths.min()} tokens")
    print(f"Max length: {lengths.max()} tokens")
    print(f"Std deviation: {lengths.std():.2f} tokens")
    print(f"\nPercentiles:")
    print(f"  25th: {np.percentile(lengths, 25):.2f} tokens")
    print(f"  50th: {np.percentile(lengths, 50):.2f} tokens")
    print(f"  75th: {np.percentile(lengths, 75):.2f} tokens")
    print(f"  90th: {np.percentile(lengths, 90):.2f} tokens")
    print(f"  95th: {np.percentile(lengths, 95):.2f} tokens")
    print(f"  99th: {np.percentile(lengths, 99):.2f} tokens")

    # Find samples exceeding 2048
    over_2048 = [s for s in sample_details if s['length'] > 2048]
    print(f"\n{'='*60}")
    print(f"Samples exceeding 2048 tokens: {len(over_2048)}")
    print(f"{'='*60}")

    if over_2048:
        print(f"Percentage of dataset: {len(over_2048)/len(lengths)*100:.2f}%")
        print("\nTop 10 longest samples:")
        sorted_details = sorted(sample_details, key=lambda x: x['length'], reverse=True)
        for i, detail in enumerate(sorted_details[:10]):
            print(f"\n{i+1}. Sample {detail['idx']}:")
            print(f"   Length: {detail['length']} tokens")
            print(f"   Turns: {detail['num_turns']}")
            print(f"   Input: {detail['sample']['input'][:100]}...")
    else:
        print("All samples fit within 2048 tokens!")
        print("\nTop 5 longest samples:")
        sorted_details = sorted(sample_details, key=lambda x: x['length'], reverse=True)
        for i, detail in enumerate(sorted_details[:5]):
            print(f"\n{i+1}. Sample {detail['idx']}:")
            print(f"   Length: {detail['length']} tokens")
            print(f"   Turns: {detail['num_turns']}")
            print(f"   Input: {detail['sample']['input'][:100]}...")

if __name__ == '__main__':
    main()