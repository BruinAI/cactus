#!/usr/bin/env python3
"""
Simple script to load a Gemma 3 1B LoRA model and immediately save it.
Used for testing and updating LoRA merging logic.
"""

import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model
import torch

def main():
    # Configuration
    base_model_name = "google/gemma-3-1b-it"
    lora_checkpoint_path = "./checkpoints/gemma3-1b-tool-calling"  # Update this path
    output_path = "./test_output/gemma3-1b-lora-merged"

    print(f"Loading base model: {base_model_name}")

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    print(f"Loading tokenizer: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Check if LoRA checkpoint exists
    if os.path.exists(lora_checkpoint_path):
        print(f"Loading LoRA weights from: {lora_checkpoint_path}")
        model = PeftModel.from_pretrained(base_model, lora_checkpoint_path)

        # Merge LoRA weights into base model
        print("Merging LoRA weights into base model...")
        model = model.merge_and_unload()
    else:
        print(f"No LoRA checkpoint found at {lora_checkpoint_path}")
        print("Using base model only (or you can create a dummy LoRA adapter)")

        # Option: Create a dummy LoRA adapter for testing
        # Uncomment the lines below if you want to test with a fresh LoRA adapter
        # lora_config = LoraConfig(
        #     r=32,
        #     lora_alpha=64,
        #     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        #     lora_dropout=0.05,
        #     bias="none",
        #     task_type="CAUSAL_LM"
        # )
        # model = get_peft_model(base_model, lora_config)
        # model = model.merge_and_unload()

        model = base_model

    # Save the model
    print(f"Saving model to: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    print("✓ Model and tokenizer saved successfully!")
    print(f"\nSaved to: {output_path}")
    print("\nYou can now test this model with:")
    print(f"  python3 tools/convert_hf.py {output_path} weights/gemma3-1b-test/ --precision INT8")

if __name__ == "__main__":
    main()
