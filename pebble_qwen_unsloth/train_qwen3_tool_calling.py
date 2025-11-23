#!/usr/bin/env python3
"""
Train Qwen 3 on tool calling using LoRA with Unsloth.

This script fine-tunes Qwen 3 models for tool calling using Unsloth,
a fast and memory-efficient library for LoRA fine-tuning.

The tool calling format uses Qwen3's native format:
- Tools provided in <tools></tools> XML tags
- Model outputs: <tool_call> with JSON containing name and arguments
- Compatible with Qwen3's apply_chat_template

Optimized for GPU training with 4-bit quantization.
"""

import os
import sys
import json
import argparse
import logging
from typing import Dict, List, Any

import torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel

# Import dataset formatting functions
from pebble_qwen_unsloth.data.format_dataset import (
    format_qwen3_dataset,
    load_tools,
    load_dataset
)

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ============================================================================
# Configuration (Unsloth Default Hyperparameters)
# ============================================================================

# Model configuration
MODEL_ID = "unsloth/Qwen3-0.6B"  # Can also use Qwen3-1.7B, Qwen3-8B, etc.

# Dataset paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DATASET_PATH = os.path.join(DATA_DIR, "synthetic_finetune_dataset.json")
TOOLS_PATH = os.path.join(DATA_DIR, "tools.json")

# Unsloth default hyperparameters
MAX_SEQ_LENGTH = 2048          # Max sequence length
LOAD_IN_4BIT = False            # Use 4-bit quantization

# LoRA hyperparameters (unsloth defaults)
LORA_RANK = 16                 # LoRA rank
LORA_ALPHA = 16                # LoRA alpha (same as rank)
LORA_DROPOUT = 0               # No dropout (unsloth recommendation)

# Training hyperparameters (unsloth defaults)
LEARNING_RATE = 2e-4           # Standard LoRA learning rate
PER_DEVICE_BATCH_SIZE = 128      # Batch size per device
GRADIENT_ACCUMULATION_STEPS = 1  # Effective batch size: 8
WARMUP_STEPS = 10              # Warmup steps
NUM_EPOCHS = 1                 # Number of epochs
OPTIMIZER = "adamw_8bit"       # 8-bit AdamW optimizer
WEIGHT_DECAY = 0.01            # Weight decay

# Output directories
OUTPUT_DIR = "outputs/qwen3_tool_calling_unsloth"
FINAL_MODEL_DIR = "models/qwen3_tool_calling_merged"


# ============================================================================
# Data Processing
# ============================================================================

def prepare_dataset(
    samples: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    tokenizer
) -> Dataset:
    """
    Prepare dataset for SFTTrainer.

    Args:
        samples: List of dataset samples
        tools: List of tool definitions
        tokenizer: HuggingFace tokenizer

    Returns:
        HuggingFace Dataset with 'text' field
    """
    formatted_data = []

    for sample in samples:
        # Format using Qwen3's native format
        formatted = format_qwen3_dataset(sample, tools, tokenizer)
        if formatted:
            # Combine all role messages into a single text string
            # The role_messages contain pre-formatted text with special tokens
            full_text = ""
            for msg in formatted:
                full_text += msg['text']

            formatted_data.append({'text': full_text})

    logger.info(f"Successfully formatted {len(formatted_data)} samples")

    return Dataset.from_list(formatted_data)


def show_training_examples(dataset: Dataset, num_examples: int = 3):
    """
    Display random training examples from the formatted dataset.

    Args:
        dataset: HuggingFace dataset with 'text' field
        num_examples: Number of examples to display
    """
    print(f"\n{'='*60}")
    print(f"Sample Training Examples ({num_examples} random samples)")
    print(f"{'='*60}")

    import random
    indices = random.sample(range(len(dataset)), min(num_examples, len(dataset)))

    for i, idx in enumerate(indices, 1):
        example = dataset[idx]
        text = example['text']

        print(f"\n{'─'*60}")
        print(f"Example {i}/{num_examples} (Index: {idx})")
        print(f"{'─'*60}")

        # Truncate long text for display
        if len(text) > 1000:
            print(text[:1000] + "...[truncated]")
        else:
            print(text)

    print(f"\n{'='*60}\n")


# ============================================================================
# Model Testing Functions
# ============================================================================

def test_model_generation(model, tokenizer, tools, label="Model"):
    """
    Test the model with examples from the synthetic dataset domain.

    Tests:
    1. Creating a note
    2. Setting an alarm
    3. Setting a timer
    """
    print(f"\n{'='*60}")
    print(f"{label} - Generation Examples")
    print(f"{'='*60}")

    # Prepare model for inference
    FastLanguageModel.for_inference(model)

    # Example test cases
    test_cases = [
        {
            'input': 'Remember to buy groceries tomorrow',
            'output': {'function_call': {'name': 'create_note', 'arguments': {}}}
        },
        {
            'input': 'Set an alarm for 7:30 AM',
            'output': {'function_call': {'name': 'set_alarm', 'arguments': {}}}
        },
        {
            'input': 'Set a timer for 10 minutes',
            'output': {'function_call': {'name': 'set_timer', 'arguments': {}}}
        }
    ]

    for i, sample in enumerate(test_cases, 1):
        # Format the example
        formatted = format_qwen3_dataset(sample, tools, tokenizer)
        if not formatted:
            print(f"Failed to format test case {i}")
            continue

        # Extract just the prompt (system + user)
        prompt = "".join(msg['text'] for msg in formatted if msg['role'] in ['system', 'user'])
        prompt += "<think>\n\n</think>\n\n"

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.0,  # Greedy decoding
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=False)

        # Extract just the generated part (after the prompt)
        generated = response[len(prompt):]

        print(f"\n--- Example {i}: {sample['input'][:50]}... ---")
        print("Generated response:")
        print(generated.strip())


# ============================================================================
# Argument Parsing
# ============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Qwen 3 on tool calling using Unsloth"
    )

    # Model arguments
    parser.add_argument(
        "--model_id",
        type=str,
        default=MODEL_ID,
        help=f"Model ID to train (default: {MODEL_ID})"
    )

    # Training hyperparameters
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=LEARNING_RATE,
        help=f"Learning rate (default: {LEARNING_RATE})"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=NUM_EPOCHS,
        help=f"Number of epochs (default: {NUM_EPOCHS})"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=PER_DEVICE_BATCH_SIZE,
        help=f"Batch size per device (default: {PER_DEVICE_BATCH_SIZE})"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=GRADIENT_ACCUMULATION_STEPS,
        help=f"Gradient accumulation steps (default: {GRADIENT_ACCUMULATION_STEPS})"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=MAX_SEQ_LENGTH,
        help=f"Max sequence length (default: {MAX_SEQ_LENGTH})"
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        default=LOAD_IN_4BIT,
        help=f"Use 4-bit quantization (default: {LOAD_IN_4BIT})"
    )
    parser.add_argument(
        "--load_in_16bit",
        action="store_true",
        help="Use 16-bit precision instead of 4-bit (overrides --load_in_4bit)"
    )

    # LoRA hyperparameters
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=LORA_RANK,
        help=f"LoRA rank (default: {LORA_RANK})"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=LORA_ALPHA,
        help=f"LoRA alpha (default: {LORA_ALPHA})"
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=DATASET_PATH,
        help=f"Path to dataset (default: {DATASET_PATH})"
    )
    parser.add_argument(
        "--tools_path",
        type=str,
        default=TOOLS_PATH,
        help=f"Path to tools JSON (default: {TOOLS_PATH})"
    )
    parser.add_argument(
        "--train_test_split",
        type=float,
        default=0.1,
        help="Train/test split ratio (default: 0.1)"
    )

    # Output directories
    parser.add_argument(
        "--output_dir",
        type=str,
        default=OUTPUT_DIR,
        help=f"Output directory for training (default: {OUTPUT_DIR})"
    )
    parser.add_argument(
        "--final_model_dir",
        type=str,
        default=FINAL_MODEL_DIR,
        help=f"Final merged model directory (default: {FINAL_MODEL_DIR})"
    )

    return parser.parse_args()


# ============================================================================
# Main Training Script
# ============================================================================

def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()

    # Handle quantization flags
    load_in_4bit = args.load_in_4bit and not args.load_in_16bit

    print("="*60)
    print("Qwen 3 Tool Calling Training Script (Unsloth)")
    print("="*60)
    print(f"Model: {args.model_id}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Tools: {args.tools_path}")
    print(f"Batch size per device: {args.batch_size}")
    print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"LoRA rank: {args.lora_rank}, alpha: {args.lora_alpha}")
    print(f"Max sequence length: {args.max_seq_length}")
    print(f"Quantization: {'4-bit' if load_in_4bit else '16-bit (full precision)'}")
    print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")

    # Load model and tokenizer
    print(f"\n{'='*60}")
    print("Loading model and tokenizer")
    print(f"{'='*60}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_id,
        max_seq_length=args.max_seq_length,
        dtype=None,  # Auto-detect
        load_in_4bit=load_in_4bit,
    )

    print(f"Model loaded: {args.model_id}")
    print(f"Tokenizer loaded with vocab size: {len(tokenizer)}")
    print(f"Quantization: {'4-bit' if load_in_4bit else '16-bit (full precision)'}")

    # Apply LoRA
    print(f"\n{'='*60}")
    print("Applying LoRA to model")
    print(f"{'='*60}")

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",  # 30% memory savings
        random_state=3407,
        max_seq_length=args.max_seq_length,
    )

    print(f"LoRA applied: rank={args.lora_rank}, alpha={args.lora_alpha}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Load tools and dataset
    print(f"\n{'='*60}")
    print("Loading synthetic dataset and tools")
    print(f"{'='*60}")

    tools = load_tools(args.tools_path)
    dataset_samples = load_dataset(args.dataset_path)

    print(f"Loaded {len(tools)} tools")
    print(f"Loaded {len(dataset_samples)} dataset samples")

    # Prepare dataset
    print("\nFormatting examples for Qwen 3 tool calling...")
    full_dataset = prepare_dataset(dataset_samples, tools, tokenizer)

    # Split dataset
    if args.train_test_split > 0:
        split = full_dataset.train_test_split(test_size=args.train_test_split, seed=42)
        train_dataset = split['train']
        eval_dataset = split['test']
    else:
        train_dataset = full_dataset
        eval_dataset = None

    print(f"Training split: {len(train_dataset)} samples")
    if eval_dataset:
        print(f"Validation split: {len(eval_dataset)} samples")

    # Show sample training examples
    show_training_examples(train_dataset, num_examples=2)

    # Test base model before training
    print(f"\n{'='*60}")
    print("Testing base model BEFORE training")
    print(f"{'='*60}")
    test_model_generation(model, tokenizer, tools, label="Base Model (Before Training)")

    # Setup training
    print(f"\n{'='*60}")
    print("Setting up training")
    print(f"{'='*60}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(args.batch_size // 4, 1),
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=WARMUP_STEPS,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim=OPTIMIZER,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type="linear",
        seed=3407,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_dataset else "no",
        report_to="none",  # Disable wandb/tensorboard
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        args=training_args,
        packing=False,  # Don't pack sequences
    )

    print("Trainer initialized")
    print(f"Total training steps: {len(train_dataset) * args.num_epochs // (args.batch_size * args.gradient_accumulation_steps)}")

    # Train!
    print(f"\n{'='*60}")
    print("Starting training")
    print(f"{'='*60}")

    trainer.train()

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}\n")

    # Test trained model
    print(f"\n{'='*60}")
    print("Testing trained model AFTER training")
    print(f"{'='*60}")
    test_model_generation(model, tokenizer, tools, label="Trained Model (After Training)")

    # Save model
    print(f"\n{'='*60}")
    print("Saving model in HuggingFace safetensors format")
    print(f"{'='*60}")

    # Save merged model (LoRA merged with base weights)
    model.save_pretrained_merged(
        args.final_model_dir,
        tokenizer,
        save_method="merged_16bit"
    )

    print(f"\nModel saved to: {args.final_model_dir}")
    print("\nSaved files:")
    for f in os.listdir(args.final_model_dir):
        size = os.path.getsize(os.path.join(args.final_model_dir, f)) / (1024 * 1024)
        print(f"  {f:<30} {size:>10.2f} MB")

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Model saved to: {args.final_model_dir}")
    print(f"Training outputs: {args.output_dir}")
    print(f"\nThe model is ready to use for inference!")
    print(f"Load it with: AutoModelForCausalLM.from_pretrained('{args.final_model_dir}')")


if __name__ == '__main__':
    main()
