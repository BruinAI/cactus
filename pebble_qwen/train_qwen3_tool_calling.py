#!/usr/bin/env python3
"""
Train Qwen 3 on tool calling using LoRA with synthetically generated dataset.

This script fine-tunes Qwen 3 models for tool calling using a synthetically generated
dataset which contains examples of creating notes, setting alarms, timers, and reminders.
The dataset is generated using a 3-phase backwards generation pipeline with Claude API.

The tool calling format uses JSON:
- Tools provided as JSON schemas
- Model outputs: JSON with function_call object containing name and arguments
- Compatible with standard function calling formats

Optimized for 4x TPU v5e chips.
"""

import os
import sys
import json
import argparse
import logging
from typing import Dict
import nest_asyncio
nest_asyncio.apply()

import jax
import wandb
import optax
import numpy as np
from datasets import Dataset
import grain.python as grain
from transformers import AutoTokenizer

# Import tunix libraries
from tunix.generate import sampler as sampler_lib
from tunix.models.qwen3 import model as qwen_lib
from tunix.models.qwen3 import params as params_lib
from tunix.sft import metrics_logger
from tunix.sft.metrics_logger import Mode
from tunix.sft import peft_trainer
from tunix.sft import utils

# Import dataset formatting functions
from pebble_qwen.data.format_dataset import (
    format_qwen3_dataset,
    load_tools,
    load_dataset
)

# Import Qwen model utilities
from pebble_qwen.qwen_utils import download_and_setup_model, create_lora_model, save_lora_weights

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ============================================================================
# Configuration (Default values - can be overridden by command-line arguments)
# ============================================================================

# Model configuration
MODEL_ID = "Qwen/Qwen3-0.6B"  # Can also use Qwen3-1.7B, Qwen3-8B, etc.
QWEN_TOKENIZER_PATH = None  # Will be loaded from the model directory

# Dataset paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DATASET_PATH = os.path.join(DATA_DIR, "synthetic_finetune_dataset.json")
TOOLS_PATH = os.path.join(DATA_DIR, "tools.json")

# Training hyperparameters
# Based on hyperparameter search results:
# - 3 epochs: val_loss=0.0108, overfit_ratio=3.3x (RECOMMENDED for production)
# - 5 epochs: val_loss=0.0074, overfit_ratio=26.7x (use for benchmarks only)
#
# 3 epochs is the sweet spot: only 32% worse validation loss but 8x less overfitting.
# For small datasets, excessive overfitting (26.7x) means the model has memorized training
# data and won't generalize. Use --num_epochs=5 only if maximizing benchmark scores on
# similar data or if you have a much larger, more diverse dataset.
NUM_EPOCHS = 1
LEARNING_RATE = 1e-4  # 1e-4 significantly outperformed 1e-5 across all configurations
MAX_TARGET_LENGTH = 1500  # Noah's dataset has shorter sequences

# Allow max_steps override
MAX_STEPS = None

BATCH_SIZE = 8  # Batch size 8 performed slightly better than 16
GRADIENT_ACCUMULATION_STEPS = 1
EVAL_EVERY_N_STEPS = 5

# LoRA hyperparameters
# Higher rank and alpha showed consistent improvements
RANK = 32  # Increased from 16 (32 performed better across configurations)
ALPHA = 64.0  # Increased from 32 (64 achieved best results)

# TPU/GPU mesh configuration
MESH_SHAPE = len(jax.devices()), 1
MESH_AXIS_NAMES = "fsdp", "tp"

# Train/validation split
TRAIN_TEST_SPLIT = 1/6

# Checkpoint and output directories
CKPT_DIR = "/tmp/qwen_tool_calling_ckpts/"
LORA_OUTPUT_DIR = f"/dev/shm/{MODEL_ID.split('/')[-1]}_tool_calling_lora"



# ============================================================================
# Data Loading and Processing
# ============================================================================

class TokenizerWrapper:
    """Wrapper to adapt HuggingFace tokenizer to tunix tokenizer API."""

    def __init__(self, hf_tokenizer):
        self._tokenizer = hf_tokenizer

    def encode(self, text: str):
        """Encode text to token IDs."""
        return self._tokenizer.encode(text, add_special_tokens=False)

    def pad_id(self):
        """Get padding token ID."""
        return self._tokenizer.pad_token_id

    def eos_id(self):
        """Get EOS token ID."""
        return self._tokenizer.eos_token_id


class _Tokenize(grain.MapTransform):
    """Tokenize role-based messages and create proper masks."""

    def __init__(self, tokenizer: TokenizerWrapper):
        self._tokenizer = tokenizer

    def map(self, element: Dict[str, str]) -> Dict[str, np.ndarray]:
        """
        Tokenize role messages and create loss mask.

        Returns dict with:
        - tokens: Full token sequence
        - mask: Loss mask (1 for model outputs, 0 for user/system inputs)
        """
        # Parse role messages
        role_messages = json.loads(element['role_messages'])

        all_tokens = []
        all_masks = []

        for turn in role_messages:
            role = turn['role']
            text = turn['text']

            # Tokenize the complete text
            tokens = self._tokenizer.encode(text)

            # Create mask based on role
            if role in ['user', 'system']:
                # Don't train on user or system input
                all_tokens.extend(tokens)
                all_masks.extend([0] * len(tokens))
            elif role == 'model':
                # TRAIN on model output
                all_tokens.extend(tokens)
                all_masks.extend([1] * len(tokens))

        return {
            'tokens': np.array(all_tokens, dtype=np.int32),
            'mask': np.array(all_masks, dtype=np.float32)
        }


class _BuildTrainInput(grain.MapTransform):
    """Build TrainingInput from tokens with proper loss masking."""

    def __init__(self, max_seq_len: int, pad_value: int):
        self._max_seq_len = max_seq_len
        self._pad_value = pad_value

    def map(self, tokenized_dict: Dict[str, np.ndarray]) -> peft_trainer.TrainingInput:
        """Build training input from tokens and mask."""
        tokens = tokenized_dict['tokens']
        mask = tokenized_dict['mask']

        # Pad or truncate to max_seq_len
        if len(tokens) > self._max_seq_len:
            tokens = tokens[:self._max_seq_len]
            mask = mask[:self._max_seq_len]
        else:
            pad_len = self._max_seq_len - len(tokens)
            tokens = np.pad(tokens, [[0, pad_len]], mode='constant', constant_values=self._pad_value)
            mask = np.pad(mask, [[0, pad_len]], mode='constant', constant_values=0)

        return peft_trainer.TrainingInput(
            input_tokens=tokens,
            input_mask=mask
        )


class _FilterOverlength(grain.FilterTransform):
    """Filter out overlength examples."""

    def __init__(self, max_seq_len: int):
        self._max_seq_len = max_seq_len

    def filter(self, element: peft_trainer.TrainingInput) -> bool:
        return element.input_tokens.shape[0] <= self._max_seq_len


def _build_data_loader(
    *,
    data_source: grain.RandomAccessDataSource,
    batch_size: int,
    num_epochs: int,
    max_seq_len: int,
    tokenizer: TokenizerWrapper,
    shuffle: bool,
    seed: int = 42
) -> grain.DataLoader:
    """Build a grain DataLoader."""
    sampler = grain.IndexSampler(
        num_records=len(data_source),
        num_epochs=num_epochs,
        shard_options=grain.NoSharding(),
        shuffle=shuffle,
        seed=seed if shuffle else None,
    )

    return grain.DataLoader(
        data_source=data_source,
        sampler=sampler,
        operations=[
            _Tokenize(tokenizer),
            _BuildTrainInput(max_seq_len, tokenizer.pad_id()),
            _FilterOverlength(max_seq_len),
            grain.Batch(batch_size=batch_size, drop_remainder=True),
        ],
    )


# ============================================================================
# Model and Training Setup
# ============================================================================

def gen_model_input_fn(x: peft_trainer.TrainingInput, tokenizer):
    """Generate model inputs from training data."""
    pad_mask = x.input_tokens != tokenizer.pad_id()
    positions = utils.build_positions_from_mask(pad_mask)
    attention_mask = utils.make_causal_attn_mask(pad_mask)
    return {
        'input_tokens': x.input_tokens,
        'input_mask': x.input_mask,
        'positions': positions,
        'attention_mask': attention_mask,
    }


# ============================================================================
# Model Testing Functions
# ============================================================================

def show_training_examples(dataset, num_examples=3):
    """
    Display random training examples from the formatted dataset.

    Args:
        dataset: HuggingFace dataset with 'role_messages' field
        num_examples: Number of examples to display (default: 3)
    """
    print(f"\n{'='*60}")
    print(f"Sample Training Examples ({num_examples} random samples)")
    print(f"{'='*60}")

    import random
    indices = random.sample(range(len(dataset)), min(num_examples, len(dataset)))

    for i, idx in enumerate(indices, 1):
        example = dataset[idx]
        role_messages = json.loads(example['role_messages'])

        print(f"\n{'─'*60}")
        print(f"Example {i}/{num_examples} (Index: {idx})")
        print(f"{'─'*60}")

        for turn_idx, turn in enumerate(role_messages):
            role = turn['role']
            text = turn['text']
            mask_indicator = "✓ TRAIN" if role == 'model' else "✗ SKIP"

            print(f"\n[{role.upper()}] [{mask_indicator}]")
            # Truncate long text (like system prompts with tool definitions)
            if len(text) > 500:
                print(text[:500] + "...[truncated]")
            else:
                print(text)

    print(f"\n{'='*60}\n")


def test_model_generation(model, tokenizer, model_config, eos_tokens, tools, label="Model"):
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

    # Create sampler with larger cache for tool calling prompts
    # Tool definitions in system prompt can be very long (>2000 tokens)
    sampler = sampler_lib.Sampler(
        transformer=model,
        tokenizer=tokenizer,
        cache_config=sampler_lib.CacheConfig(
            cache_size=4096,  # Increased from 1024 to handle long tool definitions
            num_layers=model_config.num_layers,
            num_kv_heads=model_config.num_kv_heads,
            head_dim=model_config.head_dim,
        ),
    )

    # Example 1: Create a note
    sample1 = {
        'input': 'Remember to buy groceries tomorrow',
        'output': {'function_call': {'name': 'create_note', 'arguments': {}}}
    }

    # Example 2: Set an alarm
    sample2 = {
        'input': 'Set an alarm for 7:30 AM',
        'output': {'function_call': {'name': 'set_alarm', 'arguments': {}}}
    }

    # Example 3: Set a timer
    sample3 = {
        'input': 'Set a timer for 10 minutes',
        'output': {'function_call': {'name': 'set_timer', 'arguments': {}}}
    }

    # Format examples
    formatted1 = format_qwen3_dataset(sample1, tools, tokenizer)
    formatted2 = format_qwen3_dataset(sample2, tools, tokenizer)
    formatted3 = format_qwen3_dataset(sample3, tools, tokenizer)

    if not (formatted1 and formatted2 and formatted3):
        print("Failed to format test examples")
        return

    # Extract just the user part (before assistant response)
    prompt1 = formatted1[0]['text']
    prompt2 = formatted2[0]['text']
    prompt3 = formatted3[0]['text']

    out_data = sampler(
        input_strings=[prompt1, prompt2, prompt3],
        max_generation_steps=128,
        eos_tokens=eos_tokens,
        top_k=1,
        top_p=None,
        temperature=None,
    )

    examples = [
        ("Create a note", prompt1, out_data.text[0]),
        ("Set an alarm", prompt2, out_data.text[1]),
        ("Set a timer", prompt3, out_data.text[2]),
    ]

    for title, prompt, response in examples:
        print(f"\n--- Example: {title} ---")
        print("Prompt:", prompt[:200] + "..." if len(prompt) > 200 else prompt)
        print("\nModel response:")
        print(response.strip())


def run_test_set_generation(model, tokenizer, model_config, eos_tokens, test_dataset, tools, output_file):
    """
    Run generation on the test set and save results to file.

    Args:
        model: The trained model
        tokenizer: HuggingFace tokenizer
        model_config: Model configuration
        eos_tokens: EOS token IDs
        test_dataset: Test dataset with 'role_messages' field
        tools: Tool definitions
        output_file: Path to output text file
    """
    print(f"\n{'='*60}")
    print("Running Test Set Generation")
    print(f"{'='*60}")

    # Create sampler
    sampler = sampler_lib.Sampler(
        transformer=model,
        tokenizer=tokenizer,
        cache_config=sampler_lib.CacheConfig(
            cache_size=4096,
            num_layers=model_config.num_layers,
            num_kv_heads=model_config.num_kv_heads,
            head_dim=model_config.head_dim,
        ),
    )

    results = []

    for idx, example in enumerate(test_dataset):
        role_messages = json.loads(example['role_messages'])

        # Extract user input and target output for display
        user_input = None
        target_output = None

        # The role_messages contain pre-formatted text with special tokens
        # Extract the actual user query from the formatted text
        for turn in role_messages:
            if turn['role'] == 'user':
                # Extract just the content between im_start and im_end
                user_text = turn['text']
                if '<|im_start|>user\n' in user_text:
                    user_input = user_text.split('<|im_start|>user\n')[1].split('<|im_end|>')[0].strip()
                else:
                    user_input = user_text
            elif turn['role'] == 'model':
                target_output = turn['text']

        if not user_input:
            print(f"Warning: No user input found in test example {idx}")
            continue

        # Build prompt from role_messages (system + user parts)
        # The role_messages already have the proper formatting with special tokens
        prompt = ''
        for turn in role_messages:
            if turn['role'] in ['system', 'user']:
                prompt += turn['text']
            else:
                break  # Stop before model response

        out_data = sampler(
            input_strings=[prompt],
            max_generation_steps=256,
            eos_tokens=eos_tokens,
            top_k=1,
            top_p=None,
            temperature=None,
        )

        model_output = out_data.text[0].strip()

        results.append({
            'index': idx,
            'input': user_input,
            'target_output': target_output or "(no target)",
            'model_output': model_output
        })

        print(f"Processed test example {idx + 1}/{len(test_dataset)}")

    # Write results to file
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("TEST SET GENERATION RESULTS\n")
        f.write("="*80 + "\n\n")

        for result in results:
            f.write(f"{'─'*80}\n")
            f.write(f"Test Example {result['index'] + 1}\n")
            f.write(f"{'─'*80}\n\n")

            f.write("INPUT:\n")
            f.write(result['input'] + "\n\n")

            f.write("TARGET OUTPUT:\n")
            f.write(result['target_output'] + "\n\n")

            f.write("MODEL OUTPUT:\n")
            f.write(result['model_output'] + "\n\n")

    print(f"Saved generation results to: {output_file}")
    return results


def compute_test_set_perplexity(trainer, test_loader):
    """
    Compute perplexity on the test set using tunix's built-in evaluation.

    Args:
        trainer: PeftTrainer instance
        test_loader: DataLoader for test set

    Returns:
        dict: Perplexity metrics
    """
    print(f"\n{'='*60}")
    print("Computing Test Set Perplexity")
    print(f"{'='*60}")

    # Use trainer's eval step function
    eval_step_fn = trainer.create_eval_step_fn()

    # Run evaluation loop
    total_loss = 0.0
    eval_steps = 0

    for eval_example in test_loader:
        # Prepare inputs (same as training)
        loss, _ = eval_step_fn(trainer.model, eval_example)
        loss = jax.lax.stop_gradient(loss)

        # Accumulate loss
        total_loss += float(np.array(loss))
        eval_steps += 1

        print(f"Test example {eval_steps}: loss={float(np.array(loss)):.4f}")

    # Compute average loss and perplexity
    avg_loss = total_loss / eval_steps if eval_steps > 0 else float('inf')
    perplexity = np.exp(avg_loss)

    metrics = {
        'test_loss': float(avg_loss),
        'test_perplexity': float(perplexity),
        'test_steps': eval_steps
    }

    print(f"\n{'─'*60}")
    print("TEST SET PERPLEXITY METRICS")
    print(f"{'─'*60}")
    print(f"Average loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")
    print(f"Test steps: {eval_steps}")
    print(f"{'─'*60}")

    return metrics


# ============================================================================
# Argument Parsing
# ============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Qwen 3 on tool calling using LoRA"
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
        default=BATCH_SIZE,
        help=f"Batch size (default: {BATCH_SIZE})"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=GRADIENT_ACCUMULATION_STEPS,
        help=f"Gradient accumulation steps (default: {GRADIENT_ACCUMULATION_STEPS})"
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=MAX_TARGET_LENGTH,
        help=f"Max sequence length (default: {MAX_TARGET_LENGTH})"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=MAX_STEPS,
        help="Max steps (overrides epochs if set)"
    )
    parser.add_argument(
        "--eval_every_n_steps",
        type=int,
        default=EVAL_EVERY_N_STEPS,
        help=f"Evaluate every N steps (default: {EVAL_EVERY_N_STEPS})"
    )

    # LoRA hyperparameters
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=RANK,
        help=f"LoRA rank (default: {RANK})"
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=ALPHA,
        help=f"LoRA alpha (default: {ALPHA})"
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
        default=TRAIN_TEST_SPLIT,
        help=f"Train/test split ratio (default: {TRAIN_TEST_SPLIT})"
    )

    # Output directories
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=CKPT_DIR,
        help=f"Checkpoint directory (default: {CKPT_DIR})"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="LoRA output directory (default: /dev/shm/MODEL_tool_calling_lora)"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Name for this training run (used by hyperparameter search)"
    )

    return parser.parse_args()


# ============================================================================
# Main Training Script
# ============================================================================

def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()

    # Update global variables with args
    model_id = args.model_id
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    max_target_length = args.max_target_length
    max_steps = args.max_steps
    eval_every_n_steps = args.eval_every_n_steps
    lora_rank = args.lora_rank
    lora_alpha = args.lora_alpha
    dataset_path = args.dataset_path
    tools_path = args.tools_path
    train_test_split = args.train_test_split
    ckpt_dir = args.checkpoint_dir
    lora_output_dir = args.output_dir or f"/dev/shm/{model_id.split('/')[-1]}_tool_calling_lora"

    print("="*60)
    print("Qwen 3 Tool Calling Training Script (Synthetic Dataset)")
    print("="*60)
    print(f"Model: {model_id}")
    print(f"Dataset: {dataset_path}")
    print(f"Tools: {tools_path}")
    print(f"Global batch size: {batch_size}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"LoRA rank: {lora_rank}, alpha: {lora_alpha}")
    print(f"Max sequence length: {max_target_length}")
    print(f"Devices: {len(jax.devices())} x {jax.devices()[0].platform}")
    print(f"Mesh configuration: {MESH_SHAPE} x {MESH_AXIS_NAMES}")

    # Create checkpoint directories
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Checkpoint directory: {ckpt_dir}")

    # Download model
    local_model_path, eos_tokens = download_and_setup_model(model_id)

    # Initialize tokenizer
    print(f"\n{'='*60}")
    print("Initializing tokenizer")
    print(f"{'='*60}")
    # Load HuggingFace tokenizer
    hf_tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    print(f"Loaded tokenizer from {local_model_path}")

    # Update EOS tokens
    if hf_tokenizer.eos_token_id not in eos_tokens:
        eos_tokens.append(hf_tokenizer.eos_token_id)
        print(f"Updated EOS token IDs: {eos_tokens}")

    # Wrap HF tokenizer for compatibility with training pipeline
    tokenizer = TokenizerWrapper(hf_tokenizer)

    # Load tools and dataset
    print(f"\n{'='*60}")
    print("Loading synthetic dataset and tools")
    print(f"{'='*60}")
    tools = load_tools(tools_path)
    dataset_samples = load_dataset(dataset_path)
    print(f"Loaded {len(tools)} tools")
    print(f"Loaded {len(dataset_samples)} dataset samples")

    # Format dataset
    print("\nFormatting examples for Qwen 3 tool calling...")
    formatted_samples = []
    for sample in dataset_samples:
        formatted = format_qwen3_dataset(sample, tools, hf_tokenizer)
        if formatted:
            formatted_samples.append({'role_messages': json.dumps(formatted)})

    print(f"Successfully formatted {len(formatted_samples)} samples")

    # Reserve first 10 items as final test set
    test_samples = formatted_samples[:10]
    remaining_samples = formatted_samples[10:]

    print(f"Reserved {len(test_samples)} samples for final test set")

    # Create HuggingFace dataset and split remaining data
    remaining_dataset = Dataset.from_list(remaining_samples)
    split = remaining_dataset.train_test_split(test_size=train_test_split, seed=42)
    train_dataset = split['train']
    validation_dataset = split['test']

    # Create test dataset
    test_dataset = Dataset.from_list(test_samples)

    print(f"Training split: {len(train_dataset)} samples")
    print(f"Validation split: {len(validation_dataset)} samples")
    print(f"Test split: {len(test_dataset)} samples")

    # Show sample training examples
    show_training_examples(train_dataset, num_examples=3)

    # Build grain DataLoaders
    train_loader = _build_data_loader(
        data_source=train_dataset,
        batch_size=batch_size,
        num_epochs=num_epochs,
        max_seq_len=max_target_length,
        tokenizer=tokenizer,
        shuffle=True
    )

    validation_loader = _build_data_loader(
        data_source=validation_dataset,
        batch_size=batch_size,
        num_epochs=1,
        max_seq_len=max_target_length,
        tokenizer=tokenizer,
        shuffle=False
    )

    # Build test loader (use smaller batch size to avoid OOM during evaluation)
    test_loader = _build_data_loader(
        data_source=test_dataset,
        batch_size=1,  # Use batch size of 1 for test set to avoid OOM
        num_epochs=1,
        max_seq_len=max_target_length,
        tokenizer=tokenizer,
        shuffle=False
    )

    # Calculate steps
    num_train_examples = len(train_dataset)
    steps_per_epoch = num_train_examples // batch_size
    total_steps = steps_per_epoch * num_epochs
    if max_steps is None:
        max_steps = total_steps

    print(f"\nDataset statistics:")
    print(f"  Training examples: {num_train_examples}")
    print(f"  Validation examples: {len(validation_dataset)}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total steps ({num_epochs} epochs): {total_steps}")
    print(f"  Max steps: {max_steps}")

    # Initialize model
    print(f"\n{'='*60}")
    print("Loading base model")
    print(f"{'='*60}")

    # Determine model config based on model name
    if "0.6" in model_id or "0_6" in model_id:
        model_config = qwen_lib.ModelConfig.qwen3_0_6b()
    elif "1.7" in model_id or "1_7" in model_id:
        model_config = qwen_lib.ModelConfig.qwen3_1_7b()
    elif "8" in model_id:
        model_config = qwen_lib.ModelConfig.qwen3_8b()
    elif "14" in model_id:
        model_config = qwen_lib.ModelConfig.qwen3_14b()
    elif "30" in model_id:
        model_config = qwen_lib.ModelConfig.qwen3_30b()
    else:
        raise ValueError(f"Unsupported model ID: {model_id}")

    mesh = jax.make_mesh(MESH_SHAPE, MESH_AXIS_NAMES)
    with mesh:
        base_model = params_lib.create_model_from_safe_tensors(
            local_model_path, model_config, mesh
        )
        print("Base model loaded successfully")

    # Apply LoRA
    lora_model = create_lora_model(base_model, mesh=mesh, rank=lora_rank, alpha=lora_alpha)

    # Test base model before training
    print(f"\n{'='*60}")
    print("Testing base model BEFORE training")
    print(f"{'='*60}")
    test_model_generation(base_model, hf_tokenizer, model_config, eos_tokens, tools,
                         label="Base Model (Before Training)")

    # Setup training
    print(f"\n{'='*60}")
    print("Setting up training")
    print(f"{'='*60}")

    logging_options = metrics_logger.MetricsLoggerOptions(
        log_dir=os.path.join(ckpt_dir, "tensorboard"),
        flush_every_n_steps=20
    )

    training_config = peft_trainer.TrainingConfig(
        eval_every_n_steps=eval_every_n_steps,
        max_steps=max_steps,
        metrics_logging_options=logging_options,
        checkpoint_root_directory=ckpt_dir,
    )

    print(f"Training for {max_steps:,} steps total")

    warmup_steps = int(0.05 * max_steps)
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=max_steps - warmup_steps,
        end_value=learning_rate * 0.1,
    )

    print(f"Learning rate schedule: warmup for {warmup_steps} steps, then cosine decay")

    optimizer = optax.MultiSteps(
        optax.adamw(learning_rate=lr_schedule),
        every_k_schedule=gradient_accumulation_steps
    )

    trainer = peft_trainer.PeftTrainer(
        lora_model,
        optimizer,
        training_config
    ).with_gen_model_input_fn(lambda x: gen_model_input_fn(x, tokenizer))

    # Train!
    print(f"\n{'='*60}")
    print("Starting training")
    print(f"{'='*60}")
    print("This may take several minutes for the first step...")
    print(f"TensorBoard logs will be located at: {os.path.join(ckpt_dir, 'tensorboard')}")
    print(f"\nTo view training metrics, run:")
    print(f"  tensorboard --logdir {os.path.join(ckpt_dir, 'tensorboard')}")

    with mesh:
        trainer.train(train_loader, validation_loader)
    wandb.init()

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}\n")

    # Extract and save final metrics for hyperparameter search
    print("\nExtracting metrics from trainer...")
    train_metrics = trainer.metrics_logger.get_metric_history('loss', Mode.TRAIN)
    eval_metrics = trainer.metrics_logger.get_metric_history('loss', Mode.EVAL)
    final_metrics = {
        "final_train_loss": float(train_metrics[-1]),
        "final_val_loss": float(eval_metrics[-1]),
        "best_val_loss": float(min(eval_metrics)),
        "total_steps": trainer.train_steps,
    }
    
    # Save metrics to JSON file for hyperparameter search
    metrics_file = os.path.join(ckpt_dir, "metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(final_metrics, f, indent=2)
    print(f"\nSaved training metrics to: {metrics_file}")

    # Print final summary
    print(f"\n{'─'*60}")
    print("FINAL METRICS")
    print(f"{'─'*60}")
    if final_metrics["final_train_loss"] is not None:
        print(f"Final training loss:    {final_metrics['final_train_loss']:.4f}")
    if final_metrics["final_val_loss"] is not None:
        print(f"Final validation loss:  {final_metrics['final_val_loss']:.4f}")
    if final_metrics["best_val_loss"] is not None:
        print(f"Best validation loss:   {final_metrics['best_val_loss']:.4f}")
    print(f"Total steps:            {final_metrics['total_steps']}")
    print(f"{'─'*60}")

    # Test trained model
    print(f"\n{'='*60}")
    print("Testing trained model AFTER training")
    print(f"{'='*60}")
    test_model_generation(lora_model, hf_tokenizer, model_config, eos_tokens, tools,
                          label="Trained Model (After Training)")

    # Run final test set evaluation
    print(f"\n{'='*60}")
    print("FINAL TEST SET EVALUATION")
    print(f"{'='*60}")

    # 1. Generate outputs on test set
    test_generation_file = os.path.join(ckpt_dir, "test_set_generation.txt")
    run_test_set_generation(
        lora_model, hf_tokenizer, model_config, eos_tokens,
        test_dataset, tools, test_generation_file
    )

    # Clear JAX compilation cache to free memory before perplexity computation
    print("\nClearing JAX cache to free memory...")
    jax.clear_caches()

    # 2. Compute perplexity on test set using trainer
    test_perplexity_metrics = compute_test_set_perplexity(
        trainer, test_loader
    )

    # Save perplexity metrics to JSON
    test_perplexity_file = os.path.join(ckpt_dir, "test_set_perplexity.json")
    with open(test_perplexity_file, 'w') as f:
        json.dump(test_perplexity_metrics, f, indent=2)
    print(f"\nSaved test perplexity metrics to: {test_perplexity_file}")

    # Save LoRA weights merged with base model
    saved_path = save_lora_weights(lora_model, local_model_path, lora_output_dir, rank=RANK, alpha=ALPHA)

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Model saved to: {saved_path}")
    print(f"Training checkpoints: {ckpt_dir}")
    print(f"Test generation results: {test_generation_file}")
    print(f"Test perplexity metrics: {test_perplexity_file}")
    print(f"\nThe model is ready to use for inference!")
    print(f"Load it from: {saved_path}")


if __name__ == '__main__':
    main()
