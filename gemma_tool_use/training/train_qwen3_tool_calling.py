#!/usr/bin/env python3
"""
Train Qwen 3 on tool calling using LoRA with Noah's custom dataset.

This script fine-tunes Qwen 3 models for tool calling using Noah's custom dataset
which contains examples of creating notes, setting alarms, timers, and reminders.

The tool calling format uses JSON:
- Tools provided as JSON schemas
- Model outputs: JSON with function_call object containing name and arguments
- Compatible with standard function calling formats

Optimized for 4x TPU v5e chips.
"""

import os
import json
import logging
import nest_asyncio
nest_asyncio.apply()

import jax
import optax
import wandb
import numpy as np
from datasets import Dataset
import grain.python as grain
from transformers import AutoTokenizer

# Import tunix libraries
from tunix.generate import sampler as sampler_lib
from tunix.generate import tokenizer_adapter as tokenizer_lib
from tunix.models.qwen3 import model as qwen_lib
from tunix.models.qwen3 import params as params_lib
from tunix.sft import metrics_logger
from tunix.sft import peft_trainer
from tunix.sft import utils

# Import dataset formatting functions
from format_noah_dataset import (
    format_qwen3_noah_dataset,
    load_noah_tools,
    load_noah_dataset
)

# Import Qwen model utilities
from qwen_utils import download_and_setup_model, create_lora_model, save_lora_weights

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ============================================================================
# Configuration
# ============================================================================

# Model configuration
MODEL_ID = "Qwen/Qwen3-0.6B-Chat"  # Can also use Qwen3-1.7B, Qwen3-8B, etc.
QWEN_TOKENIZER_PATH = None  # Will be loaded from the model directory

# Dataset paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DATASET_PATH = os.path.join(DATA_DIR, "noah_finetune_dataset.json")
TOOLS_PATH = os.path.join(DATA_DIR, "noah_tools.json")

# Training hyperparameters
NUM_EPOCHS = 5
LEARNING_RATE = 5e-5
MAX_TARGET_LENGTH = 2048  # Noah's dataset has shorter sequences

# Allow max_steps override
MAX_STEPS = None

BATCH_SIZE = 8
EVAL_EVERY_N_STEPS = 5

# LoRA hyperparameters
RANK = 16
ALPHA = 32.0

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

class _Tokenize(grain.MapTransform):
    """Tokenize role-based messages and create proper masks."""

    def __init__(self, tokenizer: tokenizer_lib.Tokenizer):
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
    tokenizer: tokenizer_lib.Tokenizer,
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
    Test the model with examples from Noah's dataset domain.

    Tests:
    1. Creating a note
    2. Setting an alarm
    3. Setting a timer
    """
    print(f"\n{'='*60}")
    print(f"{label} - Generation Examples")
    print(f"{'='*60}")

    # Create sampler
    sampler = sampler_lib.Sampler(
        transformer=model,
        tokenizer=tokenizer,
        cache_config=sampler_lib.CacheConfig(
            cache_size=1024,
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
    formatted1 = format_qwen3_noah_dataset(sample1, tools, tokenizer)
    formatted2 = format_qwen3_noah_dataset(sample2, tools, tokenizer)
    formatted3 = format_qwen3_noah_dataset(sample3, tools, tokenizer)

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


# ============================================================================
# Main Training Script
# ============================================================================

def main():
    """Main training function."""
    print("="*60)
    print("Qwen 3 Tool Calling Training Script (Noah's Dataset)")
    print("="*60)
    print(f"Model: {MODEL_ID}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Tools: {TOOLS_PATH}")
    print(f"Global batch size: {BATCH_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"LoRA rank: {RANK}, alpha: {ALPHA}")
    print(f"Max sequence length: {MAX_TARGET_LENGTH}")
    print(f"Devices: {len(jax.devices())} x {jax.devices()[0].platform}")
    print(f"Mesh configuration: {MESH_SHAPE} x {MESH_AXIS_NAMES}")

    # Create checkpoint directories
    os.makedirs(CKPT_DIR, exist_ok=True)
    print(f"Checkpoint directory: {CKPT_DIR}")

    # Download model
    local_model_path, eos_tokens = download_and_setup_model(MODEL_ID)

    # Initialize tokenizers
    print(f"\n{'='*60}")
    print("Initializing tokenizers")
    print(f"{'='*60}")
    # tunix tokenizer for training data loading
    tokenizer_path = os.path.join(local_model_path, "tokenizer.json")
    tokenizer = tokenizer_lib.Tokenizer(tokenizer_path=tokenizer_path)
    if tokenizer.eos_id() not in eos_tokens:
        eos_tokens.append(tokenizer.eos_id())
        print(f"Updated EOS token IDs: {eos_tokens}")

    # HuggingFace tokenizer for generation (used by sampler)
    hf_tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    print(f"Loaded tokenizers from {local_model_path}")

    # Load tools and dataset
    print(f"\n{'='*60}")
    print("Loading Noah's dataset and tools")
    print(f"{'='*60}")
    tools = load_noah_tools(TOOLS_PATH)
    dataset_samples = load_noah_dataset(DATASET_PATH)
    print(f"Loaded {len(tools)} tools")
    print(f"Loaded {len(dataset_samples)} dataset samples")

    # Format dataset
    print("\nFormatting examples for Qwen 3 tool calling...")
    formatted_samples = []
    for sample in dataset_samples:
        formatted = format_qwen3_noah_dataset(sample, tools, hf_tokenizer)
        if formatted:
            formatted_samples.append({'role_messages': json.dumps(formatted)})

    print(f"Successfully formatted {len(formatted_samples)} samples")

    # Create HuggingFace dataset and split
    full_dataset = Dataset.from_list(formatted_samples)
    split = full_dataset.train_test_split(test_size=TRAIN_TEST_SPLIT, seed=42)
    train_dataset = split['train']
    validation_dataset = split['test']

    print(f"Training split: {len(train_dataset)} samples")
    print(f"Validation split: {len(validation_dataset)} samples")

    # Show sample training examples
    show_training_examples(train_dataset, num_examples=3)

    # Build grain DataLoaders
    train_loader = _build_data_loader(
        data_source=train_dataset,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        max_seq_len=MAX_TARGET_LENGTH,
        tokenizer=tokenizer,
        shuffle=True
    )

    validation_loader = _build_data_loader(
        data_source=validation_dataset,
        batch_size=BATCH_SIZE,
        num_epochs=1,
        max_seq_len=MAX_TARGET_LENGTH,
        tokenizer=tokenizer,
        shuffle=False
    )

    # Calculate steps
    num_train_examples = len(train_dataset)
    steps_per_epoch = num_train_examples // BATCH_SIZE
    total_steps = steps_per_epoch * NUM_EPOCHS
    max_steps = MAX_STEPS if MAX_STEPS is not None else total_steps

    print(f"\nDataset statistics:")
    print(f"  Training examples: {num_train_examples}")
    print(f"  Validation examples: {len(validation_dataset)}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total steps ({NUM_EPOCHS} epochs): {total_steps}")
    print(f"  Max steps: {max_steps}")

    # Initialize model
    print(f"\n{'='*60}")
    print("Loading base model")
    print(f"{'='*60}")

    # Determine model config based on model name
    if "0.6" in MODEL_ID or "0_6" in MODEL_ID:
        model_config = qwen_lib.ModelConfig.qwen3_0_6b()
    elif "1.7" in MODEL_ID or "1_7" in MODEL_ID:
        model_config = qwen_lib.ModelConfig.qwen3_1_7b()
    elif "8" in MODEL_ID:
        model_config = qwen_lib.ModelConfig.qwen3_8b()
    elif "14" in MODEL_ID:
        model_config = qwen_lib.ModelConfig.qwen3_14b()
    elif "30" in MODEL_ID:
        model_config = qwen_lib.ModelConfig.qwen3_30b()
    else:
        raise ValueError(f"Unsupported model ID: {MODEL_ID}")

    mesh = jax.make_mesh(MESH_SHAPE, MESH_AXIS_NAMES)
    with mesh:
        base_model = params_lib.create_model_from_safe_tensors(
            local_model_path, model_config, mesh
        )
        print("Base model loaded successfully")

    # Apply LoRA
    lora_model = create_lora_model(base_model, mesh=mesh, rank=RANK, alpha=ALPHA)

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
        log_dir=os.path.join(CKPT_DIR, "tensorboard"),
        flush_every_n_steps=20
    )

    training_config = peft_trainer.TrainingConfig(
        eval_every_n_steps=EVAL_EVERY_N_STEPS,
        max_steps=max_steps,
        metrics_logging_options=logging_options,
        checkpoint_root_directory=CKPT_DIR,
    )

    print(f"Training for {max_steps:,} steps total")

    warmup_steps = int(0.05 * max_steps)
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=LEARNING_RATE,
        warmup_steps=warmup_steps,
        decay_steps=max_steps - warmup_steps,
        end_value=LEARNING_RATE * 0.1,
    )

    print(f"Learning rate schedule: warmup for {warmup_steps} steps, then cosine decay")

    optimizer = optax.MultiSteps(
        optax.adamw(learning_rate=lr_schedule),
        every_k_schedule=GRADIENT_ACCUMULATION_STEPS
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
    print(f"TensorBoard logs will be located at: {os.path.join(CKPT_DIR, 'tensorboard')}")
    print(f"\nTo view training metrics, run:")
    print(f"  tensorboard --logdir {os.path.join(CKPT_DIR, 'tensorboard')}")

    with mesh:
        trainer.train(train_loader, validation_loader)
    wandb.init()

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}\n")

    # Test trained model
    print(f"\n{'='*60}")
    print("Testing trained model AFTER training")
    print(f"{'='*60}")
    test_model_generation(lora_model, hf_tokenizer, model_config, eos_tokens, tools,
                          label="Trained Model (After Training)")

    # Save LoRA weights merged with base model
    saved_path = save_lora_weights(lora_model, local_model_path, LORA_OUTPUT_DIR)

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Model saved to: {saved_path}")
    print(f"Training checkpoints: {CKPT_DIR}")
    print(f"\nThe model is ready to use for inference!")
    print(f"Load it from: {saved_path}")


if __name__ == '__main__':
    main()
