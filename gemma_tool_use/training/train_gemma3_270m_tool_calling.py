#!/usr/bin/env python3
"""
Train Gemma 3 270M on tool calling using LoRA.

Based on tuning/lora_gemma.ipynb, this script fine-tunes Gemma 3 270M for tool calling
using the Toucan-1.5M dataset, filtered for:
- Single-turn conversations only
- ≤2 tools used per sample
- ≤3 tools available in prompt

The tool calling format follows gemma_tool_use/PLAN.md and is compatible with
the evaluation format in gemma_fc.py.
"""

import os
import json
import logging
import jax
import jax.numpy as jnp
from typing import Dict, List, Any
from datasets import load_dataset
from flax import nnx
from huggingface_hub import snapshot_download
import optax
import qwix

# Import tunix libraries
from tunix.generate import tokenizer_adapter as tokenizer_lib
from tunix.models.gemma3 import model as gemma_lib
from tunix.models.gemma3 import params_safetensors as params_safetensors_lib
from tunix.sft import metrics_logger
from tunix.sft import peft_trainer
from tunix.sft import utils

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ============================================================================
# Configuration
# ============================================================================

# Model configuration
MODEL_ID = "google/gemma-3-270m-it"
GEMMA_TOKENIZER_PATH = "gs://gemma-data/tokenizers/tokenizer_gemma3.model"

# Training hyperparameters
BATCH_SIZE = 16  # Adjust based on your TPU/GPU memory
NUM_EPOCHS = 3
LEARNING_RATE = 1e-3
MAX_TARGET_LENGTH = 512
MAX_STEPS = 1000
EVAL_EVERY_N_STEPS = 100

# LoRA hyperparameters
RANK = 16
ALPHA = 16

# TPU/GPU mesh configuration
NUM_DEVICES = len(jax.devices())
if NUM_DEVICES == 8:
    MESH_COUNTS = (1, 4)
elif NUM_DEVICES == 1:
    MESH_COUNTS = (1, 1)
else:
    MESH_COUNTS = (1, NUM_DEVICES)

MESH = [
    MESH_COUNTS,
    ("fsdp", "tp"),
]

# Dataset filtering criteria (as per PLAN.md Phase 3)
MAX_TOOLS_USED = 2
MAX_TOOLS_AVAILABLE = 3

# Checkpoint and output directories
CKPT_DIR = "/tmp/gemma_tool_calling_ckpts/"
PROFILING_DIR = "/tmp/gemma_tool_calling_profiling/"
LORA_OUTPUT_DIR = "./gemma3_270m_tool_calling_lora"


# ============================================================================
# Tool Calling Format Functions
# ============================================================================

def format_tools_for_prompt(tools: List[Dict[str, Any]]) -> str:
    """
    Format tools list for Gemma 3 prompt following PLAN.md format.

    Args:
        tools: List of tool dictionaries with 'name', 'description', 'parameters'

    Returns:
        Formatted tools string with XML tags
    """
    return f"<tools>\n{json.dumps(tools, indent=2)}\n</tools>"


def format_gemma3_tool_calling_example(sample: Dict[str, Any]) -> Dict[str, str]:
    """
    Format a Toucan dataset sample into Gemma 3 tool calling format.

    Following the format from gemma_tool_use/PLAN.md:
    - Tools are wrapped in <tools></tools> tags
    - Tool calls are wrapped in <tool_call></tool_call> tags
    - Format: {"name": "<function-name>", "args": {...}}
    - System instructions and tools are prepended to first user message

    Args:
        sample: A Toucan dataset sample with 'messages', 'tools', 'target_tools'

    Returns:
        Dictionary with 'text' (full input+output for SFT)
    """
    messages = json.loads(sample['messages'])
    tools = json.loads(sample['tools'])

    # Build the prompt following Gemma 3 chat template
    full_text = "<start_of_turn>user\n"

    # Extract system message if present (prepend to first user message)
    system_content = ""
    if len(messages) > 0 and messages[0]['role'] == 'system':
        system_content = messages[0]['content']
        messages = messages[1:]

    # Find the first user message and assistant response
    user_message = None
    assistant_message = None

    for msg in messages:
        if msg['role'] == 'user' and user_message is None:
            user_message = msg['content']
        elif msg['role'] == 'assistant' and assistant_message is None:
            assistant_message = msg
            break

    if user_message is None:
        raise ValueError("No user message found in sample")

    # Add system instructions if present
    if system_content:
        full_text += system_content + "\n\n"

    # Add tools definition
    full_text += "Here are the available tools that you can use:\n"
    full_text += format_tools_for_prompt(tools)
    full_text += "\n\n"

    # Add user query
    full_text += user_message
    full_text += "<end_of_turn>\n"

    # Build the completion (assistant response with tool calls)
    full_text += "<start_of_turn>model\n"

    if assistant_message:
        # Add any text content before tool calls
        if 'content' in assistant_message and assistant_message['content']:
            full_text += assistant_message['content'] + "\n"

        # Add tool calls
        if 'tool_calls' in assistant_message and assistant_message['tool_calls']:
            for tool_call in assistant_message['tool_calls']:
                full_text += '<tool_call>\n'
                # Handle both formats: with 'function' wrapper or direct
                if 'function' in tool_call:
                    call_data = {
                        "name": tool_call['function']['name'],
                        "args": tool_call['function'].get('arguments', {})
                    }
                else:
                    call_data = {
                        "name": tool_call['name'],
                        "args": tool_call.get('arguments', tool_call.get('args', {}))
                    }
                full_text += json.dumps(call_data)
                full_text += '\n</tool_call>\n'

    full_text += "<end_of_turn>"

    return {'text': full_text}


# ============================================================================
# Dataset Loading and Filtering
# ============================================================================

def filter_toucan_dataset(dataset, max_tools_used=2, max_tools_available=3):
    """
    Filter Toucan dataset for single-turn examples with limited tools.

    Args:
        dataset: Toucan dataset (SFT split)
        max_tools_used: Maximum number of tools used in target (default: 2)
        max_tools_available: Maximum number of tools available in prompt (default: 3)

    Returns:
        Filtered dataset
    """
    print(f"\nFiltering dataset:")
    print(f"  - Single-turn only")
    print(f"  - ≤{max_tools_used} tools used")
    print(f"  - ≤{max_tools_available} tools available")

    filtered_indices = []
    total = len(dataset)

    for idx in range(total):
        if idx % 10000 == 0:
            print(f"  Processed {idx:,}/{total:,} samples...")

        sample = dataset[idx]

        # Check single-turn
        messages = json.loads(sample['messages'])
        num_turns = len([m for m in messages if m['role'] == 'user'])
        if num_turns != 1:
            continue

        # Check tools used
        target_tools_str = sample['target_tools'].strip()
        if target_tools_str:
            target_tools_list = [t.strip() for t in target_tools_str.split(',')]
            num_target_tools = len(target_tools_list)
        else:
            num_target_tools = 0

        if num_target_tools > max_tools_used or num_target_tools == 0:
            continue

        # Check tools available
        tools = json.loads(sample['tools'])
        num_available_tools = len(tools)

        if num_available_tools > max_tools_available or num_available_tools == 0:
            continue

        filtered_indices.append(idx)

    print(f"  Processed {total:,}/{total:,} samples...")
    print(f"\nFiltered dataset size: {len(filtered_indices):,} samples")
    print(f"Percentage retained: {100 * len(filtered_indices) / total:.2f}%")

    return dataset.select(filtered_indices)


def create_tool_calling_dataset(tokenizer, global_batch_size, max_target_length, num_train_epochs):
    """
    Create and format the tool calling dataset.

    Args:
        tokenizer: Gemma tokenizer
        global_batch_size: Batch size for training
        max_target_length: Maximum sequence length
        num_train_epochs: Number of training epochs

    Returns:
        Tuple of (train_ds, validation_ds) as grain DataLoaders
    """
    print(f"\n{'='*60}")
    print("Loading Toucan-1.5M dataset")
    print(f"{'='*60}")

    # Load Toucan SFT dataset
    dataset = load_dataset('Agent-Ark/Toucan-1.5M', 'SFT', split='train')

    # Filter dataset
    filtered_dataset = filter_toucan_dataset(
        dataset,
        max_tools_used=MAX_TOOLS_USED,
        max_tools_available=MAX_TOOLS_AVAILABLE
    )

    # Split into train/validation (90/10 split)
    total_size = len(filtered_dataset)
    train_size = int(0.9 * total_size)

    train_dataset = filtered_dataset.select(range(train_size))
    validation_dataset = filtered_dataset.select(range(train_size, total_size))

    print(f"\nTrain dataset size: {len(train_dataset):,} samples")
    print(f"Validation dataset size: {len(validation_dataset):,} samples")

    # Format examples
    print("Formatting examples for Gemma 3 tool calling...")

    def format_function(examples):
        """Format a batch of examples"""
        texts = []

        for i in range(len(examples['messages'])):
            sample = {
                'messages': examples['messages'][i],
                'tools': examples['tools'][i],
                'target_tools': examples['target_tools'][i]
            }

            try:
                formatted = format_gemma3_tool_calling_example(sample)
                texts.append(formatted['text'])
            except Exception as e:
                logger.warning(f"Failed to format example {i}: {e}")
                texts.append("")

        return {'text': texts}

    train_dataset = train_dataset.map(
        format_function,
        batched=True,
        batch_size=1000,
        remove_columns=train_dataset.column_names
    )
    validation_dataset = validation_dataset.map(
        format_function,
        batched=True,
        batch_size=1000,
        remove_columns=validation_dataset.column_names
    )

    # Remove any empty examples that failed formatting
    train_dataset = train_dataset.filter(lambda x: len(x['text']) > 0)
    validation_dataset = validation_dataset.filter(lambda x: len(x['text']) > 0)

    print(f"Formatted {len(train_dataset):,} training examples")
    print(f"Formatted {len(validation_dataset):,} validation examples")

    # Convert to grain DataLoaders
    def tokenize_fn(example):
        """Tokenize a single example"""
        tokens = tokenizer.encode(example['text'])
        # Truncate to max_target_length
        tokens = tokens[:max_target_length]
        return tokens

    # Create grain datasets
    train_examples = list(train_dataset)
    validation_examples = list(validation_dataset)

    # Use grain's DataLoader-like functionality
    def create_grain_dataloader(examples, is_train=True):
        """Create a grain dataloader from examples"""
        # Simple implementation - in production, use grain's full capabilities
        def data_generator():
            import numpy as np
            indices = list(range(len(examples)))
            if is_train:
                np.random.shuffle(indices)

            for idx in indices:
                tokens = tokenize_fn(examples[idx])
                # Pad to max_target_length
                if len(tokens) < max_target_length:
                    tokens = tokens + [tokenizer.pad_id()] * (max_target_length - len(tokens))

                # Create input and target
                input_tokens = jnp.array(tokens[:-1], dtype=jnp.int32)
                target_tokens = jnp.array(tokens[1:], dtype=jnp.int32)

                # Create mask (1 for real tokens, 0 for padding)
                input_mask = (input_tokens != tokenizer.pad_id()).astype(jnp.float32)

                yield peft_trainer.TrainingInput(
                    input_tokens=input_tokens,
                    target_tokens=target_tokens,
                    input_mask=input_mask
                )

        return data_generator

    train_gen = create_grain_dataloader(train_examples, is_train=True)
    validation_gen = create_grain_dataloader(validation_examples, is_train=False)

    # Calculate steps
    num_train_examples = len(train_examples)
    steps_per_epoch = num_train_examples // global_batch_size
    total_steps = steps_per_epoch * num_train_epochs

    print(f"\nDataset statistics:")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total steps: {total_steps}")

    return train_gen, validation_gen


# ============================================================================
# Model and Training Setup
# ============================================================================

def download_and_setup_model():
    """Download model from HuggingFace and setup tokenizer."""
    print(f"\n{'='*60}")
    print(f"Downloading {MODEL_ID} from HuggingFace")
    print(f"{'='*60}")

    ignore_patterns = ["*.pth"]  # Ignore PyTorch weights
    local_model_path = snapshot_download(
        repo_id=MODEL_ID,
        ignore_patterns=ignore_patterns
    )

    print(f"Model downloaded to: {local_model_path}")

    # Load generation config for EOS tokens
    generation_config_path = os.path.join(local_model_path, "generation_config.json")
    eos_tokens = []
    if os.path.exists(generation_config_path):
        with open(generation_config_path, "r") as f:
            generation_configs = json.load(f)
        eos_tokens = generation_configs.get("eos_token_id", [])
        print(f"Using EOS token IDs: {eos_tokens}")

    return local_model_path, eos_tokens


def create_lora_model(base_model, mesh):
    """
    Apply LoRA to the base model.

    Args:
        base_model: Base Gemma model
        mesh: JAX mesh for sharding

    Returns:
        LoRA model
    """
    print(f"\n{'='*60}")
    print("Applying LoRA to model")
    print(f"{'='*60}")
    print(f"  Rank: {RANK}")
    print(f"  Alpha: {ALPHA}")

    lora_provider = qwix.LoraProvider(
        module_path=".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj",
        rank=RANK,
        alpha=ALPHA,
    )

    model_input = base_model.get_model_input()
    lora_model = qwix.apply_lora_to_model(
        base_model, lora_provider, **model_input
    )

    with mesh:
        state = nnx.state(lora_model)
        pspecs = nnx.get_partition_spec(state)
        sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
        nnx.update(lora_model, sharded_state)

    return lora_model


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


def save_lora_weights(lora_model, output_dir):
    """
    Save LoRA adapter weights to disk.

    Args:
        lora_model: Trained LoRA model
        output_dir: Directory to save weights

    Returns:
        Path to saved weights directory
    """
    print(f"\n{'='*60}")
    print("Saving LoRA weights")
    print(f"{'='*60}")

    def path_to_str(qwix_path):
        """Convert qwix path to string."""
        return '.'.join([str(field) for field in qwix_path])

    # Extract LoRA layers
    lora_layers = {}
    for layer in lora_model.layers:
        down_proj_path = path_to_str(layer.mlp.down_proj.qwix_path)
        up_proj_path = path_to_str(layer.mlp.up_proj.qwix_path)
        lora_layers[down_proj_path] = (
            layer.mlp.down_proj.kernel_lora_a,
            layer.mlp.down_proj.kernel_lora_b
        )
        lora_layers[up_proj_path] = (
            layer.mlp.up_proj.kernel_lora_a,
            layer.mlp.up_proj.kernel_lora_b
        )

    print(f"Found {len(lora_layers)} LoRA layer pairs")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving to: {output_dir}")

    # Save each LoRA layer
    for name, (lora_a, lora_b) in lora_layers.items():
        a_filename = os.path.join(output_dir, f"model.{name}.lora_a.npy")
        b_filename = os.path.join(output_dir, f"model.{name}.lora_b.npy")

        jnp.save(a_filename, lora_a.value.astype(jnp.float32))
        jnp.save(b_filename, lora_b.value.astype(jnp.float32))

    print(f"Successfully saved {len(lora_layers) * 2} weight files")
    return output_dir


# ============================================================================
# Main Training Script
# ============================================================================

def main():
    """Main training function."""
    print("="*60)
    print("Gemma 3 270M Tool Calling Training Script")
    print("="*60)
    print(f"Model: {MODEL_ID}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Max steps: {MAX_STEPS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"LoRA rank: {RANK}, alpha: {ALPHA}")
    print(f"Devices: {NUM_DEVICES} ({jax.devices()[0].platform})")

    # Create checkpoint directories
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(PROFILING_DIR, exist_ok=True)
    print(f"Checkpoint directory: {CKPT_DIR}")

    # Download model
    local_model_path, eos_tokens = download_and_setup_model()

    # Initialize tokenizer
    print(f"\n{'='*60}")
    print("Initializing tokenizer")
    print(f"{'='*60}")
    tokenizer = tokenizer_lib.Tokenizer(tokenizer_path=GEMMA_TOKENIZER_PATH)
    if tokenizer.eos_id() not in eos_tokens:
        eos_tokens.append(tokenizer.eos_id())
        print(f"Updated EOS token IDs: {eos_tokens}")

    # Create datasets
    train_gen, validation_gen = create_tool_calling_dataset(
        tokenizer,
        global_batch_size=BATCH_SIZE,
        max_target_length=MAX_TARGET_LENGTH,
        num_train_epochs=NUM_EPOCHS
    )

    # Initialize model
    print(f"\n{'='*60}")
    print("Loading base model")
    print(f"{'='*60}")

    model_config = gemma_lib.ModelConfig.gemma3_270m()
    mesh = jax.make_mesh(*MESH)

    with mesh:
        base_model = params_safetensors_lib.create_model_from_safe_tensors(
            local_model_path, (model_config), mesh
        )
        print("Base model loaded successfully")

    # Apply LoRA
    lora_model = create_lora_model(base_model, mesh=mesh)

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
        max_steps=MAX_STEPS,
        metrics_logging_options=logging_options,
        checkpoint_root_directory=CKPT_DIR,
    )

    trainer = peft_trainer.PeftTrainer(
        lora_model,
        optax.adamw(LEARNING_RATE),
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

    with jax.profiler.trace(os.path.join(PROFILING_DIR, "tool_calling_lora")):
        with mesh:
            trainer.train(train_gen, validation_gen)

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")

    # Save LoRA weights
    saved_path = save_lora_weights(lora_model, LORA_OUTPUT_DIR)

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"LoRA weights saved to: {saved_path}")
    print(f"Training checkpoints: {CKPT_DIR}")
    print(f"\nTo use the trained model, load the base Gemma 3 270M model")
    print(f"and apply these LoRA weights from: {saved_path}")


if __name__ == '__main__':
    main()
