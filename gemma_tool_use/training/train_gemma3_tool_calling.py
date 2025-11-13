#!/usr/bin/env python3
"""
Train Gemma 3 270M on tool calling using LoRA.

Based on tuning/lora_gemma.ipynb, this script fine-tunes Gemma 3 270M for tool calling
using the Toucan-1.5M dataset, filtered for:
- Single-turn conversations only
- ≤2 tools used per sample
- ≤3 tools available in prompt

The tool calling format follows BFCL simple_python test style (see gemma_tool_use/BFCL_GEMMA_FORMAT.md):
- Tools provided as JSON schemas
- Model outputs: [func_name(param1=value1, param2=value2)]
- Tool responses as Python list of dicts
- Compatible with BFCL evaluation

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

# Import tunix libraries
from tunix.generate import sampler as sampler_lib
from tunix.generate import tokenizer_adapter as tokenizer_lib
from tunix.models.gemma3 import model as gemma_lib
from tunix.models.gemma3 import params_safetensors as params_safetensors_lib
from tunix.sft import metrics_logger
from tunix.sft import peft_trainer
from tunix.sft import utils

# Import BFCL-style formatting functions
from format_bfcl_style import format_gemma3_bfcl_style

# Import data utilities
from data_utils import create_tool_calling_dataset

# Import Gemma model utilities
from gemma_utils import download_and_setup_model, create_lora_model, save_lora_weights

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ============================================================================
# Configuration
# ============================================================================

# Model configuration
MODEL_ID = "google/gemma-3-1b-it"  # Choose either "google/gemma-3-270m-it" or "google/gemma-3-1b-it"
GEMMA_TOKENIZER_PATH = "gs://gemma-data/tokenizers/tokenizer_gemma3.model"

# Training hyperparameters
# Optimized for TPU v5e-4 (even with 8, only 4 will be used)
NUM_EPOCHS = 1
LEARNING_RATE = 2e-4  # Middle ground between 1e-4 (too high) and 5e-5 (too low)
MAX_GRAD_NORM = 1.0   # Keep gradient clipping to reduce oscillations
MAX_TARGET_LENGTH = 4096  # 95th percentile w/ max 1 turn= 4,086 tokens, w/ no max turns = 7,090
MAX_STEPS = None

BATCH_SIZE = 8
DESIRED_EFFECTIVE_BATCH_SIZE = 64  # Increased from 64 to reduce gradient variance and stabilize training
EVAL_EVERY_N_EFFECTIVE_BATCHES = 125  # Adjusted to maintain similar eval frequency (every ~1000 steps)

# LoRA hyperparameters, Choose either 64 or 32 for both
RANK = 32  # Restored to 64 (achieved better eval loss of 1.17 vs 1.22 with rank=32)
ALPHA = 64.0

# TPU/GPU mesh configuration
# Optimized for 4x TPU v5e
MESH_SHAPE = len(jax.devices()), 1  # Default to all devices in FSDP, no tensor parallelism
MESH_AXIS_NAMES = "fsdp", "tp"

# Dataset filtering criteria (as per PLAN.md Phase 3)
MAX_TOOLS_USED = 10
MAX_TOOLS_AVAILABLE = 10
MAX_NUMBER_OF_TURNS = float('inf')  # No limit on number of turns

# Checkpoint and output directories
CKPT_DIR = "/tmp/gemma_tool_calling_ckpts/"
LORA_OUTPUT_DIR = f"/dev/shm/{MODEL_ID.split('/')[-1]}_tool_calling_lora"

# NOTE: SYSTEM_PROMPT is now defined in format_bfcl_style.py as BFCL_SYSTEM_PROMPT
# The BFCL-style system prompt includes instructions for Python function call format

# Calculating derived hyperparameters
assert DESIRED_EFFECTIVE_BATCH_SIZE % BATCH_SIZE == 0
GRADIENT_ACCUMULATION_STEPS = DESIRED_EFFECTIVE_BATCH_SIZE // BATCH_SIZE
EVAL_EVERY_N_STEPS = GRADIENT_ACCUMULATION_STEPS * EVAL_EVERY_N_EFFECTIVE_BATCHES


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

def show_training_examples(dataset, num_examples=5):
    """
    Display random training examples from the formatted dataset.

    Args:
        dataset: HuggingFace dataset with 'role_messages' field
        num_examples: Number of examples to display (default: 5)
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

        # Reconstruct the formatted text for display
        for turn_idx, turn in enumerate(role_messages):
            role = turn['role']
            text = turn['text']
            mask_indicator = "✓ TRAIN" if role == 'model' else "✗ SKIP"

            print(f"\n<start_of_turn>{role} [{mask_indicator}]")
            print(text)
            print(f"<end_of_turn>")

    print(f"\n{'='*60}\n")


def test_model_generation(model, tokenizer, model_config, eos_tokens, label="Model"):
    """
    Test the model with two examples:
    1. Simple tool calling (model requests weather)
    2. Tool response usage (model uses weather data to respond)

    Uses format_gemma3_bfcl_style to ensure consistency with training format.
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

    # Define tools in the format expected by format_gemma3_bfcl_style
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]

    # Example 1: Simple tool calling - user asks, model should call tool
    sample1 = {
        'messages': json.dumps([
            {"role": "user", "content": "What's the weather in Boston?"}
        ]),
        'tools': json.dumps(tools),
        'target_tools': json.dumps([])
    }

    # Example 2: Tool response usage - user asks, model calls tool, gets response
    sample2 = {
        'messages': json.dumps([
            {"role": "user", "content": "What's the weather in Boston?"},
            {"role": "tool_call", "content": "{'name': 'get_weather', 'arguments': '{\"location\": \"Boston, MA\"}'}"},
            {"role": "tool_response", "name": "get_weather", "content": {"temperature": 72, "condition": "Sunny"}}
        ]),
        'tools': json.dumps(tools),
        'target_tools': json.dumps([])
    }

    # Format both examples using BFCL-style formatting (same as training)
    formatted1 = format_gemma3_bfcl_style(sample1)
    formatted2 = format_gemma3_bfcl_style(sample2)
    assert formatted1 and formatted2, "Failed to format test examples"

    prompt1 = ''.join(turn['text'] for turn in formatted1)
    prompt2 = ''.join(turn['text'] for turn in formatted2)
    assert prompt1 and prompt2, "Formatted prompts are empty"

    out_data = sampler(
        input_strings=[prompt1, prompt2],
        max_generation_steps=128,
        eos_tokens=eos_tokens,
        top_k=1,
        top_p=None,
        temperature=None,
    )

    print("\n--- Example 1: Tool Calling ---")
    print("Prompt:", prompt1)
    print("\nModel response:")
    print(out_data.text[0].strip())

    print("\n--- Example 2: Using Tool Response ---")
    print("Prompt:", prompt2)
    print("\nModel response:")
    print(out_data.text[1].strip())


# ============================================================================
# Main Training Script
# ============================================================================

def main():
    """Main training function."""
    print("="*60)
    print("Gemma 3 270M Tool Calling Training Script")
    print("="*60)
    print(f"Model: {MODEL_ID}")
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

    # Initialize tokenizer
    print(f"\n{'='*60}")
    print("Initializing tokenizer")
    print(f"{'='*60}")
    tokenizer = tokenizer_lib.Tokenizer(tokenizer_path=GEMMA_TOKENIZER_PATH)
    if tokenizer.eos_id() not in eos_tokens:
        eos_tokens.append(tokenizer.eos_id())
        print(f"Updated EOS token IDs: {eos_tokens}")

    # Create datasets
    def format_function(examples):
        """Format a batch of examples into role-based format using BFCL style"""
        role_messages_list = []

        for i in range(len(examples['messages'])):
            sample = {
                'messages': examples['messages'][i],
                'tools': examples['tools'][i],
                'target_tools': examples['target_tools'][i]
            }

            # Use BFCL-style formatting
            formatted = format_gemma3_bfcl_style(sample)
            if formatted:  # Returns list of {'role': ..., 'text': ...} dicts
                role_messages_list.append(json.dumps(formatted))

        return {'role_messages': role_messages_list}

    train_loader, validation_loader, total_steps, train_dataset = create_tool_calling_dataset(
        tokenizer=tokenizer,
        global_batch_size=BATCH_SIZE,
        max_target_length=MAX_TARGET_LENGTH,
        num_train_epochs=NUM_EPOCHS,
        max_tools_used=MAX_TOOLS_USED,
        max_tools_available=MAX_TOOLS_AVAILABLE,
        max_number_of_turns=MAX_NUMBER_OF_TURNS,
        format_function=format_function
    )

    # Show sample training examples
    show_training_examples(train_dataset, num_examples=5)

    # Use calculated total steps if MAX_STEPS not specified
    max_steps = MAX_STEPS if MAX_STEPS is not None else total_steps

    # Initialize model
    print(f"\n{'='*60}")
    print("Loading base model")
    print(f"{'='*60}")

    if "gemma-3-270m" in MODEL_ID:
        model_config = gemma_lib.ModelConfig.gemma3_270m()
    elif "gemma-3-1b" in MODEL_ID:
        model_config = gemma_lib.ModelConfig.gemma3_1b()
    else:
        raise ValueError(f"Unsupported model ID: {MODEL_ID}")
    
    mesh = jax.make_mesh(MESH_SHAPE, MESH_AXIS_NAMES)
    with mesh:
        base_model = params_safetensors_lib.create_model_from_safe_tensors(
            local_model_path, (model_config), mesh
        )
        print("Base model loaded successfully")

    # Apply LoRA
    lora_model = create_lora_model(base_model, mesh=mesh, rank=RANK, alpha=ALPHA)

    # Test base model before training
    print(f"\n{'='*60}")
    print("Testing base model BEFORE training")
    print(f"{'='*60}")
    test_model_generation(base_model, tokenizer, model_config, eos_tokens, label="Base Model (Before Training)")

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
        optax.chain(
            optax.clip_by_global_norm(MAX_GRAD_NORM),
            optax.adamw(learning_rate=lr_schedule)
        ),
        every_k_schedule=GRADIENT_ACCUMULATION_STEPS
    )
    print(f"Gradient clipping enabled with max_norm={MAX_GRAD_NORM}")
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
    test_model_generation(lora_model, tokenizer, model_config, eos_tokens, label="Trained Model (After Training)")

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
