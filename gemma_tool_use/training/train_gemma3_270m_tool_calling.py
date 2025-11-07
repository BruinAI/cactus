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

Optimized for 4x TPU v5e chips.
"""

import os
import json
import logging
import shutil
import jax
import jax.numpy as jnp
from typing import Dict, List, Any
from datasets import load_dataset
from flax import nnx
from huggingface_hub import snapshot_download
import optax
import qwix
import grain.python as grain
import numpy as np
from tqdm import tqdm
from safetensors import numpy as safe_np
import wandb

# Import tunix libraries
from tunix.generate import sampler as sampler_lib
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
# Optimized for TPU v5e-4 (even with 8, only 4 will be used)
BATCH_SIZE = 32
NUM_EPOCHS = 1
LEARNING_RATE = 1e-4
MAX_TARGET_LENGTH = 512
MAX_STEPS = None
EVAL_EVERY_N_STEPS = 250

# LoRA hyperparameters
RANK = 32
ALPHA = 16.0

# TPU/GPU mesh configuration
# Optimized for 4x TPU v5e
NUM_DEVICES = len(jax.devices())
if NUM_DEVICES == 8:
    # 8 TPUs: use both FSDP and tensor parallelism
    MESH_COUNTS = (1, 4)
elif NUM_DEVICES == 1:
    MESH_COUNTS = (1, 1)
else:
    raise ValueError(f"Unsupported number of devices: {NUM_DEVICES}")

MESH = [
    MESH_COUNTS,
    ("fsdp", "tp"),
]

# Dataset filtering criteria (as per PLAN.md Phase 3)
MAX_TOOLS_USED = 2
MAX_TOOLS_AVAILABLE = 3

# Checkpoint and output directories
CKPT_DIR = "/tmp/gemma_tool_calling_ckpts/"
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
    print(f"\nFiltering dataset (single-turn, ≤{max_tools_used} tools used, ≤{max_tools_available} tools available)...")

    filtered_indices = []
    total = len(dataset)

    def has_tool_response(m):
        # Check if message is a tool response
        # In Toucan dataset, tool responses can be:
        # 1. role: "tool"
        # 2. role: "user" with tool_call_id or tool results
        if m.get('role') == 'tool':
            return True
        # Check for tool_call_id which indicates this is a tool response
        if 'tool_call_id' in m:
            return True
        # As a fallback, check content for tool response markers
        content = m.get('content', '')
        if isinstance(content, str) and any(marker in content for marker in ['<tool_response>', '"tool_call_id"', '"result":']):
            return True
        return False

    for idx in tqdm(range(total), desc="Filtering samples", unit="sample"):
        sample = dataset[idx]

        messages = json.loads(sample['messages'])
        import ipdb; ipdb.set_trace()

        # Count user queries (not tool responses)
        # Single-turn means one initial user query (tool responses don't count as turns)
        user_messages = [m for m in messages if m['role'] == 'user']
        if len(user_messages) == 0 or len(user_messages) > 2:
            continue
        
        assert not has_tool_response(messages[0]), "First message cannot be a tool response"
        assistant_messages = [m for m in messages if m['role'] == 'assistant']
        if not assistant_messages:
            continue

        assert len(assistant_messages) == len(user_messages), "Number of assistant messages must match user messages"

        if len(user_messages) == 2:
            # Positive example: should have tool response and tool calls
            if not has_tool_response(messages[1]):
                continue

            num_tool_calls = len(assistant_messages[0].get('tool_calls', []))
            if num_tool_calls == 0 or num_tool_calls > max_tools_used:
                continue
        else:
            # Negative example: should NOT have tool calls
            num_tool_calls = len(assistant_messages[0].get('tool_calls', []))
            if num_tool_calls > 0:
                continue

        # Check tools available
        tools = json.loads(sample['tools'])
        num_available_tools = len(tools)

        if num_available_tools > max_tools_available:
            continue

        filtered_indices.append(idx)

    print(f"Filtered dataset size: {len(filtered_indices):,} samples ({100 * len(filtered_indices) / total:.2f}% retained)")

    return dataset.select(filtered_indices)


# ============================================================================
# Grain DataLoader Implementation
# ============================================================================

class _Tokenize(grain.MapTransform):
    """Tokenize formatted text examples."""

    def __init__(self, tokenizer: tokenizer_lib.Tokenizer):
        self._tokenizer = tokenizer

    def map(self, element: Dict[str, str]) -> np.ndarray:
        """Tokenize the text field."""
        tokens = self._tokenizer.encode(element['text'])
        return np.array(tokens, dtype=np.int32)


class _BuildTrainInput(grain.MapTransform):
    """Build TrainingInput from tokens."""

    def __init__(self, max_seq_len: int, pad_value: int):
        self._max_seq_len = max_seq_len
        self._pad_value = pad_value

    def map(self, tokens: np.ndarray) -> peft_trainer.TrainingInput:
        """Build training input from tokens."""
        # Pad or truncate to max_seq_len
        if len(tokens) > self._max_seq_len:
            tokens = tokens[:self._max_seq_len]
        else:
            pad_len = self._max_seq_len - len(tokens)
            tokens = np.pad(tokens, [[0, pad_len]], mode='constant', constant_values=self._pad_value)

        # Create mask (1 for real tokens, 0 for padding)
        mask = (tokens != self._pad_value).astype(np.float32)

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
    # Create sampler
    sampler = grain.IndexSampler(
        num_records=len(data_source),
        num_epochs=num_epochs,
        shard_options=grain.NoSharding(),
        shuffle=shuffle,
        seed=seed if shuffle else None,
    )

    # Create data loader with transformations
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


def create_tool_calling_dataset(tokenizer, global_batch_size, max_target_length, num_train_epochs):
    """
    Create and format the tool calling dataset.

    Args:
        tokenizer: Gemma tokenizer
        global_batch_size: Batch size for training
        max_target_length: Maximum sequence length
        num_train_epochs: Number of training epochs

    Returns:
        Tuple of (train_loader, validation_loader, total_steps, train_dataset)
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

    # Build grain DataLoaders (HuggingFace Dataset objects work as grain data sources)
    train_loader = _build_data_loader(
        data_source=train_dataset,
        batch_size=global_batch_size,
        num_epochs=num_train_epochs,
        max_seq_len=max_target_length,
        tokenizer=tokenizer,
        shuffle=True
    )

    validation_loader = _build_data_loader(
        data_source=validation_dataset,
        batch_size=global_batch_size,
        num_epochs=1,  # Validation only runs once per eval
        max_seq_len=max_target_length,
        tokenizer=tokenizer,
        shuffle=False
    )

    # Calculate steps
    num_train_examples = len(train_dataset)
    steps_per_epoch = num_train_examples // global_batch_size
    total_steps = steps_per_epoch * num_train_epochs

    print(f"\nDataset statistics:")
    print(f"  Training examples: {num_train_examples:,}")
    print(f"  Validation examples: {len(validation_dataset):,}")
    print(f"  Steps per epoch: {steps_per_epoch:,}")
    print(f"  Total steps ({num_train_epochs} epochs): {total_steps:,}")
    print(f"  Effective batch size: {global_batch_size}")

    return train_loader, validation_loader, total_steps, train_dataset


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


def save_lora_weights(lora_model, local_model_path, output_dir):
    """
    Save LoRA weights merged with base model as safetensors.

    Args:
        lora_model: Trained LoRA model
        local_model_path: Path to base model (for loading base weights)
        output_dir: Directory to save merged weights

    Returns:
        Path to saved weights directory
    """
    print(f"\n{'='*60}")
    print("Saving model with merged LoRA weights")
    print(f"{'='*60}")

    # Remove and recreate output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    print(f"Saving to: {output_dir}")

    print("\nStep 1: Extracting LoRA weights from lora_model...")

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

    print(f"Found {len(lora_layers)} LoRA layers")
    print(f"LoRA layer names: {list(lora_layers.keys())[:3]}...")

    # Load base model state
    print("\nStep 2: Loading base model weights...")
    base_state = safe_np.load_file(local_model_path + "/model.safetensors")
    print(f"Loaded {len(base_state)} base model parameters")

    # Step 3: Apply LoRA deltas to base weights
    print("\nStep 3: Merging LoRA deltas with base weights...")
    for lora_name, (lora_a, lora_b) in lora_layers.items():
        state_key = f'model.{lora_name}.weight'
        assert state_key in base_state, \
               f"LoRA layer {lora_name} not found in base model state dict"

        lora_a_val = jnp.asarray(lora_a.value).astype(np.float32)
        lora_b_val = jnp.asarray(lora_b.value).astype(np.float32)

        combined_lora = lora_a_val @ lora_b_val
        base_state[state_key] = base_state[state_key] + combined_lora.T

    print(f"Merged {len(lora_layers)} LoRA layers into base weights")

    # Step 4: Save merged weights as safetensors
    print("\nStep 4: Saving as safetensors...")
    safetensors_path = os.path.join(output_dir, "model.safetensors")
    safe_np.save_file(base_state, safetensors_path)
    print("Model weights saved")

    # Step 5: Copy other model files
    print("\nStep 5: Copying other model files...")
    for filename in os.listdir(local_model_path):
        # Check if the file is NOT a safetensors file
        if not filename.endswith(".safetensors"):
            src = os.path.join(local_model_path, filename)
            dst = os.path.join(output_dir, filename)

            # Check if it's a file (and not a directory) before copying
            if os.path.isfile(src):
                shutil.copy(src, dst)
                print(f"  Copied {filename} from base model")

    print("\n" + "="*60)
    print("Model saved successfully!")
    print(f"Output directory: {output_dir}")
    print("="*60)

    print("\nSaved files:")
    for f in os.listdir(output_dir):
        size = os.path.getsize(os.path.join(output_dir, f)) / (1024 * 1024)
        print(f"  {f:<30} {size:>10.2f} MB")

    return output_dir


# ============================================================================
# Model Testing Functions
# ============================================================================

def show_training_examples(dataset, num_examples=5):
    """
    Display random training examples from the formatted dataset.

    Args:
        dataset: HuggingFace dataset with 'text' field
        num_examples: Number of examples to display (default: 5)
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
        print(text)

    print(f"\n{'='*60}\n")


def test_model_generation(model, tokenizer, model_config, eos_tokens, label="Model"):
    """
    Test the model with two examples:
    1. Simple tool calling (model requests weather)
    2. Tool response usage (model uses weather data to respond)
    """
    print(f"\n{'='*60}")
    print(f"{label} - Generation Examples")
    print(f"{'='*60}")

    # Create sampler
    sampler = sampler_lib.Sampler(
        transformer=model,
        tokenizer=tokenizer,
        cache_config=sampler_lib.CacheConfig(
            cache_size=512,  # Large enough for prompt + generation
            num_layers=model_config.num_layers,
            num_kv_heads=model_config.num_kv_heads,
            head_dim=model_config.head_dim,
        ),
    )

    # Example 1: Simple tool calling
    tools = [
        {
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
    ]

    prompt1 = f"""<start_of_turn>user
Here are the available tools that you can use:
<tools>
{json.dumps(tools, indent=2)}
</tools>

What's the weather in Boston?<end_of_turn>
<start_of_turn>model
"""

    print("\n--- Example 1: Tool Calling ---")
    print("Prompt: What's the weather in Boston?")
    print("\nModel response:")

    out_data1 = sampler(
        input_strings=[prompt1],
        max_generation_steps=128,
        eos_tokens=eos_tokens,
    )

    response1 = out_data1.text[0]
    # Extract just the model's response after the prompt
    if "<start_of_turn>model" in response1:
        response1 = response1.split("<start_of_turn>model")[-1]
    if "<end_of_turn>" in response1:
        response1 = response1.split("<end_of_turn>")[0]

    print(response1.strip())

    # Example 2: Using tool response
    prompt2 = f"""<start_of_turn>user
Here are the available tools that you can use:
<tools>
{json.dumps(tools, indent=2)}
</tools>

What's the weather in Boston?<end_of_turn>
<start_of_turn>model
<tool_call>
{{
  "name": "get_weather",
  "args": {{
    "location": "Boston, MA"
  }}
}}
</tool_call><end_of_turn>
<start_of_turn>user
<tool_response>
{{
  "name": "get_weather",
  "result": {{
    "temperature": 72,
    "condition": "Sunny"
  }}
}}
</tool_response><end_of_turn>
<start_of_turn>model
"""

    print("\n--- Example 2: Using Tool Response ---")
    print("Prompt: [After tool returns weather data]")
    print("\nModel response:")

    out_data2 = sampler(
        input_strings=[prompt2],
        max_generation_steps=64,
        eos_tokens=eos_tokens,
    )

    response2 = out_data2.text[0]
    # Extract just the model's response after the prompt
    if "<start_of_turn>model" in response2:
        response2 = response2.split("<start_of_turn>model")[-1]
    if "<end_of_turn>" in response2:
        response2 = response2.split("<end_of_turn>")[0]

    print(response2.strip())
    print()


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
    print(f"Devices: {NUM_DEVICES}x {jax.devices()[0].platform}")
    print(f"Mesh configuration: {MESH_COUNTS} (fsdp={MESH_COUNTS[0]}, tp={MESH_COUNTS[1]})")

    # Create checkpoint directories
    os.makedirs(CKPT_DIR, exist_ok=True)
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
    train_loader, validation_loader, total_steps, train_dataset = create_tool_calling_dataset(
        tokenizer,
        global_batch_size=BATCH_SIZE,
        max_target_length=MAX_TARGET_LENGTH,
        num_train_epochs=NUM_EPOCHS
    )

    # Show sample training examples
    show_training_examples(train_dataset, num_examples=5)

    # Use calculated total steps if MAX_STEPS not specified
    max_steps = MAX_STEPS if MAX_STEPS is not None else total_steps

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

    with mesh:
        trainer.train(train_loader, validation_loader)
    wandb.init()

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")

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
