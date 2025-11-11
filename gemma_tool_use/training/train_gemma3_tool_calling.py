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
import ast
import logging
import shutil
import nest_asyncio
nest_asyncio.apply()

import jax
import jax.numpy as jnp
from typing import Dict, List, Any, Optional
from datasets import load_dataset, Dataset
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
MODEL_ID = "google/gemma-3-1b-it"  # Choose either "google/gemma-3-270m-it" or "google/gemma-3-1b-it"
GEMMA_TOKENIZER_PATH = "gs://gemma-data/tokenizers/tokenizer_gemma3.model"

# Training hyperparameters
# Optimized for TPU v5e-4 (even with 8, only 4 will be used)
NUM_EPOCHS = 1
LEARNING_RATE = 2e-4  # Middle ground between 1e-4 (too high) and 5e-5 (too low)
MAX_GRAD_NORM = 1.0   # Keep gradient clipping to reduce oscillations
MAX_TARGET_LENGTH = 4096  # 95th percentile = 4,086 tokens
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

# Checkpoint and output directories
CKPT_DIR = "/tmp/gemma_tool_calling_ckpts/"
LORA_OUTPUT_DIR = f"/dev/shm/{MODEL_ID.split('/')[-1]}_tool_calling_lora"

SYSTEM_PROMPT = """At each turn, if you decide to invoke any of the function(s), it should be wrapped with ```tool_code```. \
The python methods described below are imported and available, you can only use defined methods. \
The generated code should be readable and efficient. \
The response to a method will be wrapped in ```tool_output``` use it to call more tools or generate a helpful, friendly response. \
When using a ```tool_call``` think step by step why and how it should be used.

The following Python methods are available:

"""

# Calculating derived hyperparameters
assert DESIRED_EFFECTIVE_BATCH_SIZE % BATCH_SIZE == 0
GRADIENT_ACCUMULATION_STEPS = DESIRED_EFFECTIVE_BATCH_SIZE // BATCH_SIZE
EVAL_EVERY_N_STEPS = GRADIENT_ACCUMULATION_STEPS * EVAL_EVERY_N_EFFECTIVE_BATCHES


# ============================================================================
# Tool Calling Format Functions
# ============================================================================
def correct_dict_type(tool: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure that the 'type' field in tool function parameters is 'dict'.
    """
    assert 'function' in tool
    function = tool['function']
    if 'parameters' in function:
        function['parameters']['type'] = 'dict'
    tool['function'] = function
    return tool

def format_tools_for_prompt_json_style(tools: List[Dict[str, Any]]) -> str:
    """
    Format tools list for Gemma 3 prompt following PLAN.md format (JSON style).
    """
    return '\n'.join(json.dumps(correct_dict_type(tool), indent=2) for tool in tools)


def format_tools_for_prompt(tools: List[Dict[str, Any]]) -> str:
    """
    Format tools list for Gemma 3 prompt following philschmid's Python function style.

    Converts JSON tool definitions to Python function signatures with docstrings.
    Following the format from https://www.philschmid.de/gemma-function-calling

    Args:
        tools: List of tool definitions in OpenAI function calling format

    Returns:
        Python code block with function signatures and docstrings
    """
    python_functions = []

    for tool in tools:
        assert 'function' in tool, "Tool must have 'function' key"
        func = tool['function']
        func_name = func['name']
        description = func.get('description', '')
        parameters = func.get('parameters', {})
        properties = parameters.get('properties', {})
        required = parameters.get('required', [])

        # Build function signature with type hints
        # Separate required and optional parameters
        required_params = []
        optional_params = []

        for param_name, param_info in properties.items():
            param_type = param_info.get('type', 'any')

            # Map JSON schema types to Python types
            type_map = {
                'string': 'str',
                'number': 'float',
                'integer': 'int',
                'boolean': 'bool',
                'array': 'list',
                'object': 'dict',
                'any': 'Any'
            }
            python_type = type_map.get(param_type, 'Any')

            # Check if parameter is required
            if param_name in required:
                required_params.append(f"{param_name}: {python_type}")
            else:
                optional_params.append(f"{param_name}: {python_type} = None")

        # Combine params: required first, then optional
        all_params = required_params + optional_params

        # Replace hyphens with underscores for valid Python identifiers
        func_name_safe = func_name.replace('-', '_')
        signature = f"def {func_name_safe}({', '.join(all_params)}):"

        # Build docstring
        docstring_lines = []
        if description:
            # Split description into lines and indent each line properly
            desc_lines = description.split('\n')
            docstring_lines.append(f'  """{desc_lines[0]}')
            for line in desc_lines[1:]:
                # Add proper indentation (2 spaces) to each line of the description
                if line.strip():  # Only indent non-empty lines
                    docstring_lines.append(f'  {line}')
                else:
                    docstring_lines.append('')
        else:
            docstring_lines.append('  """')

        # Add Args section if there are parameters
        if properties:
            docstring_lines.append('')
            docstring_lines.append('  Args:')
            for param_name, param_info in properties.items():
                param_desc = param_info.get('description', 'No description')
                # Mark required parameters
                required_marker = ' (required)' if param_name in required else ' (optional)'
                docstring_lines.append(f'    {param_name}: {param_desc}{required_marker}')

        docstring_lines.append('  """')

        # Combine signature and docstring
        function_def = signature + '\n' + '\n'.join(docstring_lines)
        python_functions.append(function_def)

    # Wrap all functions in a Python code block
    all_functions = '\n\n'.join(python_functions)
    return f'```python\n{all_functions}\n```'


def group_messages_by_turn(messages: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """
    Groups messages into turns based on their roles.
    Each user message is its own turn.
    - Each user message must follow assistant message and be first message
    Each assistant message + following consecutive tool calls are grouped together.
    - Assistant messages must follow user or tool call messages.
    - tool calls must follow assistant messages.
    Each set of consecutive tool responses are grouped together.
    - Tool responses must follow tool calls.

    Args:
        messages: List of message dictionaries with 'role' and 'content'

    Returns:
        List of message groups, where each group represents a turn
    """
    grouped_messages = []
    last_type = 'assistant'  # Initialize to assistant to allow first user message
    for msg in messages:
        role = msg['role']
        if role == 'user':
            assert last_type == 'assistant', \
                f"User message must follow assistant message, found this previous type instead: {last_type}"
            grouped_messages.append([msg])
            last_type = 'user'
        elif role == 'assistant':
            assert last_type in ['user', 'tool_response'], \
                f"Assistant message must follow user or tool response message, found this previous type instead: {last_type}"
            grouped_messages.append([msg])
            last_type = 'assistant'
        elif role == 'tool_call':
            # no assertion because it may follow any other type
            if last_type in ('user', 'tool_response'):
                grouped_messages.append([msg])
            else:
                grouped_messages[-1].append(msg)
            last_type = 'tool_call'
        elif role == 'tool_response':
            assert last_type in ('tool_call', 'tool_response'), \
                f"Tool response message must follow tool call or tool response message, found this previous type instead: {last_type}"
            if last_type == 'tool_call':
                grouped_messages.append([msg])
            else:
                grouped_messages[-1].append(msg)
            last_type = 'tool_response'
        else:
            raise ValueError(f"Unknown message role: {role}")
    return grouped_messages

def format_gemma3_tool_calling_example(sample: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
    """
    Format a Toucan dataset sample into Gemma 3 tool calling format using philschmid's approach.

    This uses the format from "Google Gemma 3 Function Calling Example" by philschmid:
    - Tools are still wrapped in JSON format in the prompt (unchanged)
    - Tool calls use ```tool_code with Python function call syntax
    - Tool responses use ```tool_output with the result
    - Multiple tool calls/responses each have their own enclosing backtick blocks

    Following the format from https://www.philschmid.de/gemma-function-calling:
    - Tool calls: ```tool_code\nfunction_name(arg1=value1, arg2=value2)\n```
    - Tool responses: ```tool_output\nresult_value\n```

    Args:
        sample: A Toucan dataset sample with 'messages', 'tools', 'target_tools'

    Returns:
        Dictionary with 'text' (full input+output for SFT)
    """
    messages = json.loads(sample['messages'])
    tools = json.loads(sample['tools'])

    # Extract system message if present
    system_content = ""
    if len(messages) > 0 and messages[0]['role'] == 'system':
        system_content = messages[0]['content']
        messages = messages[1:]

    # Group messages by turn
    grouped_messages = group_messages_by_turn(messages)

    role_messages = []
    # Process each turn
    for turn_idx, turn_group in enumerate(grouped_messages):
        assert turn_group, "Turn group should not be empty"

        first_msg = turn_group[0]
        role = first_msg['role']

        new_text = ''
        if role == 'user':
            # User turn
            # Add system instructions to first user message
            if turn_idx == 0 and system_content:
                new_text += system_content + "\n\n"

            # Add tools definition to first user message
            if turn_idx == 0:
                new_text += SYSTEM_PROMPT
                new_text += format_tools_for_prompt(tools)
                new_text += "\n\n"

            # Add user content
            assert first_msg['content'].strip(), "User message content should not be empty"
            new_text += first_msg['content']
            role_messages.append({'role': 'user', 'text': new_text})
        elif role == 'assistant' or role == 'tool_call':
            # Assistant turn (may include tool calls)
            for msg in turn_group:
                msg_role = msg['role']

                if msg_role == 'assistant':
                    # Add assistant text content
                    assert 'content' in msg and msg['content'].strip(), "Assistant message content should not be empty"
                    new_text += msg['content'] + "\n"

                elif msg_role == 'tool_call':
                    # Toucan stores tool calls as Python dict strings
                    # Format: "{'name': '...', 'arguments': '{...}'}"
                    tool_call_content = msg['content']
                    assert tool_call_content.strip(), "Tool call content should not be empty"

                    # Parse the Python dict string safely
                    tool_call_data = ast.literal_eval(tool_call_content)
                    try:
                        tool_args = json.loads(tool_call_data['arguments'])
                    except json.JSONDecodeError:
                        return None

                    # Convert to philschmid's format: function_name(arg1=value1, arg2=value2)
                    # Replace hyphens with underscores for valid Python identifiers
                    func_name = tool_call_data['name'].replace('-', '_')
                    args_str = ', '.join([f"{k}={json.dumps(v)}" for k, v in tool_args.items()])
                    new_text += f'```tool_code\n{func_name}({args_str})\n```\n'

            assert new_text.endswith('\n')
            new_text = new_text[:-1]  # Remove last newline before end_of_turn
            role_messages.append({'role': 'model', 'text': new_text})
        elif role == 'tool_response':
            # Tool response turn (wrapped as user message with ```tool_output)
            for msg in turn_group:
                # Use philschmid's format: ```tool_output\nresult\n```
                result = msg['content']
                # Handle both string and dict content (Toucan dataset can have either)
                if isinstance(result, dict):
                    result = json.dumps(result)
                elif not isinstance(result, str):
                    raise ValueError("Tool response content must be string or dict")

                if not result.strip():
                    return None
                new_text += f'```tool_output\n{result}\n```\n'

            assert new_text.endswith('\n')
            new_text = new_text[:-1]  # Remove last newline before end_of_turn
            role_messages.append({'role': 'user', 'text': new_text})
        else:
            raise ValueError(f"Unknown message role: {role}")

    def format_tags(role, message):
        if role == 'user':
            return {"role": role, "text": f"\n<start_of_turn>user\n{message}<end_of_turn>\n<start_of_turn>model\n"}
        elif role == 'model':
            return {"role": role, "text": f"{message}<end_of_turn>"}
        else:
            raise ValueError(f"Unknown role: {role}")
    role_messages = [format_tags(m['role'], m['text']) for m in role_messages]
    role_messages[0]['text'] = role_messages[0]['text'].lstrip()  # Remove leading newline from first user message
    return role_messages


# ============================================================================
# Dataset Loading and Filtering
# ============================================================================

def is_english_only(text: str) -> bool:
    """
    Fast heuristic to check if text is primarily English.
    Returns False if text contains significant non-English characters.
    Only removes ~3% of the dataset.

    Args:
        text: Text to check

    Returns:
        True if English-only, False if contains non-English scripts
    """
    if not text or len(text) < 10:
        return True

    # Count non-English script characters
    cjk_count = sum(1 for c in text if 0x4E00 <= ord(c) <= 0x9FFF)  # Chinese
    arabic_count = sum(1 for c in text if 0x0600 <= ord(c) <= 0x06FF)  # Arabic
    hangul_count = sum(1 for c in text if 0xAC00 <= ord(c) <= 0xD7AF)  # Korean

    # Latin Extended (accented characters used in Portuguese, Spanish, French, etc.)
    # This includes: á, é, í, ó, ú, ã, õ, ç, ñ, ü, etc.
    latin_extended_count = sum(1 for c in text if 0x00C0 <= ord(c) <= 0x00FF or 0x0100 <= ord(c) <= 0x017F)

    # If more than threshold non-English characters, consider it non-English
    # Allow up to 2 accented characters for place names like "São Paulo"
    return (cjk_count < 5 and
            arabic_count < 5 and
            hangul_count < 5 and
            latin_extended_count < 3)


def filter_toucan_dataset(dataset, max_tools_used, max_tools_available, max_number_of_turns=1, english_only=True) -> Dataset:
    """
    Filter Toucan dataset for single-turn examples with limited tools.

    Args:
        dataset: Toucan dataset (SFT split)
        max_tools_used: Maximum number of tools used in target (default: 2)
        max_tools_available: Maximum number of tools available in prompt (default: 3)
        english_only: Filter to English-only samples (default: True)

    Returns:
        Filtered dataset
    """
    print(f"\nFiltering dataset (single-turn, ≤{max_tools_used} tools used, ≤{max_tools_available} tools available, english_only={english_only})...")

    total = len(dataset)

    def reduce_user_messages(sample):
        """Remove all messages after max_number_of_turns user messages."""
        messages = json.loads(sample['messages'])
        user_message_idxs = [idx for idx, m in enumerate(messages) if m['role'] == 'user']
        if len(user_message_idxs) <= max_number_of_turns:
            return sample
        cutoff_idx = user_message_idxs[max_number_of_turns]
        sample['messages'] = json.dumps(messages[:cutoff_idx])
        return sample

    filtered_dataset = dataset.map(
        reduce_user_messages,
        num_proc=16,
        desc=f"Reducing messages to {max_number_of_turns} user turns"
    )

    def filter_fn(sample):
        """Filter function for a single sample."""
        messages = json.loads(sample['messages'])

        user_messages = [m for m in messages if m['role'] == 'user']
        if len(user_messages) == 0:
            return False

        num_tool_calls = sum(1 for m in messages if m['role'] == 'tool_call')
        num_tool_responses = sum(1 for m in messages if m['role'] == 'tool_response')
        assert num_tool_calls == num_tool_responses

        if num_tool_calls > max_tools_used:
            return False

        if len(json.loads(sample['tools'])) > max_tools_available:
            return False

        if english_only:
            assert all('content' in m and isinstance(m['content'], str) for m in messages)
            all_text = [msg['content'] for msg in messages]
            if not is_english_only(' '.join(all_text)):
                return False

        return True

    filtered_dataset = filtered_dataset.filter(
        filter_fn,
        num_proc=16,
        desc="Filtering samples"
    )

    print(f"Filtered dataset size: {len(filtered_dataset):,} samples ({100 * len(filtered_dataset) / total:.2f}% retained)")

    return filtered_dataset


# ============================================================================
# Grain DataLoader Implementation
# ============================================================================

class _Tokenize(grain.MapTransform):
    """Tokenize role-based messages and create proper masks."""

    def __init__(self, tokenizer: tokenizer_lib.Tokenizer):
        self._tokenizer = tokenizer

    def map(self, element: Dict[str, str]) -> Dict[str, np.ndarray]:
        """
        Tokenize role messages and create loss mask.

        The text field already contains complete formatting:
        - User: "\n<start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n"
        - Model: "{content}<end_of_turn>"

        Returns dict with:
        - tokens: Full token sequence
        - mask: Loss mask (1 for model outputs, 0 for user inputs)
        """
        # Parse role messages
        role_messages = json.loads(element['role_messages'])

        all_tokens = []
        all_masks = []

        for turn in role_messages:
            role = turn['role']
            text = turn['text']

            # Tokenize the complete text (which already has formatting tags)
            tokens = self._tokenizer.encode(text)

            # Create mask based on role
            if role == 'user':
                # Don't train on user input (including the <start_of_turn>model\n at the end)
                all_tokens.extend(tokens)
                all_masks.extend([0] * len(tokens))

            elif role == 'model':
                # TRAIN on model output (including <end_of_turn>)
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
            mask = np.pad(mask, [[0, pad_len]], mode='constant', constant_values=0)  # Pad with 0 (don't train)

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
    filtered_dataset = filter_toucan_dataset(dataset, MAX_TOOLS_USED, MAX_TOOLS_AVAILABLE)

    # Format examples
    print("Formatting examples for Gemma 3 tool calling...")

    def format_function(examples):
        """Format a batch of examples into role-based format"""
        role_messages_list = []

        for i in range(len(examples['messages'])):
            sample = {
                'messages': examples['messages'][i],
                'tools': examples['tools'][i],
                'target_tools': examples['target_tools'][i]
            }

            formatted = format_gemma3_tool_calling_example(sample)
            if formatted:  # Returns list of {'role': ..., 'text': ...} dicts
                role_messages_list.append(json.dumps(formatted))

        return {'role_messages': role_messages_list}
    
    filtered_dataset = filtered_dataset.map(
        format_function,
        batched=True,
        batch_size=1000,
        remove_columns=filtered_dataset.column_names
    )
    # filtered_dataset = filtered_dataset.filter(
    #     lambda x: len(tokenizer.encode(x['text'])) <= MAX_TARGET_LENGTH,
    #     desc="Filtering overlength samples",
    #     num_proc=16,
    # )
    

    # Split into train and validation sets: 95% train, 5% validation -> 34k, 1.8k data points
    split = filtered_dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split['train']
    validation_dataset = split['test']

    print(f"Formatted {len(train_dataset):,} training examples")
    print(f"Formatted {len(validation_dataset):,} validation examples")

    # Count examples without tool calls (need to parse JSON and check all turns)
    def has_tool_call(example):
        """Check if any turn in the role_messages contains a tool call."""
        role_messages = json.loads(example['role_messages'])
        return any('```tool_code' in turn['text'] for turn in role_messages)

    count_with_tool_call = len(train_dataset.filter(has_tool_call))
    count_no_tool_call = len(train_dataset) - count_with_tool_call
    print(f"  Training examples without tool calls: {count_no_tool_call} ({100 * count_no_tool_call / len(train_dataset):.2f}%)")

    count_with_tool_call_val = len(validation_dataset.filter(has_tool_call))
    count_no_tool_call_val = len(validation_dataset) - count_with_tool_call_val
    print(f"  Validation examples without tool calls: {count_no_tool_call_val} ({100 * count_no_tool_call_val / len(validation_dataset):.2f}%)")

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
        num_epochs=1,  # validation only runs once per eval
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

    # Validate loss masking by inspecting a sample batch
    print(f"\n{'='*60}")
    print("Validating Loss Masking")
    print(f"{'='*60}")
    sample_batch = next(iter(train_loader))
    sample_tokens = np.array(sample_batch.input_tokens[0])  # First example in batch
    sample_mask = np.array(sample_batch.input_mask[0])

    # Count masked tokens
    total_tokens = len(sample_tokens)
    train_tokens = int(np.sum(sample_mask))
    skip_tokens = total_tokens - train_tokens
    pad_tokens = int(np.sum(sample_tokens == tokenizer.pad_id()))

    print("Sample masking statistics:")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Training tokens (mask=1): {train_tokens} ({100*train_tokens/total_tokens:.1f}%)")
    print(f"  Skipped tokens (mask=0): {skip_tokens} ({100*skip_tokens/total_tokens:.1f}%)")
    print(f"  Padding tokens: {pad_tokens} ({100*pad_tokens/total_tokens:.1f}%)")

    # Show first 100 tokens with their mask values to verify correctness
    print("\nFirst 100 tokens with mask (✓=train, ✗=skip):")
    print("-" * 80)
    for i in range(min(100, len(sample_tokens))):
        token_id = int(sample_tokens[i])
        mask_val = sample_mask[i]
        if token_id == tokenizer.pad_id():
            decoded = "<PAD>"
        else:
            decoded = tokenizer.decode([token_id])
        symbol = "✓" if mask_val == 1 else "✗"
        # Truncate long decoded strings
        decoded_display = repr(decoded)[:40]
        print(f"{i:3d} {symbol} [{mask_val:.0f}] {decoded_display}")

    print("-" * 80)
    print("Expected pattern:")
    print("  ✗ for <start_of_turn>user and all user content")
    print("  ✓ for model outputs after <start_of_turn>model")
    print("  ✗ for padding tokens")
    print(f"{'='*60}\n")

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

    Uses format_gemma3_tool_calling_example to ensure consistency with training format.
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

    # Define tools in the format expected by format_gemma3_tool_calling_example
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

    # Format both examples using the same function as training
    formatted1 = format_gemma3_tool_calling_example(sample1)
    formatted2 = format_gemma3_tool_calling_example(sample2)
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

    warmup_steps = int(0.05 * max_steps)
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=LEARNING_RATE,
        warmup_steps=warmup_steps,
        decay_steps=max_steps - warmup_steps,
        end_value=LEARNING_RATE * 0.2,
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
