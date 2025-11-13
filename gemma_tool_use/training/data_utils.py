#!/usr/bin/env python3
"""
Data loading and preprocessing utilities for Gemma 3 tool calling training.

This module handles:
- Dataset loading and filtering (Toucan-1.5M)
- Tool calling format conversion
- Tokenization and masking
- Grain DataLoader creation
"""

import json
import ast
import logging
from typing import Dict, List, Any, Optional

import numpy as np
import jax.numpy as jnp
from datasets import load_dataset, Dataset
import grain.python as grain

from tunix.generate import tokenizer_adapter as tokenizer_lib
from tunix.sft import peft_trainer

logger = logging.getLogger(__name__)


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


def format_gemma3_tool_calling_example(sample: Dict[str, Any], system_prompt: str) -> Optional[List[Dict[str, str]]]:
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
        system_prompt: System prompt to prepend to first user message

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
                new_text += system_prompt
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
        max_number_of_turns: Maximum number of user turns to keep (default: 1)
        english_only: Filter to English-only samples (default: True)

    Returns:
        Filtered dataset
    """
    print(f"\nFiltering dataset (≤{max_number_of_turns} turns, ≤{max_tools_used} tools used, ≤{max_tools_available} tools available, english_only={english_only})...")

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


def create_tool_calling_dataset(
    tokenizer,
    global_batch_size,
    max_target_length,
    num_train_epochs,
    max_tools_used,
    max_tools_available,
    format_function
):
    """
    Create and format the tool calling dataset.

    Args:
        tokenizer: Gemma tokenizer
        global_batch_size: Batch size for training
        max_target_length: Maximum sequence length
        num_train_epochs: Number of training epochs
        max_tools_used: Maximum number of tools used per example
        max_tools_available: Maximum number of tools available per example
        format_function: Function to format examples (takes batched examples dict)

    Returns:
        Tuple of (train_loader, validation_loader, total_steps, train_dataset)
    """
    print(f"\n{'='*60}")
    print("Loading Toucan-1.5M dataset")
    print(f"{'='*60}")

    # Load Toucan SFT dataset
    dataset = load_dataset('Agent-Ark/Toucan-1.5M', 'SFT', split='train')

    # Filter dataset
    filtered_dataset = filter_toucan_dataset(dataset, max_tools_used, max_tools_available)

    # Format examples
    print("Formatting examples for Gemma 3 tool calling...")

    filtered_dataset = filtered_dataset.map(
        format_function,
        batched=True,
        batch_size=1000,
        remove_columns=filtered_dataset.column_names
    )

    # Split into train and validation sets: 95% train, 5% validation -> 34k, 1.8k data points
    split = filtered_dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split['train']
    validation_dataset = split['test']

    print(f"Formatted {len(train_dataset):,} training examples")
    print(f"Formatted {len(validation_dataset):,} validation examples")

    # Count examples without tool calls (need to parse JSON and check all turns)
    def has_tool_call(example):
        """Check if any turn in the role_messages contains a tool call (BFCL format: [func(...)])."""
        role_messages = json.loads(example['role_messages'])
        # Look for BFCL-style function calls: [func_name(
        import re
        return any(re.search(r'\[[\w\.]+\(', turn['text']) for turn in role_messages)

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
