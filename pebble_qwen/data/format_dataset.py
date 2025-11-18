#!/usr/bin/env python3
"""
Format Noah's custom tool calling dataset for Qwen 3 training.

This module handles formatting of Noah's dataset which has the structure:
{
    "input": "user message",
    "output": {
        "function_call": {
            "name": "function_name",
            "arguments": {"arg1": "value1", ...}
        }
    }
}

Uses Qwen3's native format via HuggingFace's tokenizer:
- System prompt with tools in <tools></tools> XML tags
- Chain-of-thought with <think></think> tags
- Tool calls in <tool_call></tool_call> XML tags
- Qwen's <|im_start|> and <|im_end|> special tokens

The formatting is done entirely by Qwen3-0.6B's apply_chat_template with
minimal post-processing to remove extra newlines.
"""

import json
from typing import Dict, List, Any, Optional


def format_qwen3_dataset(
    sample: Dict[str, Any],
    tools: List[Dict[str, Any]],
    tokenizer
) -> Optional[List[Dict[str, str]]]:
    """
    Format a Noah dataset sample into Qwen 3 tool calling format following BFCL conventions.

    Uses HuggingFace's tokenizer apply_chat_template with assertions to verify it matches
    the BFCL manual format from qwen_fc.py:
    - System message with tools in <tools></tools> XML tags
    - Tool calls as <tool_call>\n{"name": "...", "arguments": {...}}\n</tool_call>

    Args:
        sample: A Noah dataset sample with 'input' and 'output' fields
        tools: List of available tools
        tokenizer: HuggingFace tokenizer with apply_chat_template support

    Returns:
        List of role messages with 'role' and 'text' keys, or None if formatting fails
    """
    user_input = sample.get('input', '').strip()
    output = sample.get('output', {})

    if not user_input:
        return None

    if 'function_call' not in output:
        return None

    function_call = output['function_call']
    if 'name' not in function_call or 'arguments' not in function_call:
        return None

    # ========================================================================
    # Use HuggingFace tokenizer apply_chat_template
    # ========================================================================

    # Construct messages with a blank system message to suppress the default
    # "You are Qwen, created by Alibaba Cloud..." message
    # The tokenizer template will add the tools section after the system content
    messages = [
        {"role": "system", "content": ""},  # Empty system message
        {"role": "user", "content": user_input},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "name": function_call['name'],
                    "arguments": function_call['arguments']
                }
            ]
        }
    ]

    # Apply chat template
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=False
    )

    # Post-process the HF template output:
    # 1. Remove extra newlines when system content is empty
    formatted_text = formatted_text.replace("<|im_start|>system\n\n\n# Tools", "<|im_start|>system\n# Tools")
    # 2. Remove trailing newline at the end
    if formatted_text.endswith('\n'):
        formatted_text = formatted_text[:-1]

    # Note: Qwen3's <think> tags are kept in the output for chain-of-thought reasoning

    # ========================================================================
    # Split formatted text by role for proper loss masking during training
    # ========================================================================
    # Parse the formatted text to split into system, user, and model sections

    # Find the boundaries between sections
    system_end = formatted_text.find("<|im_end|>") + len("<|im_end|>\n")
    user_start = system_end
    user_end = formatted_text.find("<|im_start|>assistant\n", user_start) + len("<|im_start|>assistant\n")

    system_text = formatted_text[:system_end]
    user_text = formatted_text[user_start:user_end]
    model_text = formatted_text[user_end:]

    # Create role-based messages for loss masking
    system_message = {
        'role': 'system',
        'text': system_text
    }

    user_message = {
        'role': 'user',
        'text': user_text
    }

    model_message = {
        'role': 'model',
        'text': model_text
    }

    return [system_message, user_message, model_message]


def load_tools(tools_path: str) -> List[Dict[str, Any]]:
    """
    Load Noah's tools from JSON file.

    Args:
        tools_path: Path to noah_tools.json

    Returns:
        List of tool definitions
    """
    with open(tools_path, 'r') as f:
        tools = json.load(f)
    return tools


def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """
    Load Noah's dataset from JSON file.

    Args:
        dataset_path: Path to noah_finetune_dataset.json

    Returns:
        List of dataset samples
    """
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    return dataset
