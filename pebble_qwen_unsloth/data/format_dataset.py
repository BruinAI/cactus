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
from unittest import result


USE_OUR_FORMAT = True
SYSTEM_PROMPT = """You are a helpful personal assistant. When the user asks you to perform an action, you must call the appropriate tool. Always use tools to complete tasks or get information rather than just acknowledging the request.

Important: Do NOT call tools for general knowledge questions like "What is the capital of France?" or "How old is...?" - answer these directly.

Tool Selection Guidelines:
- For alarms: Use 'set_alarm' for requests to wake up or be alerted at a specific time, even if they mention "tomorrow". Only use timers for duration-based requests like "in 30 minutes".
- For notes: Use 'create_note' when the user wants to write something down, save a thought, remember something, or "jot down" information. If they say "remind me" WITHOUT a specific time (e.g., "remind me to buy milk"), treat it as a note to save, not a timed reminder.
- For messages: Use 'write_text_message' when the user wants to send a text, message, or tell someone something (e.g., "Tell John..." or "Text Sarah...").
- For weather: Use 'weather_lookup' when the user wants current weather or forecast information. Questions like "Will I need an umbrella?" or "How cold is it?" are weather requests."""

OUR_TOOL_SYSTEM_PROMPT = """<|im_start|>system
{system_content}

You have access to the following tools:
[
{json_tools}
]

When you need to call a tool, respond with a JSON object in this exact format:
{{\"function_call\": {{\"name\": \"function_name\", \"arguments\": {{\"arg1\": \"value1\"}}}}}}
You can call multiple tools by using multiple function_call JSON objects."""


def format_qwen3_dataset(
    sample: Dict[str, Any],
    tools: List[Dict[str, Any]],
    tokenizer,
    system_prompt_addition: str = SYSTEM_PROMPT,
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
        system_prompt_addition: Additional instruction to prepend to the system prompt

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

    # default system prompt (old):
    """<|im_start|>system
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "create_note", "description": "Creates a new note with the given text. Call this tool if asked to be reminded or to take a note.", "parameters": {"type": "object", "properties": {"text": {"type": "string", "description": "The text of the note, usually a direct quote from the user"}}, "required": ["text"]}}}
{"type": "function", "function": {"name": "set_alarm", "description": "Sets an alarm for a specific time.", "parameters": {"type": "object", "properties": {"time_hours": {"type": "integer", "description": "The hour component of the alarm time (24 hour time)"}, "time_minutes": {"type": "integer", "description": "The minute component of the alarm time (0-59)"}}, "required": ["time_hours", "time_minutes"]}}}
{"type": "function", "function": {"name": "set_timer_absolute", "description": "Sets a timer to go off at an absolute day and time.", "parameters": {"type": "object", "properties": {"day_offset": {"type": "string", "description": "The offset of the day to remind the user at e.g. 'tomorrow', 'today', 'thursday' (will be the next thursday), '3' (will be in 3 days)"}, "time_hours": {"type": "integer", "description": "The hour component of the desired end time (24 hour time)"}, "time_minutes": {"type": "integer", "description": "The minute component of the desired end time (0-59)"}}, "required": ["day_offset", "time_hours", "time_minutes"]}}}
{"type": "function", "function": {"name": "set_timer", "description": "Sets a timer for a relative duration (hours, minutes, seconds).", "parameters": {"type": "object", "properties": {"time_hours": {"type": "integer", "description": "The number of hours on the timer"}, "time_minutes": {"type": "integer", "description": "The number of minutes on the timer"}, "time_seconds": {"type": "integer", "description": "The number of seconds on the timer"}}, "required": ["time_hours", "time_minutes", "time_seconds"]}}}
{"type": "function", "function": {"name": "reminder_absolute", "description": "Creates a reminder for a specific absolute date and time.", "parameters": {"type": "object", "properties": {"day_offset": {"type": "string", "description": "The offset of the day to remind the user at e.g. 'tomorrow', 'today', 'thursday' (will be the next thursday), '3' (will be in 3 days)"}, "absolute_time_hour": {"type": "integer", "description": "The absolute time to remind the user at as a 24 hour hour part e.g. '17'"}, "absolute_time_minute": {"type": "integer", "description": "The absolute time to remind the user at as a minute part e.g. '30', or '00' for the top of the hour"}, "date_month_day": {"type": "string", "description": "The date to remind the user at if specified by the user as a date part (month-day) e.g. '12-31'"}, "date_year": {"type": "integer", "description": "The year to remind the user at if specified by the user as a year part e.g. '2022'"}, "message": {"type": "string", "description": "The message to remind the user e.g. 'Buy more milk'"}}, "required": ["day_offset", "absolute_time_hour", "absolute_time_minute", "date_month_day", "date_year", "message"]}}}
{"type": "function", "function": {"name": "create_reminder_relative", "description": "When the user requires a reminder at a relative time e.g. 'in 5 minutes' use the create_reminder_relative tool.", "parameters": {"type": "object", "properties": {"relative_time": {"type": "integer", "description": "The relative time to remind the user at as n 'time_unit's in the future"}, "time_unit": {"type": "string", "description": "The unit of time for the relative time. Must be one of: [\"seconds\", \"minutes\", \"hours\", \"days\", \"weeks\", \"months\", \"years\"]"}, "message": {"type": "string", "description": "The message to remind the user e.g. 'Buy more milk'"}}, "required": ["relative_time", "time_unit", "message"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>"""

    if USE_OUR_FORMAT:
        tool_call_dict = {"role": "assistant", "content": json.dumps({"function_call": function_call})}
    else:
        tool_call_dict = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                    "name": function_call['name'],
                    "arguments": function_call['arguments'],
            }]
        }
    messages = [
        {"role": "system", "content": system_prompt_addition},
        {"role": "user", "content": user_input},
        tool_call_dict,
    ]

    # Apply chat template
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )

    # Post-process the HF template output:
    # 1. Remove extra newlines when system content is empty
    formatted_text = formatted_text.replace("<|im_start|>system\n\n\n# Tools", "<|im_start|>system\n# Tools")
    # 2. Remove trailing newline at the end
    if formatted_text.endswith('\n'):
        formatted_text = formatted_text[:-1]

    if USE_OUR_FORMAT:
        end_of_system_prompt = formatted_text.find("<|im_end|>")
        if end_of_system_prompt == -1:
            raise ValueError("The main prompt should contain <|im_end|> tag.")
        formatted_system_prompt = OUR_TOOL_SYSTEM_PROMPT.format(
            system_content=system_prompt_addition,
            json_tools='\n'.join([json.dumps(tool) for tool in tools])
        )   
        formatted_text = formatted_system_prompt + formatted_text[end_of_system_prompt:]

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
