#!/usr/bin/env python3
"""
BFCL-style formatting functions for Gemma 3 tool calling training.

Based on the simple_python test format from BFCL (Berkeley Function Call Leaderboard).
See gemma_tool_use/BFCL_GEMMA_FORMAT.md for detailed documentation.

Key differences from philschmid format:
1. System prompt explains the Python function call format
2. Tools provided as JSON schemas (not Python function signatures)
3. Model outputs: [func_name(param1=value1, param2=value2)]
4. No ```tool_code or ```tool_output markers
5. Tool responses are presented as list of tool result dicts
"""

import json
import ast
from typing import Dict, List, Any, Optional


# BFCL-style system prompt (based on simple_python test format)
BFCL_SYSTEM_PROMPT = """You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose. If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.

You should only return the function calls in your response.

If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)] You SHOULD NOT include any other text in the response.

At each turn, you should try your best to complete the tasks requested by the user within the current turn. Continue to output functions to call until you have fulfilled the user's request to the best of your ability. Once you have no more functions to call, the system will consider the current turn complete and proceed to the next turn or task.

Here is a list of functions in json format that you can invoke.
"""


def format_tools_bfcl_style(tools: List[Dict[str, Any]]) -> str:
    """
    Format tools as JSON schemas for BFCL-style prompting.

    Args:
        tools: List of tool definitions in OpenAI function calling format

    Returns:
        JSON string with array of function definitions
    """
    # Extract function definitions and convert to BFCL format
    functions = []
    for tool in tools:
        assert 'function' in tool, "Tool must have 'function' key"
        func = tool['function']

        # BFCL format uses 'dict' instead of 'object' for parameter type
        func_def = {
            "name": func['name'],
            "description": func.get('description', ''),
            "parameters": func.get('parameters', {})
        }

        # Ensure type is 'dict' not 'object'
        if 'type' in func_def['parameters']:
            func_def['parameters']['type'] = 'dict'

        functions.append(func_def)

    return json.dumps(functions, indent=4)


def format_tool_call_bfcl_style(tool_call_content: str) -> str:
    """
    Format a tool call in BFCL Python function call syntax.

    Converts from Toucan format:
      {'name': 'get_weather', 'arguments': '{"location": "Boston"}'}

    To BFCL format:
      get_weather(location='Boston')

    Args:
        tool_call_content: Tool call content from Toucan dataset

    Returns:
        Python function call string
    """
    # Parse the Python dict string safely
    tool_call_data = ast.literal_eval(tool_call_content)

    try:
        tool_args = json.loads(tool_call_data['arguments'])
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in tool arguments: {tool_call_data['arguments']}")

    # Convert to BFCL format: func_name(param1=value1, param2=value2)
    func_name = tool_call_data['name'].replace('-', '_')

    # Format arguments with proper Python syntax
    args_parts = []
    for k, v in tool_args.items():
        # Use repr() for proper Python quoting of strings, lists, dicts, etc.
        args_parts.append(f"{k}={repr(v)}")

    args_str = ', '.join(args_parts)
    return f'{func_name}({args_str})'


def format_tool_calls_list_bfcl_style(tool_calls: List[str]) -> str:
    """
    Format multiple tool calls as a Python list.

    Args:
        tool_calls: List of formatted tool call strings

    Returns:
        Python list string: [call1, call2, ...]
    """
    if len(tool_calls) == 1:
        return f'[{tool_calls[0]}]'
    else:
        # Multi-line format for multiple calls
        calls_formatted = ', '.join(tool_calls)
        return f'[{calls_formatted}]'


def format_tool_responses_bfcl_style(tool_responses: List[Dict[str, Any]]) -> str:
    """
    Format tool responses as a Python list of dicts.

    BFCL presents tool responses as:
    [{'role': 'tool', 'name': 'func_name', 'content': 'result'}]

    Args:
        tool_responses: List of tool response dicts

    Returns:
        Python repr string of the list
    """
    return repr(tool_responses)


def format_gemma3_bfcl_style(sample: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
    """
    Format a Toucan dataset sample into BFCL-style Gemma 3 format.

    This follows the simple_python test format from BFCL:
    - System prompt explains Python function call format
    - Tools provided as JSON schemas
    - Model outputs: [func_name(param=value)]
    - Tool responses presented as list of result dicts

    Args:
        sample: A Toucan dataset sample with 'messages', 'tools', 'target_tools'

    Returns:
        List of formatted message dicts with 'role' and 'text' keys, or None if invalid
    """
    messages = json.loads(sample['messages'])
    tools = json.loads(sample['tools'])

    # Extract system message if present
    system_content = ""
    if len(messages) > 0 and messages[0]['role'] == 'system':
        system_content = messages[0]['content']
        messages = messages[1:]

    role_messages = []

    # Track state for grouping messages
    i = 0
    while i < len(messages):
        msg = messages[i]
        role = msg['role']

        if role == 'user':
            # User turn
            user_text = ''

            # Add system prompt to first user message only
            if len(role_messages) == 0:
                if system_content:
                    user_text += system_content + "\n\n"

                # Add BFCL system prompt with tools
                user_text += BFCL_SYSTEM_PROMPT
                user_text += format_tools_bfcl_style(tools)
                user_text += "\n\n"

            # Add user content
            if not msg['content'].strip():
                return None
            user_text += msg['content']

            role_messages.append({'role': 'user', 'text': user_text})
            i += 1

        elif role == 'assistant':
            # Assistant turn (may be followed by tool calls)
            model_text = ''

            # Check if there are tool calls immediately following
            j = i + 1
            tool_calls = []
            while j < len(messages) and messages[j]['role'] == 'tool_call':
                tool_call_content = messages[j]['content']
                if not tool_call_content.strip():
                    return None

                try:
                    formatted_call = format_tool_call_bfcl_style(tool_call_content)
                    tool_calls.append(formatted_call)
                except (ValueError, SyntaxError, json.JSONDecodeError):
                    return None

                j += 1

            # If assistant has content, add it first
            if msg['content'].strip():
                model_text += msg['content'] + "\n"

            # If there are tool calls, format them as a list
            if tool_calls:
                model_text += format_tool_calls_list_bfcl_style(tool_calls)

            if not model_text.strip():
                return None

            # Remove trailing newline
            model_text = model_text.rstrip('\n')

            role_messages.append({'role': 'model', 'text': model_text})
            i = j  # Skip past the tool calls we processed

        elif role == 'tool_response':
            # Tool response turn (grouped as user message)
            # Collect all consecutive tool responses
            tool_responses = []
            while i < len(messages) and messages[i]['role'] == 'tool_response':
                response_msg = messages[i]
                result = response_msg['content']

                # Handle both string and dict content
                if isinstance(result, dict):
                    result_str = json.dumps(result)
                elif isinstance(result, str):
                    result_str = result
                else:
                    return None

                if not result_str.strip():
                    return None

                # Build tool response dict (BFCL format)
                tool_response_dict = {
                    'role': 'tool',
                    'name': response_msg.get('name', 'unknown'),
                    'content': result_str
                }
                tool_responses.append(tool_response_dict)
                i += 1

            # Format as Python repr of list
            tool_response_text = format_tool_responses_bfcl_style(tool_responses)
            role_messages.append({'role': 'user', 'text': tool_response_text})

        elif role == 'tool_call':
            # Standalone tool calls without assistant message (shouldn't happen, but handle it)
            return None
        else:
            raise ValueError(f"Unknown message role: {role}")

    # Apply Gemma chat template tags
    def format_tags(role, message):
        if role == 'user':
            return {"role": role, "text": f"\n<start_of_turn>user\n{message}<end_of_turn>\n<start_of_turn>model\n"}
        elif role == 'model':
            return {"role": role, "text": f"{message}<end_of_turn>"}
        else:
            raise ValueError(f"Unknown role: {role}")

    role_messages = [format_tags(m['role'], m['text']) for m in role_messages]

    # Remove leading newline from first user message
    if role_messages:
        role_messages[0]['text'] = role_messages[0]['text'].lstrip()

    return role_messages
