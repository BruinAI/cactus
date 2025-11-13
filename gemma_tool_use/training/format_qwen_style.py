#!/usr/bin/env python3
"""
Tool calling format functions for Qwen/philschmid-style formatting.

This module handles formatting of tool definitions and messages using the
Python function signature style (as opposed to BFCL's bracket notation style).

Key formats:
- Tool definitions: Python function signatures with docstrings in ```python blocks
- Tool calls: Python function calls in ```tool_code blocks
- Tool responses: Results in ```tool_output blocks

Reference: https://www.philschmid.de/gemma-function-calling
"""

import json
import ast
from typing import Dict, List, Any, Optional


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
