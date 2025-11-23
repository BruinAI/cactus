"""
Tool Calling Evaluation Script - MLX Optimized for Mac

This script evaluates tool calling performance of fine-tuned models using MLX
for optimized inference on Apple Silicon.
"""

from gc import enable
import json
import warnings
from typing import List, Dict, Callable, Optional, get_args, get_origin, Union
import inspect
from docstring_parser import parse
from dataclasses import dataclass, field
from functools import wraps
import time
import re
import ast
import argparse

import pandas as pd
import numpy as np
import mlx.core as mx
from mlx_lm import load, generate

warnings.simplefilter('ignore')

SKIP_NONE = False
USE_OUR_FORMAT = True
SYSTEM_PROMPT = "You are a helpful personal assistant. When the user asks you to perform an action, you must call the appropriate tool. Always use tools to complete tasks rather than just acknowledging the request."
# SYSTEM_PROMPT = "You are a helpful assistant. You have access to a list of tools. You are communicating via one-shot interactions. If using a tool/function, just call it without asking follow-up questions."  # Roman's original sys prompt

OUR_TOOL_SYSTEM_PROMPT = """<|im_start|>system
{system_content}

You have access to the following tools:
[
{json_tools}
]

When you need to call a tool, respond with a JSON object in this exact format:
{{\"function_call\": {{\"name\": \"function_name\", \"arguments\": {{\"arg1\": \"value1\"}}}}}}
You can call multiple tools by using multiple function_call JSON objects."""

# ============================================================================
# Tool System - Same as original notebook
# ============================================================================

def keywords(kw: list = None):
    """A decorator to attach a list of keywords to a function."""
    def decorator(func):
        setattr(func, '_keywords', kw or [])

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return func

    return decorator


@dataclass
class ToolArgument:
    name: str
    description: str
    _type: str
    required: bool


@dataclass
class Tool:
    name: str
    description: str
    func: Callable
    args: Optional[List[ToolArgument]] = field(default_factory=list)
    keywords: Optional[List[str]] = field(default_factory=list)

    def to_openai_format(self) -> Dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        arg.name: {
                            "type": arg._type,
                            "description": arg.description
                        }
                        for arg in self.args
                    },
                    "required": [arg.name for arg in self.args if arg.required]
                }
            }
        }

    def to_string(self, transformation_type: str = "name-description-args-descriptions") -> str:
        match transformation_type:
            case 'name-description':
                return f"Function {self.name} with description: `{self.description}`"
            case 'name-description-args':
                return f"""Function {self.name} with description: `{self.description}` and arguments {', '.join([f"`{arg.name}`" for arg in self.args])}"""
            case 'name-description-args-descriptions':
                return f"""Function {self.name} with description: `{self.description}` and arguments {', '.join([f"`{arg.name}` ({arg.description})" for arg in self.args])}"""
            case _:
                raise Exception('unknown case!')


class Tools:
    def __init__(self, tools: List[Tool]) -> None:
        self.tools = tools

    def retrieve_relevant(
        self,
        user_query: str,
        return_format: str = 'openai',
        top_n: int = 1,
        embed_model = None
    ) -> List[Tool | Dict]:
        """Retrieve relevant tools based on query similarity."""
        assert return_format in ('openai', 'dict'), 'Unsupported return format!'

        if embed_model is None:
            # No embedding model available, return all tools
            if return_format == 'openai':
                return [t.to_openai_format() for t in self.tools[:top_n]]
            return self.tools[:top_n]

        # Calculate cosine similarity
        query_embedding = embed_model.embed(user_query)
        tool_cos_sims = [
            calculate_cosine_similarity(query_embedding, embed_model.embed(tool.to_string()))
            for tool in self.tools
        ]
        sorted_tools = sorted(zip(self.tools, tool_cos_sims), key=lambda x: x[1], reverse=True)
        retrieved_tools = [x[0] for x in sorted_tools[:top_n]]

        if return_format == 'openai':
            return [t.to_openai_format() for t in retrieved_tools]
        return retrieved_tools


def calculate_cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def _map_type(py_type) -> str:
    """Maps Python types to JSON schema string types."""
    if py_type == str:
        return "string"
    if py_type == int:
        return "integer"
    if py_type == float:
        return "number"
    if py_type == bool:
        return "boolean"
    return "string"


def build_tool_from_func(func: Callable) -> Tool:
    """Generates a Tool object by inspecting a Python function."""
    sig = inspect.signature(func)
    docstring = parse(inspect.getdoc(func))
    param_docs = {param.arg_name: param.description for param in docstring.params}

    tool_args = []
    for name, param in sig.parameters.items():
        is_required = (param.default == inspect.Parameter.empty)
        real_type = param.annotation
        if get_origin(real_type) in [Union, Optional]:
            args = get_args(real_type)
            real_type = next(t for t in args if t is not type(None))

        tool_args.append(
            ToolArgument(
                name=name,
                description=param_docs.get(name, ""),
                _type=_map_type(real_type),
                required=is_required
            )
        )

    return Tool(
        name=func.__name__,
        description=docstring.short_description or "",
        func=func,
        args=tool_args,
        keywords=getattr(func, '_keywords', [])
    )


# ============================================================================
# MLX Chat Model Implementation
# ============================================================================

class MLXChatModel:
    """MLX-based chat model for optimized inference on Apple Silicon."""

    def __init__(
        self,
        model_path: str,
        system_prompt: str = 'You are a helpful assistant.',
        max_tokens: int = 1024,
        temperature: float = 0.7,
        verbose: bool = False
    ):
        """
        Initialize MLX chat model.

        Args:
            model_path: Path to the model (HuggingFace model ID or local path)
            system_prompt: System prompt for the conversation
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            verbose: Whether to print verbose output
        """
        self.model_path = model_path
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.verbose = verbose

        print(f"Loading model from {model_path}...")
        self.model, self.tokenizer = load(model_path)
        print("Model loaded successfully!")

        self.reset_history()

    def reset_history(self) -> None:
        """Resets the conversation to just the initial system prompt."""
        self.message_history_raw = [{'role': 'system', 'content': self.system_prompt}] if self.system_prompt else []

    def format_messages_for_chat(self, messages: List[Dict], tools: Optional[List[Dict]] = None) -> str:
        if USE_OUR_FORMAT:
            return self.our_format_messages_for_chat(messages, tools or [])
        return self.base_format_messages_for_chat(messages, tools or [])

    def base_format_messages_for_chat(self, messages: List[Dict], tools: List[Dict]) -> str:
        """Format messages into chat template format."""
        # Convert messages to format expected by tokenizer
        formatted_messages = []
        for msg in messages:
            formatted_msg = {'role': msg['role']}

            # Ensure content field exists
            if 'content' in msg:
                formatted_msg['content'] = msg['content']
            elif 'response' in msg:
                formatted_msg['content'] = msg['response']
            else:
                formatted_msg['content'] = ''

            # # Preserve tool calls if present
            # if 'function_calls' in msg:
            #     formatted_msg['tool_calls'] = msg['function_calls']

            formatted_messages.append(formatted_msg)

        # Use tokenizer's chat template with tools support
        prompt = self.tokenizer.apply_chat_template(
            formatted_messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return prompt

    def our_format_messages_for_chat(self, messages: List[Dict], tools: List[Dict]) -> str:
        main_prompt = self.base_format_messages_for_chat(messages, tools)
        end_of_system_prompt = main_prompt.find("<|im_end|>")
        if end_of_system_prompt == -1:
            raise ValueError("The main prompt should contain <|im_end|> tag.")
        assert messages[0]['role'] == 'system'
        result = OUR_TOOL_SYSTEM_PROMPT.format(
            system_content=messages[0]['content'],
            json_tools='\n'.join([json.dumps(tool) for tool in tools])
        )
            
        result = result + main_prompt[end_of_system_prompt:]
        return result

    def parse_tool_calls(self, response_text: str) -> List[Dict]:
        """Parse tool calls from model response."""
        tool_calls = []

        # Try to find the new JSON format: {"function_call": {"name": "...", "arguments": {...}}}
        # We'll extract all JSON objects and check their structure
        json_objects = []
        brace_count = 0
        current_obj = ""
        in_json = False

        for char in response_text:
            if char == '{':
                if brace_count == 0:
                    in_json = True
                    current_obj = "{"
                else:
                    current_obj += char
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                current_obj += char
                if brace_count == 0 and in_json:
                    json_objects.append(current_obj)
                    current_obj = ""
                    in_json = False
            elif in_json:
                current_obj += char

        # Try to parse each JSON object
        for json_str in json_objects:
            try:
                data = json.loads(json_str)

                # Check for new format: {"function_call": {"name": "...", "arguments": {...}}}
                if 'function_call' in data:
                    func_call = data['function_call']
                    if 'name' in func_call:
                        tool_calls.append({
                            'name': func_call['name'],
                            'arguments': func_call.get('arguments', {})
                        })
                # Check for old format: {"name": "...", "arguments": {...}}
                elif 'name' in data and 'arguments' in data:
                    tool_calls.append({
                        'name': data['name'],
                        'arguments': data['arguments']
                    })
            except json.JSONDecodeError:
                continue

        # Fallback: Try to find <tool_call>...</tool_call> blocks (old XML format)
        if not tool_calls:
            pattern_with_close = r'<tool_call>\s*(.*?)\s*</tool_call>'
            matches = re.findall(pattern_with_close, response_text, re.DOTALL)

            # If no matches with closing tag, try without closing tag (less strict)
            if not matches:
                pattern_without_close = r'<tool_call>\s*({.*?})'
                matches = re.findall(pattern_without_close, response_text, re.DOTALL)

            for match in matches:
                try:
                    tool_data = json.loads(match)
                    tool_calls.append({
                        'name': tool_data.get('name') or tool_data.get('function'),
                        'arguments': tool_data.get('arguments', {})
                    })
                except json.JSONDecodeError:
                    if self.verbose:
                        print(f"Failed to parse tool call: {match}")
                    continue

        return tool_calls

    def generate_response(self, prompt: str) -> str:
        """Generate a response using MLX."""
        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            verbose=self.verbose
        )
        return response

    def send_message(
        self,
        message: str,
        tools: Optional[Tools] = None,
        filter_tools: bool = True,
        top_n_tools: int = 1,
        auto_call_tool: bool = True
    ) -> Dict:
        """
        Send a message and get a response.

        Args:
            message: User message
            tools: Available tools
            filter_tools: Whether to filter tools by relevance
            top_n_tools: Number of top tools to include
            auto_call_tool: Whether to automatically call tools

        Returns:
            Response dictionary
        """
        # Prepare tools
        tool_list = []
        if tools:
            if filter_tools:
                tool_list = tools.retrieve_relevant(message, return_format='openai', top_n=top_n_tools)
            else:
                tool_list = [tool.to_openai_format() for tool in tools.tools]

        # Add user message
        self.message_history_raw.append({'role': 'user', 'content': message})

        # Format prompt with tools passed to tokenizer
        prompt = self.format_messages_for_chat(self.message_history_raw, tools=tool_list if tool_list else None)

        # Generate response
        response_text = self.generate_response(prompt)

        # Parse tool calls
        tool_calls = self.parse_tool_calls(response_text)

        # Create response dict
        response_json = {
            'role': 'assistant',
            'response': response_text
        }

        if tool_calls:
            response_json['function_calls'] = tool_calls

        self.message_history_raw.append(response_json)

        # Auto-call tools if requested
        if auto_call_tool and tool_calls and tools:
            for call in tool_calls:
                tool_name = call.get('name')
                tool_args = call.get('arguments', {})

                # Find the tool
                tool = next((t for t in tools.tools if t.name == tool_name), None)
                if tool:
                    try:
                        tool_output = tool.func(**tool_args)
                        self.message_history_raw.append({'role': 'tool_response', 'content': str(tool_output)})
                    except Exception as e:
                        self.message_history_raw.append({'role': 'tool_response', 'content': f'Error calling tool: {str(e)}'})
                else:
                    self.message_history_raw.append({'role': 'tool_response', 'content': f"Error: Tool '{tool_name}' not found."})

            # Generate final response after tool calls
            prompt = self.format_messages_for_chat(self.message_history_raw, tools=tool_list if tool_list else None)
            final_response = self.generate_response(prompt)
            final_response_json = {
                'role': 'assistant',
                'response': final_response
            }
            self.message_history_raw.append(final_response_json)

        return response_json

    @property
    def messages_df(self) -> pd.DataFrame:
        """Get message history as DataFrame."""
        df = pd.DataFrame(self.message_history_raw)
        if 'content' not in df.columns:
            df['content'] = None
        if 'response' in df.columns:
            df['content'] = df['content'].combine_first(df['response'])
        if 'role' in df.columns:
            df['role'].fillna('assistant', inplace=True)
        return df[[col for col in df.columns if col != 'response']]


# ============================================================================
# Tool Definitions
# ============================================================================

def create_note(text: str):
    """
    Creates a new note with the given text. Call this tool if asked to be reminded or to take a note.

    Args:
        text: The text of the note, usually a direct quote from the user
    """
    return f"Note created with text: {text}"


def set_alarm(time_hours: int, time_minutes: int):
    """
    Sets an alarm for a specific time.

    Args:
        time_hours: The hour component of the alarm time (24 hour time)
        time_minutes: The minute component of the alarm time (0-59)
    """
    return f"Alarm set successfully!"


def set_timer_absolute(day_offset: Optional[str], time_hours: int, time_minutes: int):
    """
    Sets a timer to go off at an absolute day and time.

    Args:
        day_offset: The offset of the day to remind the user at e.g. 'tomorrow', 'today', 'thursday' (will be the next thursday), '3' (will be in 3 days)
        time_hours: The hour component of the desired end time (24 hour time)
        time_minutes: The minute component of the desired end time (0-59)
    """
    return f"Absolute timer set for {day_offset} at {time_hours}:{time_minutes}"


def set_timer(time_hours: Optional[int], time_minutes: Optional[int], time_seconds: Optional[int]):
    """
    Sets a timer for a relative duration (hours, minutes, seconds).

    Args:
        time_hours: The number of hours on the timer
        time_minutes: The number of minutes on the timer
        time_seconds: The number of seconds on the timer
    """
    return f"Timer set for {time_hours}h {time_minutes}m {time_seconds}s"


def reminder_absolute(day_offset: Optional[str], absolute_time_hour: int, absolute_time_minute: int,
                     date_month_day: Optional[str], date_year: Optional[int], message: str):
    """
    Creates a reminder for a specific absolute date and time.

    Args:
        day_offset: The offset of the day to remind the user at e.g. 'tomorrow', 'today', 'thursday' (will be the next thursday), '3' (will be in 3 days)
        absolute_time_hour: The absolute time to remind the user at as a 24 hour hour part e.g. '17'
        absolute_time_minute: The absolute time to remind the user at as a minute part e.g. '30', or '00' for the top of the hour
        date_month_day: The date to remind the user at if specified by the user as a date part (month-day) e.g. '12-31'
        date_year: The year to remind the user at if specified by the user as a year part e.g. '2022'
        message: The message to remind the user e.g. 'Buy more milk'
    """
    return f"Absolute reminder set for '{message}' on {date_month_day}-{date_year} or {day_offset} at {absolute_time_hour}:{absolute_time_minute}"


def create_reminder_relative(relative_time: int, time_unit: str, message: str):
    """
    When the user requires a reminder at a relative time e.g. 'in 5 minutes' use the create_reminder_relative tool.

    Args:
        relative_time: The relative time to remind the user at as n 'time_unit's in the future
        time_unit: The unit of time for the relative time. Must be one of: ["seconds", "minutes", "hours", "days", "weeks", "months", "years"]
        message: The message to remind the user e.g. 'Buy more milk'
    """
    return f"Relative reminder set for '{message}' in {relative_time} {time_unit}"


def weather_lookup(location: str):
    """
    Get weather information for a location.

    Args:
        location: The city or location to get weather for
    """
    return f"Weather lookup for {location}"


def write_text_message(recipient: str, message: str):
    """
    Send a text message to someone.

    Args:
        recipient: The person to send the message to
        message: The message text to send
    """
    return f"Message sent to {recipient}: {message}"


# ============================================================================
# Evaluation Data and Logic
# ============================================================================

EVAL_DATA = [
    {"query": "send Henry a message about our upcoming framework release.", "correct_tool": "write_text_message"},
    {"query": "what is the weather in London?", "correct_tool": "weather_lookup"},
    {"query": "Wake me up at 5 am tomorrow please.", "correct_tool": "set_alarm"},
    {"query": "Write down that i need to go buy groceries for the house tomorrow", "correct_tool": "create_note"},
    {"query": "Hey how are you!", "correct_tool": None},
    {"query": "Text mom I'll be home late.", "correct_tool": "write_text_message"},
    {"query": "Can you message Alex about the 3pm call?", "correct_tool": "write_text_message"},
    {"query": "Tell Henry I've finished the draft for the medium article.", "correct_tool": "write_text_message"},
    {"query": "Will I need an umbrella tomorrow in New York?", "correct_tool": "weather_lookup"},
    {"query": "How cold is it in Paris right now?", "correct_tool": "weather_lookup"},
    {"query": "Get me the forecast for San Francisco this weekend.", "correct_tool": "weather_lookup"},
    {"query": "Set an alarm for 7:30 PM.", "correct_tool": "set_alarm"},
    {"query": "I need an alarm for 6:15 tomorrow morning.", "correct_tool": "set_alarm"},
    {"query": "Remind me to buy milk and eggs.", "correct_tool": "create_note"},
    {"query": "Make a note: pick up dry cleaning on Tuesday.", "correct_tool": "create_note"},
    {"query": "Save this thought: on-device inference is key for privacy.", "correct_tool": "create_note"},
    {"query": "That's great, thanks!", "correct_tool": None},
    {"query": "What is the capital of France?", "correct_tool": None},
    {"query": "Who won the game last night?", "correct_tool": None},
    {"query": "Make a note of the weather in Berlin.", "correct_tool": "create_note"},
    {"query": "I need an alarm for 8:45 in the morning.", "correct_tool": "set_alarm"},
    {"query": "Set an alarm for 11:30 PM tonight.", "correct_tool": "set_alarm"},
    {"query": "Alarm for 6am.", "correct_tool": "set_alarm"},
    {"query": "Can you wake me up at 7:15 am?", "correct_tool": "set_alarm"},
    {"query": "What's the weather like in Boston?", "correct_tool": "weather_lookup"},
    {"query": "I'm going to Paris tomorrow, what's the forecast?", "correct_tool": "weather_lookup"},
    {"query": "Tell me the temperature in Dubai.", "correct_tool": "weather_lookup"},
    {"query": "Weather forecast for Seattle for the next 3 days.", "correct_tool": "weather_lookup"},
    {"query": "Send a message to Alice asking 'What time is dinner?'", "correct_tool": "write_text_message"},
    {"query": "Text Bob: 'I'm running about 15 minutes late.'", "correct_tool": "write_text_message"},
    {"query": "Please message my manager that I've pushed the new code.", "correct_tool": "write_text_message"},
    {"query": "Text 'On my way!' to Sarah.", "correct_tool": "write_text_message"},
    {"query": "Can you text my brother 'Happy birthday!'?", "correct_tool": "write_text_message"},
    {"query": "Note to self: buy milk.", "correct_tool": "create_note"},
    {"query": "Remember this: the new inference engine for the react-native app is a priority.", "correct_tool": "create_note"},
    {"query": "I need to make a note about the meeting... just write down 'Follow up with marketing'.", "correct_tool": "create_note"},
    {"query": "Jot this down: need to research more apps for the Cactus library.", "correct_tool": "create_note"},
    {"query": "Create a new note titled 'Gift Ideas' with 'book for mom' in it.", "correct_tool": "create_note"},
    {"query": "What time is it?", "correct_tool": None},
    {"query": "Thanks, that's perfect.", "correct_tool": None},
    {"query": "How do I set an alarm?", "correct_tool": None},
    {"query": "Who was the first person on the moon?", "correct_tool": None},
    {"query": "How old is the Eiffel Tower?", "correct_tool": None},
    {"query": "What's the alarm for?", "correct_tool": None},
    {"query": "Tell me a joke.", "correct_tool": None},
]
if SKIP_NONE:
    EVAL_DATA = [d for d in EVAL_DATA if d['correct_tool'] is not None]


def extract_tool_calls_from_response(message_history: List[Dict]) -> List[str]:
    """
    Extract tool names from message history.

    Args:
        message_history: List of message dictionaries

    Returns:
        List of tool names that were called
    """
    # Check if there are already extracted function_calls
    existing_calls = [m for m in message_history if m.get('function_calls')]
    if existing_calls:
        return [fc.get('name') for fc in existing_calls[-1].get('function_calls', [])]

    return []


def run_evaluation(
    model_path: str,
    output_file: str = 'tool_calling_eval_results.csv',
    filter_tools_options: List[bool] = [True, False],
    top_n_tools: int = 3,
    system_prompt: str = SYSTEM_PROMPT,
    ):
    """
    Run the evaluation and save results.

    Args:
        model_path: Path to the model
        output_file: Output CSV file path
        filter_tools_options: List of filter_tools settings to test
        top_n_tools: Number of top tools to retrieve
        system_prompt: System prompt for the model
    """
    # Build tools
    tools = Tools([build_tool_from_func(f) for f in [
        create_note,
        set_alarm,
        set_timer_absolute,
        set_timer,
        reminder_absolute,
        create_reminder_relative,
        weather_lookup,
        write_text_message
    ]])

    results = []

    # Initialize model
    chat_model = MLXChatModel(
        model_path=model_path,
        system_prompt=system_prompt,
        verbose=True
    )

    total_runs = len(filter_tools_options) * len(EVAL_DATA)
    current_run = 0

    for filter_tools in filter_tools_options:
        for sample in EVAL_DATA:
            current_run += 1
            query = sample['query']
            correct_tool = sample['correct_tool']

            print(f"\n[{current_run}/{total_runs}] Query: {query}")
            print(f"  Expected tool: {correct_tool}")
            print(f"  Filter tools: {filter_tools}")

            # Reset conversation
            chat_model.reset_history()

            try:
                # Send message
                start_time = time.time()
                chat_model.send_message(
                    message=query,
                    tools=tools,
                    filter_tools=filter_tools,
                    top_n_tools=top_n_tools
                )
                elapsed = time.time() - start_time

                # Extract tool calls
                tools_called = extract_tool_calls_from_response(chat_model.message_history_raw)

                # Check if correct tool was called
                correct_tool_called = (correct_tool is None and tools_called == []) or (correct_tool in tools_called)

                print(f"  Tools called: {tools_called}")
                print(f"  Correct: {correct_tool_called}")
                print(f"  Time: {elapsed:.2f}s")

                results.append({
                    "query": query,
                    "model": model_path,
                    "filter_tools": filter_tools,
                    "correct_tool": correct_tool,
                    "tools_called": tools_called,
                    "correct_tool_called": correct_tool_called,
                    "message_history": chat_model.message_history_raw,
                    "elapsed_time": elapsed
                })

            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({
                    "query": query,
                    "model": model_path,
                    "filter_tools": filter_tools,
                    "correct_tool": correct_tool,
                    "tools_called": [],
                    "correct_tool_called": False,
                    "message_history": [],
                    "error": str(e),
                    "elapsed_time": 0
                })

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")

    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)

    # Overall accuracy
    overall_accuracy = df['correct_tool_called'].mean()
    print(f"\nOverall Accuracy: {overall_accuracy:.2%}")

    # Accuracy by filter_tools setting
    print("\nAccuracy by filter_tools setting:")
    for filter_tools in filter_tools_options:
        subset = df[df['filter_tools'] == filter_tools]
        acc = subset['correct_tool_called'].mean()
        print(f"  filter_tools={filter_tools}: {acc:.2%}")

    # Accuracy by correct_tool
    print("\nAccuracy by tool type:")
    tool_accuracy = df.groupby('correct_tool')['correct_tool_called'].mean()
    for tool_name, acc in tool_accuracy.items():
        print(f"  {tool_name}: {acc:.2%}")

    # Average time
    avg_time = df['elapsed_time'].mean()
    print(f"\nAverage inference time: {avg_time:.2f}s")

    return df


def main():
    parser = argparse.ArgumentParser(description='Run tool calling evaluation with MLX')
    parser.add_argument('model_path', type=str, help='Path to the model (HuggingFace ID or local path)')
    parser.add_argument('--output', type=str, default='tool_calling_eval_results.csv',
                       help='Output CSV file path')
    parser.add_argument('--no-filter', action='store_true',
                       help='Only test without tool filtering')
    parser.add_argument('--top-n', type=int, default=3,
                       help='Number of top tools to retrieve when filtering')

    args = parser.parse_args()

    # Determine filter_tools options
    if args.no_filter:
        filter_options = [False]
    else:
        filter_options = [True, False]

    # Run evaluation
    df = run_evaluation(
        model_path=args.model_path,
        output_file=args.output,
        filter_tools_options=filter_options,
        top_n_tools=args.top_n
    )

    print(f"\nEvaluation complete! Results saved to {args.output}")


if __name__ == "__main__":
    main()
