# Tool Calling Evaluation - MLX Optimized

This directory contains a Python script for evaluating tool calling performance of fine-tuned models using MLX for optimized inference on Apple Silicon (M1/M2/M3/M4 Macs).

## Overview

The evaluation script tests how well a model can:
1. Identify when to call a tool
2. Select the correct tool for a given query
3. Extract proper arguments for tool calls

## Features

- **MLX-optimized**: Uses Apple's MLX framework for fast inference on Apple Silicon
- **Tool filtering**: Tests both filtered (top-k relevant tools) and unfiltered (all tools) scenarios
- **Comprehensive metrics**: Tracks accuracy by tool type, filtering setting, and overall performance
- **Detailed results**: Saves full message history and timing data to CSV

## Installation

Install the required dependencies:

```bash
cd pebble_qwen_unsloth/eval
pip install -r requirements.txt
```

Or using uv:

```bash
cd pebble_qwen_unsloth/eval
uv pip install -r requirements.txt
```

## Usage

### Basic Usage

Evaluate a HuggingFace model:

```bash
python tool_calling_eval.py "Qwen/Qwen3-0.6B"
```

Evaluate a local model:

```bash
python tool_calling_eval.py "/path/to/your/model"
```

### Advanced Options

```bash
# Specify output file
python tool_calling_eval.py "Qwen/Qwen3-0.6B" --output my_results.csv

# Only test without tool filtering
python tool_calling_eval.py "Qwen/Qwen3-0.6B" --no-filter

# Change number of top tools to retrieve when filtering
python tool_calling_eval.py "Qwen/Qwen3-0.6B" --top-n 5
```

### Example

```bash
# Evaluate your fine-tuned model
python tool_calling_eval.py "../models/qwen3_tool_calling_lora" --output qwen3_lora_eval.csv

# Or from HuggingFace Hub
python tool_calling_eval.py "yourusername/qwen3-tool-calling" --output results.csv
```

## Output

The script generates:

1. **CSV file** with detailed results for each test case:
   - `query`: The user query
   - `model`: Model path
   - `filter_tools`: Whether tools were filtered
   - `correct_tool`: Expected tool name
   - `tools_called`: List of tools the model called
   - `correct_tool_called`: Boolean indicating correctness
   - `message_history`: Full conversation history
   - `elapsed_time`: Inference time in seconds

2. **Console output** with:
   - Progress for each test case
   - Summary statistics (overall accuracy, per-tool accuracy, etc.)
   - Average inference time

## Evaluation Dataset

The script tests 34 queries across different tool categories:

- **create_note**: Taking notes and reminders
- **set_alarm**: Setting alarms for specific times
- **set_timer**: Setting timers for durations
- **reminder_absolute**: Absolute time reminders
- **create_reminder_relative**: Relative time reminders
- **weather_lookup**: Weather queries (placeholder)
- **write_text_message**: Sending messages (placeholder)

## Tool Filtering

The evaluation tests two scenarios:

1. **Filtered (filter_tools=True)**: Only the top-N most relevant tools are provided to the model
2. **Unfiltered (filter_tools=False)**: All available tools are provided

This helps assess:
- How the model performs with focused tool selection
- Whether the model can handle a larger tool palette
- The trade-off between context size and accuracy

## Customization

### Adding New Tools

Add new tool functions to the script:

```python
def my_custom_tool(param1: str, param2: int):
    """
    Description of what the tool does.

    Args:
        param1: Description of param1
        param2: Description of param2
    """
    return f"Tool executed with {param1} and {param2}"

# Add to tools list in run_evaluation()
tools = Tools([build_tool_from_func(f) for f in [
    create_note,
    set_alarm,
    my_custom_tool,  # Add here
    # ...
]])
```

### Adding Test Cases

Add new queries to the `EVAL_DATA` list:

```python
EVAL_DATA = [
    # ... existing cases ...
    {"query": "Your new test query", "correct_tool": "expected_tool_name"},
]
```

### Modifying System Prompt

The default system prompt emphasizes tool usage. You can modify it in the `run_evaluation()` function:

```python
system_prompt = "Your custom system prompt here..."
```

## Performance

On Apple Silicon Macs with MLX:
- **M1/M2**: Expect 30-60 tokens/sec for small models (600M-1B)
- **M3/M4**: Expect 60-120+ tokens/sec for small models

The script prints timing information for each inference, helping you track performance.

## Differences from Original Notebook

This script converts the original Jupyter notebook to use:

1. **MLX instead of Cactus bindings**: Optimized for Mac inference
2. **Standard Python script**: Can be run from command line
3. **Enhanced error handling**: Graceful failure and detailed logging
4. **Flexible model loading**: Supports HuggingFace Hub or local paths
5. **Command-line arguments**: Easy configuration without code changes

## Troubleshooting

### Model Loading Issues

If you get model loading errors:
- Ensure the model is compatible with MLX
- Check if the model has a chat template in its tokenizer config
- Try using a different model or format

### Memory Issues

If you run out of memory:
- Use a smaller model
- Reduce `max_tokens` in the `MLXChatModel` initialization
- Close other memory-intensive applications

### Slow Performance

If inference is slow:
- Ensure you're running on Apple Silicon (not Intel)
- Check that MLX is properly installed: `python -c "import mlx.core as mx; print(mx.metal.is_available())"`
- Reduce `max_tokens` or use a smaller model

## Next Steps

After running evaluation:

1. Analyze the CSV results in a spreadsheet or notebook
2. Identify failure cases and patterns
3. Fine-tune the system prompt or tool descriptions
4. Iterate on model training if accuracy is low
5. Compare different model checkpoints

## License

This evaluation script is part of the Cactus-FC project.
