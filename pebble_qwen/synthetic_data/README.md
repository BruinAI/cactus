# Synthetic Data Generation for Tool Calling

This directory contains scripts for generating synthetic tool calling examples using a backwards generation approach.

## Overview

The generation process is split into **3 independent phases** that must be run sequentially:

1. **Phase 1**: Sample random parameter values (no API required)
2. **Phase 2**: Generate realistic free-text values using Claude API
3. **Phase 3**: Generate user inputs using Claude API

Each phase reads from the previous phase's output file and produces its own output file.

## Files

- `possible_params.py` - Parameter value definitions and sampling functions
- `claude_api.py` - Claude API client with parallel request support
- `phase1_sample_params.py` - Phase 1: Sample parameters
- `phase2_generate_param_text.py` - Phase 2: Generate free-text parameter values
- `phase3_generate_inputs.py` - Phase 3: Generate user inputs
- `convert_to_training_format.py` - Convert synthetic data to training format
- `synthetic_generation.py` - Orchestrator script that runs all phases and converts to training format

## Prerequisites

### For Phase 1
- Python 3.13+
- No external dependencies

### For Phases 2 & 3
- Anthropic API key
- `anthropic` Python package

Install dependencies:
```bash
cd /Users/noahcylich/Documents/Desert/cactus-fc/pebble_qwen
uv pip install anthropic
```

Set API key:
```bash
export ANTHROPIC_API_KEY='your-api-key-here'
```

## Usage

### Option 1: Run All Phases (Orchestrator)

Run all three phases sequentially with a single command:

```bash
python3 synthetic_generation.py [samples_per_tool]
```

**Examples:**
```bash
# Generate 20 samples per tool
export ANTHROPIC_API_KEY='your-key-here'
python3 synthetic_generation.py 20

# Generate 50 samples per tool
python3 synthetic_generation.py 50
```

This will run Phase 1, Phase 2, Phase 3, then convert to training format automatically. If any phase fails, the script stops.

### Option 2: Run Phases Individually

Run each phase manually one at a time:

#### Phase 1: Sample Parameters

Generate random parameter combinations for all tools.

```bash
python3 phase1_sample_params.py [samples_per_tool] [output_file]
```

**Examples:**
```bash
# Generate 10 samples per tool (default)
python3 phase1_sample_params.py

# Generate 50 samples per tool
python3 phase1_sample_params.py 50

# Custom output file
python3 phase1_sample_params.py 10 my_samples.json
```

**Output:** `phase1_sampled_params.json`
- Dictionary mapping tool names to lists of sampled parameters
- Discrete values are randomly selected
- Free-text parameters have `"free-text"` placeholder
- Includes metadata: `_persona`, `_length_instruction`, and `_tone_style` for text generation diversity

### Phase 2: Generate Free-Text Values

Use Claude API to generate realistic text for free-text parameters.

```bash
python3 phase2_generate_param_text.py [input_file] [output_file]
```

**Examples:**
```bash
# Use default files
python3 phase2_generate_param_text.py

# Custom files
python3 phase2_generate_param_text.py my_samples.json my_text.json
```

**Input:** `phase1_sampled_params.json` (from Phase 1)
**Output:** `phase2_with_text.json`
- Same structure as input
- "free-text" placeholders replaced with generated text
- Uses parallel API calls for efficiency

### Phase 3: Generate User Inputs

Use Claude API to generate natural user inputs that would lead to the tool calls.

```bash
python3 phase3_generate_inputs.py [input_file] [output_file]
```

**Examples:**
```bash
# Use default files
python3 phase3_generate_inputs.py

# Custom files
python3 phase3_generate_inputs.py my_text.json final_examples.json
```

**Input:** `phase2_with_text.json` (from Phase 2)
**Output:** `phase3_final_examples.json`
- List of complete examples
- Each example has: `user_input`, `tool_name`, `parameters`
- Displays 5 sample examples when complete

## Complete Workflow Examples

### Using the Orchestrator (Easiest)

```bash
# Set API key
export ANTHROPIC_API_KEY='your-key-here'

# Run all phases with 20 samples per tool
python3 synthetic_generation.py 20
```

### Running Phases Individually (More Control)

```bash
# Step 1: Sample 20 parameter sets per tool (no API required)
python3 phase1_sample_params.py 20

# Step 2: Generate free-text values (requires API key)
export ANTHROPIC_API_KEY='your-key-here'
python3 phase2_generate_param_text.py

# Step 3: Generate user inputs (requires API key)
python3 phase3_generate_inputs.py

# Step 4: Convert to training format
python3 convert_to_training_format.py phase3_final_examples.json ../data/synthetic_finetune_dataset.json
```

This will produce:
- `phase1_sampled_params.json` - 120 parameter sets (20 × 6 tools)
- `phase2_with_text.json` - 120 parameter sets with generated text
- `phase3_final_examples.json` - 120 complete examples
- `../data/synthetic_finetune_dataset.json` - Training format dataset

## Output Format

### Phase 1 Output
```json
{
  "set_alarm": [
    {
      "time_hours": 9,
      "time_minutes": 30,
      "_persona": "a working professional organizing their day",
      "_length_instruction": "Write a brief message (1 short sentence)",
      "_tone_style": "professional"
    }
  ],
  "create_note": [
    {
      "text": "free-text",
      "_persona": "a busy parent managing family schedules",
      "_length_instruction": "Write a quick note (3-5 words)",
      "_tone_style": "casual"
    }
  ]
}
```

### Phase 2 Output
```json
{
  "set_alarm": [
    {
      "time_hours": 9,
      "time_minutes": 30,
      "_persona": "a working professional organizing their day",
      "_length_instruction": "Write a brief message (1 short sentence)",
      "_tone_style": "professional"
    }
  ],
  "create_note": [
    {
      "text": "groceries milk bread",
      "_persona": "a busy parent managing family schedules",
      "_length_instruction": "Write a quick note (3-5 words)",
      "_tone_style": "casual"
    }
  ]
}
```

### Phase 3 Output
```json
[
  {
    "user_input": "Set an alarm for 9:30 AM",
    "tool_name": "set_alarm",
    "parameters": {
      "time_hours": 9,
      "time_minutes": 30
    }
  },
  {
    "user_input": "Remind me to buy groceries on the way home",
    "tool_name": "create_note",
    "parameters": {
      "text": "Buy groceries on the way home"
    }
  }
]
```

### Training Format Output (Final)
```json
[
  {
    "input": "Set an alarm for 9:30 AM",
    "output": {
      "function_call": {
        "name": "set_alarm",
        "arguments": {
          "time_hours": 9,
          "time_minutes": 30
        }
      }
    }
  },
  {
    "input": "Remind me to buy groceries on the way home",
    "output": {
      "function_call": {
        "name": "create_note",
        "arguments": {
          "text": "Buy groceries on the way home"
        }
      }
    }
  }
]
```

## Parameter Validation

The sampling process includes automatic validation to ensure tool-specific constraints are met:

### `reminder_absolute` Constraint

This tool requires either:
- `day_offset` is set (e.g., "tomorrow", "monday", "3"), OR
- Both `date_month_day` AND `date_year` are set (e.g., "12-31" and 2025)

The validation function automatically filters out invalid combinations during Phase 1 sampling:

```python
# Valid examples:
{"day_offset": "tomorrow", "date_month_day": None, "date_year": None}  # ✓
{"day_offset": None, "date_month_day": "12-31", "date_year": 2025}    # ✓
{"day_offset": "monday", "date_month_day": "12-31", "date_year": 2025} # ✓

# Invalid examples (automatically rejected):
{"day_offset": None, "date_month_day": None, "date_year": None}       # ✗
{"day_offset": None, "date_month_day": "12-31", "date_year": None}   # ✗
```

The validation uses a retry mechanism (up to 100 attempts) to keep sampling until valid parameters are found.

## Text Generation Diversity Features

The pipeline includes multiple layers of variation to ensure diverse training data:

### Persona Variation
Each sample is assigned a random persona that influences the style and content:
- "a busy parent managing family schedules"
- "a college student tracking assignments and deadlines"
- "a working professional organizing their day"
- "a freelancer juggling multiple projects"
- "someone planning personal errands and tasks"
- And more...

### Length Variation
Samples vary in length based on random instructions:
- "Write a quick note (3-5 words)"
- "Write a brief message (1 short sentence)"
- "Write 1-2 basic sentences"

### Tone/Language Style Variation
Samples use different communication styles:
- **Professional**: Formal business language
- **Casual**: Relaxed, everyday conversation
- **Slang**: Informal, colloquial expressions
- **Abbreviated - minimal words**: Short, concise phrasing

These variations are sampled in Phase 1 as metadata fields (`_persona`, `_length_instruction`, `_tone_style`) and applied during text generation in Phases 2 and 3.

## Advantages of This Approach

1. **Backwards Generation**: Start with parameters, work backwards to user input
2. **Diversity**: Random sampling ensures parameter variety with multi-dimensional variation (persona, length, tone)
3. **Realism**: Claude generates natural, varied text matching specific personas and styles
4. **Modularity**: Each phase is independent and can be re-run
5. **Efficiency**: Parallel API calls for speed
6. **Checkpointing**: Save results after each phase
7. **Validation**: Automatic constraint enforcement for complex tools

## Tips

- Run Phase 1 locally to generate parameters without API costs
- Review Phase 1 output to ensure good parameter distribution
- Phase 2 & 3 use temperature=1.0 for creative outputs
- Re-run Phase 3 with different prompts if needed without redoing Phase 2
- Use smaller batches (10-20) for testing, larger (100+) for production

## Cost Estimation

For 100 samples per tool (600 total examples):
- Phase 2: ~600 API calls (~200 tokens/request)
- Phase 3: ~600 API calls (~150 tokens/request)
- Total: ~1200 API calls, ~210k tokens

At Claude 3.5 Sonnet pricing (~$3/million input tokens, ~$15/million output tokens):
- Estimated cost: $1-2 for 600 examples
