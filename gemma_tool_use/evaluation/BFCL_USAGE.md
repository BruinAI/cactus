# BFCL Usage Guide

## Setup

BFCL is installed in a uv virtual environment.

```bash
# Activate the virtual environment
cd gemma_tool_use
source .venv/bin/activate

# To reinstall or update BFCL
uv pip install -e evaluation/bfcl/gorilla/berkeley-function-call-leaderboard
```

Once activated, the `bfcl` command will be available in your shell.

## Running Evaluations

### Step 1: Generate Model Responses

For Gemma models (using prompt mode, not function calling):

```bash
# Test with simple_python category on gemma-3-1b-it
bfcl generate \
  --model google/gemma-3-1b-it \
  --test-category simple_python \
  --num-gpus 1

# For multiple categories
bfcl generate \
  --model google/gemma-3-1b-it \
  --test-category simple_python,parallel,multiple \
  --num-gpus 1
```

Results will be saved to: `result/google/gemma-3-1b-it/`

### Step 2: Evaluate Generated Responses

```bash
# Evaluate the generated responses
bfcl evaluate \
  --model google/gemma-3-1b-it \
  --test-category simple_python

# For multiple categories
bfcl evaluate \
  --model google/gemma-3-1b-it \
  --test-category simple_python,parallel,multiple
```

Results will be saved to: `score/google/gemma-3-1b-it/`

### Step 3: View Results

Results are saved in CSV files in the `score/` directory:
- `data_overall.csv` - Overall scores
- `data_live.csv` - Live category breakdown
- `data_non_live.csv` - Non-live category breakdown
- `data_multi_turn.csv` - Multi-turn category breakdown

## Test Categories

Run `bfcl test-categories` to see all available categories.

Common categories:
- `simple_python` - Simple Python function calls (fastest for quick tests)
- `simple_java` - Simple Java function calls
- `simple_javascript` - Simple JavaScript function calls
- `parallel` - Parallel function calls
- `multiple` - Multiple function calls
- `parallel_multiple` - Parallel multiple function calls
- `irrelevance` - Irrelevant function calls
- `live_simple` - Live simple function calls
- `live_parallel` - Live parallel function calls
- `live_multiple` - Live multiple function calls
- `multi_turn_base` - Multi-turn base
- `multi_turn_miss_func` - Multi-turn with missing functions
- `multi_turn_miss_param` - Multi-turn with missing parameters
- `multi_turn_long_context` - Multi-turn long context
- `memory_kv`, `memory_vector`, `memory_rec_sum` - Memory categories
- `web_search_base`, `web_search_no_snippet` - Web search categories

Category groups:
- `all` - All test categories (very long)
- `single_turn` - All single-turn categories
- `multi_turn` - All multi-turn categories
- `python` - All Python-specific categories
- `non_python` - Java and JavaScript categories
- `live` - All live categories
- `non_live` - All non-live categories
- `agentic` - Memory and web search categories

## Quick Test Command

For a quick baseline test on gemma-3-1b-it:

```bash
# Run simple_python category only (fastest)
bfcl generate --model google/gemma-3-1b-it --test-category simple_python --num-gpus 1
bfcl evaluate --model google/gemma-3-1b-it --test-category simple_python
```

## Notes

- Gemma models are supported in **Prompt mode** only (not FC mode)
- Generation requires GPU (will use vllm or sglang backend)
- Results are saved relative to the BFCL directory
- Use `--allow-overwrite` or `-o` to regenerate existing results
