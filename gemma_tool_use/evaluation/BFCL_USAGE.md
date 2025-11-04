# BFCL Usage Guide

## Setup

```bash
# 1. Clone repo and checkout branch
git clone <your-repo-url>
cd cactus
git checkout gemma-tool-use

# 2. Initialize BFCL submodule
git submodule update --init --recursive

# 3. Navigate to gemma_tool_use and install BFCL
cd gemma_tool_use
pip install -e evaluation/bfcl/berkeley-function-call-leaderboard

# 4. Verify
bfcl --help
```

## Running Evaluations

### Step 1: Generate Model Responses

For Gemma models (using prompt mode, not function calling):

```bash
# Test with simple_python category on gemma-3-270m-it
bfcl generate \
  --model google/gemma-3-270m-it \
  --test-category simple_python \
  --num-gpus 1 \
  --backend vllm

# Test with gemma-3-1b-it
bfcl generate \
  --model google/gemma-3-1b-it \
  --test-category simple_python \
  --num-gpus 1 \
  --backend vllm

# For multiple categories
bfcl generate \
  --model google/gemma-3-270m-it \
  --test-category simple_python,parallel,multiple \
  --num-gpus 1 \
  --backend vllm
```

Results will be saved to: `result/google/gemma-3-270m-it/` or `result/google/gemma-3-1b-it/`

### Step 2: Evaluate Generated Responses

```bash
# Evaluate the generated responses for 270m model
bfcl evaluate \
  --model google/gemma-3-270m-it \
  --test-category simple_python

# Evaluate for 1b model
bfcl evaluate \
  --model google/gemma-3-1b-it \
  --test-category simple_python

# For multiple categories
bfcl evaluate \
  --model google/gemma-3-270m-it \
  --test-category simple_python,parallel,multiple
```

Results will be saved to: `score/google/gemma-3-270m-it/` or `score/google/gemma-3-1b-it/`

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

For a quick baseline test:

```bash
# Run simple_python category on gemma-3-270m-it (smallest/fastest)
bfcl generate --model google/gemma-3-270m-it --test-category simple_python --num-gpus 1 --backend vllm
bfcl evaluate --model google/gemma-3-270m-it --test-category simple_python

# Or test on gemma-3-1b-it
bfcl generate --model google/gemma-3-1b-it --test-category simple_python --num-gpus 1 --backend vllm
bfcl evaluate --model google/gemma-3-1b-it --test-category simple_python
```

## Notes

- Gemma models are supported in **Prompt mode** only (not FC mode)
- Supported models: `google/gemma-3-270m-it`, `google/gemma-3-1b-it`, `google/gemma-3-4b-it`, `google/gemma-3-12b-it`, `google/gemma-3-27b-it`
- Generation requires GPU or TPU (will use vllm or sglang backend)
- **For TPU VMs**: Use `--backend vllm` and `--num-gpus 1` (tensor parallelism >1 not yet stable on v5litepod-8)
- **First run on TPU**: Expect 20-30 minutes for XLA compilation; subsequent runs are much faster (~5 min)
- Results are saved relative to the BFCL directory
- Use `--allow-overwrite` or `-o` to regenerate existing results
