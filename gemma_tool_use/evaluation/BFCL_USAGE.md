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

### Basic Usage

```bash
# Step 1: Generate model responses
bfcl generate \
  --model MODEL_NAME \
  --test-category CATEGORIES \
  --num-gpus 1 \
  --backend vllm

# Step 2: Evaluate responses
bfcl evaluate \
  --model MODEL_NAME \
  --test-category CATEGORIES
```

### Model Options

**Base models (prompt mode):**
- `google/gemma-3-270m-it`
- `google/gemma-3-1b-it`, `google/gemma-3-4b-it`, `google/gemma-3-12b-it`, `google/gemma-3-27b-it`

**Fine-tuned FC models:**
- `google/gemma-3-270m-it-FC` (requires `--local-model-path` or model in HF cache)

**Using local models:**
```bash
bfcl generate \
  --model google/gemma-3-270m-it-FC \
  --local-model-path /path/to/your/fine-tuned-model \
  --test-category simple_python,simple_java,simple_javascript \
  --num-gpus 1 \
  --backend vllm
```

### Example: Quick Test

```bash
# Base model (prompt mode)
bfcl generate --model google/gemma-3-270m-it --test-category simple_python,simple_java,simple_javascript --num-gpus 1 --backend vllm
bfcl evaluate --model google/gemma-3-270m-it --test-category simple_python,simple_java,simple_javascript

# Fine-tuned FC model
bfcl generate --model google/gemma-3-270m-it-FC --local-model-path /path/to/model --test-category simple_python --num-gpus 1 --backend vllm
bfcl evaluate --model google/gemma-3-270m-it-FC --test-category simple_python
```

**Results:**
- Generation: `result/{model-name}/{test-category}.json`
- Evaluation: `score/{model-name}/data_overall.csv` (and category-specific CSVs)

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

## Notes

- Gemma models are supported in both **Prompt mode** and **Function Calling (FC) mode**
- Supported base models: `google/gemma-3-270m-it`, `google/gemma-3-1b-it`, `google/gemma-3-4b-it`, `google/gemma-3-12b-it`, `google/gemma-3-27b-it`
- Supported FC models: `google/gemma-3-270m-it-FC` (custom fine-tuned models)
- Generation requires GPU or TPU (will use vllm or sglang backend)
- **For TPU VMs**: Use `--backend vllm` and `--num-gpus 1`
  - **Gemma-3-270m-it**: Must use `--num-gpus 1` (has 1 KV head, incompatible with tensor parallelism)
  - **Gemma-3-1b-it**: Likely also requires `--num-gpus 1` due to similar architecture
  - Larger models may support higher TP values depending on their KV head count
  - Tensor parallelism divides by `num_key_value_heads`, not `num_attention_heads`
- **First run on TPU**: Expect 20-30 minutes for XLA compilation; subsequent runs are much faster (~5 min)
- **Results location**: Saved to `result/google/gemma-3-270m-it/<test-category>.json`
- Use `--allow-overwrite` or `-o` to regenerate existing results

## Running Parallel Evaluations on TPU v5litepod-8

Since each evaluation only uses 1 TPU chip, you can run **8 parallel evaluations** to utilize all chips:

```bash
# Use the provided script
cd gemma_tool_use/evaluation
./run_parallel_bfcl.sh
```

This runs 8 different test categories simultaneously, each on its own vLLM server (ports 8000-8007). Logs are saved to `bfcl_<category>.log`.
