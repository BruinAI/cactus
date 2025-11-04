# Using Custom Fine-Tuned Gemma Models with BFCL

This guide explains how to evaluate your custom fine-tuned Gemma models with function calling support.

## Model Format

Your fine-tuned model should use the function calling format specified in `PLAN.md`:

- **Tool Definitions**: JSON list in system prompt wrapped by `<tools>`, `</tools>`
- **Tool Calling**: JSON dict wrapped by `<tool_call>`, `</tool_call>`
- **Tool Responding**: JSON dict in user role wrapped by `<tool_response>`, `</tool_response>`

This is based on Qwen's tool calling format, adapted for Gemma's chat template.

## Setup

The custom `GemmaFCHandler` has been added to BFCL to support this format.

## Usage

### Option 1: Using Local Model Path

If your fine-tuned model is stored locally:

```bash
bfcl generate \
  --model google/gemma-3-270m-it-FC \
  --test-category simple_python \
  --num-gpus 1 \
  --backend vllm \
  --local-model-path /path/to/your/finetuned/model
```

### Option 2: Using HuggingFace Model

If you've uploaded your fine-tuned model to HuggingFace:

```bash
# First, update model_name in model_config.py to point to your HF model
# Then run:
bfcl generate \
  --model google/gemma-3-270m-it-FC \
  --test-category simple_python \
  --num-gpus 1 \
  --backend vllm
```

### Parallel Evaluation

To run 8 parallel evaluations with your fine-tuned model:

```bash
# Edit run_parallel_bfcl.sh:
# Change MODEL="google/gemma-3-270m-it" to MODEL="google/gemma-3-270m-it-FC"
# Add LOCAL_MODEL_PATH if needed

cd gemma_tool_use/evaluation
./run_parallel_bfcl.sh
```

Or create a custom script:

```bash
#!/bin/bash
MODEL="google/gemma-3-270m-it-FC"
LOCAL_MODEL_PATH="/path/to/your/finetuned/model"  # Optional
NUM_GPUS=1
BACKEND="vllm"

CATEGORIES=(
    "simple_python"
    "simple_java"
    "simple_javascript"
    "parallel"
    "irrelevance"
    "multi_turn_base"
    "memory_kv"
    "web_search_base"
)

BASE_PORT=8000

for i in "${!CATEGORIES[@]}"; do
    CATEGORY="${CATEGORIES[$i]}"
    PORT=$((BASE_PORT + i))

    LOCAL_SERVER_PORT=$PORT bfcl generate \
        --model "$MODEL" \
        --test-category "$CATEGORY" \
        --num-gpus $NUM_GPUS \
        --backend $BACKEND \
        ${LOCAL_MODEL_PATH:+--local-model-path "$LOCAL_MODEL_PATH"} \
        > "bfcl_${CATEGORY}_fc.log" 2>&1 &

    sleep 2
done

wait
echo "All evaluations complete!"
```

## Evaluation

After generation completes:

```bash
bfcl evaluate \
  --model google/gemma-3-270m-it-FC \
  --test-category simple_python,simple_java,simple_javascript,parallel,irrelevance,multi_turn_base,memory_kv,web_search_base
```

Results will be saved to:
- Generations: `result/google/gemma-3-270m-it-FC/`
- Scores: `score/google/gemma-3-270m-it-FC/`

## Model Handler Details

The `GemmaFCHandler` (`gemma_fc.py`):

1. **Prompt Formatting**: Uses Gemma's `<start_of_turn>...<end_of_turn>` template with XML tool tags
2. **Tool Parsing**: Extracts `<tool_call>` tags and parses JSON
3. **Response Format**: Returns tool calls in BFCL-compatible format

Example formatted prompt:

```
<bos><start_of_turn>system
<tools>
[
  {
    "name": "get_current_weather",
    "description": "Get the current weather",
    "parameters": {...}
  }
]
</tools><end_of_turn>
<start_of_turn>user
What's the weather in Boston?<end_of_turn>
<start_of_turn>model
<tool_call>
{"name": "get_current_weather", "args": {"location": "Boston, MA"}}
</tool_call>
<end_of_turn>
```

## Troubleshooting

### Model Not Found

If you get "Model not found" errors:

1. Check that `--local-model-path` points to a directory with:
   - `config.json`
   - `tokenizer_config.json`
   - Model weights (`.safetensors` or `.bin` files)

2. Or ensure your HuggingFace model name is correct

### Parsing Errors

If you get tool call parsing errors:

1. Verify your fine-tuned model outputs exactly:
   ```
   <tool_call>
   {"name": "function_name", "args": {...}}
   </tool_call>
   ```

2. Check that it's using "args" not "arguments" in the JSON

### Performance Issues

- Use `--num-gpus 1` on TPU (tensor parallelism not supported for Gemma-3-270m-it)
- First run takes 20-30 minutes for XLA compilation
- Subsequent runs are much faster (~5 min)

## Next Steps

1. **Baseline Evaluation**: Run on base model first for comparison
2. **Fine-tune**: Train on function calling data
3. **Evaluate**: Use this guide to run BFCL evaluation
4. **Iterate**: Compare scores and refine your fine-tuning approach
