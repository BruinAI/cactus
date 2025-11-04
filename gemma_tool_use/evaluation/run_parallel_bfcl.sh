#!/bin/bash
# Script to run 8 parallel BFCL evaluations on different test categories
# Each uses a separate vLLM server on a different port

MODEL="google/gemma-3-270m-it"
NUM_GPUS=1
BACKEND="vllm"

# 8 diverse test categories covering different aspects
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

# Starting port for vLLM servers
BASE_PORT=8000

echo "Starting 8 parallel BFCL evaluation jobs..."
echo "Results will be saved to: result/$MODEL/"

# Launch each job in the background with its own port
for i in "${!CATEGORIES[@]}"; do
    CATEGORY="${CATEGORIES[$i]}"
    PORT=$((BASE_PORT + i))

    echo "[$i] Launching: $CATEGORY on port $PORT"

    # Set environment variable for this specific job and run in background
    LOCAL_SERVER_PORT=$PORT bfcl generate \
        --model "$MODEL" \
        --test-category "$CATEGORY" \
        --num-gpus $NUM_GPUS \
        --backend $BACKEND \
        > "bfcl_${CATEGORY}.log" 2>&1 &

    # Store the process ID
    PIDS[$i]=$!

    # Small delay to avoid startup race conditions
    sleep 2
done

echo ""
echo "All jobs launched! Process IDs:"
for i in "${!PIDS[@]}"; do
    echo "  [${CATEGORIES[$i]}]: PID ${PIDS[$i]} (port $((BASE_PORT + i)))"
done

echo ""
echo "Monitor logs with: tail -f bfcl_*.log"
echo "Check progress: ps aux | grep bfcl"
echo ""
echo "Waiting for all jobs to complete..."

# Wait for all background jobs
for i in "${!PIDS[@]}"; do
    wait ${PIDS[$i]}
    echo "✓ ${CATEGORIES[$i]} completed"
done

echo ""
echo "All evaluations complete! Results saved to: result/$MODEL/"
echo ""
echo "Run evaluation with:"
echo "  bfcl evaluate --model $MODEL --test-category simple_python,simple_java,simple_javascript,parallel,irrelevance,multi_turn_base,memory_kv,web_search_base"
