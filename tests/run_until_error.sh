#!/bin/bash

# Script to run tests/run_nomic.sh repeatedly until a non-finite value is detected
# Max iterations: 100

# Parse command-line arguments
PRECISION=""
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--precision)
            PRECISION="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [-p|--precision FP16|FP32]"
            exit 1
            ;;
    esac
done

MAX_ITERATIONS=100
OUTPUT_FILE="/tmp/nomic_test_output.txt"

# Build the command with precision flag if provided
NOMIC_CMD="./tests/run_nomic.sh"
if [[ -n "$PRECISION" ]]; then
    NOMIC_CMD="$NOMIC_CMD -p $PRECISION"
    echo "Running with precision: $PRECISION"
fi

for i in $(seq 1 $MAX_ITERATIONS); do
    echo "=== Iteration $i of $MAX_ITERATIONS ==="
    
    # Run the test and capture output
    $NOMIC_CMD > "$OUTPUT_FILE" 2>&1
    
    # Check if the output contains the error message
    if grep -q "Non-finite value at index" "$OUTPUT_FILE"; then
        echo ""
        echo "================================================================"
        echo "Found 'Non-finite value at index' in iteration $i!"
        echo "================================================================"
        echo ""
        cat "$OUTPUT_FILE"
        rm -f "$OUTPUT_FILE"
        exit 0
    fi
    
    echo "No error detected in iteration $i"
    echo ""
done

echo "Completed $MAX_ITERATIONS iterations without detecting the error."
rm -f "$OUTPUT_FILE"
exit 1

