#!/bin/bash

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

# Set weights suffix based on precision flag
WEIGHTS_SUFFIX=""
if [[ -n "$PRECISION" ]]; then
    case "$PRECISION" in
        FP16|fp16)
            WEIGHTS_SUFFIX="-fp16"
            ;;
        FP32|fp32)
            WEIGHTS_SUFFIX="-fp32"
            ;;
        *)
            echo "Invalid precision: $PRECISION"
            echo "Must be FP16 or FP32"
            exit 1
            ;;
    esac
fi

echo "Running Cactus Nomic test suite..."
echo "==================================="
if [[ -n "$WEIGHTS_SUFFIX" ]]; then
    echo "Precision: $PRECISION (weights suffix: $WEIGHTS_SUFFIX)"
else
    echo "Precision: Default (no suffix)"
fi
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Export weights suffix for test to use
export CACTUS_WEIGHTS_SUFFIX="$WEIGHTS_SUFFIX"

# Export sanitizer flags for both library and test builds
export CMAKE_CXX_FLAGS="-fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer -g -O1"
export ASAN_OPTIONS="detect_leaks=1:fast_unwind_on_malloc=0"
export UBSAN_OPTIONS="print_stacktrace=1"

echo ""
echo "Step 1: Building Cactus library..."
cd "$PROJECT_ROOT"
if ! cactus/build.sh; then
    echo "Failed to build cactus library"
    exit 1
fi

echo ""
echo "Step 2: Building tests..."
cd "$PROJECT_ROOT/tests"

rm -rf build
mkdir -p build
cd build

if ! cmake ..; then
    echo "Failed to configure tests"
    exit 1
fi

if ! make -j$(nproc 2>/dev/null || echo 4); then
    echo "Failed to build tests"
    exit 1
fi

echo ""
echo "Step 3: Running Nomic tests..."
echo "------------------------------"

if [ ! -x "./test_model_nomic" ]; then
    echo "test_model_nomic executable not found or not executable!"
    exit 1
fi

./test_model_nomic
