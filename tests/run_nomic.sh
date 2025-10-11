#!/bin/bash

echo "Running Cactus Nomic test suite..."
echo "==================================="

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

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
