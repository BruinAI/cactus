#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure cactus library is built first
if [ ! -f "../cactus/build/libcactus.a" ]; then
    echo "Building cactus library first..."
    cd ../cactus
    ./build.sh
    cd "$SCRIPT_DIR"
fi

# Build tools
mkdir -p build
cd build
cmake ..
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo ""
echo "Build complete. Benchmark tool available at: $SCRIPT_DIR/build/benchmark_onnx"
echo ""
echo "Usage:"
echo "  ./build/benchmark_onnx <ir_path> <input_path> [warmup_runs] [benchmark_runs]"
echo ""
echo "Example:"
echo "  ./build/benchmark_onnx ../graph.bin ../tests/assets/test_monkey.png 10 50"

