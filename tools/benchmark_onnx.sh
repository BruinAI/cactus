#!/bin/bash
echo ""
echo "Step 1: Building Cactus library..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

if [ ! -d "$PROJECT_ROOT" ]; then
    echo "ERROR: PROJECT_ROOT '$PROJECT_ROOT' does not exist"
    exit 1
fi

cd "$PROJECT_ROOT" || { echo "Failed to cd to PROJECT_ROOT='$PROJECT_ROOT'"; exit 1; }
if ! cactus/build.sh; then
    echo "Failed to build cactus library"
    exit 1
fi

echo ""
cd /home/karen/Documents/cactus/tests
mkdir -p build && cd build
cmake ..
cmake --build .

export CACTUS_CAPTURE_ENABLE=1
export CACTUS_CAPTURE_STDOUT=0        # or 0 if you only want a file
export CACTUS_CAPTURE_FILE=./cactus_capture.log
export CACTUS_CAPTURE_PREVIEW_COUNT=8
export CACTUS_CAPTURE_MAX_ELEMENTS=65536

set -e

cd "$SCRIPT_DIR" || { echo "Failed to cd to SCRIPT_DIR='$SCRIPT_DIR'"; exit 1; }

mkdir -p build
cd build
cmake ..
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

cd "$SCRIPT_DIR"

./build/benchmark_onnx ../graph.bin ../tests/assets/test_monkey.png 1 5 profile.txt