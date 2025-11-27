#!/bin/bash
echo ""
echo "Step 1: Building Cactus library..."
cd "$PROJECT_ROOT"
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
export CACTUS_CAPTURE_STDOUT=1        # or 0 if you only want a file
export CACTUS_CAPTURE_FILE=./cactus_capture.log
export CACTUS_CAPTURE_PREVIEW_COUNT=8
export CACTUS_CAPTURE_MAX_ELEMENTS=65536
./test_monkey_preprocess ../../graph.bin