#!/bin/bash
# Build and test Moonshine model

set -e  # Exit on first error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Export environment variables for tests
export CACTUS_TEST_MOONSHINE_MODEL="$SCRIPT_DIR/weights/moonshine-tiny"
export CACTUS_TEST_ASSETS="$SCRIPT_DIR/tests/assets"
# Enable graph execution capturing
export CACTUS_CAPTURE_ENABLE=1
# Choose output destination (STDOUT or a file)
# export CACTUS_CAPTURE_STDOUT=1
export CACTUS_CAPTURE_FILE="debug_dump.log"
export CACTUS_CAPTURE_DIR="$SCRIPT_DIR/tests/build/dump_cpp"
# Optional: Configure preview details
export CACTUS_CAPTURE_PREVIEW_COUNT=16       # Number of elements to print per tensor
export CACTUS_CAPTURE_MAX_ELEMENTS=1000   # Max elements to capture per tensor

# Debugging options (uncomment to enable)
# export CACTUS_CAPTURE_ENABLE=1
# export CACTUS_CAPTURE_STDOUT=1
# export CACTUS_CAPTURE_PREVIEW_COUNT=16
# export CACTUS_CAPTURE_MAX_ELEMENTS=1000000

rm -rf weights/moonshine-tiny
cactus download UsefulSensors/moonshine-tiny --precision FP16

echo "============================================="
echo "Moonshine Build & Test Script"
echo "============================================="
echo "Model path: $CACTUS_TEST_MOONSHINE_MODEL"
echo "Assets path: $CACTUS_TEST_ASSETS"
echo "Capture dir: $CACTUS_CAPTURE_DIR"
echo "============================================="

# Step 1: Build the Cactus library
echo ""
echo "[Step 1/3] Building Cactus library..."
source setup
cactus build

# Step 2: Build the tests
echo ""
echo "[Step 2/3] Building Moonshine tests..."
cd tests
rm -rf build
mkdir -p build
# Create capture local dir
mkdir -p "$CACTUS_CAPTURE_DIR"
cd build
cmake ..
make -j$(nproc) test_moonshine

# Step 3: Run the tests
echo ""
echo "[Step 3/3] Running Moonshine tests..."
# Ensure env var is passed to the test executable (exported at top)
./test_moonshine

# echo ""
# echo "[Step 4/4] Running Python reference dump..."
# cd "$SCRIPT_DIR"
# python3 tools/dump_hf_moonshine.py \
#     "$CACTUS_TEST_ASSETS/test.wav" \
#     -o "hf_moonshine_dump.log" \
#     --dump-dir "tests/build/dump_python"

# echo ""
# echo "============================================="
# echo "Done!"
# echo "C++ binary dumps: tests/build/dump_cpp"
# echo "Python binary dumps: tests/build/dump_python"
# echo "============================================="
