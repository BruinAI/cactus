#!/bin/bash

set -e


echo "Building Cactus chat..."
echo "======================="

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

WEIGHTS_DIR="$PROJECT_ROOT/weights/lfm2-1.2B"
if [ ! -d "$WEIGHTS_DIR" ] || [ ! -f "$WEIGHTS_DIR/config.txt" ]; then
    echo ""
    echo "LFM2 weights not found. Generating weights..."
    echo "============================================="
    cd "$PROJECT_ROOT"
    if command -v python3 &> /dev/null; then
        # Create temporary venv for conversion
        TEMP_VENV=$(mktemp -d -t cactus-convert-temp-venv)
        echo "Creating temporary venv at $TEMP_VENV..."

        # Trap to ensure cleanup on exit (success or failure)
        trap "rm -rf '$TEMP_VENV'" EXIT

        if python3 -m venv "$TEMP_VENV"; then
            echo "Installing dependencies..."
            if "$TEMP_VENV/bin/pip" install --quiet numpy torch transformers; then
                echo "Running: convert_hf.py LiquidAI/LFM2-1.2B weights/lfm2-1.2B/ --precision INT8"
                if "$TEMP_VENV/bin/python" tools/convert_hf.py LiquidAI/LFM2-1.2B weights/lfm2-1.2B/ --precision INT8; then
                    echo "Successfully generated weights"
                else
                    echo "Warning: Failed to generate weights. Tests may fail."
                    echo "Please run manually: python3 tools/convert_hf.py LiquidAI/LFM2-1.2B weights/lfm2-1.2B/ --precision INT8"
                fi
            else
                echo "Warning: Failed to install dependencies."
                echo "Please run manually: python3 tools/convert_hf.py LiquidAI/LFM2-1.2B weights/lfm2-1.2B/ --precision INT8"
            fi
        else
            echo "Warning: Failed to create venv."
            echo "Please run manually: python3 tools/convert_hf.py LiquidAI/LFM2-1.2B weights/lfm2-1.2B/ --precision INT8"
        fi
    else
        echo "Warning: Python3 not found. Cannot generate weights automatically."
        echo "Please run manually: python3 tools/convert_hf.py LiquidAI/LFM2-1.2B weights/lfm2-1.2B/ --precision INT8"
    fi
else
    echo ""
    echo "LFM2 weights found at $WEIGHTS_DIR"
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$SCRIPT_DIR/.."
BUILD_DIR="$SCRIPT_DIR/build"

mkdir -p "$BUILD_DIR"

cd "$ROOT_DIR/cactus"
if [ ! -f "build/libcactus.a" ]; then
    echo "Cactus library not found. Building..."
    ./build.sh
fi

cd "$BUILD_DIR"

echo "Compiling chat.cpp..."

if [[ "$OSTYPE" == "darwin"* ]]; then
    clang++ -std=c++17 -O3 \
        -I"$ROOT_DIR" \
        "$SCRIPT_DIR/chat.cpp" \
        "$ROOT_DIR/cactus/build/libcactus.a" \
        -o chat \
        -framework Accelerate
else
    g++ -std=c++17 -O3 \
        -I"$ROOT_DIR" \
        "$SCRIPT_DIR/chat.cpp" \
        "$ROOT_DIR/cactus/build/libcactus.a" \
        -o chat \
        -pthread
fi

echo "Build complete: $BUILD_DIR/chat"
echo ""
echo "Usage: $BUILD_DIR/chat <model_path>"
echo "Example: $BUILD_DIR/chat weights/lfm2-1.2B"
