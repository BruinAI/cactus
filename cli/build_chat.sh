#!/bin/bash

set -e

echo "Building Cactus chat..."

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
echo "Example: $BUILD_DIR/chat weights/gemma3-270m"
