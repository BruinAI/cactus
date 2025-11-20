# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Cactus is a fast, lightweight, cross-platform AI inference framework designed for phones and ARM-based devices. It consists of two main components:

1. **Cactus Graph**: A general numerical computing framework (like PyTorch for phones)
2. **Cactus Engine**: An AI inference engine with OpenAI-compatible APIs built on Cactus Graph

The repository also contains an active research project for fine-tuning Gemma 3 models for tool calling (`gemma_tool_use/`).

## Build System

### Core Library Build

Build the Cactus library (required before tests):

```bash
cactus/build.sh
```

This creates `cactus/build/libcactus.a` using CMake.

### Running Tests

The main test command automatically builds dependencies and runs all tests:

```bash
tests/run.sh
```

This script:
1. Generates Qwen3-600m weights if not present (`python3 tools/convert_hf.py Qwen/Qwen3-0.6B weights/qwen3-600m/ --precision INT8`)
2. Builds the Cactus library
3. Builds and runs all test executables

To run individual tests after building:

```bash
cd tests/build
./test_engine    # Engine tests (streaming, tool calls, embeddings)
./test_graph     # Graph tests
./test_kernel    # Kernel tests
```

### Building Chat Demo

```bash
./tools/build_chat.sh
./tools/build/chat weights/gemma3-270m
```

## Weight Conversion

Convert HuggingFace models to Cactus format:

```bash
# Language models (INT8)
python3 tools/convert_hf.py google/gemma-3-270m-it weights/gemma3-270m/ --precision INT8
python3 tools/convert_hf.py Qwen/Qwen3-0.6B weights/qwen3-600m/ --precision INT8
python3 tools/convert_hf.py google/gemma-3-1b-it weights/gemma3-1b/ --precision INT8

# Embedding models
python3 tools/convert_hf.py Qwen/Qwen3-Embedding-0.6B weights/qwen3-embed-600m/
```

After conversion, update the model path in `tests/test_engine.cpp` (line 12: `g_model_path`).

## Architecture

### C++ Core Structure

The codebase is organized into several key directories under `cactus/`:

- **graph/**: Computational graph implementation (graph_core.cpp, graph_builder.cpp, graph_ops.cpp, graph_file.cpp)
  - Low-level graph operations and building blocks
  - File I/O for loading graph structures

- **kernel/**: Optimized computation kernels for ARM
  - `kernel_gemm.cpp`: Matrix multiplication (GEMM)
  - `kernel_attention.cpp`: Attention mechanisms
  - `kernel_blas.cpp`: BLAS operations
  - `kernel_quants.cpp`: Quantization operations (INT8/INT4)
  - `kernel_nn.cpp`: Neural network operations
  - `kernel_reduce.cpp`: Reduction operations
  - All kernels use ARM NEON SIMD intrinsics for performance

- **engine/**: High-level inference engine
  - `engine_model.cpp`: Model loading and management
  - `engine_tokenizer.cpp`, `engine_bpe.cpp`: Tokenization
  - `engine_sp.cpp`: SentencePiece support
  - `engine_cache.cpp`: KV cache management

- **models/**: Model-specific implementations
  - `model_gemma.cpp`: Gemma architecture
  - `model_qwen.cpp`: Qwen architecture
  - `model_lfm2.cpp`: LFM2 architecture
  - `model_smol.cpp`: SmolLM architecture
  - `model_nomic.cpp`: Nomic embeddings

- **ffi/**: Foreign Function Interface
  - `cactus_ffi.cpp`: C API for language bindings
  - Provides clean C interface for SDK integrations

### Key Header Files

- `cactus/cactus.h`: Main include file that brings in all components
- `cactus/graph/graph.h`: Graph API
- `cactus/kernel/kernel.h`: Kernel operations
- `cactus/engine/engine.h`: Engine API
- `cactus/ffi/cactus_ffi.h`: C FFI declarations

### Test Architecture

Tests are in `tests/` directory:
- `test_engine.cpp`: Engine tests (streaming, tool calling, embeddings, context)
- `test_graph.cpp`: Graph operation tests
- `test_kernel.cpp`: Kernel performance tests
- `test_utils.cpp/h`: Shared test utilities

Each test file is built as a separate executable via CMake.

## Gemma Tool Use Research Project

Location: `gemma_tool_use/`

This is an active research project for fine-tuning Gemma 3 models (270M/1B) on tool calling using LoRA. It uses the Toucan-1.5M dataset and targets the Berkeley Function Calling Leaderboard (BFCL) evaluation.

### Key Files

- `PLAN.md`: Detailed execution plan and format specifications
- `training/train_gemma3_tool_calling.py`: Main training script (TPU-optimized)
- `training/data_utils.py`: Dataset loading and preprocessing
- `training/format_bfcl_style.py`: BFCL-style format (Python function signatures)
- `training/format_qwen_style.py`: Qwen-style XML format (deprecated)
- `training/gemma_utils.py`: Model utilities
- `evaluation/bfcl/`: Berkeley Function Calling Leaderboard evaluation suite (git submodule)

### Tool Calling Format

The current active format is **Philschmid Python-style** (not Qwen-style XML):

- **Tool definitions**: Python function signatures with docstrings in ` ```python ` blocks
- **Tool calls**: Python function calls in ` ```tool_code ` blocks (e.g., `get_weather(location="Boston, MA", unit="fahrenheit")`)
- **Tool responses**: Results in ` ```tool_output ` blocks

See `PLAN.md` for complete format specification and examples.

### Training Configuration

- Optimized for TPU v5e-4 chips (4x TPU setup)
- LoRA fine-tuning (rank 32-64, alpha 64)
- Batch size: 8, effective batch size: 64 (via gradient accumulation)
- Learning rate: 2e-4
- Max sequence length: 4096 tokens
- Filtering: Currently trains on samples with ≤10 tools used, ≤10 tools available, ≤2 turns

### Running Training

Training scripts assume TPU environment with JAX:

```bash
cd gemma_tool_use/training
# Activate venv if needed
python3 train_gemma3_tool_calling.py
```

### Evaluation

BFCL evaluation is integrated as a git submodule:

```bash
cd gemma_tool_use/evaluation/bfcl/berkeley-function-call-leaderboard
bfcl generate --model <model_name> --test-category <category>
```

## DCO Requirements

All commits MUST be signed-off with Developer Certificate of Origin:

```bash
# Set up automatic sign-off (recommended)
./tools/setup-dco.sh

# Or sign-off manually with each commit
git commit -s -m "Your message"
```

## Performance Context

- Target: 60+ tokens/sec on M3 CPU with Qwen3-600m-INT8
- Optimized for ARM NEON SIMD instructions
- INT8 quantization for reduced memory and faster inference
- KV cache management for efficient generation

## Common Patterns

### Model Path Configuration

Model paths are typically hardcoded in test files as constants (e.g., `g_model_path` in test_engine.cpp). Update these when testing different models.

### C++ FFI Usage Pattern

```cpp
#include "cactus.h"

cactus_model_t model = cactus_init("path/to/weights", 2048);
char response[4096];
int result = cactus_complete(model, messages_json, response, sizeof(response),
                             options_json, tools_json, callback, user_data);
cactus_destroy(model);
```

See `docs/cactus_engine.md` for complete FFI documentation.

### Adding New Model Support

1. Create new model file in `cactus/models/model_<name>.cpp`
2. Implement model-specific architecture
3. Register model in engine
4. Add weight conversion support in `tools/convert_hf.py`

## Development Notes

- **Platform**: ARM-optimized, runs on Apple Silicon, iOS, Android
- **Build system**: CMake with shell script wrappers
- **C++ Standard**: C++17
- **Key dependencies**: ARM NEON intrinsics, Accelerate framework (macOS)
- **Thread safety**: Each model instance should be used from a single thread