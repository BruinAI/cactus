# Gemma 3 270M Tool Use Fine-Tuning

This directory contains the complete pipeline for fine-tuning Gemma 3 270M on tool calling tasks.

## Overview

Multi-stage fine-tuning pipeline:
1. **Stage 1**: Define function calling format (documented in PLAN.md)
2. **Stage 2**: Setup BFCL (Berkeley Function Calling Leaderboard) evaluation
3. **Stage 3**: SFT on 1-3 tasks (initial focused training)
4. **Stage 4**: SFT on broader task set (generalization)
5. **Stage 5**: GRPO (Group Relative Policy Optimization) - optional

## Directory Structure

```
gemma_tool_use/
├── config/                      # Training configurations for each stage
├── data/                        # Data preparation scripts
├── evaluation/                  # Evaluation harness (BFCL)
├── training/                    # Training scripts (SFT, GRPO)
├── orchestration/              # Pipeline orchestration
├── PLAN.md                     # Detailed execution plan
└── README.md                   # This file
```

## Quick Start

### 1. Setup Environment

Ensure you have the TPU environment set up as described in `tuning/README.md`:
- Python 3.12
- JAX with TPU support
- Tunix library

### 2. Run the Pipeline

```bash
# Run all stages
python orchestration/run_pipeline.py

# Run specific stages
python orchestration/run_pipeline.py --stages 2 3 4

# Resume from checkpoint
python orchestration/run_pipeline.py --stages 4 5 --resume checkpoints/stage3
```

## Tool Calling Format

Following Qwen's approach, the format uses XML tags:

### System Prompt
```
<tools>
[
  {
    "name": "function_name",
    "description": "Function description",
    "parameters": { ... }
  }
]
</tools>
```

### Model Tool Calls
```
<tool_call>
{
  "name": "function_name",
  "args": { ... }
}
</tool_call>
```

### Tool Responses (user role)
```
<tool_response>
{
  "name": "function_name",
  "result": { ... }
}
</tool_response>
```

See PLAN.md for complete example conversation.

## Configuration

Each stage has a YAML config file in `config/`:
- `stage1_sft_small.yaml`: Initial SFT on 1-3 tasks
- `stage2_sft_full.yaml`: Full SFT on broader task set
- `stage3_grpo.yaml`: GRPO reinforcement learning
- `pipeline.yaml`: Overall pipeline settings

## Evaluation

BFCL (Berkeley Function Calling Leaderboard) evaluation runs after each training stage to track progress.

## Expected Performance

Target metrics:
- Stage 2 (baseline): ~X% BFCL accuracy
- Stage 3 (small SFT): ~Y% BFCL accuracy
- Stage 4 (full SFT): ~Z% BFCL accuracy
- Stage 5 (GRPO): ~W% BFCL accuracy

(Values TBD after initial runs)

## Hardware Requirements

- TPU v5litepod-8 recommended for training
- Training time estimates:
  - Stage 3: ~X hours
  - Stage 4: ~Y hours
  - Stage 5: ~Z hours

## Notes

- All training uses LoRA for efficiency
- Checkpoints saved after each stage
- Automatic evaluation and tracking
- Supports resuming from any stage
