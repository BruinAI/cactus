# Gemma 3 270M Tool Calling Training

Training script for fine-tuning Gemma 3 270M on tool calling using LoRA.

## Overview

This script fine-tunes Gemma 3 270M for function calling using:
- **Dataset**: [Toucan-1.5M](https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M) (SFT subset)
- **Method**: LoRA (Low-Rank Adaptation)
- **Format**: Gemma 3 tool calling format as specified in `gemma_tool_use/PLAN.md`

## Dataset Filtering

Following Phase 3 of `PLAN.md`, the script filters the Toucan SFT dataset for:
- **Single-turn conversations only** (no multi-turn interactions)
- **≤2 tools used** per sample (target tools)
- **≤3 tools available** in the prompt

This creates a focused training set for teaching basic tool selection and usage patterns.

## Tool Calling Format

The script implements the Gemma 3 function calling format from `PLAN.md`:

### Tools Definition
```
<bos><start_of_turn>user
Here are the available tools that you can use:
<tools>
[
  {
    "name": "function_name",
    "description": "...",
    "parameters": {...}
  }
]
</tools>

User query here<end_of_turn>
```

### Tool Calls
```
<start_of_turn>model
Optional reasoning text
<tool_call>
{
  "name": "function_name",
  "args": {
    "param1": "value1"
  }
}
</tool_call><end_of_turn>
```

## Configuration

Key hyperparameters in `train_gemma3_270m_tool_calling.py`:

```python
# Model
MODEL_ID = "google/gemma-3-270m-it"

# Training
BATCH_SIZE = 4
NUM_EPOCHS = 3
LEARNING_RATE = 1e-4
MAX_TARGET_LENGTH = 512

# LoRA
RANK = 16
ALPHA = 16

# Dataset filtering
MAX_TOOLS_USED = 2
MAX_TOOLS_AVAILABLE = 3
```

## Requirements

Install required packages:

```bash
pip install kagglehub
pip install safetensors
pip install tensorflow
pip install tensorflow_datasets
pip install tensorboardX
pip install transformers
pip install grain
pip install datasets
pip install wandb
pip install git+https://github.com/google/tunix
pip install git+https://github.com/google/qwix
pip install git+https://github.com/google/flax
```

## Usage

### Basic Training

```bash
python gemma_tool_use/training/train_gemma3_270m_tool_calling.py
```

### Environment Setup

Set environment variables for Kaggle, Weights & Biases, and Hugging Face:

```bash
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_key"
export WANDB_API_KEY="your_wandb_key"  # Optional
export HF_TOKEN="your_hf_token"  # Optional
```

Or use a `.env` file in the project root.

### Kaggle Setup

1. Accept the Gemma license on [Kaggle](https://www.kaggle.com/models/google/gemma/flax/)
2. Get your Kaggle API credentials from your [account settings](https://www.kaggle.com/settings)
3. Place `kaggle.json` in `~/.kaggle/` or set environment variables

## Expected Dataset Size

Based on analysis in `gemma_tool_use/data/toucan_analysis_phase1.csv`:

| Filter | Sample Count | Percentage |
|--------|--------------|------------|
| ≤2 tools + ≤3 available | ~X,XXX | ~X.XX% |

The exact count depends on the Toucan SFT subset size (~119k samples).

## Training Output

The script will:
1. Download Gemma 3 270M from HuggingFace
2. Load and filter the Toucan-1.5M dataset
3. Format examples according to the Gemma 3 tool calling format
4. Apply LoRA adapters to the model
5. Train for the specified number of steps
6. Save LoRA adapter weights to `./gemma3_270m_tool_calling_lora/`

### Saved Outputs

After training completes, you'll find:
- **LoRA weights**: `./gemma3_270m_tool_calling_lora/`
  - Contains `model.*.lora_a.npy` and `model.*.lora_b.npy` files
  - These are the trained adapter weights that can be loaded onto the base model
- **Training checkpoints**: `/tmp/gemma_tool_calling_ckpts/`
- **TensorBoard logs**: `/tmp/gemma_tool_calling_ckpts/tensorboard/`

## Format Compatibility

This format is compatible with:
- `gemma_tool_use/evaluation/bfcl/berkeley-function-call-leaderboard/bfcl_eval/model_handler/local_inference/gemma_fc.py`
- BFCL (Berkeley Function-Calling Leaderboard) evaluation

## Using the Trained Model

The LoRA weights are saved in NumPy format in `./gemma3_270m_tool_calling_lora/`. To use them:

1. Load the base Gemma 3 270M model
2. Apply the saved LoRA weights using qwix
3. Use for inference with the tool calling format

See the `lora_gemma.ipynb` notebook for examples of loading and using LoRA weights.

## Next Steps

After training:
1. Evaluate using BFCL: Test the model on the Berkeley Function-Calling Leaderboard
2. Iterate: Based on evaluation results, adjust hyperparameters or dataset filtering
3. Deploy: Use the LoRA weights for inference on tool calling tasks

## References

- Base training approach: `tuning/lora_gemma.ipynb`
- Tool calling format: `gemma_tool_use/PLAN.md`
- Evaluation format: `gemma_tool_use/evaluation/bfcl/.../gemma_fc.py`
- Dataset analysis: `gemma_tool_use/data/analyze_toucan_phase1.py`
