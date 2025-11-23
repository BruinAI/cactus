# Pebble Qwen - Unsloth Training

This directory contains the Unsloth-based implementation for fine-tuning Qwen 3 models on tool calling.

## Overview

This is a reimplementation of the original `pebble_qwen` training script using [Unsloth](https://github.com/unslothai/unsloth), which provides:
- **2x faster training** compared to standard HuggingFace Trainer
- **80% less memory usage** with 4-bit quantization
- **Simpler codebase** (~450 lines vs 1000 lines)
- **GPU-optimized** instead of TPU-optimized

## Key Differences from Original

| Aspect | Original (JAX/Tunix) | Unsloth (PyTorch) |
|--------|---------------------|-------------------|
| Framework | JAX + Tunix | PyTorch + Transformers |
| Hardware | TPU-optimized | GPU-optimized (CUDA) |
| Model Loading | Custom params_lib | FastLanguageModel |
| LoRA | qwix.LoraProvider | FastLanguageModel.get_peft_model |
| Data Loading | Grain with custom transforms | HuggingFace datasets |
| Training | tunix PeftTrainer | TRL SFTTrainer |
| Quantization | Manual INT8 | Built-in 4bit/8bit |
| Memory | Manual JAX sharding | Automatic gradient checkpointing |

## Hyperparameters (Unsloth Defaults)

All hyperparameters use Unsloth's recommended defaults:

### LoRA Configuration
- **rank**: 16
- **lora_alpha**: 16 (same as rank)
- **lora_dropout**: 0 (no dropout)
- **target_modules**: `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`

### Training Configuration
- **learning_rate**: 2e-4
- **per_device_batch_size**: 2
- **gradient_accumulation_steps**: 4 (effective batch size: 8)
- **warmup_steps**: 10
- **num_epochs**: 3
- **max_seq_length**: 2048
- **optimizer**: adamw_8bit
- **weight_decay**: 0.01

## Installation

### 1. Install PyTorch with CUDA support

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2. Install Unsloth and dependencies

```bash
cd pebble_qwen_unsloth
pip install -r requirements.txt
```

## Usage

### Basic Training

```bash
python3 train_qwen3_tool_calling.py
```

### With Custom Parameters

```bash
python3 train_qwen3_tool_calling.py \
    --model_id Qwen/Qwen3-0.6B \
    --num_epochs 3 \
    --learning_rate 2e-4 \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lora_rank 16 \
    --lora_alpha 16 \
    --max_seq_length 2048 \
    --output_dir outputs/my_training \
    --final_model_dir models/my_model
```

### Available Arguments

- `--model_id`: HuggingFace model ID (default: Qwen/Qwen3-0.6B)
- `--learning_rate`: Learning rate (default: 2e-4)
- `--num_epochs`: Number of training epochs (default: 3)
- `--batch_size`: Batch size per device (default: 2)
- `--gradient_accumulation_steps`: Gradient accumulation (default: 4)
- `--max_seq_length`: Maximum sequence length (default: 2048)
- `--lora_rank`: LoRA rank (default: 16)
- `--lora_alpha`: LoRA alpha (default: 16)
- `--dataset_path`: Path to dataset JSON (default: data/synthetic_finetune_dataset.json)
- `--tools_path`: Path to tools JSON (default: data/tools.json)
- `--train_test_split`: Train/test split ratio (default: 0.1)
- `--output_dir`: Training output directory (default: outputs/qwen3_tool_calling_unsloth)
- `--final_model_dir`: Final merged model directory (default: models/qwen3_tool_calling_merged)

## Output

The training script produces:

1. **Training checkpoints**: Saved to `--output_dir` (default: `outputs/qwen3_tool_calling_unsloth`)
2. **Final merged model**: Saved to `--final_model_dir` (default: `models/qwen3_tool_calling_merged`)

The final model is saved in HuggingFace safetensors format and includes:
- `model.safetensors` - Merged weights (LoRA + base model)
- `config.json` - Model configuration
- `tokenizer.json` - Fast tokenizer
- `tokenizer_config.json` - Tokenizer configuration
- `special_tokens_map.json` - Special tokens
- Other configuration files

## Loading the Trained Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("models/qwen3_tool_calling_merged")
tokenizer = AutoTokenizer.from_pretrained("models/qwen3_tool_calling_merged")
```

Or with Unsloth for continued training:

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="models/qwen3_tool_calling_merged",
    max_seq_length=2048,
    load_in_4bit=True,
)
```

## Dataset Format

The training script uses the same dataset as the original `pebble_qwen`:
- **Input**: `data/synthetic_finetune_dataset.json`
- **Tools**: `data/tools.json`

Format:
```json
{
  "input": "user message",
  "output": {
    "function_call": {
      "name": "function_name",
      "arguments": {"arg1": "value1"}
    }
  }
}
```

The dataset is automatically formatted using Qwen3's native tool calling format with `<tools>` and `<tool_call>` XML tags.

## Hardware Requirements

- **Minimum**: GPU with 8GB VRAM (for Qwen3-0.6B with 4-bit quantization)
- **Recommended**: GPU with 16GB+ VRAM (for larger models or longer sequences)
- **CPU training**: Not recommended (very slow)

## Memory Usage

With 4-bit quantization and gradient checkpointing:
- **Qwen3-0.6B**: ~6GB VRAM
- **Qwen3-1.7B**: ~10GB VRAM
- **Qwen3-8B**: ~20GB VRAM

## Training Speed

On NVIDIA RTX 4090 with Qwen3-0.6B:
- **Tokens/sec**: ~3000-4000
- **Time per epoch**: ~5-10 minutes (1000 samples)
- **Total training time**: ~15-30 minutes (3 epochs)

## Troubleshooting

### CUDA Out of Memory

Reduce memory usage:
```bash
python3 train_qwen3_tool_calling.py \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_seq_length 1024
```

### Slow Training

Increase batch size if you have more memory:
```bash
python3 train_qwen3_tool_calling.py \
    --batch_size 4 \
    --gradient_accumulation_steps 2
```

## References

- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [Unsloth Documentation](https://docs.unsloth.ai/)
- [Qwen3 How to Run & Fine-tune](https://docs.unsloth.ai/models/qwen3-how-to-run-and-fine-tune)
- [LoRA Hyperparameters Guide](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide)
- [Original Qwen3 Alpaca Notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(14B)-Alpaca.ipynb)
