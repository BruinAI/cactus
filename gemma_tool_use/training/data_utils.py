#!/usr/bin/env python3
"""
Data loading and preprocessing utilities for Gemma 3 tool calling training.

This module handles:
- Dataset loading and filtering (Toucan-1.5M)
- Tokenization and masking
- Grain DataLoader creation

Note: Tool calling format functions have been moved to format_qwen_style.py
"""

import json
import logging
from typing import Dict, Any

import numpy as np
from datasets import load_dataset, Dataset
import grain.python as grain

from tunix.generate import tokenizer_adapter as tokenizer_lib
from tunix.sft import peft_trainer

logger = logging.getLogger(__name__)


# ============================================================================
# Dataset Loading and Filtering
# ============================================================================

def is_english_only(text: str) -> bool:
    """
    Fast heuristic to check if text is primarily English.
    Returns False if text contains significant non-English characters.
    Only removes ~3% of the dataset.

    Args:
        text: Text to check

    Returns:
        True if English-only, False if contains non-English scripts
    """
    if not text or len(text) < 10:
        return True

    # Count non-English script characters
    cjk_count = sum(1 for c in text if 0x4E00 <= ord(c) <= 0x9FFF)  # Chinese
    arabic_count = sum(1 for c in text if 0x0600 <= ord(c) <= 0x06FF)  # Arabic
    hangul_count = sum(1 for c in text if 0xAC00 <= ord(c) <= 0xD7AF)  # Korean

    # Latin Extended (accented characters used in Portuguese, Spanish, French, etc.)
    # This includes: á, é, í, ó, ú, ã, õ, ç, ñ, ü, etc.
    latin_extended_count = sum(1 for c in text if 0x00C0 <= ord(c) <= 0x00FF or 0x0100 <= ord(c) <= 0x017F)

    # If more than threshold non-English characters, consider it non-English
    # Allow up to 2 accented characters for place names like "São Paulo"
    return (cjk_count < 5 and
            arabic_count < 5 and
            hangul_count < 5 and
            latin_extended_count < 3)


def filter_toucan_dataset(dataset, max_tools_used, max_tools_available, max_number_of_turns, english_only=True) -> Dataset:
    """
    Filter Toucan dataset for single-turn examples with limited tools.

    Args:
        dataset: Toucan dataset (SFT split)
        max_tools_used: Maximum number of tools used in target (default: 2)
        max_tools_available: Maximum number of tools available in prompt (default: 3)
        max_number_of_turns: Maximum number of user turns to keep (default: 1)
        english_only: Filter to English-only samples (default: True)

    Returns:
        Filtered dataset
    """
    print(f"\nFiltering dataset (≤{max_number_of_turns} turns, ≤{max_tools_used} tools used, ≤{max_tools_available} tools available, english_only={english_only})...")

    total = len(dataset)

    def reduce_user_messages(sample):
        """Remove all messages after max_number_of_turns user messages."""
        messages = json.loads(sample['messages'])
        user_message_idxs = [idx for idx, m in enumerate(messages) if m['role'] == 'user']
        if len(user_message_idxs) <= max_number_of_turns:
            return sample
        cutoff_idx = user_message_idxs[max_number_of_turns]
        sample['messages'] = json.dumps(messages[:cutoff_idx])
        return sample

    filtered_dataset = dataset.map(
        reduce_user_messages,
        num_proc=16,
        desc=f"Reducing messages to {max_number_of_turns} user turns"
    )

    def filter_fn(sample):
        """Filter function for a single sample."""
        messages = json.loads(sample['messages'])

        user_messages = [m for m in messages if m['role'] == 'user']
        if len(user_messages) == 0:
            return False

        num_tool_calls = sum(1 for m in messages if m['role'] == 'tool_call')
        num_tool_responses = sum(1 for m in messages if m['role'] == 'tool_response')
        assert num_tool_calls == num_tool_responses

        if num_tool_calls > max_tools_used:
            return False

        if len(json.loads(sample['tools'])) > max_tools_available:
            return False

        if english_only:
            assert all('content' in m and isinstance(m['content'], str) for m in messages)
            all_text = [msg['content'] for msg in messages]
            if not is_english_only(' '.join(all_text)):
                return False

        return True

    filtered_dataset = filtered_dataset.filter(
        filter_fn,
        num_proc=16,
        desc="Filtering samples"
    )

    print(f"Filtered dataset size: {len(filtered_dataset):,} samples ({100 * len(filtered_dataset) / total:.2f}% retained)")

    return filtered_dataset


# ============================================================================
# Grain DataLoader Implementation
# ============================================================================

class _Tokenize(grain.MapTransform):
    """Tokenize role-based messages and create proper masks."""

    def __init__(self, tokenizer: tokenizer_lib.Tokenizer):
        self._tokenizer = tokenizer

    def map(self, element: Dict[str, str]) -> Dict[str, np.ndarray]:
        """
        Tokenize role messages and create loss mask.

        The text field already contains complete formatting:
        - User: "\n<start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n"
        - Model: "{content}<end_of_turn>"

        Returns dict with:
        - tokens: Full token sequence
        - mask: Loss mask (1 for model outputs, 0 for user inputs)
        """
        # Parse role messages
        role_messages = json.loads(element['role_messages'])

        all_tokens = []
        all_masks = []

        for turn in role_messages:
            role = turn['role']
            text = turn['text']

            # Tokenize the complete text (which already has formatting tags)
            tokens = self._tokenizer.encode(text)

            # Create mask based on role
            if role == 'user':
                # Don't train on user input (including the <start_of_turn>model\n at the end)
                all_tokens.extend(tokens)
                all_masks.extend([0] * len(tokens))

            elif role == 'model':
                # TRAIN on model output (including <end_of_turn>)
                all_tokens.extend(tokens)
                all_masks.extend([1] * len(tokens))

        return {
            'tokens': np.array(all_tokens, dtype=np.int32),
            'mask': np.array(all_masks, dtype=np.float32)
        }


class _BuildTrainInput(grain.MapTransform):
    """Build TrainingInput from tokens with proper loss masking."""

    def __init__(self, max_seq_len: int, pad_value: int):
        self._max_seq_len = max_seq_len
        self._pad_value = pad_value

    def map(self, tokenized_dict: Dict[str, np.ndarray]) -> peft_trainer.TrainingInput:
        """Build training input from tokens and mask."""
        tokens = tokenized_dict['tokens']
        mask = tokenized_dict['mask']

        # Pad or truncate to max_seq_len
        if len(tokens) > self._max_seq_len:
            tokens = tokens[:self._max_seq_len]
            mask = mask[:self._max_seq_len]
        else:
            pad_len = self._max_seq_len - len(tokens)
            tokens = np.pad(tokens, [[0, pad_len]], mode='constant', constant_values=self._pad_value)
            mask = np.pad(mask, [[0, pad_len]], mode='constant', constant_values=0)  # Pad with 0 (don't train)

        return peft_trainer.TrainingInput(
            input_tokens=tokens,
            input_mask=mask
        )


class _FilterOverlength(grain.FilterTransform):
    """Filter out overlength examples."""

    def __init__(self, max_seq_len: int):
        self._max_seq_len = max_seq_len

    def filter(self, element: peft_trainer.TrainingInput) -> bool:
        return element.input_tokens.shape[0] <= self._max_seq_len


def _build_data_loader(
    *,
    data_source: grain.RandomAccessDataSource,
    batch_size: int,
    num_epochs: int,
    max_seq_len: int,
    tokenizer: tokenizer_lib.Tokenizer,
    shuffle: bool,
    seed: int = 42
) -> grain.DataLoader:
    """Build a grain DataLoader."""
    # Create sampler
    sampler = grain.IndexSampler(
        num_records=len(data_source),
        num_epochs=num_epochs,
        shard_options=grain.NoSharding(),
        shuffle=shuffle,
        seed=seed if shuffle else None,
    )

    # Create data loader with transformations
    return grain.DataLoader(
        data_source=data_source,
        sampler=sampler,
        operations=[
            _Tokenize(tokenizer),
            _BuildTrainInput(max_seq_len, tokenizer.pad_id()),
            _FilterOverlength(max_seq_len),
            grain.Batch(batch_size=batch_size, drop_remainder=True),
        ],
    )


def create_tool_calling_dataset(
    tokenizer,
    global_batch_size,
    max_target_length,
    num_train_epochs,
    max_tools_used,
    max_tools_available,
    max_number_of_turns,
    format_function,
    use_full_dataset=False
):
    """
    Create and format the tool calling dataset.

    Args:
        tokenizer: Gemma tokenizer
        global_batch_size: Batch size for training
        max_target_length: Maximum sequence length
        num_train_epochs: Number of training epochs
        max_tools_used: Maximum number of tools used per example
        max_tools_available: Maximum number of tools available per example
        format_function: Function to format examples (takes batched examples dict)
        use_full_dataset: If True, load all configurations (Kimi-K2, OSS, Qwen3, SFT).
                          If False, only load SFT split (default: False)

    Returns:
        Tuple of (train_loader, validation_loader, total_steps, train_dataset)
    """
    print(f"\n{'='*60}")
    print("Loading Toucan-1.5M dataset")
    print(f"{'='*60}")

    if use_full_dataset:
        # Load all configurations: Kimi-K2, OSS, Qwen3, and SFT
        print("Loading ALL dataset configurations (Kimi-K2, OSS, Qwen3, SFT)...")

        # Load each configuration
        from datasets import concatenate_datasets

        kimi_dataset = load_dataset('Agent-Ark/Toucan-1.5M', 'Kimi-K2', split='train')
        oss_dataset = load_dataset('Agent-Ark/Toucan-1.5M', 'OSS', split='train')
        qwen_dataset = load_dataset('Agent-Ark/Toucan-1.5M', 'Qwen3', split='train')
        sft_dataset = load_dataset('Agent-Ark/Toucan-1.5M', 'SFT', split='train')

        print(f"  Kimi-K2: {len(kimi_dataset):,} samples")
        print(f"  OSS: {len(oss_dataset):,} samples")
        print(f"  Qwen3: {len(qwen_dataset):,} samples")
        print(f"  SFT: {len(sft_dataset):,} samples")
        print(f"  Total raw samples: {len(kimi_dataset) + len(oss_dataset) + len(qwen_dataset) + len(sft_dataset):,}")

        # Filter each dataset separately
        print("\nFiltering each dataset configuration...")
        filtered_kimi = filter_toucan_dataset(kimi_dataset, max_tools_used, max_tools_available, max_number_of_turns)
        filtered_oss = filter_toucan_dataset(oss_dataset, max_tools_used, max_tools_available, max_number_of_turns)
        filtered_qwen = filter_toucan_dataset(qwen_dataset, max_tools_used, max_tools_available, max_number_of_turns)
        filtered_sft = filter_toucan_dataset(sft_dataset, max_tools_used, max_tools_available, max_number_of_turns)

        print(f"\nFiltered sizes:")
        print(f"  Kimi-K2: {len(filtered_kimi):,} samples")
        print(f"  OSS: {len(filtered_oss):,} samples")
        print(f"  Qwen3: {len(filtered_qwen):,} samples")
        print(f"  SFT: {len(filtered_sft):,} samples")

        # Concatenate non-SFT datasets first, then SFT
        # This ensures non-SFT data comes before SFT data in training
        non_sft_dataset = concatenate_datasets([filtered_kimi, filtered_oss, filtered_qwen])
        print(f"\nCombined non-SFT dataset: {len(non_sft_dataset):,} samples")

        # Split the SFT dataset: 95% for training, 5% for validation
        sft_split = filtered_sft.train_test_split(test_size=0.05, seed=42)
        sft_train = sft_split['train']
        validation_dataset = sft_split['test']

        print(f"SFT training split: {len(sft_train):,} samples")
        print(f"SFT validation split: {len(validation_dataset):,} samples")

        # Combine non-SFT data with SFT training data (non-SFT first)
        filtered_dataset = concatenate_datasets([non_sft_dataset, sft_train])
        print(f"\nFinal combined training dataset: {len(filtered_dataset):,} samples")
        print(f"  - Non-SFT portion: {len(non_sft_dataset):,} samples (first)")
        print(f"  - SFT portion: {len(sft_train):,} samples (second)")

    else:
        # Original behavior: Load only SFT dataset
        dataset = load_dataset('Agent-Ark/Toucan-1.5M', 'SFT', split='train')

        # Filter dataset
        filtered_dataset = filter_toucan_dataset(dataset, max_tools_used, max_tools_available, max_number_of_turns)

        # Split into train and validation sets: 95% train, 5% validation
        split = filtered_dataset.train_test_split(test_size=0.05, seed=42)
        filtered_dataset = split['train']
        validation_dataset = split['test']

        print(f"SFT training split: {len(filtered_dataset):,} samples")
        print(f"SFT validation split: {len(validation_dataset):,} samples")
    
    # Format examples
    print("\nFormatting examples for Gemma 3 tool calling...")
    train_dataset = filtered_dataset.map(
        format_function,
        batched=True,
        batch_size=1000,
        remove_columns=filtered_dataset.column_names
    )

    validation_dataset = validation_dataset.map(
        format_function,
        batched=True,
        batch_size=1000,
        remove_columns=validation_dataset.column_names
    )

    print(f"Formatted {len(train_dataset):,} training examples")
    print(f"Formatted {len(validation_dataset):,} validation examples")

    # Count examples without tool calls (need to parse JSON and check all turns)
    def has_tool_call(example):
        """Check if any turn in the role_messages contains a tool call (BFCL format: [func(...)])."""
        role_messages = json.loads(example['role_messages'])
        # Look for BFCL-style function calls: [func_name(
        import re
        return any(re.search(r'\[[\w\.]+\(', turn['text']) for turn in role_messages)

    count_with_tool_call = len(train_dataset.filter(has_tool_call))
    count_no_tool_call = len(train_dataset) - count_with_tool_call
    print(f"  Training examples without tool calls: {count_no_tool_call} ({100 * count_no_tool_call / len(train_dataset):.2f}%)")

    count_with_tool_call_val = len(validation_dataset.filter(has_tool_call))
    count_no_tool_call_val = len(validation_dataset) - count_with_tool_call_val
    print(f"  Validation examples without tool calls: {count_no_tool_call_val} ({100 * count_no_tool_call_val / len(validation_dataset):.2f}%)")

    # Build grain DataLoaders (HuggingFace Dataset objects work as grain data sources)
    train_loader = _build_data_loader(
        data_source=train_dataset,
        batch_size=global_batch_size,
        num_epochs=num_train_epochs,
        max_seq_len=max_target_length,
        tokenizer=tokenizer,
        shuffle=True
    )

    validation_loader = _build_data_loader(
        data_source=validation_dataset,
        batch_size=global_batch_size,
        num_epochs=1,  # validation only runs once per eval
        max_seq_len=max_target_length,
        tokenizer=tokenizer,
        shuffle=False
    )

    # Calculate steps
    num_train_examples = len(train_dataset)
    steps_per_epoch = num_train_examples // global_batch_size
    total_steps = steps_per_epoch * num_train_epochs

    print(f"\nDataset statistics:")
    print(f"  Training examples: {num_train_examples:,}")
    print(f"  Validation examples: {len(validation_dataset):,}")
    print(f"  Steps per epoch: {steps_per_epoch:,}")
    print(f"  Total steps ({num_train_epochs} epochs): {total_steps:,}")
    print(f"  Effective batch size: {global_batch_size}")

    # Validate loss masking by inspecting a sample batch
    print(f"\n{'='*60}")
    print("Validating Loss Masking")
    print(f"{'='*60}")
    sample_batch = next(iter(train_loader))
    sample_tokens = np.array(sample_batch.input_tokens[0])  # First example in batch
    sample_mask = np.array(sample_batch.input_mask[0])

    # Count masked tokens
    total_tokens = len(sample_tokens)
    train_tokens = int(np.sum(sample_mask))
    skip_tokens = total_tokens - train_tokens
    pad_tokens = int(np.sum(sample_tokens == tokenizer.pad_id()))

    print("Sample masking statistics:")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Training tokens (mask=1): {train_tokens} ({100*train_tokens/total_tokens:.1f}%)")
    print(f"  Skipped tokens (mask=0): {skip_tokens} ({100*skip_tokens/total_tokens:.1f}%)")
    print(f"  Padding tokens: {pad_tokens} ({100*pad_tokens/total_tokens:.1f}%)")

    # Show first 100 tokens with their mask values to verify correctness
    print("\nFirst 100 tokens with mask (✓=train, ✗=skip):")
    print("-" * 80)
    for i in range(min(100, len(sample_tokens))):
        token_id = int(sample_tokens[i])
        mask_val = sample_mask[i]
        if token_id == tokenizer.pad_id():
            decoded = "<PAD>"
        else:
            decoded = tokenizer.decode([token_id])
        symbol = "✓" if mask_val == 1 else "✗"
        # Truncate long decoded strings
        decoded_display = repr(decoded)[:40]
        print(f"{i:3d} {symbol} [{mask_val:.0f}] {decoded_display}")

    print("-" * 80)
    print("Expected pattern:")
    print("  ✗ for <start_of_turn>user and all user content")
    print("  ✓ for model outputs after <start_of_turn>model")
    print("  ✗ for padding tokens")
    print(f"{'='*60}\n")

    return train_loader, validation_loader, total_steps, train_dataset
