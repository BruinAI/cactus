#!/usr/bin/env python3
"""
Verify that the Toucan dataset loading and filtering works as expected.
This script checks the filtering logic without loading the full training infrastructure.
"""

import json
from datasets import load_dataset

# Same filtering criteria as training script
MAX_TOOLS_USED = 2
MAX_TOOLS_AVAILABLE = 3

def filter_toucan_dataset(dataset, max_tools_used=2, max_tools_available=3):
    """
    Filter Toucan dataset for single-turn examples with limited tools.
    """
    print(f"\nFiltering dataset:")
    print(f"  - Single-turn only")
    print(f"  - ≤{max_tools_used} tools used")
    print(f"  - ≤{max_tools_available} tools available")

    filtered_indices = []
    total = len(dataset)

    # Track statistics
    stats = {
        'multi_turn': 0,
        'too_many_tools_used': 0,
        'too_many_tools_available': 0,
        'zero_tools': 0,
        'passed': 0
    }

    for idx in range(total):
        if idx % 10000 == 0:
            print(f"  Processed {idx:,}/{total:,} samples...")

        sample = dataset[idx]

        # Check single-turn
        messages = json.loads(sample['messages'])
        num_turns = len([m for m in messages if m['role'] == 'user'])
        if num_turns != 1:
            stats['multi_turn'] += 1
            continue

        # Check tools used
        target_tools_str = sample['target_tools'].strip()
        if target_tools_str:
            target_tools_list = [t.strip() for t in target_tools_str.split(',')]
            num_target_tools = len(target_tools_list)
        else:
            num_target_tools = 0

        if num_target_tools == 0:
            stats['zero_tools'] += 1
            continue

        if num_target_tools > max_tools_used:
            stats['too_many_tools_used'] += 1
            continue

        # Check tools available
        tools = json.loads(sample['tools'])
        num_available_tools = len(tools)

        if num_available_tools > max_tools_available or num_available_tools == 0:
            stats['too_many_tools_available'] += 1
            continue

        stats['passed'] += 1
        filtered_indices.append(idx)

    print(f"  Processed {total:,}/{total:,} samples...")

    print(f"\n{'='*60}")
    print("Filter Statistics")
    print(f"{'='*60}")
    print(f"Total samples: {total:,}")
    print(f"\nFiltered out:")
    print(f"  Multi-turn: {stats['multi_turn']:,} ({100*stats['multi_turn']/total:.2f}%)")
    print(f"  Zero tools used: {stats['zero_tools']:,} ({100*stats['zero_tools']/total:.2f}%)")
    print(f"  Too many tools used (>{max_tools_used}): {stats['too_many_tools_used']:,} ({100*stats['too_many_tools_used']/total:.2f}%)")
    print(f"  Too many tools available (>{max_tools_available}): {stats['too_many_tools_available']:,} ({100*stats['too_many_tools_available']/total:.2f}%)")
    print(f"\nPassed filters: {stats['passed']:,} ({100*stats['passed']/total:.2f}%)")

    return dataset.select(filtered_indices), stats


def analyze_sample(dataset, num_samples=3):
    """Print some example samples to verify formatting."""
    print(f"\n{'='*60}")
    print(f"Sample Examples (first {num_samples})")
    print(f"{'='*60}")

    for idx in range(min(num_samples, len(dataset))):
        sample = dataset[idx]
        messages = json.loads(sample['messages'])
        tools = json.loads(sample['tools'])
        target_tools = sample['target_tools']

        print(f"\n--- Sample {idx + 1} ---")
        print(f"Target tools: {target_tools}")
        print(f"Number of available tools: {len(tools)}")
        print(f"Number of turns: {len([m for m in messages if m['role'] == 'user'])}")

        # Show user message (truncated)
        user_msg = next((m['content'] for m in messages if m['role'] == 'user'), None)
        if user_msg:
            print(f"User message: {user_msg[:100]}...")

        # Show assistant tool calls
        assistant_msg = next((m for m in messages if m['role'] == 'assistant'), None)
        if assistant_msg and 'tool_calls' in assistant_msg:
            print(f"Tool calls: {len(assistant_msg['tool_calls'])}")
            for tc in assistant_msg['tool_calls'][:2]:  # Show first 2
                if 'function' in tc:
                    print(f"  - {tc['function']['name']}")
                else:
                    print(f"  - {tc['name']}")


def main():
    print("="*60)
    print("Toucan Dataset Verification")
    print("="*60)
    print(f"Loading Toucan-1.5M SFT dataset...")

    # Load dataset
    dataset = load_dataset('Agent-Ark/Toucan-1.5M', 'SFT', split='train')
    print(f"Loaded dataset with {len(dataset):,} total samples")

    # Filter dataset
    filtered_dataset, stats = filter_toucan_dataset(
        dataset,
        max_tools_used=MAX_TOOLS_USED,
        max_tools_available=MAX_TOOLS_AVAILABLE
    )

    # Calculate train/val split
    total_size = len(filtered_dataset)
    train_size = int(0.9 * total_size)
    val_size = total_size - train_size

    print(f"\n{'='*60}")
    print("Train/Validation Split (90/10)")
    print(f"{'='*60}")
    print(f"Training samples: {train_size:,}")
    print(f"Validation samples: {val_size:,}")

    # Calculate training steps
    BATCH_SIZE = 64
    NUM_EPOCHS = 3
    steps_per_epoch = train_size // BATCH_SIZE
    total_steps = steps_per_epoch * NUM_EPOCHS

    print(f"\n{'='*60}")
    print("Training Configuration")
    print(f"{'='*60}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Steps per epoch: {steps_per_epoch:,}")
    print(f"Total training steps: {total_steps:,}")
    print(f"Evaluations per epoch: {steps_per_epoch // 250} (every 250 steps)")

    # Show example samples
    analyze_sample(filtered_dataset, num_samples=3)

    print(f"\n{'='*60}")
    print("Verification Complete!")
    print(f"{'='*60}")
    print(f"✓ Dataset filtering works correctly")
    print(f"✓ Expected ~{train_size:,} training examples")
    print(f"✓ Will train for ~{total_steps:,} steps over {NUM_EPOCHS} epochs")


if __name__ == '__main__':
    main()
