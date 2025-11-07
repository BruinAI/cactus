#!/usr/bin/env python3
"""
Analyze Toucan-1.5M dataset for Phase 1 training.
Focus: Single-turn data with tools used and available tools cumulative analysis.
"""

from datasets import load_dataset
from collections import Counter, defaultdict
import csv
import json

def analyze_phase1(dataset):
    """Analyze single-turn data for Phase 1 training."""
    print(f"\n{'='*60}")
    print("Phase 1 Analysis: Single-Turn Data Only")
    print(f"{'='*60}")

    total_samples = len(dataset)
    print(f"Total samples in dataset: {total_samples:,}")

    # Counters
    target_tool_counts = []
    available_tool_counts = []
    single_turn_indices = []

    # First pass: identify single-turn samples
    print(f"\nIdentifying single-turn samples...")
    for idx in range(total_samples):
        if idx % 10000 == 0:
            print(f"  Processed {idx:,}/{total_samples:,} samples...")

        sample = dataset[idx]
        messages = json.loads(sample['messages'])
        num_turns = len([m for m in messages if m['role'] == 'user'])

        # Only analyze single-turn samples
        if num_turns == 1:
            single_turn_indices.append(idx)

    print(f"  Processed {total_samples:,}/{total_samples:,} samples...")

    single_turn_count = len(single_turn_indices)
    print(f"\nSingle-turn samples: {single_turn_count:,} ({100 * single_turn_count / total_samples:.2f}%)")

    # Second pass: analyze single-turn samples in detail
    print(f"\nAnalyzing single-turn samples in detail...")
    for i, idx in enumerate(single_turn_indices):
        if i % 10000 == 0:
            print(f"  Processed {i:,}/{single_turn_count:,} single-turn samples...")

        sample = dataset[idx]

        # Parse target tools (comma-separated string)
        target_tools_str = sample['target_tools'].strip()
        target_tools_list = [t.strip() for t in target_tools_str.split(',')]
        num_target_tools = len(target_tools_list)
        target_tool_counts.append(num_target_tools)

        # Parse available tools (JSON)
        tools = json.loads(sample['tools'])
        num_available_tools = len(tools)
        available_tool_counts.append(num_available_tools)

    print(f"  Processed {single_turn_count:,}/{single_turn_count:,} single-turn samples...")

    # Generate statistics
    print(f"\n{'='*60}")
    print("Results")
    print(f"{'='*60}")

    print(f"\nTarget Tools per Sample (tools actually used):")
    print(f"  Min: {min(target_tool_counts)}")
    print(f"  Max: {max(target_tool_counts)}")
    print(f"  Mean: {sum(target_tool_counts)/len(target_tool_counts):.2f}")

    print(f"\nAvailable Tools per Sample:")
    print(f"  Min: {min(available_tool_counts)}")
    print(f"  Max: {max(available_tool_counts)}")
    print(f"  Mean: {sum(available_tool_counts)/len(available_tool_counts):.2f}")

    # Build results for CSV
    results = []

    # 1. Overall stats
    results.append({
        'filter_type': 'total',
        'filter_value': 'all_single_turn',
        'sample_count': single_turn_count,
        'percentage': '100.00%',
        'notes': 'All single-turn samples'
    })

    # 2. Target tools - exact counts
    target_tool_distribution = Counter(target_tool_counts)
    print(f"\nTarget Tools Distribution:")
    for num_tools in sorted(target_tool_distribution.keys()):
        count = target_tool_distribution[num_tools]
        print(f"  {num_tools} tools: {count:,} ({100 * count / single_turn_count:.2f}%)")
        results.append({
            'filter_type': 'target_tools_exact',
            'filter_value': f'{num_tools}_tools',
            'sample_count': count,
            'percentage': f'{100 * count / single_turn_count:.2f}%',
            'notes': f'Tasks requiring exactly {num_tools} tool(s)'
        })

    # 3. Target tools - cumulative
    for threshold in [1, 2, 3, 4, 5]:
        cumulative_count = sum(count for num, count in target_tool_distribution.items() if num <= threshold)
        results.append({
            'filter_type': 'target_tools_cumulative',
            'filter_value': f'{threshold}_or_fewer_tools',
            'sample_count': cumulative_count,
            'percentage': f'{100 * cumulative_count / single_turn_count:.2f}%',
            'notes': f'Tasks requiring ≤{threshold} tool(s)'
        })

    # 4. Available tools - cumulative only (1, 3, 5, 10)
    available_tool_distribution = Counter(available_tool_counts)
    print(f"\nAvailable Tools (Cumulative):")
    for threshold in [1, 3, 5, 10]:
        cumulative_count = sum(count for num, count in available_tool_distribution.items() if num <= threshold)
        print(f"  ≤{threshold} available: {cumulative_count:,} ({100 * cumulative_count / single_turn_count:.2f}%)")
        results.append({
            'filter_type': 'available_tools_cumulative',
            'filter_value': f'{threshold}_or_fewer_available',
            'sample_count': cumulative_count,
            'percentage': f'{100 * cumulative_count / single_turn_count:.2f}%',
            'notes': f'Prompts with ≤{threshold} available tools'
        })

    # 5. Combined: target tools × available tools
    print(f"\nCombined Filters (Target × Available):")
    for target_threshold in [1, 2, 3]:
        for avail_threshold in [1, 3, 5, 10]:
            combined_count = sum(1 for target, available in zip(target_tool_counts, available_tool_counts)
                                if target <= target_threshold and available <= avail_threshold)
            percentage = 100 * combined_count / single_turn_count
            print(f"  ≤{target_threshold} tools + ≤{avail_threshold} available: {combined_count:,} ({percentage:.2f}%)")
            results.append({
                'filter_type': 'combined',
                'filter_value': f'{target_threshold}tools_AND_{avail_threshold}available',
                'sample_count': combined_count,
                'percentage': f'{percentage:.2f}%',
                'notes': f'≤{target_threshold} target tools + ≤{avail_threshold} available tools'
            })

    return results

def main():
    print("Loading Toucan-1.5M SFT dataset from HuggingFace...")
    print("Analyzing for Phase 1 training (single-turn only)...\n")

    print("#" * 60)
    print("Loading SFT subset...")
    print("#" * 60)

    dataset = load_dataset('Agent-Ark/Toucan-1.5M', 'SFT', split='train')
    results = analyze_phase1(dataset)

    # Write to CSV
    import os
    output_path = os.path.join(os.path.dirname(__file__), 'toucan_analysis_phase1.csv')
    print(f"\n{'='*60}")
    print(f"Writing results to {output_path}...")
    print(f"{'='*60}")

    with open(output_path, 'w', newline='') as f:
        fieldnames = ['filter_type', 'filter_value', 'sample_count', 'percentage', 'notes']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✓ Analysis complete!")
    print(f"  Total entries: {len(results)}")
    print(f"  Output: {output_path}")

if __name__ == '__main__':
    main()
