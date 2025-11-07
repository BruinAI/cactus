#!/usr/bin/env python3
"""
Analyze the Toucan-1.5M dataset to understand filtering options and sample counts.
Generates a comprehensive analysis CSV for planning training data selection.
"""

from datasets import load_dataset
from collections import Counter, defaultdict
import csv
import json

def analyze_subset(subset_name, dataset):
    """Analyze a single subset of the Toucan dataset."""
    print(f"\n{'='*60}")
    print(f"Analyzing subset: {subset_name}")
    print(f"{'='*60}")

    results = []
    total_samples = len(dataset)
    print(f"Total samples: {total_samples:,}")

    # Print available fields
    print(f"\nAvailable fields: {list(dataset.features.keys())}")

    # Show sample record
    sample = dataset[0]
    print(f"\nSample record:")
    for key, value in sample.items():
        print(f"  {key}: {str(value)[:100]}")

    # Initialize counters
    tool_counts = []
    message_turn_counts = []
    subsets_counter = Counter()
    unique_tools = set()
    subset_tool_combos = defaultdict(int)

    # Analyze ALL samples
    print(f"\nAnalyzing all {total_samples:,} samples...")

    for idx in range(total_samples):
        if idx % 10000 == 0:
            print(f"  Processed {idx:,}/{total_samples:,} samples...")

        sample = dataset[idx]

        # Parse tools JSON
        tools = json.loads(sample['tools'])
        num_available_tools = len(tools)

        # Track unique tool names
        for tool in tools:
            if 'function' in tool and 'name' in tool['function']:
                unique_tools.add(tool['function']['name'])

        # Parse target_tools (comma-separated string)
        target_tools_str = sample['target_tools'].strip()
        target_tools_list = [t.strip() for t in target_tools_str.split(',')]
        num_target_tools = len(target_tools_list)

        tool_counts.append({
            'available': num_available_tools,
            'target': num_target_tools
        })

        # Parse messages JSON and count user turns
        messages = json.loads(sample['messages'])
        num_turns = len([m for m in messages if m['role'] == 'user'])
        message_turn_counts.append(num_turns)

        # Track subset category
        subset_cat = sample['subset_name']
        subsets_counter[subset_cat] += 1

        # Track subset + tool count combinations
        subset_tool_combos[f"{subset_cat}_{num_target_tools}tools"] += 1

    print(f"  Processed {total_samples:,}/{total_samples:,} samples...")

    # Generate statistics
    print(f"\n{'='*60}")
    print(f"Results for {subset_name}")
    print(f"{'='*60}")

    available_tool_counts = [t['available'] for t in tool_counts]
    target_tool_counts = [t['target'] for t in tool_counts]

    print(f"\nUnique Tools Found: {len(unique_tools)}")

    print(f"\nAvailable Tools per Sample:")
    print(f"  Min: {min(available_tool_counts)}")
    print(f"  Max: {max(available_tool_counts)}")
    print(f"  Mean: {sum(available_tool_counts)/len(available_tool_counts):.2f}")

    print(f"\nTarget Tools per Sample (tools actually used):")
    print(f"  Min: {min(target_tool_counts)}")
    print(f"  Max: {max(target_tool_counts)}")
    print(f"  Mean: {sum(target_tool_counts)/len(target_tool_counts):.2f}")

    print(f"\nConversation Turns per Sample:")
    print(f"  Min: {min(message_turn_counts)}")
    print(f"  Max: {max(message_turn_counts)}")
    print(f"  Mean: {sum(message_turn_counts)/len(message_turn_counts):.2f}")

    print(f"\nSubset Distribution:")
    for subset, count in subsets_counter.most_common():
        print(f"  {subset}: {count:,} ({100 * count / total_samples:.2f}%)")

    # Build results for CSV

    # 1. Overall stats
    results.append({
        'subset': subset_name,
        'filter_type': 'total',
        'filter_value': 'all_samples',
        'sample_count': total_samples,
        'percentage': '100.00%',
        'notes': f'Unique tools: {len(unique_tools)}'
    })

    # 2. Target tool count distribution
    target_tool_distribution = Counter(target_tool_counts)
    for num_tools in sorted(target_tool_distribution.keys()):
        count = target_tool_distribution[num_tools]
        results.append({
            'subset': subset_name,
            'filter_type': 'target_tools_count',
            'filter_value': f'{num_tools}_tools',
            'sample_count': count,
            'percentage': f'{100 * count / total_samples:.2f}%',
            'notes': f'Tasks requiring exactly {num_tools} tool(s)'
        })

    # 3. Cumulative: 1-3 tools (Step 3 target)
    tasks_1_to_3_tools = sum(count for num_tools, count in target_tool_distribution.items() if 1 <= num_tools <= 3)
    results.append({
        'subset': subset_name,
        'filter_type': 'target_tools_cumulative',
        'filter_value': '1_to_3_tools',
        'sample_count': tasks_1_to_3_tools,
        'percentage': f'{100 * tasks_1_to_3_tools / total_samples:.2f}%',
        'notes': 'Ideal for Step 3 training'
    })

    # 4. Cumulative: 1 tool only (simplest)
    tasks_1_tool = target_tool_distribution.get(1, 0)
    results.append({
        'subset': subset_name,
        'filter_type': 'target_tools_cumulative',
        'filter_value': '1_tool_only',
        'sample_count': tasks_1_tool,
        'percentage': f'{100 * tasks_1_tool / total_samples:.2f}%',
        'notes': 'Simplest: single tool tasks'
    })

    # 5. Conversation type
    turn_distribution = Counter(message_turn_counts)
    single_turn_count = sum(count for turns, count in turn_distribution.items() if turns <= 1)
    multi_turn_count = sum(count for turns, count in turn_distribution.items() if turns > 1)

    results.append({
        'subset': subset_name,
        'filter_type': 'conversation_type',
        'filter_value': 'single_turn',
        'sample_count': single_turn_count,
        'percentage': f'{100 * single_turn_count / total_samples:.2f}%',
        'notes': '≤1 user turn'
    })

    results.append({
        'subset': subset_name,
        'filter_type': 'conversation_type',
        'filter_value': 'multi_turn',
        'sample_count': multi_turn_count,
        'percentage': f'{100 * multi_turn_count / total_samples:.2f}%',
        'notes': '>1 user turns'
    })

    # 6. Detailed turn distribution
    for num_turns in sorted(turn_distribution.keys()):
        count = turn_distribution[num_turns]
        results.append({
            'subset': subset_name,
            'filter_type': 'turn_count_detailed',
            'filter_value': f'{num_turns}_turns',
            'sample_count': count,
            'percentage': f'{100 * count / total_samples:.2f}%',
            'notes': f'Exactly {num_turns} user turn(s)'
        })

    # 7. Subset category distribution
    for subset_cat, count in subsets_counter.items():
        results.append({
            'subset': subset_name,
            'filter_type': 'subset_category',
            'filter_value': subset_cat,
            'sample_count': count,
            'percentage': f'{100 * count / total_samples:.2f}%',
            'notes': 'Data generation pipeline type'
        })

    # 8. Combined: single-turn + 1-3 tools (Step 3 optimal)
    step3_samples = sum(1 for turns, tools in zip(message_turn_counts, target_tool_counts)
                        if turns <= 1 and 1 <= tools <= 3)
    results.append({
        'subset': subset_name,
        'filter_type': 'combined_step3',
        'filter_value': 'single_turn_AND_1to3tools',
        'sample_count': step3_samples,
        'percentage': f'{100 * step3_samples / total_samples:.2f}%',
        'notes': 'Perfect for Step 3: simple single-turn tasks'
    })

    # 9. Combined: single-turn + 1 tool (simplest)
    simplest_samples = sum(1 for turns, tools in zip(message_turn_counts, target_tool_counts)
                           if turns <= 1 and tools == 1)
    results.append({
        'subset': subset_name,
        'filter_type': 'combined_simplest',
        'filter_value': 'single_turn_AND_1tool',
        'sample_count': simplest_samples,
        'percentage': f'{100 * simplest_samples / total_samples:.2f}%',
        'notes': 'Absolute simplest: single turn, single tool'
    })

    # 10. Subset × tool count combinations
    for combo, count in sorted(subset_tool_combos.items(), key=lambda x: x[1], reverse=True):
        results.append({
            'subset': subset_name,
            'filter_type': 'subset_x_toolcount',
            'filter_value': combo,
            'sample_count': count,
            'percentage': f'{100 * count / total_samples:.2f}%',
            'notes': 'Subset category × tool count'
        })

    # 11. Available tools cumulative (X or fewer)
    available_tool_distribution = Counter(available_tool_counts)
    for threshold in [1, 2, 3, 4, 5, 10, 15, 20, 30]:
        cumulative_count = sum(count for num, count in available_tool_distribution.items() if num <= threshold)
        results.append({
            'subset': subset_name,
            'filter_type': 'available_tools_cumulative',
            'filter_value': f'{threshold}_or_fewer_available',
            'sample_count': cumulative_count,
            'percentage': f'{100 * cumulative_count / total_samples:.2f}%',
            'notes': f'Prompts with ≤{threshold} available tools'
        })

    # 12. Combined: single-turn + 1-3 target tools + various available tool thresholds
    for avail_threshold in [3, 5, 10, 15, 20]:
        combined_count = sum(1 for turns, target, available in zip(message_turn_counts, target_tool_counts, available_tool_counts)
                            if turns <= 1 and 1 <= target <= 3 and available <= avail_threshold)
        results.append({
            'subset': subset_name,
            'filter_type': 'combined_step3_detailed',
            'filter_value': f'single_turn_AND_1to3tools_AND_<={avail_threshold}available',
            'sample_count': combined_count,
            'percentage': f'{100 * combined_count / total_samples:.2f}%',
            'notes': f'Step 3 with ≤{avail_threshold} available tools'
        })

    return results

def main():
    print("Loading Toucan-1.5M SFT dataset from HuggingFace...")
    print("Analyzing the complete SFT subset (119k samples)...\n")

    # Load SFT subset
    print("#" * 60)
    print("Loading SFT subset...")
    print("#" * 60)

    dataset = load_dataset('Agent-Ark/Toucan-1.5M', 'SFT', split='train')
    results = analyze_subset('SFT', dataset)

    # Write to CSV
    import os
    output_path = os.path.join(os.path.dirname(__file__), 'toucan_analysis.csv')
    print(f"\n{'='*60}")
    print(f"Writing results to {output_path}...")
    print(f"{'='*60}")

    with open(output_path, 'w', newline='') as f:
        fieldnames = ['subset', 'filter_type', 'filter_value', 'sample_count', 'percentage', 'notes']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(results, key=lambda x: (x['filter_type'], -x['sample_count'])))

    print(f"\n✓ Analysis complete!")
    print(f"  Total entries: {len(results)}")
    print(f"  Output: {output_path}")

if __name__ == '__main__':
    main()
