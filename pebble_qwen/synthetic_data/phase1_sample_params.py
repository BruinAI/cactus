"""
PHASE 1: Sample Random Parameters

This script samples random parameter values for each tool.
- Uses discrete values from possible_params
- Sets "free-text" placeholders for text parameters
- No API calls required

Usage:
    python3 phase1_sample_params.py [samples_per_tool] [output_file]

Examples:
    python3 phase1_sample_params.py 10
    python3 phase1_sample_params.py 20 my_samples.json
"""

import json
import sys
from typing import Dict, List, Any
from possible_params import get_all_tools, sample_multiple_tools


def save_json(data: Any, path: str):
    """Save data to JSON file."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved to: {path}")


def phase1_sample_parameters(
    samples_per_tool: int = 10,
    output_path: str = "phase1_sampled_params.json"
) -> Dict[str, List[Dict[str, Any]]]:
    """
    PHASE 1: Sample random parameter values for each tool.

    This phase generates random parameter combinations using discrete values
    and "free-text" placeholders for text parameters.

    Args:
        samples_per_tool: Number of parameter sets to sample per tool
        output_path: Path to save the sampled parameters

    Returns:
        Dictionary mapping tool names to lists of sampled parameters
    """
    print("="*60)
    print("PHASE 1: Sampling Random Parameters")
    print("="*60)
    print(f"Samples per tool: {samples_per_tool}")
    print()

    sampled_data = {}

    for tool_name in get_all_tools():
        print(f"Sampling {tool_name}...")
        sampled_params = sample_multiple_tools(tool_name, samples_per_tool)
        sampled_data[tool_name] = sampled_params
        print(f"  ✓ Generated {len(sampled_params)} parameter sets")

    print(f"\nTotal parameter sets: {sum(len(v) for v in sampled_data.values())}")
    save_json(sampled_data, output_path)
    print("="*60)

    return sampled_data


if __name__ == "__main__":
    # Parse command line arguments
    samples_per_tool = 10
    output_file = "phase1_sampled_params.json"

    if len(sys.argv) > 1:
        samples_per_tool = int(sys.argv[1])
    if len(sys.argv) > 2:
        output_file = sys.argv[2]

    # Run Phase 1
    try:
        phase1_sample_parameters(samples_per_tool, output_file)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
