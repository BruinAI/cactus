#!/usr/bin/env python3
"""
Convert synthetic data to training format.

Converts from synthetic generation format:
{
  "user_input": "...",
  "tool_name": "...",
  "parameters": {...}
}

To training format:
{
  "input": "...",
  "output": {
    "function_call": {
      "name": "...",
      "arguments": {...}
    }
  }
}
"""

import json
import sys
import os
from typing import List, Dict, Any


def convert_to_training_format(synthetic_examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert synthetic examples to training format.

    Args:
        synthetic_examples: List of examples in synthetic format

    Returns:
        List of examples in training format
    """
    training_examples = []

    for example in synthetic_examples:
        training_example = {
            "input": example["user_input"],
            "output": {
                "function_call": {
                    "name": example["tool_name"],
                    "arguments": example["parameters"]
                }
            }
        }
        training_examples.append(training_example)

    return training_examples


def main():
    """Convert synthetic data file to training format."""
    if len(sys.argv) < 2:
        print("Usage: python3 convert_to_training_format.py <mode> <input_file> [output_file]")
        print("\nArguments:")
        print("  mode: Either 'append' or 'overwrite'")
        print("  input_file: Input file with synthetic examples")
        print("  output_file: Optional output file (default: input_file_training.json)")
        print("\nExamples:")
        print("  python3 convert_to_training_format.py append phase3_final_examples.json")
        print("  python3 convert_to_training_format.py overwrite phase3_final_examples.json training_data.json")
        sys.exit(1)

    mode = sys.argv[1]
    input_file = sys.argv[2] if len(sys.argv) > 2 else None
    output_file = sys.argv[3] if len(sys.argv) > 3 else None

    # Validate inputs
    if input_file is None:
        print("❌ Error: Missing input_file argument")
        sys.exit(1)

    # Validate mode
    if mode not in ['append', 'overwrite']:
        print(f"❌ Error: Invalid mode '{mode}'. Must be 'append' or 'overwrite'.")
        sys.exit(1)

    if output_file is None:
        output_file = input_file.replace('.json', '_training.json')

    print(f"Mode: {mode}")
    print(f"Loading synthetic data from: {input_file}")
    with open(input_file, 'r') as f:
        synthetic_examples = json.load(f)

    print(f"Converting {len(synthetic_examples)} examples...")
    training_examples = convert_to_training_format(synthetic_examples)

    # Handle append vs overwrite
    if mode == 'append' and os.path.exists(output_file):
        print(f"Appending to existing file: {output_file}")
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_examples = json.load(f)

        original_count = len(existing_examples)
        existing_examples.extend(training_examples)
        training_examples = existing_examples

        print(f"✓ Original dataset: {original_count} examples")
        print(f"✓ Added: {len(training_examples) - original_count} examples")
        print(f"✓ Total: {len(training_examples)} examples")
    else:
        if mode == 'append':
            print(f"Output file does not exist. Creating new file: {output_file}")
        else:
            print(f"Overwriting file: {output_file}")

    print(f"Saving to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(training_examples, f, indent=2, ensure_ascii=False)

    print(f"✓ Final dataset contains {len(training_examples)} examples")
    print(f"\nSample converted example:")
    print(json.dumps(training_examples[-1], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
