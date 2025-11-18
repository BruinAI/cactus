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
        print("Usage: python3 convert_to_training_format.py <input_file> [output_file]")
        print("\nExample:")
        print("  python3 convert_to_training_format.py phase3_final_examples.json")
        print("  python3 convert_to_training_format.py phase3_final_examples.json training_data.json")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.replace('.json', '_training.json')

    print(f"Loading synthetic data from: {input_file}")
    with open(input_file, 'r') as f:
        synthetic_examples = json.load(f)

    print(f"Converting {len(synthetic_examples)} examples...")
    training_examples = convert_to_training_format(synthetic_examples)

    print(f"Saving to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(training_examples, f, indent=2, ensure_ascii=False)

    print(f"✓ Converted {len(training_examples)} examples")
    print(f"\nSample converted example:")
    print(json.dumps(training_examples[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
