"""
PHASE 3: Generate User Inputs

This script takes complete parameter sets from Phase 2 and uses Claude API
to generate natural user inputs that would result in those tool calls.

Requirements:
    - ANTHROPIC_API_KEY environment variable must be set
    - Input file from Phase 2

Usage:
    python3 phase3_generate_inputs.py [input_file] [output_file]

Examples:
    python3 phase3_generate_inputs.py
    python3 phase3_generate_inputs.py phase2_with_text.json phase3_final_examples.json
"""

import json
import os
import sys
from typing import List, Dict, Any, Optional
from claude_api import ClaudeAPIClient


def load_json(path: str) -> Any:
    """Load data from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def save_json(data: Any, path: str):
    """Save data to JSON file."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved to: {path}")


def load_tools_schema(tools_path: str = "../data/tools.json") -> List[Dict[str, Any]]:
    """Load the tools schema from JSON file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, tools_path)

    with open(full_path, 'r') as f:
        return json.load(f)


def phase3_generate_user_inputs(
    input_path: str = "phase2_with_text.json",
    output_path: str = "phase3_final_examples.json",
    api_key: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    PHASE 3: Generate user inputs that would lead to the tool calls.

    Takes the complete parameter sets from Phase 2 and uses Claude API to
    generate natural user inputs that would result in those tool calls.

    Args:
        input_path: Path to Phase 2 output (parameters with text)
        output_path: Path to save final examples
        api_key: Optional API key (uses env var if not provided)

    Returns:
        List of complete examples with user_input, tool_name, and parameters
    """
    print("="*60)
    print("PHASE 3: Generating User Inputs")
    print("="*60)
    print(f"Loading from: {input_path}")
    print()

    # Load Phase 2 data
    data_with_text = load_json(input_path)
    client = ClaudeAPIClient(api_key=api_key)
    tools_schema = load_tools_schema()

    all_examples = []

    for tool_name, params_list in data_with_text.items():
        print(f"\nProcessing {tool_name}...")

        # Find tool schema
        tool_schema = next(
            (t for t in tools_schema if t['function']['name'] == tool_name),
            None
        )
        if not tool_schema:
            raise ValueError(f"Tool schema not found for {tool_name}")

        # Build prompts for generating user inputs
        requests = []
        for params in params_list:
            # Format the tool call
            tool_call_str = f"{tool_name}({', '.join(f'{k}={repr(v)}' for k, v in params.items())})"

            prompt = f"""Generate a natural user input that would result in this exact tool call.

Tool: {tool_name}
Description: {tool_schema['function']['description']}

Target tool call:
{tool_call_str}

Parameters:
{json.dumps(params, indent=2)}

Generate a natural, conversational user message that would lead to exactly this tool call.
- Be creative and varied in phrasing
- Use natural language (not robotic or templated)
- Keep it concise (1-2 sentences)
- Make it sound like something a real user would say
- DO NOT just repeat the parameter values verbatim

Return ONLY the user input text, nothing else:"""

            requests.append({
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 150,
                "temperature": 1.0
            })

        print(f"  Generating {len(requests)} user inputs...")

        # Call API in parallel
        responses = client.call_parallel(requests)

        # Build complete examples
        tool_examples = []
        for params, response in zip(params_list, responses):
            if "error" not in response:
                user_input = response["content"].strip()
                # Filter out metadata fields (prefixed with _) from final output
                clean_params = {k: v for k, v in params.items() if not k.startswith("_")}
                tool_examples.append({
                    "user_input": user_input,
                    "tool_name": tool_name,
                    "parameters": clean_params
                })
            else:
                print(f"    Warning: Error generating input: {response['error']}")

        all_examples.extend(tool_examples)
        print(f"  ✓ Generated {len(tool_examples)} complete examples")

    print(f"\n{'='*60}")
    print(f"Total examples generated: {len(all_examples)}")
    save_json(all_examples, output_path)
    print("="*60)

    return all_examples


def display_sample_examples(examples: List[Dict[str, Any]], count: int = 3):
    """Display a few sample examples."""
    print(f"\nSample Examples (showing {count} of {len(examples)}):")
    print("="*60)

    for i, example in enumerate(examples[:count], 1):
        print(f"\nExample {i}:")
        print(f"User: {example['user_input']}")
        print(f"Tool: {example['tool_name']}")
        print(f"Parameters: {json.dumps(example['parameters'], indent=2)}")
        print("-" * 40)


if __name__ == "__main__":
    # Parse command line arguments
    input_file = "phase2_with_text.json"
    output_file = "phase3_final_examples.json"

    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("❌ Error: ANTHROPIC_API_KEY environment variable not set")
        print("Please set your API key: export ANTHROPIC_API_KEY='your-key-here'")
        sys.exit(1)

    # Run Phase 3
    try:
        examples = phase3_generate_user_inputs(input_file, output_file)

        # Display sample examples
        display_sample_examples(examples, count=5)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
