"""
PHASE 2: Generate Free-Text Parameter Values

This script takes sampled parameters from Phase 1 and uses Claude API
to generate realistic free-text values (notes, reminder messages, etc.).

Requirements:
    - ANTHROPIC_API_KEY environment variable must be set
    - Input file from Phase 1

Usage:
    python3 phase2_generate_param_text.py [input_file] [output_file]

Examples:
    python3 phase2_generate_param_text.py
    python3 phase2_generate_param_text.py phase1_sampled_params.json phase2_with_text.json
"""

import json
import os
import sys
from typing import List, Dict, Any, Optional
from possible_params import is_free_text, possible_params
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


def phase2_generate_free_text(
    input_path: str = "phase1_sampled_params.json",
    output_path: str = "phase2_with_text.json",
    api_key: Optional[str] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    PHASE 2: Generate realistic free-text values for parameters.

    Takes the sampled parameters from Phase 1 and uses Claude API to generate
    realistic text for all "free-text" placeholders.

    Args:
        input_path: Path to Phase 1 output (sampled parameters)
        output_path: Path to save parameters with generated text
        api_key: Optional API key (uses env var if not provided)

    Returns:
        Dictionary mapping tool names to parameters with generated text
    """
    print("="*60)
    print("PHASE 2: Generating Free-Text Values")
    print("="*60)
    print(f"Loading from: {input_path}")
    print()

    # Load Phase 1 data
    sampled_data = load_json(input_path)
    client = ClaudeAPIClient(api_key=api_key)
    tools_schema = load_tools_schema()

    data_with_text = {}

    for tool_name, sampled_params in sampled_data.items():
        print(f"\nProcessing {tool_name}...")

        # Find tool schema
        tool_schema = next(
            (t for t in tools_schema if t['function']['name'] == tool_name),
            None
        )
        if not tool_schema:
            raise ValueError(f"Tool schema not found for {tool_name}")

        # Identify which parameters need free-text generation
        free_text_param = [
            param for param in possible_params[tool_name].keys()
            if is_free_text(tool_name, param)
        ]
        if not free_text_param:
            # No free-text parameters, keep as-is
            print(f"  No free-text parameters, keeping {len(sampled_params)} sets")
            data_with_text[tool_name] = sampled_params
            continue
        assert len(free_text_param) == 1
        free_text_param_name = free_text_param[0]

        # Build prompts for generating free-text values
        requests = []
        for params in sampled_params:
            # Extract persona, length instruction, and tone style (sampled in Phase 1)
            persona = params.get("_persona", "a casual user")
            length_instruction = params.get("_length_instruction", "Write a brief message")
            tone_style = params.get("_tone_style", "casual")

            # Build context about the tool call (exclude metadata fields)
            context_parts = []
            for param, value in params.items():
                if not param.startswith("_") and value != "free-text":
                    context_parts.append(f"  - {param}: {value}")

            context = "\n".join(context_parts) if context_parts else "  (no other parameters)"

            # Build system prompt with persona and tone
            system_prompt = f"You are {persona}. Generate a realistic message they would write using a {tone_style} tone/language style."

            # Build user prompt
            user_prompt = f"""Generate a message for this tool call parameter:

Tool: {tool_name}
Description: {tool_schema['function']['description']}

Parameter: {free_text_param_name}
Description: {tool_schema['function']['parameters']['properties'][free_text_param_name]['description']}

Context:
{context}

{length_instruction} that is natural and realistic for this persona.
Use a {tone_style} tone/language style.
Write ONLY the text value for the "{free_text_param_name}" parameter. Do not use emojis.
Return ONLY the text value, nothing else."""

            requests.append({
                "messages": [
                    {"role": "user", "content": user_prompt}
                ],
                "system": system_prompt,
                "max_tokens": 200,
                "temperature": 1.0
            })

        print(f"  Generating {len(requests)} free-text values for '{free_text_param_name}'...")

        # Call API in parallel
        responses = client.call_parallel(requests)

        # Fill in the generated values
        updated_params = []
        for params, response in zip(sampled_params, responses):
            updated = params.copy()
            if "error" not in response:
                generated_text = response["content"].strip()
                # Replace the free-text placeholder with generated value
                updated[free_text_param_name] = generated_text
            else:
                print(f"    Warning: Error generating text: {response['error']}")
                # Keep placeholder
            updated_params.append(updated)

        data_with_text[tool_name] = updated_params
        print(f"  ✓ Generated text for {len(updated_params)} parameter sets")

    print(f"\n{'='*60}")
    print(f"Total parameter sets with text: {sum(len(v) for v in data_with_text.values())}")
    save_json(data_with_text, output_path)
    print("="*60)

    return data_with_text


if __name__ == "__main__":
    # Parse command line arguments
    input_file = "phase1_sampled_params.json"
    output_file = "phase2_with_text.json"

    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("❌ Error: ANTHROPIC_API_KEY environment variable not set")
        print("Please set your API key: export ANTHROPIC_API_KEY='your-key-here'")
        sys.exit(1)

    # Run Phase 2
    try:
        phase2_generate_free_text(input_file, output_file)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
