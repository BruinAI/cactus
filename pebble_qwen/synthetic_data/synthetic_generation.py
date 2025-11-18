"""
Synthetic data generation orchestrator.

This script orchestrates the 3-phase synthetic data generation process
by calling each phase script in sequence.

Usage:
    python3 synthetic_generation.py <samples_per_tool>

Example:
    python3 synthetic_generation.py 20
"""

import sys
import subprocess


def run_phase(script_name: str, *args) -> int:
    """Run a phase script and return its exit code."""
    cmd = ["python3", script_name] + list(args)
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print('='*60)

    result = subprocess.run(cmd)
    return result.returncode


def run_all_phases(samples_per_tool: str):
    """Run all three phases sequentially."""
    print("="*60)
    print("SYNTHETIC DATA GENERATION - ALL PHASES")
    print("="*60)
    print(f"Samples per tool: {samples_per_tool}")

    # Phase 1: Sample parameters
    exit_code = run_phase("phase1_sample_params.py", samples_per_tool)
    if exit_code != 0:
        print(f"\n❌ Phase 1 failed with exit code {exit_code}")
        return exit_code

    # Phase 2: Generate free-text
    exit_code = run_phase("phase2_generate_param_text.py")
    if exit_code != 0:
        print(f"\n❌ Phase 2 failed with exit code {exit_code}")
        return exit_code

    # Phase 3: Generate user inputs
    exit_code = run_phase("phase3_generate_inputs.py")
    if exit_code != 0:
        print(f"\n❌ Phase 3 failed with exit code {exit_code}")
        return exit_code

    print("\n" + "="*60)
    print("✅ ALL PHASES COMPLETED SUCCESSFULLY")
    print("="*60)
    print("\nOutput files:")
    print("  - phase1_sampled_params.json")
    print("  - phase2_with_text.json")
    print("  - phase3_final_examples.json")
    print()

    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 synthetic_generation.py <samples_per_tool>")
        print("Example: python3 synthetic_generation.py 20")
        sys.exit(1)

    samples_per_tool = sys.argv[1]

    exit_code = run_all_phases(samples_per_tool)
    sys.exit(exit_code)
