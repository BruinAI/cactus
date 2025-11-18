"""
Synthetic data generation orchestrator.

This script orchestrates the 3-phase synthetic data generation process
by calling each phase script in sequence, then converts the output to
training format.

Usage:
    python3 synthetic_generation.py <mode> <samples_per_tool>

Arguments:
    mode: Either 'append' or 'overwrite'
    samples_per_tool: Number of samples to generate per tool

Examples:
    python3 synthetic_generation.py append 20
    python3 synthetic_generation.py overwrite 50
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


def run_all_phases(mode: str, samples_per_tool: str):
    """Run all three phases sequentially."""
    print("="*60)
    print("SYNTHETIC DATA GENERATION - ALL PHASES")
    print("="*60)
    print(f"Mode: {mode}")
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

    # Convert to training format
    exit_code = run_phase(
        "convert_to_training_format.py",
        mode,
        "phase3_final_examples.json",
        "../data/synthetic_finetune_dataset.json"
    )
    if exit_code != 0:
        print(f"\n❌ Conversion to training format failed with exit code {exit_code}")
        return exit_code

    print("\n" + "="*60)
    print("✅ ALL PHASES COMPLETED SUCCESSFULLY")
    print("="*60)
    print("\nOutput files:")
    print("  - phase1_sampled_params.json")
    print("  - phase2_with_text.json")
    print("  - phase3_final_examples.json")
    print(f"  - ../data/synthetic_finetune_dataset.json ({mode} mode)")
    print()

    return 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 synthetic_generation.py <mode> <samples_per_tool>")
        print("\nArguments:")
        print("  mode: Either 'append' or 'overwrite'")
        print("  samples_per_tool: Number of samples to generate per tool")
        print("\nExamples:")
        print("  python3 synthetic_generation.py append 20")
        print("  python3 synthetic_generation.py overwrite 50")
        sys.exit(1)

    mode = sys.argv[1]
    samples_per_tool = sys.argv[2]

    # Validate mode
    if mode not in ['append', 'overwrite']:
        print(f"❌ Error: Invalid mode '{mode}'. Must be 'append' or 'overwrite'.")
        sys.exit(1)

    exit_code = run_all_phases(mode, samples_per_tool)
    sys.exit(exit_code)
