#!/usr/bin/env python3
"""
Hyperparameter search for Qwen 3 tool calling training.

This script runs multiple training jobs with different hyperparameter configurations
to find the optimal settings. Supports both grid search and random search strategies.

Usage:
    # Grid search with all combinations
    python hparam_search.py --strategy grid --config hparam_config.json

    # Random search with N trials
    python hparam_search.py --strategy random --config hparam_config.json --num_trials 20

    # Resume from previous run
    python hparam_search.py --strategy grid --config hparam_config.json --resume
"""

import os
import sys
import json
import argparse
import itertools
import random
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HyperparameterSearch:
    """Manages hyperparameter search for training runs."""

    def __init__(
        self,
        config_path: str,
        strategy: str = "grid",
        num_trials: Optional[int] = None,
        output_dir: str = "./hparam_search_results",
        resume: bool = False,
        keep_checkpoints: bool = False
    ):
        """
        Initialize hyperparameter search.

        Args:
            config_path: Path to JSON config file with search space
            strategy: "grid" for grid search, "random" for random search
            num_trials: Number of trials for random search (ignored for grid)
            output_dir: Directory to save results
            resume: Whether to resume from previous run
            keep_checkpoints: Whether to keep checkpoints after each run (default: False to save space)
        """
        self.config_path = config_path
        self.strategy = strategy
        self.num_trials = num_trials
        self.output_dir = Path(output_dir)
        self.resume = resume
        self.keep_checkpoints = keep_checkpoints

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Results tracking
        self.results_file = self.output_dir / "results.json"
        self.completed_runs = self._load_completed_runs() if resume else []

        logger.info(f"Initialized hyperparameter search")
        logger.info(f"Strategy: {strategy}")
        logger.info(f"Output directory: {output_dir}")
        if resume:
            logger.info(f"Resuming: {len(self.completed_runs)} runs already completed")

    def _load_completed_runs(self) -> List[Dict]:
        """Load previously completed runs from results file."""
        if not self.results_file.exists():
            return []

        with open(self.results_file, 'r') as f:
            return json.load(f)

    def _save_results(self, results: List[Dict]):
        """Save results to JSON file."""
        with open(self.results_file, 'w') as f:
            json.dump(results, indent=2, fp=f)
        logger.info(f"Saved {len(results)} results to {self.results_file}")

    def _expand_lora_params(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Expand LoRA rank and alpha_rank_ratio into rank and alpha.

        If config has 'lora_rank' and 'alpha_rank_ratio', compute lora_alpha.
        Otherwise, use lora_rank and lora_alpha directly.
        """
        if "alpha_rank_ratio" in config and "lora_rank" in config:
            expanded = config.copy()
            rank = expanded.pop("lora_rank")
            ratio = expanded.pop("alpha_rank_ratio")
            expanded["lora_rank"] = rank
            expanded["lora_alpha"] = float(rank * ratio)
            return expanded
        return config

    def _generate_grid_configs(self) -> List[Dict[str, Any]]:
        """Generate all hyperparameter combinations for grid search."""
        search_space = self.config["search_space"]

        # Get all parameter names and their values
        param_names = list(search_space.keys())
        param_values = [search_space[name] for name in param_names]

        # Generate all combinations
        configs = []
        for combination in itertools.product(*param_values):
            config = dict(zip(param_names, combination))
            # Expand LoRA params if using ratio notation
            config = self._expand_lora_params(config)
            configs.append(config)

        logger.info(f"Generated {len(configs)} configurations for grid search")
        return configs

    def _generate_random_configs(self, num_trials: int) -> List[Dict[str, Any]]:
        """Generate random hyperparameter configurations."""
        search_space = self.config["search_space"]

        configs = []
        for _ in range(num_trials):
            config = {
                param: random.choice(values)
                for param, values in search_space.items()
            }
            # Expand LoRA params if using ratio notation
            config = self._expand_lora_params(config)
            configs.append(config)

        logger.info(f"Generated {num_trials} random configurations")
        return configs

    def _is_already_completed(self, hparams: Dict[str, Any]) -> bool:
        """Check if this configuration has already been run."""
        for run in self.completed_runs:
            if run.get("hyperparameters") == hparams:
                return True
        return False

    def _create_run_name(self, hparams: Dict[str, Any]) -> str:
        """Create a descriptive name for this run."""
        parts = []
        for key, value in sorted(hparams.items()):
            # Abbreviate parameter names
            abbrev = {
                "learning_rate": "lr",
                "num_epochs": "ep",
                "lora_rank": "r",
                "lora_alpha": "a",
                "batch_size": "bs",
                "gradient_accumulation_steps": "gas"
            }.get(key, key)

            # Format value
            if isinstance(value, float):
                val_str = f"{value:.0e}" if value < 0.001 else f"{value:.4f}".rstrip('0').rstrip('.')
            else:
                val_str = str(value)

            parts.append(f"{abbrev}{val_str}")

        return "_".join(parts)

    def _run_training(
        self,
        hparams: Dict[str, Any],
        run_name: str,
        run_idx: int,
        total_runs: int
    ) -> Dict[str, Any]:
        """
        Run a single training job with given hyperparameters.

        Args:
            hparams: Hyperparameter configuration
            run_name: Name for this run
            run_idx: Index of this run (1-indexed)
            total_runs: Total number of runs

        Returns:
            Dictionary with run results
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Run {run_idx}/{total_runs}: {run_name}")
        logger.info(f"Hyperparameters: {json.dumps(hparams, indent=2)}")
        logger.info(f"{'='*80}\n")

        # Create run-specific directories
        run_dir = self.output_dir / run_name
        run_dir.mkdir(exist_ok=True)

        ckpt_dir = run_dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        ckpt_dir = ckpt_dir.absolute()

        # Build command with hyperparameters
        # Use python -m to run as module (required for imports to work)
        cmd = [
            sys.executable,
            "-m",
            "pebble_qwen.train_qwen3_tool_calling",
            "--learning_rate", str(hparams.get("learning_rate", 5e-5)),
            "--num_epochs", str(hparams.get("num_epochs", 5)),
            "--lora_rank", str(hparams.get("lora_rank", 16)),
            "--lora_alpha", str(hparams.get("lora_alpha", 32.0)),
            "--batch_size", str(hparams.get("batch_size", 8)),
            "--gradient_accumulation_steps", str(hparams.get("gradient_accumulation_steps", 1)),
            "--checkpoint_dir", str(ckpt_dir),
            "--run_name", run_name,
        ]

        # Add any fixed parameters from config
        if "fixed_params" in self.config:
            for key, value in self.config["fixed_params"].items():
                cmd.extend([f"--{key}", str(value)])

        # Save command to file
        cmd_file = run_dir / "command.txt"
        with open(cmd_file, 'w') as f:
            f.write(" ".join(cmd))

        # Run training
        start_time = datetime.now()
        log_file = run_dir / "training.log"

        try:
            with open(log_file, 'w') as f:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )

                # Stream output to both file and console
                for line in process.stdout:
                    f.write(line)
                    print(line, end='')
                    f.flush()

                process.wait()
                return_code = process.returncode

        except Exception as e:
            logger.error(f"Training failed with exception: {e}")
            return_code = -1

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Parse results from log file or checkpoint
        metrics = self._extract_metrics(run_dir, log_file)

        result = {
            "run_name": run_name,
            "hyperparameters": hparams,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "return_code": return_code,
            "success": return_code == 0,
            "metrics": metrics,
            "output_dir": str(run_dir),
        }

        # Save individual run result
        result_file = run_dir / "result.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)

        # Clean up checkpoints if not keeping them (saves disk space)
        if not self.keep_checkpoints:
            import shutil
            if ckpt_dir.exists():
                logger.info("Cleaning up checkpoints to save disk space...")
                try:
                    shutil.rmtree(ckpt_dir)
                    logger.info(f"Removed checkpoint directory: {ckpt_dir}")
                except Exception as e:
                    logger.warning(f"Failed to remove checkpoints: {e}")

        return result

    def _extract_metrics(self, run_dir: Path, log_file: Path) -> Dict[str, Any]:
        """
        Extract final metrics from training.

        Tries multiple sources in order of preference:
        1. metrics.json file (saved by training script)
        2. TensorBoard event files
        3. Parsing training logs

        Returns:
            Dictionary with final_train_loss, final_val_loss, and best_val_loss
        """
        metrics = {
            "final_train_loss": None,
            "final_val_loss": None,
            "best_val_loss": None,
        }

        # Method 1: Try to load from metrics.json (most reliable)
        metrics_json = run_dir / "checkpoints" / "metrics.json"
        if metrics_json.exists():
            try:
                with open(metrics_json, 'r') as f:
                    saved_metrics = json.load(f)
                logger.info(f"Loaded metrics from {metrics_json}")
                return {
                    "final_train_loss": saved_metrics.get("final_train_loss"),
                    "final_val_loss": saved_metrics.get("final_val_loss"),
                    "best_val_loss": saved_metrics.get("best_val_loss"),
                }
            except Exception as e:
                logger.warning(f"Failed to load metrics.json: {e}")

        # Method 2: Try to read from TensorBoard events
        try:
            tb_dir = run_dir / "checkpoints" / "tensorboard"
            if tb_dir.exists():
                metrics_from_tb = self._extract_metrics_from_tensorboard(tb_dir)
                if metrics_from_tb["best_val_loss"] is not None:
                    logger.info("Loaded metrics from TensorBoard events")
                    return metrics_from_tb
        except Exception as e:
            logger.warning(f"Failed to extract from TensorBoard: {e}")

        # Method 3: Parse training logs (least reliable, but fallback)
        if log_file.exists():
            try:
                metrics_from_logs = self._extract_metrics_from_logs(log_file)
                if metrics_from_logs["best_val_loss"] is not None:
                    logger.info("Loaded metrics from training logs")
                    return metrics_from_logs
            except Exception as e:
                logger.warning(f"Failed to extract from logs: {e}")

        logger.warning(f"Could not extract metrics for run in {run_dir}")
        return metrics

    def _extract_metrics_from_tensorboard(self, tb_dir: Path) -> Dict[str, Any]:
        """
        Extract metrics from TensorBoard event files.

        Args:
            tb_dir: Directory containing TensorBoard event files

        Returns:
            Dictionary with extracted metrics
        """
        metrics = {
            "final_train_loss": None,
            "final_val_loss": None,
            "best_val_loss": None,
        }

        try:
            from tensorboard.backend.event_processing import event_accumulator

            # Find all event files
            event_files = list(tb_dir.glob("events.out.tfevents.*"))
            if not event_files:
                return metrics

            # Load events from the most recent file
            event_file = max(event_files, key=lambda p: p.stat().st_mtime)
            ea = event_accumulator.EventAccumulator(str(event_file))
            ea.Reload()

            # Extract scalars
            train_losses = []
            val_losses = []

            # Try common metric names
            for tag in ea.Tags().get('scalars', []):
                if 'train' in tag.lower() and 'loss' in tag.lower():
                    events = ea.Scalars(tag)
                    train_losses.extend([e.value for e in events])
                elif 'val' in tag.lower() and 'loss' in tag.lower():
                    events = ea.Scalars(tag)
                    val_losses.extend([e.value for e in events])
                elif 'eval' in tag.lower() and 'loss' in tag.lower():
                    events = ea.Scalars(tag)
                    val_losses.extend([e.value for e in events])

            if train_losses:
                metrics["final_train_loss"] = train_losses[-1]
            if val_losses:
                metrics["final_val_loss"] = val_losses[-1]
                metrics["best_val_loss"] = min(val_losses)

        except ImportError:
            logger.debug("tensorboard package not available for reading event files")
        except Exception as e:
            logger.debug(f"Error reading TensorBoard events: {e}")

        return metrics

    def _extract_metrics_from_logs(self, log_file: Path) -> Dict[str, Any]:
        """
        Extract metrics by parsing training log file.

        Args:
            log_file: Path to training log file

        Returns:
            Dictionary with extracted metrics
        """
        metrics = {
            "final_train_loss": None,
            "final_val_loss": None,
            "best_val_loss": None,
        }

        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()

            train_losses = []
            val_losses = []

            # Improved patterns to match various log formats
            import re
            train_patterns = [
                r'train[_\s]loss[:\s=]+([0-9.]+)',
                r'training[_\s]loss[:\s=]+([0-9.]+)',
                r'Final training loss:\s*([0-9.]+)',
            ]
            val_patterns = [
                r'val[_\s]loss[:\s=]+([0-9.]+)',
                r'validation[_\s]loss[:\s=]+([0-9.]+)',
                r'eval[_\s]loss[:\s=]+([0-9.]+)',
                r'Final validation loss:\s*([0-9.]+)',
                r'Best validation loss:\s*([0-9.]+)',
            ]

            for line in lines:
                line_lower = line.lower()

                # Try to match training loss
                for pattern in train_patterns:
                    match = re.search(pattern, line_lower)
                    if match:
                        try:
                            train_losses.append(float(match.group(1)))
                        except (ValueError, IndexError):
                            pass

                # Try to match validation loss
                for pattern in val_patterns:
                    match = re.search(pattern, line_lower)
                    if match:
                        try:
                            val_losses.append(float(match.group(1)))
                        except (ValueError, IndexError):
                            pass

            if train_losses:
                metrics["final_train_loss"] = train_losses[-1]

            if val_losses:
                metrics["final_val_loss"] = val_losses[-1]
                metrics["best_val_loss"] = min(val_losses)

        except Exception as e:
            logger.debug(f"Error parsing log file: {e}")

        return metrics

    def run(self):
        """Run hyperparameter search."""
        # Generate configurations
        if self.strategy == "grid":
            configs = self._generate_grid_configs()
        elif self.strategy == "random":
            if self.num_trials is None:
                raise ValueError("num_trials must be specified for random search")
            configs = self._generate_random_configs(self.num_trials)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        # Filter out already completed runs if resuming
        if self.resume:
            configs = [
                config for config in configs
                if not self._is_already_completed(config)
            ]
            logger.info(f"After filtering completed runs: {len(configs)} configurations remaining")

        if not configs:
            logger.info("No configurations to run!")
            return

        # Run all configurations
        all_results = list(self.completed_runs)  # Start with existing results
        total_runs = len(configs)

        for idx, hparams in enumerate(configs, start=1):
            run_name = self._create_run_name(hparams)

            result = self._run_training(hparams, run_name, idx, total_runs)
            all_results.append(result)

            # Save results after each run
            self._save_results(all_results)

            # Print summary
            if result["success"]:
                logger.info(f"\n✓ Run {idx}/{total_runs} completed successfully")
                if result["metrics"].get("final_val_loss"):
                    logger.info(f"  Final validation loss: {result['metrics']['final_val_loss']:.4f}")
            else:
                logger.info(f"\n✗ Run {idx}/{total_runs} failed (return code: {result['return_code']})")

        # Print final summary
        self._print_summary(all_results)

    def _print_summary(self, results: List[Dict]):
        """Print summary of all results."""
        successful_runs = [r for r in results if r["success"]]
        failed_runs = [r for r in results if not r["success"]]

        logger.info(f"\n{'='*80}")
        logger.info(f"HYPERPARAMETER SEARCH SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"Total runs: {len(results)}")
        logger.info(f"Successful: {len(successful_runs)}")
        logger.info(f"Failed: {len(failed_runs)}")

        if successful_runs:
            # Find best run by validation loss
            runs_with_val_loss = [
                r for r in successful_runs
                if r["metrics"].get("best_val_loss") is not None
            ]

            if runs_with_val_loss:
                best_run = min(runs_with_val_loss, key=lambda r: r["metrics"]["best_val_loss"])

                logger.info(f"\nBest run: {best_run['run_name']}")
                logger.info(f"  Best validation loss: {best_run['metrics']['best_val_loss']:.4f}")
                logger.info(f"  Hyperparameters:")
                for key, value in best_run["hyperparameters"].items():
                    logger.info(f"    {key}: {value}")
                logger.info(f"  Output: {best_run['output_dir']}")

        logger.info(f"\nFull results saved to: {self.results_file}")
        logger.info(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter search for Qwen 3 tool calling training"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to hyperparameter config JSON file"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["grid", "random"],
        default="grid",
        help="Search strategy: grid or random"
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        help="Number of trials for random search (required if strategy=random)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./hparam_search_results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous search (skip completed runs)"
    )
    parser.add_argument(
        "--keep_checkpoints",
        action="store_true",
        help="Keep checkpoints after each run (default: False, checkpoints are deleted to save space)"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.strategy == "random" and args.num_trials is None:
        parser.error("--num_trials is required when using random search")

    # Run search
    search = HyperparameterSearch(
        config_path=args.config,
        strategy=args.strategy,
        num_trials=args.num_trials,
        output_dir=args.output_dir,
        resume=args.resume,
        keep_checkpoints=args.keep_checkpoints
    )

    search.run()


if __name__ == "__main__":
    main()
