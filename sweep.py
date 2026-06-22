#!/usr/bin/env python3
"""Hyperparameter sweep for gbagent PPO training.

Launches sequential training runs with different hyperparameter
configurations and records results for comparison.

Usage
-----
    # Run sweep with default parameter grid
    python sweep.py

    # Run specific configs
    python sweep.py --params learning_rate entropy_coef
    python sweep.py --params learning_rate --values 1e-4 2.5e-4 5e-4

    # Use a base config file
    python sweep.py --base configs/pokemon_red.yaml

    # Dry run (print configs without training)
    python sweep.py --dry-run

    # Resume interrupted sweep
    python sweep.py --resume sweep_results.json

Grid Definition
---------------
The default grid covers the most impactful PPO hyperparameters:

    learning_rate:   [1e-4, 2.5e-4, 5e-4, 1e-3]
    entropy_coef:    [0.001, 0.005, 0.01, 0.02]
    clip_epsilon:    [0.1, 0.2, 0.3]
    gae_lambda:      [0.9, 0.95, 0.97, 0.99]
    update_epochs:   [3, 4, 8]
    num_minibatches: [4, 8, 16]
    n_steps:         [64, 128, 256]

Results
-------
Written to ``sweep_results.json`` after each run (append-mode so partial
results survive crashes).  Each entry includes:

    {
        "params": { ... hyperparameter dict ... },
        "timestamp": "2026-01-01T12:00:00",
        "global_step": 500000,
        "mean_return": 12.3,
        "mean_value_loss": 0.05,
        "mean_policy_loss": -0.01,
        "mean_entropy": 1.2,
        "approx_kl": 0.003,
        "clip_frac": 0.05,
        "sps": 450,
        "elapsed_s": 1111
    }
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Default parameter grid
# ---------------------------------------------------------------------------

DEFAULT_GRID: dict[str, list[Any]] = {
    "learning_rate": [1e-4, 2.5e-4, 5e-4],
    "entropy_coef": [0.005, 0.01, 0.02],
    "clip_epsilon": [0.1, 0.2],
    "gae_lambda": [0.95, 0.97],
    "update_epochs": [4, 8],
    "num_minibatches": [4, 8],
    "n_steps": [128, 256],
}

# Maximum number of grid points to run (0 = unlimited)
_MAX_RUNS = 0


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------


class SweepRunner:
    """Run a sequence of training jobs with different hyperparameters."""

    def __init__(
        self,
        base_config: str | None = None,
        results_file: str = "sweep_results.json",
        dry_run: bool = False,
        params_to_sweep: list[str] | None = None,
        param_values: dict[str, list[Any]] | None = None,
        max_runs: int = 0,
    ):
        self.base_config = base_config
        self.results_file = Path(results_file)
        self.dry_run = dry_run
        self.params_to_sweep = params_to_sweep
        self.param_values = param_values or {}
        self.max_runs = max_runs

        self.results: list[dict[str, Any]] = []
        self.completed_keys: set[str] = set()

        # Load existing results (for resume)
        if self.results_file.is_file():
            try:
                data = json.loads(self.results_file.read_text())
                if isinstance(data, list):
                    self.results = data
                    self.completed_keys = {
                        _param_key(r["params"]) for r in self.results
                    }
                    msg = f"  ✓ Loaded {len(self.results)} existing results"
                    print(f"{msg} from {self.results_file}")
            except (json.JSONDecodeError, KeyError):
                print("  ⚠ Could not parse existing results file, starting fresh")

    # ------------------------------------------------------------------
    def build_grid(self) -> list[dict[str, Any]]:
        """Build the Cartesian product of parameter ranges."""
        grid_params: dict[str, list[Any]] = {}

        if self.param_values:
            # Use explicit values if provided
            grid_params.update(self.param_values)
        else:
            grid_params.update(DEFAULT_GRID)

        if self.params_to_sweep:
            # Filter to only requested params
            for key in self.params_to_sweep:
                if key not in grid_params:
                    print(f"  ⚠ Unknown parameter '{key}', skipping")
            grid_params = {k: v for k, v in grid_params.items() if k in self.params_to_sweep}

        param_names = list(grid_params.keys())
        param_values_list = list(grid_params.values())

        points = []
        for values in itertools.product(*param_values_list):
            point = dict(zip(param_names, values, strict=False))
            key = _param_key(point)
            if key not in self.completed_keys:
                points.append(point)

        # Limit
        if self.max_runs > 0:
            points = points[:self.max_runs]

        return points

    # ------------------------------------------------------------------
    def run_sweep(self) -> None:
        """Execute all pending grid points sequentially."""
        grid = self.build_grid()
        if not grid:
            print("  ✓ All grid points already completed. Nothing to do.")
            return

        print(f"\n  Sweep grid: {len(grid)} pending runs")
        for i, params in enumerate(grid):
            print(f"\n{'=' * 60}")
            print(f"  Run {i + 1}/{len(grid)}: {params}")
            print(f"{'=' * 60}")

            if self.dry_run:
                print("  [dry-run] Would train with:")
                for k, v in sorted(params.items()):
                    print(f"    {k}: {v}")
                continue

            result = self._run_single(params)

            # Append to results
            saved = self._save_result(result)
            if saved:
                print(f"  → Saved to {self.results_file}")

        self._print_summary()

    # ------------------------------------------------------------------
    def _run_single(self, params: dict[str, Any]) -> dict[str, Any]:
        """Run one training session with the given hyperparameters."""
        # Build CLI arguments
        cmd = [sys.executable, "train.py"]

        if self.base_config:
            cmd.extend(["--config", self.base_config])

        # Map parameter names to CLI flags
        flag_map = {
            "learning_rate": "--learning-rate",
            "entropy_coef": None,  # set via config or env var
            "clip_epsilon": None,
            "gae_lambda": None,
            "update_epochs": None,
            "num_minibatches": None,
            "n_steps": None,
            "total_timesteps": "--total-timesteps",
            "num_envs": "--num-envs",
        }

        # Use environment variable override for non-CLI params
        env_override: str = json.dumps({
            k: v for k, v in params.items() if k not in flag_map or flag_map[k] is None
        })

        env = dict(
            GBAGENT_SWEEP_PARAMS=env_override,
            # Reduce total timesteps for sweep to iterate faster
            # Override via --total-timesteps CLI if needed
        )

        for k, v in params.items():
            flag = flag_map.get(k)
            if flag:
                cmd.extend([flag, str(v)])

        # Reduce sweep run to 2M timesteps by default
        cmd.extend(["--total-timesteps", "2000000"])
        param_tag = "_".join(f"{k}{v}" for k, v in params.items())
        cmd.extend(["--log-dir", f"logs/sweep/{param_tag}"])

        start_time = time.time()
        print(f"  $ {' '.join(cmd)}")
        print(f"  Env override: {env_override}")
        print()

        try:
            result = subprocess.run(
                cmd,
                env={**__import__("os").environ, **env},
                capture_output=True,
                text=True,
                timeout=3600 * 12,  # 12h max per run
            )
            success = result.returncode == 0
            if result.stdout:
                print(result.stdout[-2000:])  # tail of output
            if result.stderr:
                print("stderr:", result.stderr[-1000:])
        except subprocess.TimeoutExpired:
            print("  ⚠ Run timed out after 12h")
            success = False
            result = None
        except Exception as exc:
            print(f"  ⚠ Run failed: {exc}")
            success = False
            result = None

        elapsed = time.time() - start_time

        # Parse training log for final metrics
        metrics = self._extract_final_metrics(result)

        return {
            "params": params,
            "success": success,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": round(elapsed, 1),
            **metrics,
        }

    # ------------------------------------------------------------------
    def _extract_final_metrics(self, result: subprocess.CompletedProcess | None) -> dict[str, Any]:
        """Extract final metrics from training stdout."""
        if result is None or not result.stdout:
            return {"note": "no output captured"}
        tail = result.stdout
        # Look for the final "Training complete" line and surrounding metrics
        import re
        metrics: dict[str, Any] = {}

        # Parse final step count: "step 5,000,000 (50.0%)"
        m = re.search(r'step\s+([\d,]+)\s+\(', tail)
        if m:
            metrics["global_step"] = int(m.group(1).replace(",", ""))

        # Parse SPS: "SPS 450"
        m = re.search(r'SPS\s+([\d.]+)', tail)
        if m:
            metrics["sps"] = float(m.group(1))

        # Parse return: "R 12.34"
        m = re.search(r'R\s+([\d.-]+)', tail)
        if m:
            metrics["mean_return"] = float(m.group(1))

        # Parse policy loss: "PG 0.012"
        m = re.search(r'PG\s+([\d.-]+)', tail)
        if m:
            metrics["mean_policy_loss"] = float(m.group(1))

        # Parse value loss: "VF 0.050"
        m = re.search(r'VF\s+([\d.-]+)', tail)
        if m:
            metrics["mean_value_loss"] = float(m.group(1))

        # Parse entropy: "Ent 1.200"
        m = re.search(r'Ent\s+([\d.-]+)', tail)
        if m:
            metrics["mean_entropy"] = float(m.group(1))

        # Parse KL: "KL 0.003"
        m = re.search(r'KL\s+([\d.-]+)', tail)
        if m:
            metrics["approx_kl"] = float(m.group(1))

        # Parse clip frac: "CF 0.050"
        m = re.search(r'CF\s+([\d.-]+)', tail)
        if m:
            metrics["clip_frac"] = float(m.group(1))

        if not metrics:
            metrics["note"] = "no metrics matched in output"

        return metrics

    # ------------------------------------------------------------------
    def _save_result(self, result: dict[str, Any]) -> bool:
        """Append a result to the JSON results file."""
        self.results.append(result)
        try:
            self.results_file.write_text(json.dumps(self.results, indent=2))
            return True
        except OSError as exc:
            print(f"  ⚠ Failed to write results: {exc}")
            return False

    # ------------------------------------------------------------------
    def _print_summary(self) -> None:
        """Print a sweep summary."""
        if not self.results:
            print("\n  No results to summarise.")
            return

        print(f"\n{'=' * 60}")
        print(f"  Sweep Summary ({len(self.results)} runs)")
        print(f"{'=' * 60}")
        print(f"  Results file: {self.results_file}")

        # Count success/failure
        successful = sum(1 for r in self.results if r.get("success"))
        failed = sum(1 for r in self.results if not r.get("success"))
        print(f"  Successful: {successful}, Failed: {failed}")

        # Show best runs by mean_return if available
        with_return = [r for r in self.results if r.get("mean_return") is not None]
        if with_return:
            with_return.sort(key=lambda r: r["mean_return"], reverse=True)
            print("\n  Top 3 runs by mean return:")
            for i, r in enumerate(with_return[:3]):
                print(f"    {i+1}. R={r['mean_return']:.2f}  params={r['params']}")
        else:
            print("\n  Tip: to see metrics, parse TensorBoard event files")
            print("  or implement _extract_final_metrics() with tbparse.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _param_key(params: dict[str, Any]) -> str:
    """Return a canonical string key for a parameter dict."""
    return json.dumps(params, sort_keys=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hyperparameter sweep for gbagent PPO training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base", type=str, default=None,
        help="Base YAML config file (game, reward, etc.)",
    )
    parser.add_argument(
        "--params", type=str, nargs="*", default=None,
        help="Which hyperparameters to sweep (default: all)",
    )
    parser.add_argument(
        "--values", type=str, nargs="+", action="append",
        dest="param_values_list",
        help="Custom values for each --param (in order). "
             "Multiple --values args match corresponding --params.",
    )
    parser.add_argument(
        "--max-runs", type=int, default=0,
        help="Maximum number of sweep runs (0 = unlimited)",
    )
    parser.add_argument(
        "--results", type=str, default="sweep_results.json",
        help="Path to results file (append-mode)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print grid points without running",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Resume from an existing results file",
    )
    parser.add_argument(
        "--list-defaults", action="store_true",
        help="Print the default parameter grid and exit",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_defaults:
        print("Default sweep grid:")
        for param, values in DEFAULT_GRID.items():
            print(f"  {param}: {values}")
        sys.exit(0)

    # Convert --values args to dict keyed by --params
    param_values: dict[str, list[Any]] = {}
    if args.params and args.param_values_list:
        if len(args.params) != len(args.param_values_list):
            print("Error: --params and --values counts must match")
            sys.exit(1)
        for param, values in zip(args.params, args.param_values_list, strict=False):
            param_values[param] = [parse_value(v) for v in values]

    # Resume uses existing results file
    results_file = args.results
    if args.resume:
        results_file = args.resume

    runner = SweepRunner(
        base_config=args.base,
        results_file=results_file,
        dry_run=args.dry_run,
        params_to_sweep=args.params,
        param_values=param_values,
        max_runs=args.max_runs,
    )

    runner.run_sweep()


def parse_value(s: str) -> Any:
    """Parse a CLI string into int, float, or str."""
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


if __name__ == "__main__":
    main()
