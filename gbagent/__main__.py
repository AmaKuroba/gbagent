#!/usr/bin/env python3
"""gbagent — Run inference, export models, and check model info from the CLI.

Usage
-----
    # Run headless inference in an environment
    python -m gbagent --model checkpoints/model.keras \\
                      --game PokemonRed-GB --state Level1 --steps 5000

    # Export to TFLite
    python -m gbagent export checkpoints/model.keras export/model.tflite

    # Export to ONNX
    python -m gbagent export checkpoints/model.keras export/model.onnx --format onnx

    # Show model summary
    python -m gbagent info checkpoints/model.keras
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="gbagent — inference, export, and model utilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # ── run (default) ────────────────────────────────────────────────
    run_p = sub.add_parser(
        "run",
        help="Run headless inference in a game environment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    run_p.add_argument("--model", "-m", type=str, required=True,
                       help="Path to model file (.keras / .tflite / .onnx)")
    run_p.add_argument("--backend", "-b", type=str, default=None,
                       choices=["keras", "tflite", "onnx"],
                       help="Inference backend (inferred from extension if omitted)")
    run_p.add_argument("--game", "-g", type=str, default="PokemonRed-GB",
                       help="ROM/game name")
    run_p.add_argument("--state", "-s", type=str, default="Level1",
                       help="Start state")
    run_p.add_argument("--steps", type=int, default=1000,
                       help="Maximum inference steps")
    run_p.add_argument("--frame-skip", type=int, default=4)
    run_p.add_argument("--frame-stack", type=int, default=4)
    run_p.add_argument("--render", action="store_true",
                       help="Render game window (requires GUI)")
    run_p.add_argument("--deterministic", action="store_true",
                       help="Greedy (argmax) action selection")
    run_p.add_argument("--seed", type=int, default=None,
                       help="Environment seed")

    # ── export ───────────────────────────────────────────────────────
    export_p = sub.add_parser(
        "export",
        help="Export model to TFLite or ONNX",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    export_p.add_argument("model", type=str,
                          help="Path to .keras model file")
    export_p.add_argument("output", type=str, nargs="?", default=None,
                          help="Output path (default: export/<stem>.<ext>)")
    export_p.add_argument("--format", "-f", choices=["tflite", "onnx"],
                          default="tflite",
                          help="Export format")
    export_p.add_argument("--quantize", "-q", choices=["fp16", "int8"],
                          default=None,
                          help="Quantization mode (TFLite only)")
    export_p.add_argument("--repr-data", type=str, default=None,
                          help="Path to .npy representative dataset (INT8 quant)")
    export_p.add_argument("--opset", type=int, default=17,
                          help="ONNX opset version")

    # ── info ─────────────────────────────────────────────────────────
    info_p = sub.add_parser(
        "info",
        help="Show model summary and metadata",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    info_p.add_argument("model", type=str,
                        help="Path to model file (.keras / .tflite / .onnx)")
    info_p.add_argument("--backend", "-b", type=str, default=None,
                        choices=["keras", "tflite", "onnx"],
                        help="Inference backend (inferred from extension if omitted)")

    return parser


# ===================================================================
# Commands
# ===================================================================


def cmd_run(args: argparse.Namespace) -> None:
    """Run headless inference in a game environment."""
    from gbagent.env import GBAGEnv
    from gbagent.inference import InferenceEngine

    # Load engine
    print(f"› Loading model from {args.model} …")
    engine = InferenceEngine(args.model, backend=args.backend)
    print(f"  Backend: {engine.backend}")
    print()

    # Create env
    print(f"› Creating environment: game={args.game}, state={args.state}")
    env = GBAGEnv(
        game=args.game,
        state=args.state,
        frame_skip=args.frame_skip,
        frame_stack=args.frame_stack,
    )
    if args.seed is not None:
        env.reset(seed=args.seed)
    else:
        env.reset()

    # Timing
    import time
    start_time = time.time()

    # Rollout
    print(f"› Running inference for {args.steps} steps (deterministic={args.deterministic}) …")
    result = engine.rollout(
        env,
        n_steps=args.steps,
        render=args.render,
        deterministic=args.deterministic,
        callback=lambda step, *_: (step % 500 == 0 and print(f"  Step {step} ..."), True)[1],
    )

    elapsed = time.time() - start_time

    # Summary
    print()
    print("╔════════════════════════════════════════════╗")
    print("║           Inference Results                ║")
    print("╚════════════════════════════════════════════╝")
    print(f"  Steps:          {result['steps']:,}")
    print(f"  Total reward:   {result['total_reward']:.2f}")
    print(f"  Mean value:     {result['mean_value']:.4f}")
    print(f"  Duration:       {result['duration_s']:.1f}s")
    print(f"  FPS:            {result['fps']:.0f}")
    print(f"  Wall time:      {elapsed:.1f}s")
    print()

    env.close()


def cmd_export(args: argparse.Namespace) -> None:
    """Export model to TFLite or ONNX."""
    from gbagent.export import export_to_onnx, export_to_tflite

    repr_data = None
    if args.repr_data:
        print(f"› Loading representative data from {args.repr_data} …")
        repr_data = np.load(args.repr_data)
        print(f"  Shape: {repr_data.shape}")

    if args.format == "tflite":
        export_to_tflite(args.model, args.output or _out_path(args.model, ".tflite"),
                         quantize=args.quantize, repr_data=repr_data)
    else:
        export_to_onnx(args.model, args.output or _out_path(args.model, ".onnx"),
                       opset=args.opset)


def cmd_info(args: argparse.Namespace) -> None:
    """Show model summary and metadata."""
    from gbagent.inference import InferenceEngine

    engine = InferenceEngine(args.model, backend=args.backend)
    info = engine.summary()

    print()
    print("╔════════════════════════════════════════════╗")
    print("║           Model Summary                    ║")
    print("╚════════════════════════════════════════════╝")
    for key, value in info.items():
        print(f"  {key}: {value}")
    print()


def _out_path(model_path: str, ext: str) -> str:
    """Derive default output path."""
    p = Path(model_path)
    stem = p.stem
    while stem.endswith(".keras"):
        stem = stem.rsplit(".keras", 1)[0]
    return str(Path("export") / f"{stem}{ext}")


# ===================================================================
# Main
# ===================================================================


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "export":
        cmd_export(args)
    elif args.command == "info":
        cmd_info(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
