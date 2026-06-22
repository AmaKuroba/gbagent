"""Model export — TFLite (optionally INT8 quantized) and ONNX.

Usage
-----
    # From Python:
    from gbagent.export import export_to_tflite, export_to_onnx

    export_to_tflite("checkpoints/model.keras", "export/model.tflite")
    export_to_tflite("checkpoints/model.keras", "export/model_int8.tflite",
                     quantize="int8", repr_data=repr_data)

    export_to_onnx("checkpoints/model.keras", "export/model.onnx")

    # From CLI:
    python -m gbagent.export checkpoint.keras export/model.tflite
    python -m gbagent.export checkpoint.keras export/model.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Lazy-imported helpers
# ---------------------------------------------------------------------------

_IMPORT_MISSING: list[str] = []


def _import_tf():
    """Import TensorFlow, with a friendly message on failure."""
    try:
        import tensorflow as tf

        return tf
    except ImportError:
        _IMPORT_MISSING.append("tensorflow")
        return None


# ---------------------------------------------------------------------------
# TFLite export
# ---------------------------------------------------------------------------


def export_to_tflite(
    keras_path: str | Path,
    output_path: str | Path,
    quantize: str | None = None,
    repr_data: np.ndarray | None = None,
) -> Path:
    """Convert a saved ``.keras`` model to a TFLite flatbuffer.

    Parameters
    ----------
    keras_path : str | Path
        Path to a ``.keras`` model saved by ``model.save()`` (Keras v3 format).
    output_path : str | Path
        Destination for the ``.tflite`` file.
    quantize : str | None
        Quantization mode:

        * ``None`` (default) — FP32, no quantization.
        * ``"fp16"`` — FP16 quantization (weights only).
        * ``"int8"`` — Full INT8 quantization (weights + activations).
          Requires *repr_data* for calibration.
    repr_data : np.ndarray | None
        Representative dataset for INT8 calibration.
        Shape ``(N, 84, 84, 4)`` — a batch of preprocessed observations.
        Required when ``quantize="int8"``; ignored otherwise.

    Returns
    -------
    Path
        Absolute path to the written ``.tflite`` file.

    Raises
    ------
    FileNotFoundError
        If the ``.keras`` file does not exist.
    ValueError
        If *quantize* is ``"int8"`` but no *repr_data* is provided.
    """
    tf = _import_tf()

    keras_path = Path(keras_path)
    output_path = Path(output_path).resolve()

    if not keras_path.is_file():
        raise FileNotFoundError(f"Keras model not found: {keras_path}")

    # Load the Keras model
    print(f"› Loading model from {keras_path} …")
    inner = tf.keras.models.load_model(str(keras_path))
    print(f"  ✓ Loaded ({inner.count_params():,} params)")

    # Wrap into a Functional model so TFLite converter can trace the graph.
    # Subclassed Keras models (like ActorCritic) don't expose input_shape
    # in a way TFLite requires; the explicit Input layer fixes this.
    inp = tf.keras.layers.Input(shape=(84, 84, 4), batch_size=1)
    out = inner(inp, training=False)
    model = tf.keras.models.Model(inputs=inp, outputs=out, name=inner.name + "_export")

    # ── Convert ──────────────────────────────────────────────────────
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantize is None:
        # FP32 — default
        converter.optimizations = []
        print("  Mode: FP32")

    elif quantize == "fp16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        print("  Mode: FP16 (weight-only quantization)")

    elif quantize == "int8":
        if repr_data is None:
            raise ValueError(
                "INT8 quantization requires a representative dataset (repr_data). "
                "Provide a NumPy array of shape (N, 84, 84, 4) sampled from "
                "actual observations."
            )
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = _make_repr_gen(repr_data)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        print("  Mode: INT8 (full integer quantization)")

    else:
        raise ValueError(f"Unknown quantization mode: {quantize!r}")

    # Convert
    print("  Converting …")
    tflite_model = converter.convert()

    # Write
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f"  ✓ TFLite exported → {output_path}  ({size_kb:.1f} KB)")

    return output_path


def _make_repr_gen(repr_data: np.ndarray):
    """Return a representative dataset generator for INT8 calibration.

    Wraps the provided observations into a generator that yields one
    batch at a time, as required by ``TFLiteConverter.representative_dataset``.
    """
    import tensorflow as tf

    dataset = tf.data.Dataset.from_tensor_slices(repr_data).batch(1)

    def gen():
        for batch in dataset:
            yield [batch]

    return gen


# ---------------------------------------------------------------------------
# ONNX export  (via tf2onnx)
# ---------------------------------------------------------------------------


def export_to_onnx(
    keras_path: str | Path,
    output_path: str | Path,
    opset: int = 17,
) -> Path:
    """Convert a saved ``.keras`` model to ONNX format.

    Requires the ``tf2onnx`` package::

        pip install tf2onnx

    Parameters
    ----------
    keras_path : str | Path
        Path to a ``.keras`` model.
    output_path : str | Path
        Destination for the ``.onnx`` file.
    opset : int
        ONNX opset version (default 17 — good balance of operator support).

    Returns
    -------
    Path
        Absolute path to the written ``.onnx`` file.

    Raises
    ------
    ImportError
        If ``tf2onnx`` is not installed.
    FileNotFoundError
        If the ``.keras`` file does not exist.
    """
    tf = _import_tf()

    try:
        import tf2onnx  # ty: ignore[unresolved-import]
    except ImportError as err:
        raise ImportError(
            "tf2onnx is required for ONNX export. Install it with:\n"
            "    pip install tf2onnx"
        ) from err

    keras_path = Path(keras_path)
    output_path = Path(output_path).resolve()

    if not keras_path.is_file():
        raise FileNotFoundError(f"Keras model not found: {keras_path}")

    print(f"› Loading model from {keras_path} …")
    model = tf.keras.models.load_model(str(keras_path))

    # Wrap model in a SignatureDef so tf2onnx knows input/output names
    # The model takes a single tensor input (obs) and returns 3 tensors.
    spec = (tf.TensorSpec((None, 84, 84, 4), tf.float32, name="obs"),)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("  Converting …")
    model_proto, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=spec,
        opset=opset,
        output_path=str(output_path),
    )
    size_kb = output_path.stat().st_size / 1024
    print(f"  ✓ ONNX exported → {output_path}  ({size_kb:.1f} KB)")

    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _infer_output_path(keras_path: Path, fmt: str) -> Path:
    """Derive a default output path from the input name + format."""
    stem = keras_path.stem
    # If the stem still ends with .keras (Keras v3), strip it
    while stem.endswith(".keras"):
        stem = stem.rsplit(".keras", 1)[0]
    ext = ".tflite" if fmt == "tflite" else ".onnx"
    return Path("export") / f"{stem}{ext}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export gbagent model to TFLite or ONNX",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("model", type=str, help="Path to .keras model file")
    parser.add_argument("output", type=str, nargs="?", default=None,
                        help="Output path (default: export/<model_stem>.tflite/.onnx)")
    parser.add_argument("--format", "-f", choices=["tflite", "onnx"], default="tflite",
                        help="Export format")
    parser.add_argument("--quantize", "-q", choices=["fp16", "int8"], default=None,
                        help="Quantization mode (TFLite only)")
    parser.add_argument("--repr-data", type=str, default=None,
                        help="Path to .npy file with representative data (INT8 quant)")
    parser.add_argument("--opset", type=int, default=17,
                        help="ONNX opset version (ONNX only)")

    args = parser.parse_args()
    keras_path = Path(args.model)

    # Default output path
    output_path = args.output
    if output_path is None:
        output_path = _infer_output_path(keras_path, args.format)

    # Load repr data if requested
    repr_data = None
    if args.repr_data:
        repr_data = np.load(args.repr_data)

    if args.format == "tflite":
        export_to_tflite(keras_path, output_path, quantize=args.quantize, repr_data=repr_data)
    else:
        export_to_onnx(keras_path, output_path, opset=args.opset)


if __name__ == "__main__":
    main()
