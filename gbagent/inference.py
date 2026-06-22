"""Headless inference — load model and run without training or dashboard.

Supports three backends:

* **Keras (TF)** — loads the full ``.keras`` model file.
* **TFLite** — loads a FP32/FP16/INT8 quantised ``.tflite`` flatbuffer.
* **ONNX** — loads an ``.onnx`` model via ONNX Runtime (optional dep).

Usage
-----
    from gbagent.inference import InferenceEngine

    engine = InferenceEngine("checkpoints/model.keras", backend="keras")
    # or engine = InferenceEngine("export/model.tflite", backend="tflite")
    # or engine = InferenceEngine("export/model.onnx", backend="onnx")

    dpad_logits, btn_logits, value = engine.predict(obs)

    # Rollout helper — runs in environment without training
    engine.rollout(env, n_steps=1000, render=True)
"""

from __future__ import annotations

import contextlib
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from gbagent.action import combine_actions, sample_actions

logger = logging.getLogger("gbagent.inference")


# ---------------------------------------------------------------------------
# Inference Engine
# ---------------------------------------------------------------------------


class InferenceEngine:
    """Load a trained model and run inference in a headless environment.

    Parameters
    ----------
    model_path : str | Path
        Path to the model file (``.keras``, ``.tflite``, or ``.onnx``).
    backend : str
        One of ``"keras"``, ``"tflite"``, or ``"onnx"``.
        If omitted, inferred from the file extension.
    """

    def __init__(self, model_path: str | Path, backend: str | None = None,
                 btn_size: int = 6):
        self._model_path = Path(model_path)
        self._btn_size = btn_size  # 6 (GB) or 8 (GBA)

        if backend is None:
            backend = self._infer_backend()

        self._backend = backend.lower()
        self._interpreter = None  # TFLite interpreter
        self._session = None  # ONNX Runtime session
        self._model = None  # Keras model
        self._input_details = None
        self._output_details = None

        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        obs: np.ndarray,  # (B, 84, 84, 4)
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run inference on a batch of observations.

        Parameters
        ----------
        obs : np.ndarray
            Shape ``(B, 84, 84, 4)``, float32 in ``[0, 1]``.

        Returns
        -------
        dpad_logits : (B, 5) float32
        btn_logits  : (B, N) float32 where N=6 (GB) or 8 (GBA)
        value       : (B, 1) float32
        """
        # Ensure batch dimension
        if obs.ndim == 3:
            obs = obs[np.newaxis, ...]

        if self._backend == "keras":
            return self._predict_keras(obs)
        elif self._backend == "tflite":
            return self._predict_tflite(obs)
        elif self._backend == "onnx":
            return self._predict_onnx(obs)
        else:
            raise ValueError(f"Unknown backend: {self._backend}")

    def predict_single(
        self,
        obs: np.ndarray,  # (84, 84, 4)
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run inference on a single observation.

        Returns
        -------
        dpad_logits : (5,) float32
        btn_logits  : (N,) float32 where N=6 (GB) or 8 (GBA)
        value       : (1,) float32
        """
        dp, bp, v = self.predict(obs[np.newaxis, ...])
        return dp[0], bp[0], v[0]

    def rollout(
        self,
        env,
        n_steps: int = 1000,
        render: bool = False,
        deterministic: bool = False,
        callback=None,
    ) -> dict[str, Any]:
        """Run a headless rollout in the given environment.

        Parameters
        ----------
        env : GBAGEnv
            An environment instance (already reset).
        n_steps : int
            Maximum number of steps.
        render : bool
            If True, call ``env.render()`` each step (may slow things).
        deterministic : bool
            If True, use argmax action selection (greedy).
        callback : callable, optional
            Called as ``callback(step, obs, dpad_logits, btn_logits, value,
            dpad_action, btn_action, env_action, reward, done)``.
            Return ``False`` to stop early.

        Returns
        -------
        dict with keys ``steps``, ``total_reward``, ``mean_value``,
        ``episode_length``, ``duration_s``, ``fps``.
        """
        obs, _ = env.reset()
        total_reward = 0.0
        values: list[float] = []
        start_time = time.time()

        for step in range(n_steps):
            dpad_logits, btn_logits, val = self.predict_single(obs)
            dpad_action, btn_action, _ = sample_actions(
                dpad_logits[np.newaxis, ...],
                btn_logits[np.newaxis, ...],
                training=not deterministic,
            )
            env_action = combine_actions(
                dpad_action, btn_action, btn_size=self._btn_size
            )[0]

            next_obs, reward, terminated, truncated, _ = env.step(int(env_action))
            total_reward += reward
            values.append(float(val[0]))
            done = terminated or truncated

            if callback is not None:
                should_continue = callback(
                    step, obs, dpad_logits, btn_logits, val,
                    int(dpad_action[0]), int(btn_action[0]), int(env_action),
                    reward, done,
                )
                if should_continue is False:
                    break

            if render:
                with contextlib.suppress(Exception):
                    env.render()

            obs = next_obs

            if done:
                break

        duration = time.time() - start_time
        return {
            "steps": step + 1,
            "total_reward": total_reward,
            "mean_value": float(np.mean(values)) if values else 0.0,
            "episode_length": step + 1,
            "duration_s": round(duration, 3),
            "fps": round((step + 1) / duration, 1) if duration > 0 else 0,
        }

    # ------------------------------------------------------------------
    # Backend-specific prediction
    # ------------------------------------------------------------------

    def _predict_keras(self, obs: np.ndarray):
        dpad_logits, btn_logits, value, _ = self._model(obs, training=False)
        return (
            dpad_logits.numpy(),
            btn_logits.numpy(),
            value.numpy(),
        )

    def _predict_tflite(self, obs: np.ndarray):
        interp = self._interpreter
        input_dtype = self._input_details[0]["dtype"]

        # Quantized models expect uint8 input; cast and scale
        if input_dtype == np.uint8:
            # The model was trained with [0, 1] float;
            # TFLite INT8 expects [0, 255] uint8.
            obs_quant = (obs * 255.0).clip(0, 255).astype(np.uint8)
        else:
            obs_quant = obs.astype(np.float32)

        interp.set_tensor(self._input_details[0]["index"], obs_quant)
        interp.invoke()

        # TFLite output tensor order may differ from Keras model output order;
        # match by the last dimension of each output tensor.
        outputs = {}
        for detail in self._output_details:
            out = interp.get_tensor(detail["index"])
            last_dim = detail["shape"][-1]
            if last_dim == 5:
                outputs["dpad"] = out
            elif last_dim in (6, 8):  # btn_size: 6 (GB) or 8 (GBA)
                outputs["btn"] = out
            elif last_dim == 1:
                outputs["value"] = out

        dpad_out = outputs.get("dpad")
        btn_out = outputs.get("btn")
        value_out = outputs.get("value")

        # Convert uint8 outputs back to float logits if needed
        if dpad_out is not None and dpad_out.dtype == np.uint8:
            dpad_out = dpad_out.astype(np.float32)
        if btn_out is not None and btn_out.dtype == np.uint8:
            btn_out = btn_out.astype(np.float32)
        if value_out is not None and value_out.dtype == np.uint8:
            value_out = value_out.astype(np.float32)

        return dpad_out, btn_out, value_out

    def _predict_onnx(self, obs: np.ndarray):
        input_name = self._session.get_inputs()[0].name
        output_names = [o.name for o in self._session.get_outputs()]
        outputs = self._session.run(output_names, {input_name: obs.astype(np.float32)})
        # Model outputs: dpad_logits, btn_logits, value
        return tuple(outputs)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _infer_backend(self) -> str:
        ext = self._model_path.suffix.lower()
        if ext == ".tflite":
            return "tflite"
        elif ext == ".onnx":
            return "onnx"
        elif ext == ".keras" or ext == ".h5":
            return "keras"
        else:
            raise ValueError(
                f"Cannot infer backend from extension '{ext}'. "
                f"Supported: .keras, .h5, .tflite, .onnx. "
                f"Set 'backend' explicitly."
            )

    def _load(self) -> None:
        path = self._model_path
        if not path.is_file():
            raise FileNotFoundError(f"Model not found: {path}")

        logger.info("Loading model (%s) via backend=%s", path, self._backend)

        if self._backend == "keras":
            self._load_keras(path)
        elif self._backend == "tflite":
            self._load_tflite(path)
        elif self._backend == "onnx":
            self._load_onnx(path)
        else:
            raise ValueError(f"Unknown backend: {self._backend}")

    def _load_keras(self, path: Path) -> None:
        import tensorflow as tf

        self._model = tf.keras.models.load_model(str(path))

        # Keras 3 models may not expose input_shape until explicitly built;
        # handle gracefully.
        try:
            in_shape = self._model.input_shape
        except (AttributeError, NotImplementedError):
            in_shape = "(built at inference time)"
        try:
            out_shapes = [o.shape for o in self._model.outputs]
        except (AttributeError, NotImplementedError):
            out_shapes = "(built at inference time)"

        logger.info(
            "  Keras model loaded — %s params, input %s, output %s",
            f"{self._model.count_params():,}",
            in_shape,
            out_shapes,
        )

    def _load_tflite(self, path: Path) -> None:
        import tensorflow as tf

        self._interpreter = tf.lite.Interpreter(model_path=str(path))
        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()

        logger.info(
            "  TFLite model loaded — inputs: %s, outputs: %s",
            [d["shape"] for d in self._input_details],
            [d["shape"] for d in self._output_details],
        )

    def _load_onnx(self, path: Path) -> None:
        try:
            import onnxruntime as ort
        except ImportError as err:
            raise ImportError(
                "onnxruntime is required for ONNX inference. Install with:\n"
                "    pip install onnxruntime"
            ) from err

        self._session = ort.InferenceSession(str(path))
        logger.info(
            "  ONNX model loaded — inputs: %s, outputs: %s",
            [(i.name, i.shape) for i in self._session.get_inputs()],
            [(o.name, o.shape) for o in self._session.get_outputs()],
        )

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    @property
    def backend(self) -> str:
        return self._backend

    def summary(self) -> dict[str, Any]:
        """Return a summary dict with model metadata."""
        info: dict[str, Any] = {
            "model_path": str(self._model_path),
            "backend": self._backend,
        }
        if self._model is not None:
            info["params"] = self._model.count_params()
            try:
                info["input_shape"] = self._model.input_shape
            except (AttributeError, NotImplementedError):
                info["input_shape"] = "(N, 84, 84, 4)"
        if self._input_details:
            info["tflite_inputs"] = [
                {"shape": d["shape"], "dtype": str(d["dtype"])}
                for d in self._input_details
            ]
            info["tflite_outputs"] = [
                {"shape": d["shape"], "dtype": str(d["dtype"])}
                for d in self._output_details
            ]
        if self._session is not None:
            info["onnx_inputs"] = [
                {"name": i.name, "shape": i.shape} for i in self._session.get_inputs()
            ]
            info["onnx_outputs"] = [
                {"name": o.name, "shape": o.shape} for o in self._session.get_outputs()
            ]
        return info
