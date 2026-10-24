"""One predict interface over every model artefact.

The benchmark, the device simulator and the API all need "windows in,
class probabilities out" without caring whether the artefact is a pickled
feature model, a PyTorch state dict or an ONNX graph. Keeping the three
adapters here also keeps heavy imports local: an ONNX device never imports
torch, which is exactly the memory saving the benchmark measures.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Protocol

import numpy as np


class ModelKind(StrEnum):
    """Artefact formats a predictor can be built from."""

    BASELINE = "baseline"
    TORCH = "torch"
    ONNX = "onnx"


class Predictor(Protocol):
    """Anything that scores windows."""

    name: str

    def predict_proba(self, x: np.ndarray, rpm: np.ndarray) -> np.ndarray:
        """Probabilities of shape (batch, n_classes) for windows of shape (batch, n_samples)."""
        ...


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)


class BaselinePredictor:
    """Feature extraction plus boosted trees; the only predictor that uses the speed.

    Thread pools are capped explicitly: scikit-learn opens an OpenMP region
    per tree during prediction, and on a many-core host an unpinned pool
    turns a single-window prediction into seconds of spin-waiting.
    """

    def __init__(self, path: Path, threads: int = 1) -> None:
        from threadpoolctl import threadpool_limits

        from pulse.models.baseline import FeatureBaseline

        self.name = "baseline_features"
        self._limits = threadpool_limits(limits=threads)
        self._model = FeatureBaseline.load(path)

    def predict_proba(self, x: np.ndarray, rpm: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(x, rpm)


class TorchPredictor:
    """The float PyTorch model, run with a fixed thread count."""

    def __init__(self, path: Path, threads: int = 1) -> None:
        import torch

        from pulse.models.cnn import load_cnn

        torch.set_num_threads(threads)
        self.name = "cnn_torch_fp32"
        self._model = load_cnn(path)

    def predict_logits(self, x: np.ndarray) -> np.ndarray:
        from pulse.models.cnn import predict_logits

        return predict_logits(self._model, x)

    def predict_proba(self, x: np.ndarray, rpm: np.ndarray) -> np.ndarray:  # noqa: ARG002 (speed is not an input)
        return softmax(self.predict_logits(x))


class OnnxPredictor:
    """Any exported or quantised graph, run on the onnxruntime CPU provider."""

    def __init__(self, source: Path | bytes, threads: int = 1, name: str = "cnn_onnx") -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        self.name = name
        model = source if isinstance(source, bytes) else str(source)
        self._session = ort.InferenceSession(model, options, providers=["CPUExecutionProvider"])
        self._input = self._session.get_inputs()[0].name

    def predict_logits(self, x: np.ndarray) -> np.ndarray:
        batch = np.ascontiguousarray(x, dtype=np.float32)
        return self._session.run(None, {self._input: batch})[0]

    def predict_proba(self, x: np.ndarray, rpm: np.ndarray) -> np.ndarray:  # noqa: ARG002 (speed is not an input)
        return softmax(self.predict_logits(x))


def load_predictor(kind: ModelKind, path: Path, threads: int = 1, name: str | None = None) -> Predictor:
    """Build the right adapter for an artefact on disk."""
    if kind is ModelKind.BASELINE:
        return BaselinePredictor(path, threads)
    if kind is ModelKind.TORCH:
        return TorchPredictor(path, threads)
    return OnnxPredictor(path, threads, name or "cnn_onnx")
