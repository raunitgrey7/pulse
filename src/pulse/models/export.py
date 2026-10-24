"""ONNX export and int8 quantisation with onnxruntime.

The TorchScript-based exporter (``dynamo=False``) is used deliberately: in
this environment (torch 2.14, onnxruntime 1.31) the dynamo exporter emits a
graph whose Gemm weight annotation fails onnxruntime's shape inference
inside the quantiser, while the legacy exporter folds BatchNorm into the
convolutions and quantises cleanly. Two int8 recipes are produced because
they trade differently: dynamic quantisation needs no calibration data and
computes activation scales at runtime; static quantisation calibrates
activation ranges once on training windows and runs integer kernels
(QLinearConv, QGemm), which is what an edge CPU benefits from.

Static quantisation is restricted to the convolution, linear and pooling
operators. The first attempt quantised the whole graph, including the
in-graph standardisation, and accuracy fell from 96 percent to 61 percent:
the squared-amplitude node spans several orders of magnitude across
windows, so an 8-bit MinMax range crushed quiet windows to zero. Keeping
the few scalar normalisation ops in float costs nothing measurable and
restores parity with the float model.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import onnx
import torch
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from pulse.models.cnn import PulseCNN

OPSET = 17
INPUT_NAME = "window"
OUTPUT_NAME = "logits"
STATIC_QUANT_OPS: tuple[str, ...] = ("Conv", "Gemm", "MaxPool")


def export_onnx(model: PulseCNN, path: Path, n_samples: int) -> int:
    """Export with a dynamic batch axis, validate the graph and return its size in bytes."""
    model.eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, n_samples, dtype=torch.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            model,
            (dummy,),
            str(path),
            input_names=[INPUT_NAME],
            output_names=[OUTPUT_NAME],
            dynamic_axes={INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}},
            opset_version=OPSET,
            dynamo=False,
            do_constant_folding=True,
        )
    onnx.checker.check_model(onnx.load(str(path)))
    return path.stat().st_size


class WindowCalibrationReader(CalibrationDataReader):
    """Feeds calibration windows to the static quantiser in fixed-size batches."""

    def __init__(self, windows: np.ndarray, batch_size: int = 32) -> None:
        self._batches = iter(
            np.ascontiguousarray(windows[i : i + batch_size], dtype=np.float32)
            for i in range(0, windows.shape[0], batch_size)
        )

    def get_next(self) -> dict[str, np.ndarray] | None:
        batch = next(self._batches, None)
        return None if batch is None else {INPUT_NAME: batch}


def _quiet_quantizer() -> None:
    """The quantiser logs advice about pre-processing that we already apply."""
    logging.getLogger("root").setLevel(logging.ERROR)
    logging.getLogger("onnxruntime").setLevel(logging.ERROR)


def preprocess_for_quantisation(src: Path, dst: Path) -> int:
    """Run onnxruntime's recommended shape inference and graph optimisation pass."""
    quant_pre_process(str(src), str(dst), skip_symbolic_shape=True)
    return dst.stat().st_size


def quantize_dynamic_int8(src: Path, dst: Path) -> int:
    """Int8 weights, activation scales computed per inference."""
    _quiet_quantizer()
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8, per_channel=True)
    onnx.checker.check_model(onnx.load(str(dst)))
    return dst.stat().st_size


def quantize_static_int8(
    src: Path,
    dst: Path,
    calibration_windows: np.ndarray,
    method: CalibrationMethod = CalibrationMethod.MinMax,
) -> int:
    """Int8 weights and activations with ranges calibrated on real windows."""
    _quiet_quantizer()
    quantize_static(
        str(src),
        str(dst),
        WindowCalibrationReader(calibration_windows),
        quant_format=QuantFormat.QOperator,
        op_types_to_quantize=list(STATIC_QUANT_OPS),
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=method,
    )
    onnx.checker.check_model(onnx.load(str(dst)))
    return dst.stat().st_size
