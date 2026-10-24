"""ONNX export parity, quantisation and the common runtime."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from pulse.models.cnn import PulseCNN, predict_logits
from pulse.models.export import (
    INPUT_NAME,
    WindowCalibrationReader,
    export_onnx,
    preprocess_for_quantisation,
    quantize_dynamic_int8,
    quantize_static_int8,
)
from pulse.models.runtime import ModelKind, OnnxPredictor, TorchPredictor, load_predictor, softmax
from pulse.signals.dataset import Dataset


@pytest.fixture(scope="module")
def quantised(tiny_onnx: Path, small_train: Dataset, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    out = tmp_path_factory.mktemp("quant")
    pre = out / "pre.onnx"
    preprocess_for_quantisation(tiny_onnx, pre)
    quantize_dynamic_int8(pre, out / "dyn.onnx")
    quantize_static_int8(pre, out / "static.onnx", small_train.x[:64])
    return {"fp32": tiny_onnx, "pre": pre, "dynamic": out / "dyn.onnx", "static": out / "static.onnx"}


def test_exported_graph_is_valid_with_dynamic_batch(tiny_onnx: Path) -> None:
    model = onnx.load(str(tiny_onnx))
    onnx.checker.check_model(model)
    assert model.graph.input[0].name == INPUT_NAME
    assert model.graph.input[0].type.tensor_type.shape.dim[0].dim_param == "batch"
    assert "BatchNormalization" not in {n.op_type for n in model.graph.node}


def test_onnx_matches_torch_logits(tiny_model: PulseCNN, tiny_onnx: Path, small_test: Dataset) -> None:
    ref = predict_logits(tiny_model, small_test.x[:16])
    got = OnnxPredictor(tiny_onnx).predict_logits(small_test.x[:16])
    np.testing.assert_allclose(got, ref, atol=1e-4)


def test_quantised_models_are_much_smaller(quantised: dict[str, Path]) -> None:
    fp32 = quantised["fp32"].stat().st_size
    assert quantised["dynamic"].stat().st_size < 0.4 * fp32
    assert quantised["static"].stat().st_size < 0.4 * fp32


def test_static_graph_uses_integer_kernels(quantised: dict[str, Path]) -> None:
    ops = {n.op_type for n in onnx.load(str(quantised["static"])).graph.node}
    assert "QLinearConv" in ops and "Conv" not in ops


@pytest.mark.parametrize("kind", ["dynamic", "static"])
def test_quantised_predictions_track_fp32(quantised: dict[str, Path], small_test: Dataset, kind: str) -> None:
    ref = OnnxPredictor(quantised["fp32"]).predict_logits(small_test.x)
    got = OnnxPredictor(quantised[kind]).predict_logits(small_test.x)
    agreement = float((ref.argmax(1) == got.argmax(1)).mean())
    correlation = float(np.corrcoef(ref.ravel(), got.ravel())[0, 1])
    assert agreement >= 0.75
    assert correlation > 0.95


def test_calibration_reader_yields_fixed_batches_then_none(small_train: Dataset) -> None:
    reader = WindowCalibrationReader(small_train.x[:70], batch_size=32)
    sizes = []
    while (batch := reader.get_next()) is not None:
        assert set(batch) == {INPUT_NAME}
        sizes.append(batch[INPUT_NAME].shape[0])
    assert sizes == [32, 32, 6]


def test_onnx_predictor_from_bytes_matches_path(tiny_onnx: Path, small_test: Dataset) -> None:
    from_path = OnnxPredictor(tiny_onnx).predict_proba(small_test.x[:4], small_test.rpm[:4])
    from_bytes = OnnxPredictor(tiny_onnx.read_bytes()).predict_proba(small_test.x[:4], small_test.rpm[:4])
    np.testing.assert_array_equal(from_path, from_bytes)


def test_torch_predictor_matches_direct_model(tiny_model: PulseCNN, small_test: Dataset, tmp_path: Path) -> None:
    from pulse.models.cnn import save_cnn

    save_cnn(tiny_model, tmp_path / "m.pt")
    predictor = TorchPredictor(tmp_path / "m.pt")
    np.testing.assert_allclose(
        predictor.predict_proba(small_test.x[:3], small_test.rpm[:3]),
        softmax(predict_logits(tiny_model, small_test.x[:3])),
        atol=1e-6,
    )


def test_load_predictor_dispatches_on_kind(tiny_onnx: Path) -> None:
    predictor = load_predictor(ModelKind.ONNX, tiny_onnx, name="x")
    assert isinstance(predictor, OnnxPredictor) and predictor.name == "x"


@settings(max_examples=30, deadline=None)
@given(hnp.arrays(np.float32, (3, 6), elements=st.floats(-50, 50, width=32)))
def test_softmax_rows_are_distributions(logits: np.ndarray) -> None:
    probs = softmax(logits)
    assert probs.shape == logits.shape
    assert np.all(probs >= 0.0)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_export_overwrites_and_returns_size(tiny_model: PulseCNN, tmp_path: Path) -> None:
    size = export_onnx(tiny_model, tmp_path / "x.onnx", 2048)
    assert size == (tmp_path / "x.onnx").stat().st_size > 100_000
