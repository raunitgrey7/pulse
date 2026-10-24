"""Bearing kinematics: published orders, identities and validation."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pulse.signals.bearing import BearingGeometry, fault_frequencies

geometries = st.builds(
    BearingGeometry,
    n_balls=st.integers(min_value=3, max_value=30),
    ball_diameter_mm=st.floats(min_value=1.0, max_value=20.0),
    pitch_diameter_mm=st.floats(min_value=25.0, max_value=200.0),
    contact_angle_deg=st.floats(min_value=0.0, max_value=45.0),
)
speeds = st.floats(min_value=1.0, max_value=500.0)


def test_skf_6205_orders_match_published_values() -> None:
    orders = fault_frequencies(BearingGeometry(), 1.0)
    assert orders.bpfo == pytest.approx(3.5848, abs=1e-3)
    assert orders.bpfi == pytest.approx(5.4152, abs=1e-3)
    assert orders.bsf == pytest.approx(2.3568, abs=1e-3)
    assert orders.ftf == pytest.approx(0.3983, abs=1e-3)


@settings(max_examples=60, deadline=None)
@given(geometries, speeds)
def test_bpfo_plus_bpfi_equals_ball_count_times_shaft(geometry: BearingGeometry, shaft_hz: float) -> None:
    f = fault_frequencies(geometry, shaft_hz)
    assert f.bpfo + f.bpfi == pytest.approx(geometry.n_balls * shaft_hz, rel=1e-9)


@settings(max_examples=60, deadline=None)
@given(geometries, speeds)
def test_frequency_ordering_holds(geometry: BearingGeometry, shaft_hz: float) -> None:
    f = fault_frequencies(geometry, shaft_hz)
    assert f.bpfi > f.bpfo > 0
    assert 0 < f.ftf < shaft_hz / 2
    assert f.bsf > 0


def test_as_orders_divides_by_shaft_frequency() -> None:
    f = fault_frequencies(BearingGeometry(), 25.0)
    orders = f.as_orders()
    assert orders.shaft_hz == 1.0
    assert orders.bpfo == pytest.approx(f.bpfo / 25.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_balls": 1},
        {"ball_diameter_mm": 0.0},
        {"ball_diameter_mm": 50.0, "pitch_diameter_mm": 40.0},
        {"contact_angle_deg": 90.0},
        {"contact_angle_deg": -1.0},
    ],
)
def test_invalid_geometry_is_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        BearingGeometry(**kwargs)


def test_non_positive_speed_is_rejected() -> None:
    with pytest.raises(ValueError):
        fault_frequencies(BearingGeometry(), 0.0)
