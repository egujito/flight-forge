import numpy as np
import pytest

from flightForge.utils import _func_from_csv, _load_curve, _unit_norm


# --- _unit_norm ---

def test_unit_norm_standard_vector():
    v = np.array([3.0, 4.0, 0.0])
    u = _unit_norm(v)
    assert np.linalg.norm(u) == pytest.approx(1.0)
    assert u[0] == pytest.approx(0.6)
    assert u[1] == pytest.approx(0.8)


def test_unit_norm_zero_vector_unchanged():
    v = np.zeros(3)
    u = _unit_norm(v)
    assert np.array_equal(u, v)


def test_unit_norm_already_unit_unchanged():
    v = np.array([1.0, 0.0, 0.0])
    u = _unit_norm(v)
    assert u[0] == pytest.approx(1.0)
    assert u[1] == pytest.approx(0.0)
    assert u[2] == pytest.approx(0.0)


def test_unit_norm_2d_vector():
    v = np.array([1.0, 1.0])
    u = _unit_norm(v)
    assert np.linalg.norm(u) == pytest.approx(1.0)


def test_unit_norm_negative_components():
    v = np.array([-1.0, 0.0, 0.0])
    u = _unit_norm(v)
    assert np.linalg.norm(u) == pytest.approx(1.0)
    assert u[0] == pytest.approx(-1.0)


# --- _func_from_csv ---

def test_func_from_csv_basic_interpolation(tmp_path):
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("0.0,0.0\n1.0,10.0\n2.0,20.0\n")
    interp, x, y = _func_from_csv(str(csv_file))
    assert x == pytest.approx([0.0, 1.0, 2.0])
    assert y == pytest.approx([0.0, 10.0, 20.0])
    assert float(interp(1.0)) == pytest.approx(10.0)
    assert float(interp(0.5)) == pytest.approx(5.0)


def test_func_from_csv_skips_malformed_lines(tmp_path):
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("# header\n0.0,0.0\nbad line\n1.0,10.0\n")
    interp, x, y = _func_from_csv(str(csv_file))
    assert len(x) == 2
    assert float(interp(0.5)) == pytest.approx(5.0)


def test_func_from_csv_skips_header_row(tmp_path):
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("time,thrust\n0.0,500.0\n1.0,1000.0\n")
    _interp, x, y = _func_from_csv(str(csv_file))
    assert len(x) == 2
    assert x[0] == pytest.approx(0.0)
    assert y[0] == pytest.approx(500.0)


# --- _load_curve ---

def test_load_curve_callable_passthrough():
    fn = lambda t: t * 2.0
    loaded = _load_curve(fn)
    assert loaded is fn


def test_load_curve_from_csv_returns_callable(tmp_path):
    csv_file = tmp_path / "thrust.csv"
    csv_file.write_text("0.0,1000.0\n5.0,500.0\n")
    loaded = _load_curve(str(csv_file))
    assert callable(loaded)
    assert float(loaded(0.0)) == pytest.approx(1000.0)
    assert float(loaded(5.0)) == pytest.approx(500.0)
