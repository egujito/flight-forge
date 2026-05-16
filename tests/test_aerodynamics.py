import math

import pytest

from flightForge import Motor, Parachute, Rocket
from flightForge.utils import logarithmic_thrust


@pytest.fixture
def rocket():
    thrust_fn = logarithmic_thrust(5.0, 2000.0, 0.5)
    motor = Motor(thrust_fn, burn_time=5.0, initial_grain_mass=2.0)
    r = Rocket(10.0, lambda _: 0.5, 0.1)
    r.add_motor(motor)
    return r


def _ref_area(dim: float = 0.1) -> float:
    return (dim / 2) ** 2 * math.pi


# --- Baseline (no parachutes) ---

def test_e_cd_no_parachutes_returns_drag_source(rocket):
    cd = rocket._e_cd(0.5, {}, 1000.0, 10.0)
    assert cd == pytest.approx(0.5)


def test_e_cd_no_parachutes_any_mach(rocket):
    for mach in [0.0, 0.5, 1.0, 2.0]:
        assert rocket._e_cd(mach, {}, 500.0, 5.0) == pytest.approx(0.5)


def test_e_cd_returns_float(rocket):
    assert isinstance(rocket._e_cd(0.5, {}, 1000.0, 1.0), float)


# --- Parachute not triggered ---

def test_e_cd_parachute_signal_t_none_no_contribution(rocket):
    chute = Parachute("drogue", cd_s=2.0, lag=1.0, trigger="apogee")
    rocket.add_parachute(chute)
    # signal_t is None by default — must not add drag
    cd = rocket._e_cd(0.2, {}, 1000.0, 10.0)
    assert cd == pytest.approx(0.5)


# --- Parachute at exact signal time (not yet contributing) ---

def test_e_cd_at_exact_signal_t_no_drag(rocket):
    chute = Parachute("drogue", cd_s=2.0, lag=2.0, trigger="apogee")
    chute.signal_t = 5.0
    rocket.add_parachute(chute)
    # condition is t > signal_t (strict), so at t=signal_t there's no contribution
    cd = rocket._e_cd(0.0, {}, 500.0, 5.0)
    assert cd == pytest.approx(0.5)


# --- Parachute ramp ---

def test_e_cd_parachute_half_open(rocket):
    chute = Parachute("drogue", cd_s=2.0, lag=2.0, trigger="apogee")
    chute.signal_t = 5.0
    rocket.add_parachute(chute)
    # t=6.0: tau = (6-5)/2 = 0.5
    cd = rocket._e_cd(0.0, {}, 500.0, 6.0)
    ref = _ref_area()
    expected = (0.5 * ref + 0.5 * 2.0) / ref
    assert cd == pytest.approx(expected, rel=1e-6)


def test_e_cd_parachute_fully_open(rocket):
    chute = Parachute("drogue", cd_s=1.0, lag=1.0, trigger="apogee")
    chute.signal_t = 5.0
    rocket.add_parachute(chute)
    # t=20.0 >> signal_t + lag: tau capped at 1.0
    cd = rocket._e_cd(0.0, {}, 500.0, 20.0)
    ref = _ref_area()
    expected = (0.5 * ref + 1.0) / ref
    assert cd == pytest.approx(expected, rel=1e-6)


# --- Multiple parachutes ---

def test_e_cd_two_parachutes_additive(rocket):
    chute1 = Parachute("drogue", cd_s=1.0, lag=1.0, trigger="apogee")
    chute2 = Parachute("main", cd_s=3.0, lag=1.0, trigger=500.0)
    chute1.signal_t = 5.0
    chute2.signal_t = 10.0
    rocket.add_parachute(chute1)
    rocket.add_parachute(chute2)
    # t=20.0: both fully open (tau=1 for both)
    cd = rocket._e_cd(0.0, {}, 400.0, 20.0)
    ref = _ref_area()
    expected = (0.5 * ref + 1.0 + 3.0) / ref
    assert cd == pytest.approx(expected, rel=1e-6)


def test_e_cd_only_triggered_chute_contributes(rocket):
    chute1 = Parachute("drogue", cd_s=1.0, lag=1.0, trigger="apogee")
    chute2 = Parachute("main", cd_s=5.0, lag=1.0, trigger=500.0)
    chute1.signal_t = 5.0
    # chute2.signal_t stays None
    rocket.add_parachute(chute1)
    rocket.add_parachute(chute2)
    cd = rocket._e_cd(0.0, {}, 400.0, 20.0)
    ref = _ref_area()
    expected = (0.5 * ref + 1.0) / ref
    assert cd == pytest.approx(expected, rel=1e-6)
