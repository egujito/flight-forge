import pytest

from flightForge import Motor
from flightForge.utils import logarithmic_thrust


def _make_motor(
    burn_time: float = 5.0,
    peak: float = 2000.0,
    ramp: float = 0.5,
    grain_mass: float = 2.0,
) -> Motor:
    thrust_fn = logarithmic_thrust(burn_time, peak, ramp)
    return Motor(thrust_fn, burn_time=burn_time, initial_grain_mass=grain_mass)


# --- logarithmic_thrust factory ---

def test_log_thrust_before_ignition():
    f = logarithmic_thrust(5.0, 1000.0, 0.5)
    assert f(-0.1) == 0.0


def test_log_thrust_after_burnout():
    f = logarithmic_thrust(5.0, 1000.0, 0.5)
    assert f(5.1) == 0.0


def test_log_thrust_exactly_at_burn_end():
    f = logarithmic_thrust(5.0, 1000.0, 0.5)
    assert f(5.0) == pytest.approx(1000.0)


def test_log_thrust_mid_ramp():
    f = logarithmic_thrust(5.0, 1000.0, 0.5)
    assert f(0.25) == pytest.approx(500.0)


def test_log_thrust_constant_phase():
    f = logarithmic_thrust(5.0, 1000.0, 0.5)
    assert f(1.0) == pytest.approx(1000.0)
    assert f(3.0) == pytest.approx(1000.0)


def test_log_thrust_zero_ramp_immediate_peak():
    f = logarithmic_thrust(5.0, 1000.0, ramp_time=0.0)
    assert f(0.0) == pytest.approx(1000.0)


def test_log_thrust_invalid_ramp_equals_burn_raises():
    with pytest.raises(ValueError):
        logarithmic_thrust(5.0, 1000.0, ramp_time=5.0)


def test_log_thrust_negative_burn_time_raises():
    with pytest.raises(ValueError):
        logarithmic_thrust(-1.0, 1000.0)


def test_log_thrust_zero_peak_raises():
    with pytest.raises(ValueError):
        logarithmic_thrust(5.0, 0.0)


# --- Motor._get_thrust ---

def test_get_thrust_at_peak():
    m = _make_motor()
    assert m._get_thrust(2.5) == pytest.approx(2000.0)


def test_get_thrust_at_t0_during_ramp():
    m = _make_motor()
    assert m._get_thrust(0.0) == pytest.approx(0.0)


def test_get_thrust_mid_ramp():
    m = _make_motor()
    assert m._get_thrust(0.25) == pytest.approx(1000.0)


def test_get_thrust_after_burnout():
    m = _make_motor()
    assert m._get_thrust(6.0) == pytest.approx(0.0)


def test_get_thrust_returns_float():
    m = _make_motor()
    assert isinstance(m._get_thrust(1.0), float)


# --- Motor._get_mdot ---

def test_mdot_not_burning_returns_zeros():
    m = _make_motor()
    tot, grain = m._get_mdot(2.5, burning=False)
    assert tot == 0.0
    assert grain == 0.0


def test_mdot_burning_total_positive():
    m = _make_motor()
    tot, grain = m._get_mdot(2.5, burning=True)
    assert tot > 0.0


def test_mdot_grain_non_negative():
    m = _make_motor()
    _tot, grain = m._get_mdot(2.5, burning=True)
    assert grain >= 0.0


def test_mdot_matches_thrust_over_ve():
    m = _make_motor()
    thrust = m._get_thrust(2.5)
    tot, _ = m._get_mdot(2.5, burning=True)
    assert tot == pytest.approx(thrust / m.ve)


def test_mdot_grain_clamped_for_ox_heavy_motor():
    # ox_mdot > total mdot → grain would be negative without clamp
    thrust_fn = logarithmic_thrust(5.0, 100.0, 0.5)
    m = Motor(thrust_fn, burn_time=5.0, initial_grain_mass=0.5,
              initial_ox_mass=5.0, ox_mdot=1.0)
    _, grain = m._get_mdot(2.5, burning=True)
    assert grain >= 0.0


# --- Exhaust velocity ---

def test_exhaust_velocity_positive():
    m = _make_motor()
    assert m.ve > 0.0


def test_total_impulse_positive():
    m = _make_motor()
    assert m.i_tot > 0.0


def test_ve_equals_impulse_over_propellant():
    m = _make_motor(grain_mass=2.0)
    total_propellant = m.initial_ox_mass + m.initial_grain_mass
    assert m.ve == pytest.approx(m.i_tot / total_propellant)
