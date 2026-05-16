import math

import numpy as np
import pytest

from flightForge import Environment


@pytest.fixture
def env():
    return Environment()


# --- ISA temperature ---

def test_isa_temperature_sea_level(env):
    assert float(env._get_isa_temperature(0.0)) == pytest.approx(288.15)


def test_isa_temperature_mid_troposphere(env):
    h = 5000.0
    expected = 288.15 - 0.0065 * h
    assert float(env._get_isa_temperature(h)) == pytest.approx(expected)


def test_isa_temperature_at_tropopause(env):
    assert float(env._get_isa_temperature(11000.0)) == pytest.approx(216.65)


def test_isa_temperature_stratosphere_constant(env):
    assert float(env._get_isa_temperature(15000.0)) == pytest.approx(216.65)
    assert float(env._get_isa_temperature(25000.0)) == pytest.approx(216.65)


def test_isa_temperature_vectorized(env):
    h = np.array([0.0, 5000.0, 11000.0, 20000.0])
    T = env._get_isa_temperature(h)
    assert T.shape == (4,)
    assert T[0] == pytest.approx(288.15)
    assert T[2] == pytest.approx(216.65)
    assert T[3] == pytest.approx(216.65)


# --- Density ---

def test_density_sea_level(env):
    assert float(env.density(0.0)) == pytest.approx(1.225, rel=1e-3)


def test_density_decreases_with_altitude(env):
    assert env.density(1000.0) < env.density(0.0)
    assert env.density(5000.0) < env.density(1000.0)
    assert env.density(11000.0) < env.density(5000.0)


def test_density_always_positive(env):
    for h in [0, 1000, 5000, 11000, 20000]:
        assert env.density(h) > 0


# --- Speed of sound ---

def test_speed_of_sound_sea_level(env):
    expected = math.sqrt(1.4 * 287.05 * 288.15)
    assert float(env.speed_of_sound(0.0)) == pytest.approx(expected, rel=1e-6)


def test_speed_of_sound_decreases_in_troposphere(env):
    assert env.speed_of_sound(5000.0) < env.speed_of_sound(0.0)


def test_speed_of_sound_constant_in_stratosphere(env):
    a1 = float(env.speed_of_sound(12000.0))
    a2 = float(env.speed_of_sound(20000.0))
    assert a1 == pytest.approx(a2)


def test_speed_of_sound_vectorized(env):
    h = np.array([0.0, 5000.0, 11000.0])
    a = env.speed_of_sound(h)
    assert a.shape == (3,)
    assert all(a > 0)


# --- Dynamic viscosity ---

def test_dynamic_viscosity_sea_level(env):
    T = 288.15
    expected = (1.458e-6 * T**1.5) / (T + 110.4)
    assert float(env.dynamic_viscosity(0.0)) == pytest.approx(expected, rel=1e-6)


def test_dynamic_viscosity_positive_everywhere(env):
    for h in [0, 5000, 11000, 20000]:
        assert env.dynamic_viscosity(h) > 0


def test_dynamic_viscosity_vectorized(env):
    h = np.array([0.0, 5000.0])
    mu = env.dynamic_viscosity(h)
    assert mu.shape == (2,)


# --- Wind ---

def test_wind_default_zero_at_surface(env):
    u, v = env.wind(0.0)
    assert u == 0.0
    assert v == 0.0


def test_wind_default_zero_at_altitude(env):
    for h in [500.0, 2000.0, 8000.0]:
        u, v = env.wind(h)
        assert u == 0.0
        assert v == 0.0


def test_custom_wind_profile_applied():
    custom = Environment(wind_u=lambda _: 5.0, wind_v=lambda _: -3.0)
    u, v = custom.wind(1000.0)
    assert u == pytest.approx(5.0)
    assert v == pytest.approx(-3.0)
