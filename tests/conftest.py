import pytest
from flightForge import Environment, Motor, Rocket, Simulation
from flightForge.utils import logarithmic_thrust


@pytest.fixture
def motor():
    thrust_fn = logarithmic_thrust(5.0, 2000.0, 0.5)
    return Motor(thrust_fn, burn_time=5.0, initial_grain_mass=2.0)


@pytest.fixture
def rocket(motor):
    r = Rocket(10.0, lambda _: 0.5, 0.1)
    r.add_motor(motor)
    return r


@pytest.fixture
def env():
    return Environment()


@pytest.fixture
def sim(env, rocket):
    return Simulation(env, rocket, rail_length=5.0, inclination=90.0, heading=0.0)
