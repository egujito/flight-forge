import numpy as np
import pytest

from flightForge import Environment, Motor, Rocket, Simulation
from flightForge.utils import logarithmic_thrust


def _make_vertical_sim() -> Simulation:
    thrust_fn = logarithmic_thrust(5.0, 2000.0, 0.5)
    motor = Motor(thrust_fn, burn_time=5.0, initial_grain_mass=2.0)
    rocket = Rocket(10.0, lambda _: 0.5, 0.1)
    rocket.add_motor(motor)
    env = Environment()
    return Simulation(env, rocket, rail_length=5.0, inclination=90.0, heading=0.0)


# --- _t_target_interpolation ---

def test_t_target_interp_midpoint(sim):
    s_prev = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 2.0, 2.0])
    s = np.array([0.0, 0.0, 5.0, 0.0, 0.0, 12.0, 1.5, 1.5])
    t_out, s_out = sim._t_target_interpolation(1.0, 0.0, s, s_prev, 0.5)
    assert t_out == pytest.approx(0.5)
    assert s_out[2] == pytest.approx(2.5)
    assert s_out[5] == pytest.approx(11.0)


def test_t_target_interp_equal_times_returns_current_state(sim):
    s_prev = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 2.0, 2.0])
    s = np.array([0.0, 0.0, 5.0, 0.0, 0.0, 12.0, 1.5, 1.5])
    t_out, s_out = sim._t_target_interpolation(1.0, 1.0, s, s_prev, 1.0)
    assert t_out == pytest.approx(1.0)
    assert s_out[2] == pytest.approx(5.0)


def test_t_target_interp_at_start_returns_prev_state(sim):
    s_prev = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0])
    s = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 5.0, 1.5, 1.5])
    t_out, s_out = sim._t_target_interpolation(1.0, 0.0, s, s_prev, 0.0)
    assert t_out == pytest.approx(0.0)
    assert s_out[2] == pytest.approx(0.0)


# --- _linear_state ---

def test_linear_state_target_in_bracket(sim):
    s_prev = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 2.0, 2.0])
    s = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 12.0, 1.5, 1.5])
    # tau for z=5 between z0=0 and z1=10 → tau=0.5, t_out=0+0.5*2=1.0
    t_out, s_out = sim._linear_state(2.0, 0.0, s, s_prev, 2, 5.0)
    assert t_out == pytest.approx(1.0)
    assert s_out[2] == pytest.approx(5.0)
    assert s_out[5] == pytest.approx(11.0)


def test_linear_state_target_below_bracket_clamped_to_zero(sim):
    s_prev = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0])
    s = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 1.5, 1.5])
    # target=-5 is below z0=0: tau clipped to 0
    t_out, s_out = sim._linear_state(1.0, 0.0, s, s_prev, 2, -5.0)
    assert t_out == pytest.approx(0.0)
    assert s_out[2] == pytest.approx(0.0)


def test_linear_state_constant_component_tau_zero(sim):
    s_prev = np.array([0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 2.0, 2.0])
    s = np.array([0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 1.5, 1.5])
    # z0 == z1: tau is defined as 0 to avoid division by zero
    t_out, s_out = sim._linear_state(1.0, 0.0, s, s_prev, 2, 5.0)
    assert t_out == pytest.approx(0.0)
    assert s_out[2] == pytest.approx(5.0)


# --- _ode_rhs physics ---

def test_ode_rhs_output_shape(sim):
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      sim.motor.initial_ox_mass, sim.motor.initial_grain_mass])
    d = sim._ode_rhs(0.0, state)
    assert d.shape == (8,)


def test_ode_rhs_position_derivative_equals_projected_velocity(sim):
    # on_rail: velocity is projected onto rail direction (=[0,0,1] for vertical)
    state = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 2.0])
    d = sim._ode_rhs(2.5, state)
    # on rail: vel = dot([1,2,3],[0,0,1])*[0,0,1] = [0, 0, 3]
    assert d[2] == pytest.approx(3.0)
    assert d[0] == pytest.approx(0.0)
    assert d[1] == pytest.approx(0.0)


def test_ode_rhs_no_drag_at_zero_velocity(sim):
    # zero velocity → only gravity → a_z = -g on vertical rail
    state = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 2.0])
    d = sim._ode_rhs(6.0, state)  # t=6 > burn_time, thrust curve returns 0
    assert d[5] == pytest.approx(-sim.env.g, rel=1e-4)


def test_ode_rhs_post_burnout_no_thrust(sim):
    # set events to mark off-rail and burned out
    sim.events["burn_out"] = (5.0, np.zeros(8))
    sim.events["rail_departure"] = (1.0, np.zeros(8))
    # rocket moving upward at 50 m/s at 500 m altitude
    state = np.array([0.0, 0.0, 500.0, 0.0, 0.0, 50.0, 0.0, 0.0])
    d = sim._ode_rhs(10.0, state)
    rho = sim.env.density(500.0)
    drag_decel = 0.5 * sim.rocket.ref_area * 0.5 * rho * 50.0**2 / sim.rocket.dry_mass
    # drag opposes upward motion, gravity is also downward
    assert d[5] == pytest.approx(-sim.env.g - drag_decel, rel=1e-3)


def test_ode_rhs_negative_propellant_mass_rates_zero(sim):
    # negative propellant masses should clamp to 0 (no negative grain consumption)
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -0.5])
    d = sim._ode_rhs(0.0, state)
    assert d[6] == pytest.approx(0.0)
    assert d[7] == pytest.approx(0.0)


# --- Full integration tests ---

def test_run_apogee_is_positive():
    s = _make_vertical_sim()
    s.run(terminate_on="apogee")
    apogee = s.linear_params["apogee"]
    assert apogee is not None
    assert apogee > 0.0


def test_run_apogee_is_finite():
    s = _make_vertical_sim()
    s.run(terminate_on="apogee")
    apogee = s.linear_params["apogee"]
    assert apogee is not None
    assert np.isfinite(apogee)


def test_run_impact_event_altitude_is_zero():
    s = _make_vertical_sim()
    s.run(terminate_on="impact")
    impact_event = s.events["impact"]
    assert impact_event is not None
    # interpolated state at impact: z component should be at ground level
    assert impact_event[1][2] == pytest.approx(0.0, abs=1.0)


def test_run_flightdata_arrays_consistent_length():
    s = _make_vertical_sim()
    flight = s.run(terminate_on="apogee")
    n = len(flight.t)
    assert len(flight.z) == n
    assert len(flight.vz) == n
    assert len(flight.mass) == n


def test_run_mass_monotonically_decreasing_during_burn():
    s = _make_vertical_sim()
    flight = s.run(terminate_on="burn_out")
    # mass should never increase during the burn
    assert all(np.diff(flight.mass) <= 1e-9)


def test_rk4_vs_rk45_apogee_within_one_percent():
    s45 = _make_vertical_sim()
    s45.run(terminate_on="apogee")
    apogee_45 = s45.linear_params["apogee"]
    assert apogee_45 is not None

    s4 = _make_vertical_sim()
    s4.run(terminate_on="apogee", method="RK4", dt=0.005)
    apogee_4 = s4.linear_params["apogee"]
    assert apogee_4 is not None

    assert apogee_4 == pytest.approx(apogee_45, rel=0.01)
