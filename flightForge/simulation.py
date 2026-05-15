from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
from scipy.integrate import RK45

from .flight import FlightData
from .logger import bcolors, logger
from .utils import unit_norm

if TYPE_CHECKING:
    from .environment import Environment
    from .motor import Motor
    from .rocket import Rocket

_POS = slice(0, 3)
_VEL = slice(3, 6)
_M_OX = 6
_M_GRN = 7


class Simulation:
    """3DOF translational flight simulator with adaptive RK45 or fixed RK4 integration."""

    def __init__(
        self,
        environment: Environment,
        rocket: Rocket,
        rail_length: float,
        inclination: float,
        heading: float,
    ) -> None:
        self.env = environment
        self.rocket = rocket
        self.motor: Motor = self.rocket.motor
        self.rail_length = rail_length
        self.inc = np.radians(inclination)
        self.heading = np.radians(heading)
        self.results: Optional[FlightData] = None

        self.dir = np.array(
            [
                np.cos(self.inc) * np.cos(self.heading),
                np.cos(self.inc) * np.sin(self.heading),
                np.sin(self.inc),
            ]
        )

        self.events: dict[str, Optional[tuple[float, np.ndarray]]] = {
            "rail_departure": None,
            "burn_out": None,
            "apogee": None,
            "impact": None,
        }
        for p in self.rocket.parachutes:
            self.events[f"{p.name}_signal"] = None
            self.events[f"{p.name}_opening"] = None

        self.linear_params: dict[str, Optional[float]] = {
            "out_of_rail_velocity": None,
            "apogee": None,
        }

    def _t_target_interpolation(
        self,
        t: float,
        t_prev: float,
        state: np.ndarray,
        state_prev: np.ndarray,
        t_target: float,
    ) -> tuple[float, np.ndarray]:
        if t == t_prev:
            return t_target, state.copy()
        tau = (t_target - t_prev) / (t - t_prev)
        return t_target, state_prev + tau * (state - state_prev)

    def _linear_state(
        self,
        t: float,
        t_prev: float,
        state: np.ndarray,
        state_prev: np.ndarray,
        i: int,
        target: float,
    ) -> tuple[float, np.ndarray]:
        z0, z1 = state_prev[i], state[i]
        tau = 0.0 if z0 == z1 else np.clip((target - z0) / (z1 - z0), 0.0, 1.0)
        return t_prev + tau * (t - t_prev), state_prev + tau * (state - state_prev)

    def _cmd_log(self, t: float, s: np.ndarray, si: str) -> None:
        logger.info("-------------------------------------------")
        logger.info(f"Event {bcolors.BOLD}{bcolors.OKGREEN}{si}{bcolors.ENDC} at {t:.2f} s")
        logger.info(f"Position: ({s[0]:.2f}, {s[1]:.2f}, {s[2]:.2f}) m")
        logger.info(f"Velocity: ({s[3]:.2f}, {s[4]:.2f}, {s[5]:.2f}) m/s")
        logger.info(f"Mass: {self.rocket.dry_mass + max(s[_M_OX], 0.0) + max(s[_M_GRN], 0.0):.2f} kg")
        logger.info("-------------------------------------------")

    def _event_check(
        self,
        t: float,
        t_prev: float,
        state: np.ndarray,
        state_prev: np.ndarray,
    ) -> None:
        tl: Optional[float] = None
        sl: Optional[np.ndarray] = None
        state_info = ""

        if self.events["rail_departure"] is None:
            s_prev = np.dot(state_prev[_POS], self.dir)
            s = np.dot(state[_POS], self.dir)
            if s_prev < self.rail_length <= s:
                tl, sl = self._linear_state(
                    t, t_prev, state, state_prev, 2, self.dir[2] * self.rail_length
                )
                self.events["rail_departure"] = (tl, sl)
                self.linear_params["out_of_rail_velocity"] = float(np.linalg.norm(sl[_VEL]))
                state_info = "rail_departure"

        burning = state[_M_OX] > 0 or state[_M_GRN] > 0
        if self.events["burn_out"] is None and not burning:
            tl, sl = self._t_target_interpolation(t, t_prev, state, state_prev, t)
            self.events["burn_out"] = (tl, sl)
            state_info = "burn_out"
        elif self.events["burn_out"] is None and t_prev < self.motor.burn_time <= t:
            tl, sl = self._t_target_interpolation(
                t, t_prev, state, state_prev, self.motor.burn_time
            )
            self.events["burn_out"] = (tl, sl)
            state_info = "burn_out"

        if self.events["apogee"] is None and self.events["rail_departure"] is not None:
            if state_prev[5] > 0 and state[5] <= 0:
                tl, sl = self._linear_state(t, t_prev, state, state_prev, 5, 0)
                self.events["apogee"] = (tl, sl)
                self.linear_params["apogee"] = float(sl[2])
                state_info = "apogee"

        if self.events["impact"] is None and self.events["rail_departure"] is not None:
            if state_prev[2] > 0 and state[2] <= 0:
                tl, sl = self._linear_state(t, t_prev, state, state_prev, 2, 0)
                self.events["impact"] = (tl, sl)
                state_info = "impact"

        if sl is not None:
            self._cmd_log(tl, sl, state_info)

        for p in self.rocket.parachutes:
            if p.signal_t is None:
                triggered = False
                if p.trigger == "apogee" and self.events["apogee"] is not None:
                    p.signal_t = self.events["apogee"][0]
                    triggered = True
                elif isinstance(p.trigger, (int, float)):
                    if state[2] <= p.trigger and state_prev[2] > p.trigger and state[5] < 0:
                        p.signal_t = t
                        triggered = True
                if triggered:
                    p.opening_t = p.signal_t + p.lag
                    self.events[f"{p.name}_signal"] = (p.signal_t, state.copy())
                    if not p.logged:
                        p.logged = True
                        logger.info(
                            f"{bcolors.OKCYAN}{p.name} signal at: {p.signal_t:.2f} s, "
                            f"opening at: {p.opening_t:.2f} s{bcolors.ENDC}"
                        )

            if p.opening_t is not None and self.events[f"{p.name}_opening"] is None:
                if t >= p.opening_t:
                    self.events[f"{p.name}_opening"] = (p.opening_t, state.copy())

    def _ode_rhs(self, t: float, state: np.ndarray) -> np.ndarray:
        pos = state[_POS]
        vel = state[_VEL]
        m_ox = max(state[_M_OX], 0.0)
        m_grain = max(state[_M_GRN], 0.0)
        m_total = self.rocket.dry_mass + m_ox + m_grain

        rho = self.env.density(pos[2])
        wind = np.array([*self.env.wind(pos[2]), 0.0])
        rel_v = vel - wind
        v_mag = float(np.linalg.norm(rel_v))
        mach = v_mag / self.env.speed_of_sound(pos[2])

        cd = self.rocket.e_cd(mach, self.events, pos[2], t)

        on_rail = self.events["rail_departure"] is None
        v_dir = self.dir if on_rail else unit_norm(rel_v)

        drag_mag = -cd * self.rocket.ref_area * 0.5 * rho * v_mag**2
        drag = drag_mag * v_dir

        burning = self.events["burn_out"] is None
        thrust = self.motor.get_thrust(t) * v_dir if burning else np.zeros(3)

        weight = m_total * np.array([0.0, 0.0, -self.env.g])
        total_force = thrust + drag + weight

        _, g_mdot = self.motor.get_mdot(t, burning)
        ox_dot = -self.motor.ox_mdot if m_ox > 0 else 0.0
        grain_dot = -g_mdot if m_grain > 0 else 0.0

        if on_rail:
            total_force = np.dot(total_force, self.dir) * self.dir
            vel = np.dot(vel, self.dir) * self.dir

        accel = total_force / m_total

        return np.concatenate((vel, accel, [ox_dot, grain_dot]))

    def run(
        self,
        terminate_on: str = "impact",
        method: str = "RK45",
        rtol: float = 1e-6,
        atol: float = 1e-9,
        max_step: float = np.inf,
        dt: float = 0.01,
        t_max: float = 1000,
        initial_state: Optional[np.ndarray] = None,
    ) -> FlightData:
        """Run the simulation and return a FlightData object.

        Args:
            terminate_on: Event key to stop at ('impact', 'apogee', 'burn_out', 'rail_departure').
            method:       Integration method — 'RK45' (adaptive, default) or 'RK4' (fixed-step).
            rtol:         Relative tolerance (RK45 only).
            atol:         Absolute tolerance (RK45 only).
            max_step:     Maximum step size in seconds (RK45 only).
            dt:           Fixed time step in seconds (RK4 only).
            t_max:        Maximum simulation time in seconds.
            initial_state: Optional 8-element initial state array [x,y,z,vx,vy,vz,m_ox,m_grain].
        """
        if terminate_on not in self.events:
            raise ValueError(
                f"Invalid terminate_on: '{terminate_on}'. "
                f"Valid keys: {list(self.events.keys())}"
            )
        if method not in ("RK45", "RK4"):
            raise ValueError(f"Invalid method: '{method}'. Use 'RK45' or 'RK4'.")

        if initial_state is not None:
            state = initial_state.astype(float)
        else:
            m_ox = self.motor.initial_ox_mass
            m_grain = self.motor.initial_grain_mass
            state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, m_ox, m_grain])

        logger.info(
            f"Initial Mass: {self.rocket.dry_mass + state[_M_OX] + state[_M_GRN]:.2f} kg"
        )

        hist_t: list[float] = []
        hist_pos: list[np.ndarray] = []
        hist_vel: list[np.ndarray] = []
        hist_accel: list[np.ndarray] = []
        hist_mass: list[float] = []
        hist_thrust_mag: list[float] = []
        hist_drag_mag: list[float] = []
        hist_mdot: list[float] = []
        hist_g_mdot: list[float] = []
        hist_mach: list[float] = []

        def record(t: float, state: np.ndarray, d_state: np.ndarray) -> None:
            pos = state[_POS]
            vel = state[_VEL]
            m_ox = max(state[_M_OX], 0.0)
            m_grain = max(state[_M_GRN], 0.0)
            m_total = self.rocket.dry_mass + m_ox + m_grain

            rho = self.env.density(pos[2])
            wind = np.array([*self.env.wind(pos[2]), 0.0])
            rel_v = vel - wind
            v_mag = float(np.linalg.norm(rel_v))
            mach = v_mag / self.env.speed_of_sound(pos[2])

            burning = self.events["burn_out"] is None
            thrust_f = self.motor.get_thrust(t)
            on_rail = self.events["rail_departure"] is None
            v_dir = self.dir if on_rail else unit_norm(rel_v)
            thrust_vec = thrust_f * v_dir if burning else np.zeros(3)

            cd = self.rocket.e_cd(mach, self.events, pos[2], t)
            drag_vec = -cd * self.rocket.ref_area * 0.5 * rho * v_mag**2 * v_dir

            mdot, g_mdot = self.motor.get_mdot(t, burning)

            hist_t.append(t)
            hist_pos.append(pos.copy())
            hist_vel.append(vel.copy())
            hist_accel.append(d_state[3:6].copy())
            hist_mass.append(m_total)
            hist_thrust_mag.append(float(np.linalg.norm(thrust_vec)))
            hist_drag_mag.append(float(np.linalg.norm(drag_vec)))
            hist_mdot.append(mdot)
            hist_g_mdot.append(g_mdot)
            hist_mach.append(float(mach))

        if method == "RK45":
            stepper = RK45(
                self._ode_rhs,
                t0=0.0,
                y0=state,
                t_bound=t_max,
                rtol=rtol,
                atol=atol,
                max_step=max_step,
            )
            t_prev = 0.0
            state_prev = state.copy()
            while stepper.status == "running" and self.events[terminate_on] is None:
                d_state = self._ode_rhs(stepper.t, stepper.y)
                record(stepper.t, stepper.y, d_state)
                stepper.step()
                t = stepper.t
                state = stepper.y.copy()
                self._event_check(t, t_prev, state, state_prev)
                t_prev = t
                state_prev = state.copy()
        else:
            t = 0.0
            while t < t_max and self.events[terminate_on] is None:
                state_prev = state.copy()
                t_prev = t

                d_state = self._ode_rhs(t, state)
                record(t, state, d_state)

                k1 = d_state
                k2 = self._ode_rhs(t + dt / 2, state + k1 * dt / 2)
                k3 = self._ode_rhs(t + dt / 2, state + k2 * dt / 2)
                k4 = self._ode_rhs(t + dt, state + k3 * dt)

                state = state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
                t += dt

                self._event_check(t, t_prev, state, state_prev)

        self.results = FlightData(
            np.array(hist_t),
            np.array(hist_pos),
            np.array(hist_vel),
            np.array(hist_accel),
            np.array(hist_mass),
            np.array(hist_thrust_mag),
            np.array(hist_drag_mag),
            np.array(hist_mdot),
            np.array(hist_g_mdot),
            np.array(hist_mach),
        )

        return self.results
