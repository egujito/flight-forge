from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import matplotlib.pyplot as plt
import numpy as np

from .logger import bcolors, logger
from .utils import ResultField, compute_vec, unit_norm

if TYPE_CHECKING:
    from .environment import Environment
    from .motor import Motor
    from .rocket import Rocket


class SimulationResults:
    def __init__(
        self,
        time: np.ndarray,
        pos: np.ndarray,
        vel: np.ndarray,
        accel: np.ndarray,
        mass: np.ndarray,
    ) -> None:
        self.x = ResultField(time, pos[:, 0], "X Position", "m", "blue")
        self.y = ResultField(time, pos[:, 1], "Y Position", "m", "green")
        self.z = ResultField(time, pos[:, 2], "Altitude", "m", "red")

        self.vx = ResultField(time, vel[:, 0], "X Velocity", "m/s", "blue")
        self.vy = ResultField(time, vel[:, 1], "Y Velocity", "m/s", "green")
        self.vz = ResultField(time, vel[:, 2], "Z Velocity", "m/s", "red")
        self.speed = ResultField(
            time, np.linalg.norm(vel, axis=1), "Speed", "m/s", "black"
        )

        self.ax = ResultField(time, accel[:, 0], "X Acceleration", "m/s²", "cyan")
        self.ay = ResultField(time, accel[:, 1], "Y Acceleration", "m/s²", "cyan")
        self.az = ResultField(time, accel[:, 2], "Z Acceleration", "m/s²", "cyan")
        self.acceleration = ResultField(
            time, np.linalg.norm(accel, axis=1), "Acceleration Mag", "m/s²", "black"
        )

        self.mass = ResultField(time, mass, "Mass", "kg", "purple")

    def trajectory_3d(self) -> None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(
            self.x.y_data, self.y.y_data, self.z.y_data, label="Trajectory", linewidth=2
        )
        ax.scatter(
            self.x.y_data[0],
            self.y.y_data[0],
            self.z.y_data[0],
            color="green",
            marker="o",
            s=50,
            label="Launch",
        )
        ax.scatter(
            self.x.y_data[-1],
            self.y.y_data[-1],
            self.z.y_data[-1],
            color="red",
            marker="x",
            s=50,
            label="Impact",
        )
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Altitude (m)")
        ax.set_title("3D Trajectory")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_zlim(zmin=0)
        ax.view_init(elev=20, azim=45)
        plt.tight_layout()
        plt.show()


class Simulation:
    def __init__(
        self,
        environment: Environment,
        rocket: Rocket,
        rail_length: float,
        inclination: float,
        heading: float,
        e_log: bool = False,
        plotter=None,
    ) -> None:
        self.e_log = e_log
        self.env = environment
        self.rocket = rocket
        self.motor: Motor = self.rocket.motor
        self.rail_length = rail_length
        self.inc = np.radians(inclination)
        self.heading = np.radians(heading)
        self.plotter = plotter
        self.results: Optional[SimulationResults] = None

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
        if z0 == z1:
            tau = 0.0
        else:
            tau = (target - z0) / (z1 - z0)
        tau = np.clip(tau, 0.0, 1.0)
        return t_prev + tau * (t - t_prev), state_prev + tau * (state - state_prev)

    def _cmd_log(self, t: float, s: np.ndarray, si: str) -> None:
        logger.info("-------------------------------------------")
        logger.info(f"Event {bcolors.BOLD}{bcolors.OKGREEN}{si}{bcolors.ENDC} at {t:.2f} s")
        logger.info(f"Position: ({s[0]:.2f}, {s[1]:.2f}, {s[2]:.2f}) m")
        logger.info(f"Velocity: ({s[3]:.2f}, {s[4]:.2f}, {s[5]:.2f}) m/s")
        logger.info(f"Mass: {s[6]:.2f} kg")
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
            s_prev = np.dot(state_prev[:3], self.dir)
            s = np.dot(state[:3], self.dir)
            if s_prev < self.rail_length <= s:
                tl, sl = self._linear_state(
                    t, t_prev, state, state_prev, 2, self.dir[2] * self.rail_length
                )
                self.events["rail_departure"] = (tl, sl)
                self.linear_params["out_of_rail_velocity"] = float(np.linalg.norm(sl[3:6]))
                state_info = "rail_departure"

        if self.events["burn_out"] is None and t_prev < self.motor.burn_time <= t:
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

        if sl is not None and self.e_log:
            self._cmd_log(tl, sl, state_info)

        for p in self.rocket.parachutes:
            active = False
            if p.deploy_t is None:
                if p.trigger == "apogee" and self.events["apogee"] is not None:
                    p.deploy_t = t
                    active = True
                elif isinstance(p.trigger, (int, float)):
                    if state[2] <= p.trigger and state_prev[2] > p.trigger and state[5] < 0:
                        p.deploy_t = t
                        active = True
            if self.e_log and not p.logged and active:
                pt = tl if tl is not None else t
                p.logged = True
                logger.info(f"{bcolors.OKCYAN}{p.name} deployed at: {pt:.2f} s{bcolors.ENDC}")

    def _compute_physics(
        self, t: float, state: np.ndarray
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, float, float]]:
        pos, vel, m = state[:3], state[3:6], state[6]

        rho = self.env.density(pos[2])
        wind = np.array([*self.env.wind(pos[2]), 0.0])
        rel_v = vel - wind
        v_mag = float(np.linalg.norm(rel_v))
        mach = v_mag / self.env.speed_of_sound(pos[2])

        cd = self.rocket.e_cd(mach, self.events, pos[2], t)

        on_rail = self.events["rail_departure"] is None
        v_dir = self.dir if on_rail else unit_norm(rel_v)

        drag_mag = -cd * self.rocket.ref_area * 0.5 * rho * v_mag**2
        drag = compute_vec(drag_mag, v_dir)

        burning = self.events["burn_out"] is None
        thrust = (
            compute_vec(self.motor.get_thrust(t), v_dir) if burning else np.zeros(3)
        )

        weight = m * np.array([0, 0, -self.env.g])
        total_force = thrust + drag + weight

        mdot, g_mdot = self.motor.get_mdot(t, burning)

        if on_rail:
            total_force = np.dot(total_force, self.dir) * self.dir
            vel = np.dot(vel, self.dir) * self.dir

        accel = total_force / m

        return np.concatenate((vel, accel, [-mdot])), (thrust, drag, mdot, g_mdot)

    def run(
        self,
        terminate_on: str = "impact",
        dt: float = 0.01,
        t_max: float = 1000,
        initial_state: Optional[np.ndarray] = None,
    ) -> SimulationResults:
        if terminate_on not in self.events:
            raise ValueError(
                f"Invalid terminate_on: '{terminate_on}'. "
                f"Valid keys: {list(self.events.keys())}"
            )

        t = 0.0
        if initial_state is not None:
            state = initial_state
        else:
            m0 = (
                self.rocket.dry_mass
                + self.motor.initial_ox_mass
                + self.motor.initial_grain_mass
            )
            state = np.array([0, 0, 0, 0, 0, 0, m0])

        if self.e_log:
            logger.info(f"Initial Mass: {state[6]:.2f} kg")

        hist_t, hist_pos, hist_vel, hist_accel, hist_mass = [], [], [], [], []
        hist_thrust_mag, hist_drag_mag, hist_mdot, hist_g_mdot = [], [], [], []

        while t < t_max and self.events[terminate_on] is None:
            state_prev = state.copy()
            t_prev = t

            d_state, forces = self._compute_physics(t, state)
            thrust_vec, drag_vec, mdot, g_mdot = forces

            hist_t.append(t)
            hist_pos.append(state[:3])
            hist_vel.append(state[3:6])
            hist_mass.append(state[6])
            hist_accel.append(d_state[3:6])
            hist_thrust_mag.append(float(np.linalg.norm(thrust_vec)))
            hist_drag_mag.append(float(np.linalg.norm(drag_vec)))
            hist_mdot.append(mdot)
            hist_g_mdot.append(g_mdot)

            k1 = d_state
            k2, _ = self._compute_physics(t + dt / 2, state + k1 * dt / 2)
            k3, _ = self._compute_physics(t + dt / 2, state + k2 * dt / 2)
            k4, _ = self._compute_physics(t + dt, state + k3 * dt)

            state = state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            t += dt

            self._event_check(t, t_prev, state, state_prev)
            if self.plotter:
                self.plotter.update(t, state, self.events)

        if self.plotter:
            self.plotter.finalize()

        time_arr = np.array(hist_t)

        self.results = SimulationResults(
            time_arr,
            np.array(hist_pos),
            np.array(hist_vel),
            np.array(hist_accel),
            np.array(hist_mass),
        )

        self.motor.thrust = ResultField(
            time_arr, np.array(hist_thrust_mag), "Thrust Force", "N", "orange"
        )
        self.motor.total_mdot = ResultField(
            time_arr, np.array(hist_mdot), "Total Mdot", "kg/s", "red"
        )
        self.motor.grain_mdot = ResultField(
            time_arr, np.array(hist_g_mdot), "Grain Mdot", "kg/s", "darkred"
        )
        self.rocket.drag = ResultField(
            time_arr, np.array(hist_drag_mag), "Drag Force", "N", "magenta"
        )

        return self.results
