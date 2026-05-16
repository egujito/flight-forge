from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


class FlightData:
    """Simulation output containing time-series arrays for all flight quantities."""

    def __init__(
        self,
        t: np.ndarray,
        pos: np.ndarray,
        vel: np.ndarray,
        accel: np.ndarray,
        mass: np.ndarray,
        thrust_mag: np.ndarray,
        drag_mag: np.ndarray,
        mdot: np.ndarray,
        ox_mdot: np.ndarray,
        g_mdot: np.ndarray,
        mach: np.ndarray,
    ) -> None:
        self.t = t

        self.x = pos[:, 0]
        self.y = pos[:, 1]
        self.z = pos[:, 2]

        self.vx = vel[:, 0]
        self.vy = vel[:, 1]
        self.vz = vel[:, 2]
        self.speed = np.linalg.norm(vel, axis=1)

        self.ax = accel[:, 0]
        self.ay = accel[:, 1]
        self.az = accel[:, 2]
        self.acceleration = np.linalg.norm(accel, axis=1)

        self.mass = mass
        self.thrust = thrust_mag
        self.drag = drag_mag
        self.total_mdot = mdot
        self.ox_mdot = ox_mdot
        self.grain_mdot = g_mdot
        self.mach = mach

    def at_time(self, t: float, array: np.ndarray) -> float:
        """Interpolate any stored array at a given time in seconds."""
        return float(np.interp(t, self.t, array))

    def at_height(self, h: float, array: np.ndarray) -> float:
        """Interpolate any stored array at a given altitude on the ascending phase."""
        apogee_idx = int(np.argmax(self.z))
        z_asc = self.z[:apogee_idx + 1]
        arr_asc = array[:apogee_idx + 1]
        return float(np.interp(h, z_asc, arr_asc))

    def plot(
        self,
        x: np.ndarray,
        y: np.ndarray,
        xlim: Optional[tuple[float, float]] = None,
        ylim: Optional[tuple[float, float]] = None,
        xlabel: str = "",
        ylabel: str = "",
        title: str = "",
    ) -> None:
        """Plot any two stored arrays against each other."""
        _, ax = plt.subplots(figsize=(10, 5))
        ax.plot(x, y)
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.4)
        plt.tight_layout()
        plt.show()

    def trajectory_3d(self) -> None:
        """Plot the 3D trajectory from launch to impact."""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(self.x, self.y, self.z, label="Trajectory", linewidth=2)
        ax.scatter(self.x[0], self.y[0], self.z[0], color="green", marker="o", s=50, label="Launch")
        ax.scatter(self.x[-1], self.y[-1], self.z[-1], color="red", marker="x", s=50, label="Impact")
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
