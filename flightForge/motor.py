from __future__ import annotations

from typing import Callable, Union

import numpy as np

from .logger import bcolors, logger
from .utils import func_from_csv, load_curve


class Motor:
    """Rocket motor model supporting solid and hybrid propellants."""

    def __init__(
        self,
        thrust_source: Union[str, Callable[[float], float]],
        burn_time: float,
        initial_grain_mass: float,
        initial_ox_mass: float = 0.0,
        ox_mdot: float = 0.0,
    ) -> None:
        """Initialise motor from a thrust curve (CSV path or callable).

        Args:
            thrust_source:      Path to a two-column CSV or a callable f(t) -> N.
            burn_time:          Nominal burn duration in seconds.
            initial_grain_mass: Initial solid-fuel grain mass in kg.
            initial_ox_mass:    Initial oxidiser mass in kg (0 for solid motors).
            ox_mdot:            Constant oxidiser mass-flow rate in kg/s.
        """
        if burn_time <= 0:
            raise ValueError("burn_time must be positive.")
        if initial_grain_mass <= 0:
            raise ValueError("initial_grain_mass must be positive.")
        if initial_ox_mass < 0:
            raise ValueError("initial_ox_mass must be non-negative.")
        if ox_mdot < 0:
            raise ValueError("ox_mdot must be non-negative.")

        self.burn_time = burn_time
        self.initial_ox_mass = initial_ox_mass
        self.initial_grain_mass = initial_grain_mass
        self.ox_mdot = ox_mdot
        self.type = "Solid" if initial_ox_mass <= 0.0 else "Hybrid"

        self.thrust_curve = load_curve(thrust_source)

        if isinstance(thrust_source, str):
            _, self.t_data, self.thrust_data = func_from_csv(thrust_source)
        else:
            self.t_data = np.linspace(0.0, burn_time, max(int(burn_time / 0.001), 500))
            self.thrust_data = np.array([self.thrust_curve(t) for t in self.t_data])

        self.peak_thrust = float(self.thrust_data.max())
        self._assert_flow_rates()
        self._compute_exhaust_velocity()

        logger.info("------- MOTOR INFO --------")
        if self.type == "Hybrid":
            logger.info(f"Initial Oxidiser Mass: {self.initial_ox_mass:.2f} kg")
            logger.info(f"Oxidiser Mass Flow:    {self.ox_mdot:.2f} kg/s")
        logger.info(f"Initial Grain Mass:    {self.initial_grain_mass:.2f} kg")
        logger.info(f"Total Impulse:         {self.i_tot:.2f} Ns")
        logger.info(f"Peak Thrust:           {self.peak_thrust:.2f} N")
        logger.info(f"Eff. Exhaust Vel (Ve): {self.ve:.2f} m/s")
        logger.info("------------------------------------")

    def _compute_exhaust_velocity(self) -> None:
        self.i_tot = float(np.trapezoid(self.thrust_data, self.t_data))
        total_propellant = self.initial_ox_mass + self.initial_grain_mass
        self.ve = self.i_tot / total_propellant

    def _assert_flow_rates(self) -> None:
        if self.burn_time * self.ox_mdot > self.initial_ox_mass:
            raise ValueError(f"{bcolors.FAIL}[ERROR]{bcolors.ENDC} Tank underfilled.")

    def get_thrust(self, t: float) -> float:
        """Return thrust in Newtons at time t."""
        return float(self.thrust_curve(t))

    def get_mdot(self, t: float, burning: bool) -> tuple[float, float]:
        """Return (total_mdot, grain_mdot) in kg/s at time t."""
        if burning:
            tot = self.get_thrust(t) / self.ve if self.ve > 0 else 0.0
            g = max(tot - self.ox_mdot, 0.0)
            return tot, g
        return 0.0, 0.0
