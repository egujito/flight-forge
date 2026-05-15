from __future__ import annotations

import numpy as np

from .logger import bcolors, logger
from .utils import ResultField, func_from_csv


class Motor:
    def __init__(
        self,
        thrust_source: str,
        burn_time: float,
        initial_grain_mass: float,
        initial_ox_mass: float = 0.0,
        ox_mdot: float = 0.0,
        e_log: bool = False,
    ) -> None:
        self.thrust_curve, self.t, self.thrust_arr = func_from_csv(thrust_source)

        self.ox_mdot = ox_mdot
        self.burn_time = burn_time
        self.initial_ox_mass = initial_ox_mass
        self.initial_grain_mass = initial_grain_mass
        self.ve: float = 0.0
        self.i_tot: float = 0.0
        self.type = "Solid" if initial_ox_mass <= 0.0 else "Hybrid"
        self.peak_thrust = float(max(self.thrust_arr))

        self._assert_flow_rates()
        self._compute_exhaust_velocity()

        tot_mdot_arr = np.zeros_like(self.thrust_arr)
        grain_mdot_arr = np.zeros_like(self.thrust_arr)

        if self.ve > 0:
            tot_mdot_arr = self.thrust_arr / self.ve
            grain_mdot_arr = np.maximum(tot_mdot_arr - self.ox_mdot, 0.0)

        self.thrust = ResultField(self.t, self.thrust_arr, "Thrust Force", "N", "orange")
        self.total_mdot = ResultField(self.t, tot_mdot_arr, "Total Mass Flow", "kg/s", "red")
        self.grain_mdot = ResultField(self.t, grain_mdot_arr, "Grain Mass Flow", "kg/s", "darkred")

        if e_log:
            self._cmd_log()

    def _cmd_log(self) -> None:
        logger.info("------- MOTOR INFO --------")
        if self.type == "Hybrid":
            logger.info(f"Initial Oxidizer Mass: {self.initial_ox_mass:.2f} kg")
            logger.info(f"Oxidizer Mass Flow:    {self.ox_mdot:.2f} kg/s")
        logger.info(f"Initial Grain Mass:    {self.initial_grain_mass:.2f} kg")
        logger.info(f"Total Impulse:         {self.i_tot:.2f} Ns")
        logger.info(f"Peak Thrust:           {self.peak_thrust:.2f} N")
        logger.info(f"Eff. Exhaust Vel (Ve): {self.ve:.2f} m/s")
        logger.info("------------------------------------")

    def _compute_exhaust_velocity(self) -> None:
        self.i_tot = float(np.trapezoid(self.thrust_arr, self.t))
        total_propellant = self.initial_ox_mass + self.initial_grain_mass
        self.ve = self.i_tot / total_propellant

    def _assert_flow_rates(self) -> None:
        if self.burn_time * self.ox_mdot > self.initial_ox_mass:
            raise ValueError(f"{bcolors.FAIL}[ERROR]{bcolors.ENDC} Tank underfilled.")

    def get_thrust(self, t: float) -> float:
        return float(self.thrust_curve(t))

    def get_mdot(self, t: float, burning: bool) -> tuple[float, float]:
        if burning:
            tot = self.get_thrust(t) / self.ve if self.ve > 0 else 0.0
            g = max(tot - self.ox_mdot, 0.0)
            return tot, g
        return 0.0, 0.0
