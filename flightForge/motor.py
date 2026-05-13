
import numpy as np

from .logger import logger
from .utils import ResultField, bcolors, func_from_csv


class Motor:
    def __init__(
        self,
        thrust_source,
        burn_time,
        initial_ox_mass=0,
        ox_mdot=0,
        initial_grain_mass=None,
        e_log=False,
    ):
        self.thrust_curve, self.t, self.thrust_arr = func_from_csv(thrust_source)
        
        self.ox_mdot = ox_mdot
        self.burn_time = burn_time
        self.initial_ox_mass = initial_ox_mass
        self.initial_grain_mass = initial_grain_mass
        self.ve = 0
        self.i_tot = 0
        self.type = "Solid" if initial_ox_mass == 0 else "Hybrid"
        self.peak_thrust = max(self.thrust_arr)

        if self.initial_grain_mass is None:
            raise Exception("Grain Mass needs to be specified.")

        self._assert_flow_rates()
        self._compute_exhaust_velocity()

        tot_mdot_arr = np.zeros_like(self.thrust_arr)
        grain_mdot_arr = np.zeros_like(self.thrust_arr)

        if self.ve > 0:
            tot_mdot_arr = self.thrust_arr / self.ve
            grain_mdot_arr = tot_mdot_arr - self.ox_mdot
            grain_mdot_arr = np.maximum(grain_mdot_arr, 0)

        self.thrust = ResultField(
            self.t, self.thrust_arr, "Thrust Force", "N", "orange"
        )
        self.total_mdot = ResultField(
            self.t, tot_mdot_arr, "Total Mass Flow", "kg/s", "red"
        )
        self.grain_mdot = ResultField(
            self.t, grain_mdot_arr, "Grain Mass Flow", "kg/s", "darkred"
        )

        if e_log:
            self._cmd_log()
    
    def _cmd_log(self):
        logger.info("------- MOTOR INFO --------")
        if self.type == "Hybrid":
            logger.info(f"Initial Oxidizer Mass: {self.initial_ox_mass:.2f} kg")
            logger.info(f"Oxidizer Mass Flow:    {self.ox_mdot:.2f} kg/s")

        logger.info(f"Initial Grain Mass:    {self.initial_grain_mass:.2f} kg")
        logger.info(f"Total Impulse:         {self.i_tot:.2f} Ns")
        logger.info(f"Peak Thrust:           {self.peak_thrust:.2f} N")
        logger.info(f"Eff. Exhaust Vel (Ve): {self.ve:.2f} m/s")
        logger.info("------------------------------------")

    def _compute_exhaust_velocity(self):
        self.i_tot = np.trapezoid(self.thrust_arr, self.t)
        total_propellant = self.initial_ox_mass + self.initial_grain_mass
        self.ve = self.i_tot / total_propellant

    def _assert_flow_rates(self):
        if self.burn_time * self.ox_mdot > self.initial_ox_mass:
            raise Exception(f"{bcolors.FAIL}[ERROR]{bcolors.ENDC} Tank underfilled.")
    
    def get_thrust(self, t):
        return self.thrust_curve(t)
    
    def get_mdot(self, t, burning):
        if burning:
            tot = self.get_thrust(t) / self.ve if self.ve > 0 else 0
            g = tot - self.ox_mdot
            return tot, g
        return 0, 0
