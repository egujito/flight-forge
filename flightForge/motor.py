from .utils import func_from_csv
import numpy as np

class Motor:
    def __init__(self, thrust_source, burn_time, ox_mass=0, ox_mdot=0, grain_mass=None, mass_ot=None, e_log=False):
        self.thrust_curve = func_from_csv(thrust_source)
        self.mass_curve = None
        self.ox_mdot = ox_mdot
        self.burn_time = burn_time
        self.ox_mass = ox_mass
        self.grain_mass = grain_mass
        self.ve = 0
        self.i_tot = 0
        self.type = "Solid"

        if ox_mass != 0:
            self.type = "Hybrid"

        if self.grain_mass is None:
            raise Exception(f"Grain Mass needs to be especified. Only oxidizer load and oxidizer mass flow are optional")

        if mass_ot is not None:
            self.mass_curve = func_from_csv(mass_ot)

        self._assert_flow_rates()
        self._compute_exhaust_velocity(thrust_source)
    
        if e_log==True:
            self._cmd_log()

    def _cmd_log(self):
        print(f"-------{self.type.capitalize()} MOTOR INFO --------")
        print(f"Oxidizer Mass: {self.ox_mass} kg")
        print(f"Grain Mass:    {self.grain_mass} kg")
        print(f"Total Impulse: {self.i_tot:.2f} Ns")
        print(f"Eff. Exhaust Velocity (Ve): {self.ve:.2f} m/s")
        print("------------------------------------")

    def _compute_exhaust_velocity(self, ts):
        x, y = func_from_csv(ts, get_arrs=True)
        self.i_tot = np.trapz(y, x)
        self.ve = self.i_tot / (self.ox_mass + self.grain_mass)

    def _assert_flow_rates(self):
        if self.burn_time*self.ox_mdot > self.ox_mass:
            raise Exception(f"!Tank will be underfilled. burn_time * ox_mdot = {self.burn_time*self.ox_mdot} kg ox_mass = {self.ox_mass} kg")

    def thrust(self, t):
        return self.thrust_curve(t)
    
    def mdot(self, t):
        return self.thrust(t) / self.ve
