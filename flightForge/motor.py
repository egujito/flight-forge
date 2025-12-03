from .utils import func_from_csv

class Motor:
    def __init__(self, thrust_source, burn_time, fuel_load, fuel_mdot, mass_ot=None):
        self.thrust_curve = func_from_csv(thrust_source)
        self.mass_curve = None
        self.fuel_mdot = fuel_mdot
        self.burn_time = burn_time
        self.fuel_load = fuel_load 
        if mass_ot is not None:
            self.mass_curve = func_from_csv(mass_ot)


    def thrust(self, t):
        return self.thrust_curve(t)
    
    def mdot(self):
        return self.mdot
