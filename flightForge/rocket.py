from .utils import func_from_csv, bcolors, ResultField
import math 
import numpy as np
from typing import Optional

class Rocket:
    def __init__(self, dry_mass, drag_source, dim, e_log=False):
        self.dim = dim
        self.ref_area = (dim/2)**2 * math.pi
        self.dry_mass = dry_mass
        
        self._cd_func, self.mach_arr, self.cd_arr = func_from_csv(drag_source)
        
        self.parachutes = []
        self.motor = None
        
        self.cd = ResultField(np.array(self.mach_arr), np.array(self.cd_arr), 
                              "Drag Coefficient", "-", "purple", x_label="Mach Number")
        
        self.drag: Optional[ResultField] = None

        if e_log:
            self._cmd_log()

    def e_cd(self, mach, events, z, t):
        cd_rocket = self._cd_func(mach)
        total_drag_area = cd_rocket * self.ref_area
        
        for p in self.parachutes:
            if p.deploy_t is not None:
                tau = min((t - p.deploy_t) / p.lag, 1.0)
                total_drag_area += tau * p.cd_s
        
        return total_drag_area / self.ref_area

    def add_parachute(self, chute):
        self.parachutes.append(chute)

    def add_motor(self, m):
        self.motor = m

    def _cmd_log(self):
        print(f"-------- ROCKET INFO ---------")
        print(f"Dry Mass:       {self.dry_mass:.2f} kg")
        print(f"Reference Area: {self.ref_area:.6f} m²")
        print(f"Diameter:       {self.dim:.3f} m")
        print(f"-------------------------------")
