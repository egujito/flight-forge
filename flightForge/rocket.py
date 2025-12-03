from .utils import func_from_csv
import math 

class Rocket:
    def __init__(self, dry_mass, drag_source, dim):

        self.ref_area = (dim/2)**2 * math.pi
        self.dry_mass = dry_mass
        self._cd_func = func_from_csv(drag_source)
        self.parachutes = []

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
