import numpy as np

class Environment:  
    def __init__(self, wind_profile=None, rho_profile=None):
        
        self.wind_profile = wind_profile
        self.rho_profile = rho_profile
        if wind_profile == None:
            self.wind_profile = self._def_wind_profile
        if rho_profile == None:
            self.rho_profile = self._def_rho_profile

    def density(self, h):
        return self.rho_profile(h)
    
    def wind(self, h):
        return self.wind_profile(h)
    

    @staticmethod
    def _def_wind_profile(h):
        # Magnitude increases with height
        speed = 3 + 0.01 * h           # 3 m/s at ground, +0.01 m/s per meter

        # Direction rotates clockwise with altitude
        angle = np.radians(10 + 0.02*h)  # 10° at ground, veers 0.02° per meter

        wx = speed * np.cos(angle)  # x = north
        wy = speed * np.sin(angle)  # y = east

        return (wx, wy)

    @staticmethod
    def _def_rho_profile(h):
        rho0 = 1.225
        H = 8500.0
        return rho0 * np.exp(-h / H)
