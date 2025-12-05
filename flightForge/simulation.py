import numpy as np
from .utils import *
import math
import matplotlib.pyplot as plt 

class _SimulationResults:
    def __init__(self, output_data):
        self.data = np.array(output_data)
        self.time = self.data[:, 0]
        
        self._raw_data = {}
        self._interpolators = {}
        
        var_specs = [
            ('x', 1, 'X Position', 'X (m)', 'blue'),
            ('y', 2, 'Y Position', 'Y (m)', 'green'),
            ('z', 3, 'Altitude', 'Altitude (m)', 'red'),
            ('vx', 4, 'X Velocity', 'Vx (m/s)', 'blue'),
            ('vy', 5, 'Y Velocity', 'Vy (m/s)', 'green'),
            ('vz', 6, 'Z Velocity', 'Vz (m/s)', 'red'),
            ('m', 7, 'Mass', 'Mass (kg)', 'purple'),
        ]
        
        for name, col, title, ylabel, color in var_specs:
            data = self.data[:, col]
            self._raw_data[name] = (data, title, ylabel, color)
            self._interpolators[name] = interp1d(
                self.time, data, kind='cubic',
                bounds_error=False, fill_value='extrapolate'
            )
    
    def get(self, var_name, t=None):
        if var_name not in self._raw_data:
            raise ValueError(f"Unknown variable: {var_name}")
        
        if t is None:
            data, title, ylabel, color = self._raw_data[var_name]
            return self._plot_variable(data, title, ylabel, color)
        else:
            return float(self._interpolators[var_name](t))
    
    def __getattr__(self, name):
        if name in self._raw_data:
            return lambda t=None: self.get(name, t)
        raise AttributeError(f"No attribute '{name}'")
    
    def _plot_variable(self, var_data, title, ylabel, color):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.time, var_data, color=color, linewidth=2)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'{title} vs Time', fontsize=14)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

class Simulation:
    def __init__(self, environment, rocket, motor, rail_length, inclination, heading, e_log=False, plotter=None):

        self.e_log = e_log
        self.env = environment
        self.motor = motor
        self.rocket = rocket
        self.plotter = plotter
        self.rail_length = rail_length
        self.inc = np.radians(inclination)
        self.heading = np.radians(heading)

        self.linear_params = {
            "apogee": None,
            "out_of_rail_velocity": None,
        }

        # unit vector representing the direction of the launch
        self.dir = np.array([
            np.cos(self.inc) * np.cos(self.heading), 
            np.cos(self.inc) * np.sin(self.heading), 
            np.sin(self.inc)
        ])

        self.events = {
            "rail_departure": None,
            "burn_out": None,
            "apogee": None,
            "parachute": None,
            "impact": None
        }
        
        self.outs = self._run()
        self.results = _SimulationResults(self.outs)

    def _t_target_interpolation(self, t, t_prev, state, state_prev, t_target):

        if t == t_prev:
            return state.copy()
    
        tau = (t_target - t_prev) / (t - t_prev)
    
        state_lin = state_prev + tau * (state - state_prev)
    
        return t_target, state_lin

    def _linear_state(self, t, t_prev, state, state_prev, i, target):

        z0 = state_prev[i]
        z1 = state[i]

        if z0 == z1:
            tau = 0.0
        else:
            tau = (target - z0) / (z1 - z0)

        tau = np.clip(tau, 0.0, 1.0)

        t_imp = t_prev + tau * (t - t_prev)
        state_lin = state_prev + tau * (state - state_prev)
        state_lin[i] = target 

        return t_imp, state_lin
    
    def _calc_speed(self, state):
        return math.sqrt(state[3]**2 + state[4]**2 + state[5]**2) 
    
    def _dump_linear_state(self, tl, sl, out):
        out.append([tl, *sl])
        
    def _cmd_log(self, t, s, si):
        print("-------------------------------------------")
        print(f"Event {si} occurred at {t:.2f} s.")
        print(f"{si} conditions:")
        print(f"(x, y, z) = ({s[0]:.2f}, {s[1]:.2f}, {s[2]:.2f}) [m]")
        print(f"(vx, vy, vz) = ({s[3]:.2f}, {s[4]:.2f}, {s[5]:.2f}) [m/s]")
        print(f"mass = {s[6]:.2f} kg")
        print("-------------------------------------------")
        

    def _event_check(self, t, t_prev, state, state_prev, out):

        tl = None
        sl = None
        state_info = ""

        if self.events["rail_departure"] is None:
            s_prev = np.dot(np.array([state_prev[0], state_prev[1], state_prev[2]]), self.dir)
            s = np.dot(np.array([state[0], state[1], state[2]]), self.dir)
            if s_prev < self.rail_length <= s:
                tl, sl = self._linear_state(t, t_prev, state, state_prev, 2, self.dir[2]*self.rail_length)
                self.events["rail_departure"] = (tl, sl)
                self.linear_params["out_of_rail_velocity"] = self._calc_speed(sl) 
                state_info = "rail_departure"

        if self.events["apogee"] is None and self.events["rail_departure"] is not None and state_prev[5] > 0 and state[5] <= 0:
            tl, sl = self._linear_state(t, t_prev, state, state_prev, 5, 0) 
            self.events["apogee"] = (tl, sl)
            self.linear_params["apogee"] = sl[2]
            state_info = "apogee"

        if self.events["burn_out"] is None and t_prev < self.motor.burn_time <= t:
            tl, sl = self._t_target_interpolation(t, t_prev, state, state_prev, self.motor.burn_time)
            self.events["burn_out"] = (tl, sl)
            state_info = "burn_out"

        if self.events["impact"] is None and state_prev[2] > 0 and state[2] <= 0 and self.events["burn_out"] != None:
            tl, sl = self._linear_state(t, t_prev, state, state_prev, 2, 0)
            self.events["impact"] = (tl, sl)
            state_info = "impact"

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
                pt = t
                if tl is not None:
                    pt = tl
                p.logged = True
                print(f"{p.name} parachute deployed at: {pt:.2f} [s]")

        if sl is not None and tl is not None:
            if self.e_log:
                self._cmd_log(tl, sl, state_info)
        
            self._dump_linear_state(tl, sl, out)

    def _compute_drag(self, rho, v_mag, cd) -> float:
        return -cd * self.rocket.ref_area * 0.5 * rho * v_mag**2

    def _dstate_dt(self, t, state):

        x, y, z, vx, vy, vz, m = state
        pos = np.array([x, y, z])
        vel = np.array([vx, vy, vz])
        
        rho = self.env.density(z)
        wind = np.array([*self.env.wind(z), 0.0])
        
        rel_v = vel - wind
        v_mag = np.linalg.norm(rel_v)
        mach = v_mag / 340  
        cd = self.rocket.e_cd(mach, self.events, z, t)
        
        on_rail = self.events["rail_departure"] is None
        v_dir = self.dir if on_rail else unit_norm(rel_v)
        
        drag = compute_vec(self._compute_drag(rho, v_mag, cd), v_dir)
        
        burning = self.events["burn_out"] is None
        thrust = compute_vec(self.motor.thrust_curve(t), v_dir) if burning else np.zeros(3)
        
        weight = m * np.array([0, 0, -9.81])
        
        total_force = thrust + drag + weight
        
        mdot = self.motor.fuel_mdot
        if not burning:
            mdot = 0
        
        if on_rail:
            vel = np.dot(vel, self.dir) * self.dir
            total_force = np.dot(total_force, self.dir) * self.dir
        
        accel = total_force / m
    
        return np.array([*vel, *accel, -mdot])  


    def _run(self, dt=0.01, t_max=200):
        t = 0.0
        state = np.array([0, 0, 0, 0, 0, 0, self.rocket.dry_mass + self.motor.fuel_load])
        output = []

        # til max iter or impact
        while t < t_max and self.events["impact"] is None:            
            state_prev = state.copy()
            t_prev = t

            output.append([t_prev, *state_prev])
  
            # RK4 step
            k1 = self._dstate_dt(t, state)
            k2 = self._dstate_dt(t + dt/2, state + k1*dt/2)
            k3 = self._dstate_dt(t + dt/2, state + k2*dt/2)
            k4 = self._dstate_dt(t + dt, state + k3*dt)
            state = state + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
            t += dt
            if self.motor.mass_curve is not None:
                if self.events["burn_out"] is None:
                    state[6] = self.motor.mass_curve(t)
                else:
                    state[6] = self.motor.mass_curve(self.motor.burn_time)
            
            self._event_check(t, t_prev, state, state_prev, output)

            if self.plotter is not None:
                self.plotter.update(t, state, self.events)

        if self.plotter is not None:
            self.plotter.finalize()

        return output   
