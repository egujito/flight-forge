import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


#Slow performance, the ideal solution is to use SimulationResults plots
class LivePlotter:
    def __init__(self, update_interval=1000):
        self.update_interval = update_interval
        self.iteration = 0
        
        self.time = []
        self.x = []
        self.y = []
        self.z = []
        self.velocity = []
        
        self.events = {
            'rail_departure': None,
            'burn_out': None,
            'apogee': None
        }
        self.event_indices = {
            'rail_departure': None,
            'burn_out': None,
            'apogee': None
        }
        
        self._create_3d_trajectory_window()
        self._create_altitude_window()
        self._create_velocity_window()
        
        plt.ion()  
    
    def _create_3d_trajectory_window(self):

        self.fig_3d = plt.figure(figsize=(10, 8))
        self.fig_3d.canvas.manager.set_window_title('3D Trajectory')
        
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')
        self.line_3d, = self.ax_3d.plot([], [], [], 'b-', linewidth=2, label='Trajectory')
        self.point_3d, = self.ax_3d.plot([], [], [], 'ro', markersize=8, label='Current Position')
        
        self.rail_marker_3d, = self.ax_3d.plot([], [], [], 'go', markersize=10, label='Rail Departure')
        self.burnout_marker_3d, = self.ax_3d.plot([], [], [], 'yo', markersize=10, label='Burn Out')
        self.apogee_marker_3d, = self.ax_3d.plot([], [], [], 'mo', markersize=12, label='Apogee')
        
        self.ax_3d.set_xlabel('X (m)')
        self.ax_3d.set_ylabel('Y (m)')
        self.ax_3d.set_zlabel('Altitude (m)')
        self.ax_3d.set_title('Rocket 3D Trajectory')
        
        self.ax_3d.view_init(elev=20, azim=45)
        
        self.ax_3d.legend(loc='upper left')
        self.ax_3d.grid(True, alpha=0.3)
        
        plt.tight_layout()
    
    def _create_altitude_window(self):
        self.fig_alt = plt.figure(figsize=(10, 6))
        self.fig_alt.canvas.manager.set_window_title('Altitude vs Time')
        
        self.ax_alt = self.fig_alt.add_subplot(111)
        self.line_alt, = self.ax_alt.plot([], [], 'g-', linewidth=2)
        
        self.rail_line_alt = self.ax_alt.axvline(x=0, color='g', linestyle='--', linewidth=1.5, 
                                                   label='Rail Departure')
        self.burnout_line_alt = self.ax_alt.axvline(x=0, color='orange', linestyle='--', linewidth=1.5, 
                                                      label='Burn Out')
        self.apogee_line_alt = self.ax_alt.axvline(x=0, color='m', linestyle='--', linewidth=1.5, 
                                                     label='Apogee')
        
        self.ax_alt.set_xlabel('Time (s)', fontsize=12)
        self.ax_alt.set_ylabel('Altitude (m)', fontsize=12)
        self.ax_alt.set_title('Altitude vs Time', fontsize=14)
        self.ax_alt.grid(True, alpha=0.3)
        self.ax_alt.legend(loc='upper left')
        
        plt.tight_layout()
    
    def _create_velocity_window(self):
        """Create velocity vs time plot window."""
        self.fig_vel = plt.figure(figsize=(10, 6))
        self.fig_vel.canvas.manager.set_window_title('Velocity vs Time')
        
        self.ax_vel = self.fig_vel.add_subplot(111)
        self.line_vel, = self.ax_vel.plot([], [], 'r-', linewidth=2)
        
        # Event lines
        self.rail_line_vel = self.ax_vel.axvline(x=0, color='g', linestyle='--', linewidth=1.5, 
                                                   label='Rail Departure')
        self.burnout_line_vel = self.ax_vel.axvline(x=0, color='orange', linestyle='--', linewidth=1.5, 
                                                      label='Burn Out')
        self.apogee_line_vel = self.ax_vel.axvline(x=0, color='m', linestyle='--', linewidth=1.5, 
                                                     label='Apogee')
        
        self.ax_vel.set_xlabel('Time (s)', fontsize=12)
        self.ax_vel.set_ylabel('Velocity (m/s)', fontsize=12)
        self.ax_vel.set_title('Velocity Magnitude vs Time', fontsize=14)
        self.ax_vel.grid(True, alpha=0.3)
        self.ax_vel.legend(loc='upper right')
        
        plt.tight_layout()
    def update(self, t, state, events):
        self.iteration += 1

        x, y, z, vx, vy, vz, m = state
        vel_mag = np.linalg.norm([vx, vy, vz])

        self.time.append(t)
        self.x.append(x)
        self.y.append(y)
        self.z.append(z)
        self.velocity.append(vel_mag)

        for event_name in ['rail_departure', 'burn_out', 'apogee']:
            ev = events.get(event_name)
            if ev is not None and self.events[event_name] is None:
                if isinstance(ev, (tuple, list)):
                    ev_time = ev[0]
                else:
                    ev_time = ev

                self.events[event_name] = ev_time

                idx = self._find_time_index(ev_time)
                if idx is None:
                    idx = len(self.time) - 1
                self.event_indices[event_name] = idx

        if self.iteration % self.update_interval == 0:
            self._redraw()

   
    def _redraw(self):
        time_array = np.array(self.time)
        x_array = np.array(self.x)
        y_array = np.array(self.y)
        z_array = np.array(self.z)
        vel_array = np.array(self.velocity)
        
        self._update_3d_plot(x_array, y_array, z_array)
        
        self._update_altitude_plot(time_array, z_array)
        
        self._update_velocity_plot(time_array, vel_array)
        
        self.fig_3d.canvas.draw()
        self.fig_alt.canvas.draw()
        self.fig_vel.canvas.draw()
        
        plt.pause(0.001)
    
    def _update_3d_plot(self, x_array, y_array, z_array):
        self.line_3d.set_data(x_array, y_array)
        self.line_3d.set_3d_properties(z_array)
        
        if len(x_array) > 0:
            self.point_3d.set_data([x_array[-1]], [y_array[-1]])
            self.point_3d.set_3d_properties([z_array[-1]])
        
        if self.event_indices['rail_departure'] is not None:
            idx = self.event_indices['rail_departure']
            self.rail_marker_3d.set_data([x_array[idx]], [y_array[idx]])
            self.rail_marker_3d.set_3d_properties([z_array[idx]])
        
        if self.event_indices['burn_out'] is not None:
            idx = self.event_indices['burn_out']
            self.burnout_marker_3d.set_data([x_array[idx]], [y_array[idx]])
            self.burnout_marker_3d.set_3d_properties([z_array[idx]])
        
        if self.event_indices['apogee'] is not None:
            idx = self.event_indices['apogee']
            self.apogee_marker_3d.set_data([x_array[idx]], [y_array[idx]])
            self.apogee_marker_3d.set_3d_properties([z_array[idx]])
        
        if len(x_array) > 0:
            x_range = x_array.max() - x_array.min() if x_array.max() != x_array.min() else 1
            y_range = y_array.max() - y_array.min() if y_array.max() != y_array.min() else 1
            z_max = z_array.max()
            
            self.ax_3d.set_xlim(x_array.min() - 0.1*x_range, x_array.max() + 0.1*x_range)
            self.ax_3d.set_ylim(y_array.min() - 0.1*y_range, y_array.max() + 0.1*y_range)
            self.ax_3d.set_zlim(0, z_max * 1.1)
    
    def _update_altitude_plot(self, time_array, z_array):
        self.line_alt.set_data(time_array, z_array)
        self.ax_alt.relim()
        self.ax_alt.autoscale_view()
        
        if self.events['rail_departure'] is not None:
            self.rail_line_alt.set_xdata([self.events['rail_departure']])
            self.rail_line_alt.set_visible(True)
        
        if self.events['burn_out'] is not None:
            self.burnout_line_alt.set_xdata([self.events['burn_out']])
            self.burnout_line_alt.set_visible(True)
        
        if self.events['apogee'] is not None:
            self.apogee_line_alt.set_xdata([self.events['apogee']])
            self.apogee_line_alt.set_visible(True)
    
    def _update_velocity_plot(self, time_array, vel_array):
        self.line_vel.set_data(time_array, vel_array)
        self.ax_vel.relim()
        self.ax_vel.autoscale_view()
        
        if self.events['rail_departure'] is not None:
            self.rail_line_vel.set_xdata([self.events['rail_departure']])
            self.rail_line_vel.set_visible(True)
        
        if self.events['burn_out'] is not None:
            self.burnout_line_vel.set_xdata([self.events['burn_out']])
            self.burnout_line_vel.set_visible(True)
        
        if self.events['apogee'] is not None:
            self.apogee_line_vel.set_xdata([self.events['apogee']])
            self.apogee_line_vel.set_visible(True)
    
    def _find_time_index(self, event_time):
        if len(self.time) == 0:
            return None
        time_array = np.array(self.time)
        idx = np.argmin(np.abs(time_array - event_time))
        return idx
    
    def finalize(self):
        plt.ioff()
        self._redraw()  
        plt.show()
    
    def close(self):
        plt.close(self.fig_3d)
        plt.close(self.fig_alt)
        plt.close(self.fig_vel)