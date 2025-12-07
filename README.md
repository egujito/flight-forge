# **🚀 Flight Forge**

![](img/3dof-example.png)

### **3DOF Rocket Flight Simulator**

**Flight Forge** is a Python-based open-source library designed for simulating the flight dynamics of high-power rockets. It utilizes a 3 Degrees of Freedom (3DOF) physics engine to simulate translational motion ($x, y, z$) while treating the rocket as a variable-mass point object.

This tool is optimized for the early design phase of rocketry projects, allowing the prediction of the apogee, velocity profiles, and recovery events before inertia tensors are defined.

## **Core Concepts**

### **Physics & Linearization**

Flight Forge solves the Equations of Motion (EOM) using a Runge-Kutta 4th Order (RK4) numerical integrator. This provides high accuracy for the rocket's trajectory by sampling derivatives at four points within each time step ($dt$).

The simulator employs a State Vector ($state$) representing the rocket's instantaneous condition:

$$
Y = [r_x, r_y, r_z, v_x, v_y, v_z, m]
$$

A key feature of Flight Forge is state linearization. When an event occurs between two time steps (e.g., the rocket crosses the rail length or hits apogee), the engine does not simply take the nearest step. Instead, it calculates the exact fraction of time ($\tau$) required to hit the target condition assuming linear change between steps $t_{i}$ and $t_{i+1}$. This interpolated state is added to the output of the simulation, so we keep the previous step, the interpolated step and the next step stored in the output array.  

* Method: `_linear_state` calculates $\tau = (target - z_0) / (z_1 - z_0)$.
* Result: The simulation generates an "interpolated state" at the exact micro-second the event occurred, ensuring high-precision logs for deployment and stage transitions.

### **Propulsion**

You can define if the motor is hybrid or solid simply by adjusting the parameter `ox_mass`. The `ox_mass` parameter should be 0 for solid motors (default option). If it is not 0, the parameter `ox_mdot` should be adjusted. This represents the oxidizer flow rate to the combustion chamber. 

The instantaneous mass change rate (solid + liquid (if hybrid)) is calculated by the diving the thrust at that time by the effective exhaust velocity:

$$
\dot{m_t} = \frac{Thrust(t)}{v_e} [kg/s]
$$

Where:

$$
v_e = \frac{I_t}{M_p} [m/s]
$$

And:

$$
M_p = m_l + m_s
$$

### **Event Detection**

The engine actively monitors for critical flight phases using specific triggers:

* Rail Departure: interpolation of position vector along the launch guide.
* Burnout: Time-based interpolation matching the motor's burn duration.
* Apogee: Detected when vertical velocity ($v_z$) crosses zero (positive to negative).
* Impact: Detected when altitude ($z$) returns to zero after burnout.
* Parachute Deployment: Triggered by specific events ("apogee") or altitude thresholds (e.g., 450m).

-----

## **Workflow**

Flight Forge is designed with a modular, object-oriented workflow (inspired by RocketPY). You build the rocket subsystem by subsystem.

1.  Environment: Define the atmosphere (wind and density profiles).
2.  Motor: Load thrust curves (CSV) and define fuel properties (mass, burn time). Must then be added to the Rocket.
3.  Rocket: Define the rocket diameter, ($C_d$ vs Mach from CSV) and dry mass.
4.  Parachutes: Attach recovery devices to the Rocket object, defining the $cd_s$ ($C_d * A_p$), the lag (time until fully deployed) and trigger event (can be height in meters or an event like "apogee").
5.  **Simulation**: combine all objects into the engine. And define the launch rail length, inclination and heading. 

⚠️ The simulation runs immediately upon initialization of this class.

-----

🔍 Flight Forge is built to integrate seamlessly with Jupyter Notebooks, offering two distinct ways to visualize data.

### **1. Live Plotting**

Useful for demonstrations or debugging long simulations. You can watch the flight in real-time as the math executes.

* How to use: Pass a `LivePlotter` instance to the `Simulation` constructor.
* Behavior: Opens interactive `matplotlib` windows (3D Trajectory, Altitude, Velocity) that update every $N$ iterations.

```python
from flightForge import LivePlotter
# Updates plot every 10 steps
sim = Simulation(..., plotter=LivePlotter(update_interval=1000))
```

⚠️ Should only be used in a .py script. Do not use live plotting on a jupyter notebook.

### **2. Post-Flight Analysis (Static Plotting)**

Once the simulation completes, the results are stored in `sim.results`. This object uses a dynamic access system that allows you to either plot a variable or get its value at a specific time using the same attribute name.

The available variables are: `x`, `y`, `z` (altitude), `vx`, `vy`, `vz`, `m` (mass).

  * Generate a Plot: Call the attribute as a function with no arguments.

    ```python
    # Plots Vertical Velocity (Vz) vs Time
    sim.results.vz()
    ```

  * Get Specific Value: Call the attribute with a time argument. The system uses cubic interpolation to give you the precise value at that time, even if it wasn't an exact simulation step.

    ```python
    # Returns the vertical velocity exactly at t=120.0 seconds
    v_ex = sim.results.vz(120)
    print(v_ex) # returns x m/s
    ```

-----

## **Installation**

Clone the repository and install dependencies (requires `numpy`, `scipy`, `matplotlib`).

```bash
git clone [https://github.com/egujito/flight-forge.git](https://github.com/egujito/flight-forge.git)
cd flight-forge
pip install numpy matplotlib scipy
```

-----

## **Example Usage**

```python
from flightForge import Environment, Motor, Rocket, Simulation, LivePlotter, Parachute

# 1. Setup Environment
env = Environment()

# 2. Define Motor
# (Thrust CSV, Fuel Mass, Burn Time, Flow Rate, Mass Curve CSV (Optional))
motor = Motor("curves/thrust(2).csv", 4.2, 9, 1.8, mass_ot="curves/mass.csv")

# 3. Define Rocket
# (Dry Mass, Cd vs Mach CSV, Diameter)
rocket = Rocket(40.8, "curves/MaCd.csv", 0.163)

# 4. Add motor to rocket
rocket.add_motor(motor)

# 5. Add Recovery System
rocket.add_parachute(Parachute("drogue", 0.7354, 1, "apogee"))
rocket.add_parachute(Parachute("main", 13.8991, 1, 450)) # Deploys at 450m

# 6. Run Simulation
# (env, motor, rocket, rail_len, inclination, heading, logging)
sim = Simulation(env, rocket, 12, 84, 144, e_log=True)

# 7. Analyze Results
sim.results.z()       # Plots Altitude vs Time
print(sim.results.vz(10.5)) # Prints velocity at t=10.5s
```
