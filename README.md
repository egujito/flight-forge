## 🚀 Flight Forge: 3DOF Rocket Flight Simulator

![](img/3dof-example.png)

Flight Forge is a Python-based open-source library designed for simulating the flight dynamics of high-power rockets. It utilizes a 3 Degrees of Freedom (3DOF) physics engine to simulate translational motion ($x, y, z$) while treating the rocket as a variable-mass point object.

-----

### Why 3DOF?

In the early stages of rocket design, detailed inertia tensors are often unknown. A 3DOF simulator is crucial for rapidly iterating on:

  * Apogee prediction: Estimating peak altitude based on mass and drag.
  * Velocity profiling: Ensuring the rocket leaves the launch rail at a safe velocity.
  * Recovery planning: Predicting terminal velocities for parachutes with set configuration.

-----

### Core Classes

The framework is built around four main interacting classes that define the simulation components.

#### 1\. Environment

The Environment class handles atmospheric conditions, including air density and wind profiles. It integrates directly with the Windy.com API to fetch real-world forecast data for specific locations and dates.

  * Custom Models: You can define a specific location (lat, lon), model (e.g., 'gfs'), and even a specific date to retrieve historical or forecast weather data.
  * Profiles: It generates altitude-dependent profiles for density ($\rho$) and wind vectors ($\vec{u}$, $\vec{v}$).

-----

#### 2\. Motor

The Motor class models the propulsion system, supporting both Solid and Hybrid engines. It handles the math to determine mass depletion in the system.

##### Performance Metrics

The engine's performance is derived from the provided thrust curve and propellant mass:

  * Total Impulse ($I_{total}$): Calculated by integrating the thrust curve over time.

$$
I_{total} = \int_{0}^{t_{burn}} Thrust(t) dt
$$

  * Effective Exhaust Velocity ($v_e$): The average velocity of exhaust gases, derived from the total impulse and total propellant mass ($M_p = m_{ox} + m_{grain}$).

$$
v_e = \frac{I_{total}}{M_p}
$$

  * Total Mass Flow Rate ($\dot{m}_{tot}$): The instantaneous rate at which mass is ejected from the rocket, assumed proportional to thrust for a constant $v_e$.

$$
\dot{m}_{tot}(t) = \frac{Thrust(t)}{v_e}
$$

##### Hybrid vs. Solid Propulsion

The system distinguishes between motor types based on the oxidizer mass (ox\_mass):

  * Solid Motors: ox\_mass = 0. The entire flow rate comes from the solid grain.
  * Hybrid Motors: ox\_mass \> 0. You must define a constant oxidizer flow rate (ox\_mdot). The system derives the solid grain regression rate ($\dot{m}_{grain}$) by subtracting the oxidizer flow from the total flow required to match the thrust curve.

$$
\dot{m}_{grain}(t) = \dot{m}_{tot}(t) - \dot{m}_{ox}
$$

Note on Negative Grain Flow: In hybrid configurations, if the thrust $F(t)$ drops significantly (e.g., during startup or shutdown) while the constant oxidizer flow $\dot{m}_{ox}$ continues, the calculated $\dot{m}_{grain}$ may become negative. This physical interpretation implies that unburnt oxidizer is accumulating in the chamber or that the simplified constant $v_e$ model assumes a higher efficiency than is occurring at that moment.

-----

#### 3\. Rocket

The Rocket class defines the vehicle's physical properties, including:

  * Dry Mass: Mass of the rocket without propellant.
  * Drag Coefficient ($C_d$): Defined as a function of Mach number via CSV input.
  * Diameter: To calculate reference area.

The flight simulation runs as soon as the class is initialized.

-----

#### 4\. Simulation

The Simulation class orchestrates the interaction between the Environment, Motor, and Rocket. It initializes the physics engine, sets the launch rail parameters (length, inclination, heading), and executes the time-stepping loop.

-----

### Core Physics & Simulation Engine

#### Equations of Motion (EOM)

Flight Forge solves the translational EOM using a Runge-Kutta 4th Order (RK4) numerical integrator. This method samples the state derivatives at four points within each time step ($dt$) to ensure high accuracy.

The state vector is defined as:

$$
Y = [r_x, r_y, r_z, v_x, v_y, v_z, m]
$$

Forces considered include:

  * $\vec{F}_{thrust}$

  * $\vec{F}_{drag}$ 

  * $\vec{F}_{gravity}$

#### Event Detection & Linearization

The simulator actively monitors for discrete events such as Rail Departure, Apogee, and Burnout.

To maintain precision independent of the time step size ($dt$), the engine employs State Linearization. When an event is detected between steps $t_i$ and $t_{i+1}$ (e.g., $v_z$ changes from positive to negative), the engine calculates the exact time fraction $\tau$ where the event occurred:

$$
\tau = \frac{Target - Z_0}{Z_1 - Z_0}
$$

An interpolated state $S_{event}$ is then generated and inserted into the output log:

$$
S_{event} = S_{prev} + \tau (S_{curr} - S_{prev})
$$

-----

#### Outputs

Flight Forge is designed for seamless integration with Jupyter Notebooks. The sim.results object provides a dynamic interface to access and plot simulation data.

  * Quick Plotting: Simply call the variable attribute as a function.

```python
sim.results.vz()  # Plots vertical velocity vs time
sim.results.z()   # Plots altitude vs time
```

  * Data Interpolation: Retrieve precise values at any arbitrary time $t$, even if it wasn't a simulation step.

```python
# Get velocity exactly at t=5.0s
vel_at_5 = sim.results.vz(5.0)
```

  * Phase Plotting: Plot one variable against another.

```python
sim.results.plot_vs('x', 'z') # Plot trajectory (Z vs X)
```
All features:

```python
sim.results.x(),
            .y()
            .z()
            .vx()
            .vy()
            .vz()
            .speed() (#magnitude)
            .ax
            .ay
            .az
            .acceleration (#magnitude)
            .mass()


motor.thrust()
motor.grain_mdot()
motor.total_mdot()
rocket.drag() #only populated after simulation runs
rocket.cd()

```

### Example Usage: (Jupyter Notebook)

```python
from flightForge import Environment, Motor, Rocket, Simulation, LivePlotter, Parachute
from dotenv import load_dotenv
import os
import datetime
```


```python
load_dotenv()
api_key = os.environ.get("API_KEY")
env = Environment(e_log=True)
tomorrow = datetime.date.today() + datetime.timedelta(days=1)
date_info = (tomorrow.day, tomorrow.month, tomorrow.year)
env.set_model(api_key=api_key, model="iconEu", lat=39.389700, lon=-8.288964, date=date_info)
```

    -------ENVIRONMENT INFO --------
    Coordinates:   39.3897, -8.288964
    Model Used:    iconEu
    Surface Wind:  U=-2.73 m/s, V=1.56 m/s
                   Mag=3.14 m/s
    --------------------------------


![png](example_files/example_1_1.png)
    
```python
motor = Motor("curves/thrust(2).csv", burn_time=4.2, ox_mass=7.33, ox_mdot=1.5, grain_mass=3, e_log=True)
```

    -------Hybrid MOTOR INFO --------
    Oxidizer Mass: 7.33 kg
    Grain Mass:    3 kg
    Total Impulse: 14543.04 Ns
    Eff. Exhaust Velocity (Ve): 1407.84 m/s
    ------------------------------------



```python
rocket = Rocket(40.8, "curves/MaCd.csv", 0.163)
rocket.add_parachute(Parachute("drogue", 0.7354, 1, "apogee"))
rocket.add_parachute(Parachute("main", 13.8991, 1, 450))
rocket.add_motor(motor)
```


```python
sim = Simulation(env, rocket, 12, 84, 144, e_log=True) 
```

    -------------------------------------------
    Event rail_departure occurred at 0.78 s.
    rail_departure conditions:
    (x, y, z) = (-1.01, 0.74, 11.93) [m]
    (vx, vy, vz) = (-3.11, 2.26, 36.53) [m/s]
    mass = 49.54 kg
    -------------------------------------------
    -------------------------------------------
    Event burn_out occurred at 4.20 s.
    burn_out conditions:
    (x, y, z) = (-72.91, 73.99, 498.92) [m]
    (vx, vy, vz) = (-40.05, 42.19, 254.16) [m/s]
    mass = 41.01 kg
    -------------------------------------------
    -------------------------------------------
    Event apogee occurred at 26.15 s.
    apogee conditions:
    (x, y, z) = (-811.45, 857.95, 3125.65) [m]
    (vx, vy, vz) = (-29.19, 31.14, 0.00) [m/s]
    mass = 40.99 kg
    -------------------------------------------
    drogue parachute deployed at: 26.15 [s]
    main parachute deployed at: 111.70 [s]
    -------------------------------------------
    Event impact occurred at 176.61 s.
    impact conditions:
    (x, y, z) = (-685.50, 724.15, 0.00) [m]
    (vx, vy, vz) = (1.56, -2.73, -6.69) [m/s]
    mass = 40.99 kg
    -------------------------------------------

```python
sim.results.trajectory_3d()
```

![](example_files/example_5_0.png)

Or in a python script (main.py)

```python
from flightForge import Environment, Motor, Rocket, Simulation, LivePlotter, Parachute
from dotenv import load_dotenv
import os
import datetime

load_dotenv()

api_key = os.environ.get("API_KEY")

# Add e_log=True to any core class to get more outputs

env = Environment()
tomorrow = datetime.date.today() + datetime.timedelta(days=1)
date_info = (tomorrow.day, tomorrow.month, tomorrow.year)
env.set_model(api_key=api_key, model="gfs", lat=39.389700, lon=-8.288964, date=date_info)

motor = Motor("curves/thrust(2).csv", 4.2, ox_mass=7.33, ox_mdot=1.5, grain_mass=3)

rocket = Rocket(40.8, "curves/MaCd.csv", 0.163)
rocket.add_motor(motor)
rocket.add_parachute(Parachute("drogue", 0.7354, 1, "apogee"))
rocket.add_parachute(Parachute("main", 13.8991, 1, 450))

# sim = Simulation(env, rocket, 12, 84, 144, e_log=True, plotter=LivePlotter()) # Add plotter=LivePlotter(update_interval=x) to see live plotting
sim = Simulation(env, rocket, 12, 84, 144, e_log=True) # Add plotter=LivePlotter(update_interval=x) to see live plotting
#sim.results.z()
#sim.results.vz()
# (...)
```
