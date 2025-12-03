from flightForge import Environment, Motor, Rocket, Simulation, LivePlotter, Parachute

env = Environment()
motor = Motor("thrust(2).csv", 4.2, 9, 1.8, fuel_mass_ot="mass.csv")
rocket = Rocket(40.8, "MaCd.csv", 0.163)
rocket.add_parachute(Parachute("drogue", 0.7354, 1, "apogee"))
rocket.add_parachute(Parachute("main", 13.8991, 1, 450))


sim = Simulation(env, motor, rocket, 12, 84, 144, e_log = True) # Add plotter=LivePlotter(update_interval=x) to see live plotting

import numpy as np

np.savetxt("output.csv", sim.outs, delimiter=",")
