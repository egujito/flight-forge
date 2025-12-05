from flightForge import Environment, Motor, Rocket, Simulation, LivePlotter, Parachute

env = Environment()
motor = Motor("curves/thrust(2).csv", 4.2, 9, 1.8, mass_ot="curves/mass.csv")
rocket = Rocket(40.8, "curves/MaCd.csv", 0.163)
rocket.add_motor(motor)
rocket.add_parachute(Parachute("drogue", 0.7354, 1, "apogee"))
rocket.add_parachute(Parachute("main", 13.8991, 1, 450))


sim = Simulation(env, rocket, 12, 84, 144, e_log=True, plotter=LivePlotter(update_interval=1000)) # Add plotter=LivePlotter(update_interval=x) to see live plotting

