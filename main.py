from flightForge import Environment, Motor, Rocket, Simulation, LivePlotter, Parachute
from dotenv import load_dotenv
import os
import datetime

load_dotenv()

api_key = os.environ.get("API_KEY")

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
