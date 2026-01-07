from flightForge import Environment, Motor, Rocket, Simulation, Parachute
from dotenv import load_dotenv
import os
import datetime

load_dotenv()

api_key = os.environ.get("API_KEY")

env = Environment()
tomorrow = datetime.date.today() + datetime.timedelta(days=1)
date_info = (tomorrow.day, tomorrow.month, tomorrow.year)
env.set_model(api_key=api_key, model="iconEu", lat=39.389700, lon=-8.288964, date=date_info)

motor = Motor("curves/thrust(2).csv", 4.2, ox_mdot=1.5, initial_ox_mass=7.33, initial_grain_mass=3, e_log=True)

rocket = Rocket(40.8, "curves/MaCd.csv", 0.163, e_log=True)
rocket.add_motor(motor)
rocket.add_parachute(Parachute("drogue", 0.7354, 1, "apogee"))
rocket.add_parachute(Parachute("main", 13.8991, 1, 450))

sim = Simulation(env, rocket, 12, 84, 144, e_log=True) 

#TODO: predict_cd -> prediction of the cd based on definition of aero surfaces