import datetime
import os

from dotenv import load_dotenv

from flightForge import Environment, Motor, Parachute, Rocket, Simulation

load_dotenv()

api_key = os.environ.get("API_KEY")

env = Environment()
tomorrow = datetime.date.today() + datetime.timedelta(days=1)
date_info = (tomorrow.day, tomorrow.month, tomorrow.year)
env.set_model(
    api_key=api_key, model="iconEu", lat=39.389700,
    lon=-8.288964, date=date_info
)

motor = Motor(
    "curves/thrust_4kN.csv",
    10.5,
    ox_mdot=1.5155,
    initial_ox_mass=15.913,
    initial_grain_mass=2.448,
)

rocket = Rocket(45, "curves/CD_PowerOff_Mach3.csv", 0.160)
rocket.add_motor(motor)
rocket.add_parachute(Parachute("drogue", 0.7354, 1, "apogee"))
rocket.add_parachute(Parachute("main", 13.8991, 1, 450))

sim = Simulation(env, rocket, 12, 84, 144)
flight = sim.run(terminate_on="impact")
