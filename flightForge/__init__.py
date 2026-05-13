import logging

from .environment import Environment
from .motor import Motor
from .rocket import Rocket
from .simulation import Simulation
from .parachutes import Parachute
from .plotting import LivePlotter

# Setup logger
logger = logging.getLogger("flightForge")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(handler)

__all__ = [
    "Environment",
    "Motor",
    "Rocket",
    "Simulation",
    "Parachute",
    "LivePlotter",
    "logger",
]
