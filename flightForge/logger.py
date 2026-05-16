import logging


logger = logging.getLogger("flightForge")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(handler)


def set_verbose(verbose: bool) -> None:
    """Set package-wide verbosity.

    Pass ``False`` to silence all informational output from Motor, Rocket,
    Environment, and Simulation. Pass ``True`` to restore it.
    """
    logger.setLevel(logging.INFO if verbose else logging.WARNING)


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
