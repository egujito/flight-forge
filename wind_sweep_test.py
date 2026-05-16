"""
Wind sweep: 2D grid of surface wind (u0, v0) ∈ [−9, 9] m/s.
Profile shape: power law — W(h) = W0 * (1 + h / 500)^(1/7).
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from flightForge import Environment, Motor, Parachute, Rocket
from flightForge import utils
from flightForge.extras import Campaign


class PowerLawWindProfile:
    """
    Power-law atmospheric wind profile, picklable for multiprocessing.

    W(h) = W0 * (1 + h / H_REF) ** ALPHA
    At h=0 the profile equals the surface value W0.
    """

    ALPHA = 1.0 / 7.0
    H_REF = 500.0

    def __init__(self, u0: float, v0: float) -> None:
        self.u0 = u0
        self.v0 = v0

    def __call__(self, h: float) -> tuple[float, float]:
        scale = (1.0 + h / self.H_REF) ** self.ALPHA
        return (self.u0 * scale, self.v0 * scale)


def main() -> None:
    env = Environment()

    motor = Motor(
        utils.logarithmic_thrust(9.7, 4000),
        burn_time=9.7,
        ox_mdot=1.517,
        initial_ox_mass=14.72,
        initial_grain_mass=2.65,
    )

    rocket = Rocket(43.56, "curves/CD_PowerOff_Mach3.csv", 0.160)
    rocket.add_motor(motor)
    rocket.add_parachute(Parachute("drogue", 0.7354, 1, "apogee"))
    rocket.add_parachute(Parachute("main", 13.8991, 1, 450))

    WIND_VALS = np.linspace(-9, 9, 36)

    campaign = Campaign(
        environment=env,
        rocket=rocket,
        sim_kwargs={"rail_length": 12, "inclination": 84, "heading": 144},
        run_kwargs={"terminate_on": "apogee"},
        label="wind_sweep",
    )

    for u0 in WIND_VALS:
        for v0 in WIND_VALS:
            campaign.add_run(
                overrides={"env.wind_profile": PowerLawWindProfile(u0, v0)},
                label=f"u={u0:+.0f}_v={v0:+.0f}",
            )

    results = campaign.run(n_workers=1)

    df = results.summary()
    apogee = df["apogee_m"].to_numpy()

    N = len(WIND_VALS)
    apogee_grid = apogee.reshape(N, N)
    ext = (float(WIND_VALS[0]), float(WIND_VALS[-1]), float(WIND_VALS[0]), float(WIND_VALS[-1]))
    
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "smooth_blue_green", 
        ["#e6f2ff", "#418bf0", "#00b37e", "#00ff66"]
    )

    _, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(
        apogee_grid, origin="lower", extent=ext, aspect="auto",
        cmap=cmap, interpolation="bicubic",
    )
    plt.colorbar(im, ax=ax, label="Apogee (m)")
    ax.set_title("Apogee vs surface wind  —  power-law profile (α = 1/7)")
    ax.set_xlabel("U surface (m/s)")
    ax.set_ylabel("V surface (m/s)")
    ax.axhline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.axvline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig("wind_sweep_results.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()