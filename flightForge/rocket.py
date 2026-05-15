from __future__ import annotations

import math
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter

from .logger import logger
from .utils import ResultField, func_from_csv


class Rocket:
    def __init__(
        self,
        dry_mass: float,
        drag_source: Any,
        dim: float,
        e_log: bool = False,
    ) -> None:
        self.dim = dim
        self.ref_area = (dim / 2) ** 2 * math.pi
        self.dry_mass = dry_mass

        self.parachutes: list = []
        self.motor: Optional[Any] = None

        if not isinstance(drag_source, str):
            self._cd_func = drag_source.get_cd_function()
            self.mach_arr = np.linspace(0.01, 3.0, 100)
            self.cd_arr = [self._cd_func(m) for m in self.mach_arr]
            self.plot_range_locked = False
        else:
            self._cd_func, self.mach_arr, self.cd_arr = func_from_csv(drag_source)
            self.plot_range_locked = True

        self.cd = ResultField(
            np.array(self.mach_arr),
            np.array(self.cd_arr),
            "Drag Coefficient",
            "-",
            "purple",
            x_label="Mach Number",
        )

        if e_log:
            self._cmd_log()

    def e_cd(self, mach: float, events: dict, z: float, t: float) -> float:
        cd_rocket = self._cd_func(mach)
        total_drag_area = cd_rocket * self.ref_area

        for p in self.parachutes:
            if p.deploy_t is None:
                continue
            if t > p.deploy_t:
                tau = min((t - p.deploy_t) / p.lag, 1.0)
                total_drag_area += tau * p.cd_s

        return float(total_drag_area / self.ref_area)

    def add_parachute(self, chute: Any) -> None:
        self.parachutes.append(chute)

    def add_motor(self, m: Any) -> None:
        self.motor = m

    def _cmd_log(self) -> None:
        logger.info("-------- ROCKET INFO ---------")
        logger.info(f"Dry Mass:       {self.dry_mass:.2f} kg")
        logger.info(f"Reference Area: {self.ref_area:.6f} m²")
        logger.info(f"Diameter:       {self.dim:.3f} m")
        logger.info("-------------------------------")

    def plot_aerodynamics(self) -> None:
        mach = np.array(self.mach_arr)
        cd = np.array(self.cd_arr)

        min_m, max_m = float(min(mach)), float(max(mach))
        upper_limit_view = max(max_m, 3.0) if not self.plot_range_locked else max_m

        fig, axes = plt.subplots(
            4, 1,
            figsize=(12, 18),
            gridspec_kw={"height_ratios": [2, 1, 1, 1]},
            constrained_layout=True,
        )

        ax_main = axes[0]
        ax_main.plot(mach, cd, color="#4B0082", linewidth=2.5, label="Drag Coefficient ($C_d$)")

        if 0 < upper_limit_view:
            ax_main.axvspan(0, min(0.8, upper_limit_view), color="green", alpha=0.1)
            if 0.4 < upper_limit_view:
                ax_main.text(
                    min(0.4, upper_limit_view), float(min(cd)), "SUBSONIC",
                    color="green", alpha=0.5, ha="center", va="bottom",
                    fontsize=12, fontweight="bold", rotation=90,
                )

        if 0.8 < upper_limit_view:
            ax_main.axvspan(0.8, min(1.2, upper_limit_view), color="orange", alpha=0.15)
            if 1.0 < upper_limit_view:
                ax_main.text(
                    1.0, float(min(cd)), "TRANSONIC",
                    color="darkorange", alpha=0.6, ha="center", va="bottom",
                    fontsize=12, fontweight="bold", rotation=90,
                )

        if 1.2 < upper_limit_view:
            ax_main.axvspan(1.2, upper_limit_view, color="red", alpha=0.1)
            text_pos = (1.2 + upper_limit_view) / 2
            if text_pos < upper_limit_view:
                ax_main.text(
                    text_pos, float(min(cd)), "SUPERSONIC",
                    color="firebrick", alpha=0.5, ha="center", va="bottom", fontsize=12,
                    fontweight="bold",
                )

        ax_main.set_title("Full Aerodynamic Profile", fontsize=16, fontweight="bold")
        ax_main.set_ylabel("Drag Coefficient ($C_d$)", fontsize=12)
        ax_main.set_xlim(min_m, upper_limit_view)
        ax_main.grid(True, which="major", alpha=0.6)
        ax_main.legend(loc="upper right")

        def style_subplot(
            ax: plt.Axes,
            x_data: np.ndarray,
            y_data: np.ndarray,
            title: str,
            color: str,
            bg_color: str,
        ) -> None:
            if len(x_data) > 0:
                ax.plot(x_data, y_data, color=color, linewidth=2.5)
                ax.set_facecolor(bg_color)
                y_min, y_max = float(min(y_data)), float(max(y_data))
                y_span = y_max - y_min if y_max != y_min else 0.1
                ax.set_ylim(y_min - y_span * 0.1, y_max + y_span * 0.1)
                ax.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
                ax.yaxis.set_minor_locator(AutoMinorLocator(5))
                ax.xaxis.set_minor_locator(AutoMinorLocator(5))
                ax.grid(True, which="major", linestyle="-", alpha=0.7)
                ax.grid(True, which="minor", linestyle=":", alpha=0.4)
                ax.set_title(title, fontsize=14, fontweight="bold", color=color)
                ax.set_ylabel("$C_d$", fontsize=10)
            else:
                ax.text(0.5, 0.5, "No Data in Range", ha="center", va="center")
                ax.set_title(title, fontsize=14, color="grey")
                ax.set_facecolor("#f0f0f0")

        mask_sub = mach < 0.8
        style_subplot(axes[1], mach[mask_sub], cd[mask_sub], "SUBSONIC (M < 0.8)", "green", "#f0fff0")

        mask_trans = (mach >= 0.8) & (mach <= 1.2)
        style_subplot(axes[2], mach[mask_trans], cd[mask_trans], "TRANSONIC (0.8 <= M <= 1.2)", "darkorange", "#fffaf0")

        mask_super = mach > 1.2
        style_subplot(axes[3], mach[mask_super], cd[mask_super], "SUPERSONIC (M > 1.2)", "firebrick", "#fff0f0")

        axes[3].set_xlabel("Mach Number", fontsize=12)

        plt.show()
