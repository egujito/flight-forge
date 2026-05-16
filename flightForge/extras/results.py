from __future__ import annotations

from typing import Any, Iterator, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..flight import FlightData
from .runner import RunSpec


class CampaignResults:
    """Container for the output of :meth:`Campaign.run`.

    Wraps a list of ``(spec, flight)`` pairs and exposes:

    - :meth:`summary` — pandas DataFrame with key flight metrics per run.
    - :meth:`get` — look up a single flight by run label.
    - :meth:`plot_envelope` — percentile-banded plot of any flight channel.

    Supports ``len()``, iteration, and integer/label indexing for direct access.
    """

    def __init__(self, runs: list[tuple[RunSpec, FlightData]]) -> None:
        self.runs = list(runs)

    def __len__(self) -> int:
        return len(self.runs)

    def __iter__(self) -> Iterator[tuple[RunSpec, FlightData]]:
        return iter(self.runs)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.runs[key]
        if isinstance(key, str):
            return self.get(key)
        raise TypeError(f"Index must be int or str, got {type(key).__name__}.")

    def get(self, label: str) -> FlightData:
        """Return the :class:`FlightData` for the run with the given label."""
        for spec, flight in self.runs:
            if spec.label == label:
                return flight
        raise KeyError(f"No run with label '{label}'.")

    def summary(self) -> pd.DataFrame:
        """Return a DataFrame of override values and key flight metrics per run.

        Columns include every override path encountered across runs, plus
        ``apogee_m``, ``apogee_t``, ``max_speed_ms``, ``max_mach``,
        ``max_accel_ms2``, ``final_t``, ``final_x``, ``final_y``, ``final_z``,
        and ``final_range_m``.
        """
        rows = []
        for spec, flight in self.runs:
            row: dict[str, Any] = {"label": spec.label}
            row.update(spec.overrides)
            row.update(_flight_metrics(flight))
            rows.append(row)
        return pd.DataFrame(rows)

    def plot_envelope(
        self,
        channel: str = "z",
        x_channel: str = "t",
        percentiles: tuple[float, float] = (5.0, 95.0),
        n_points: int = 200,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """Plot median and percentile band of a flight channel across all runs.

        Args:
            channel:     :class:`FlightData` attribute to plot on the y-axis.
            x_channel:   :class:`FlightData` attribute used as the common axis
                         (typically ``"t"``).
            percentiles: Lower and upper percentile of the shaded band.
            n_points:    Resolution of the resampled grid.
            ax:          Optional existing axis to draw into.
        """
        if not self.runs:
            raise RuntimeError("No runs to plot.")
        lo_pct, hi_pct = percentiles

        x_min = max(float(getattr(f, x_channel).min()) for _, f in self.runs)
        x_max = min(float(getattr(f, x_channel).max()) for _, f in self.runs)
        if x_max <= x_min:
            raise RuntimeError(
                f"Runs have no overlapping range on '{x_channel}'; cannot build envelope."
            )
        grid = np.linspace(x_min, x_max, n_points)

        stacked = np.vstack(
            [
                np.interp(grid, getattr(f, x_channel), getattr(f, channel))
                for _, f in self.runs
            ]
        )
        median = np.median(stacked, axis=0)
        lo = np.percentile(stacked, lo_pct, axis=0)
        hi = np.percentile(stacked, hi_pct, axis=0)

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 5))
        ax.fill_between(grid, lo, hi, alpha=0.25, label=f"{lo_pct:g}–{hi_pct:g}%")
        ax.plot(grid, median, color="black", linewidth=2.0, label="median")
        ax.set_xlabel(x_channel)
        ax.set_ylabel(channel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        return ax


def _flight_metrics(flight: FlightData) -> dict[str, float]:
    apogee_idx = int(np.argmax(flight.z))
    return {
        "apogee_m": float(flight.z[apogee_idx]),
        "apogee_t": float(flight.t[apogee_idx]),
        "max_speed_ms": float(np.max(flight.speed)),
        "max_mach": float(np.max(flight.mach)),
        "max_accel_ms2": float(np.max(flight.acceleration)),
        "final_t": float(flight.t[-1]),
        "final_x": float(flight.x[-1]),
        "final_y": float(flight.y[-1]),
        "final_z": float(flight.z[-1]),
        "final_range_m": float(np.hypot(flight.x[-1], flight.y[-1])),
    }
