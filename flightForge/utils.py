from __future__ import annotations

from typing import Callable, Union

import numpy as np
from scipy.interpolate import interp1d


def _func_from_csv(path: str) -> tuple[interp1d, np.ndarray, np.ndarray]:
    """Load a two-column CSV and return (interpolator, x_array, y_array)."""
    x_vals: list[float] = []
    y_vals: list[float] = []

    with open(path, newline="") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            try:
                x_vals.append(float(parts[0]))
                y_vals.append(float(parts[1]))
            except ValueError:
                continue

    x_arr = np.array(x_vals)
    y_arr = np.array(y_vals)
    return (
        interp1d(x_arr, y_arr, kind="linear", fill_value=0, bounds_error=False),
        x_arr,
        y_arr,
    )


def _load_curve(
    source: Union[str, Callable[[float], float]],
) -> Callable[[float], float]:
    """Return a callable from either a CSV path or an existing callable."""
    if callable(source):
        return source
    interp, _, _ = _func_from_csv(source)
    return interp


def logarithmic_thrust(
    burn_time: float,
    peak_thrust: float,
    ramp_time: float = 0.2,
) -> Callable[[float], float]:
    """Return a thrust-vs-time callable with a linear ramp and logarithmic decay.

    Args:
        burn_time:  Total burn duration in seconds.
        peak_thrust: Peak thrust in Newtons, reached at end of ramp.
        ramp_time:  Duration of the linear ramp-up phase (default 0.2 s).
    """
    if burn_time <= 0:
        raise ValueError("burn_time must be positive.")
    if peak_thrust <= 0:
        raise ValueError("peak_thrust must be positive.")
    if ramp_time < 0 or ramp_time >= burn_time:
        raise ValueError("ramp_time must be in [0, burn_time).")

    decay_duration = burn_time - ramp_time

    def _thrust(t: float) -> float:
        if t < 0 or t > burn_time:
            return 0.0
        if t <= ramp_time:
            return peak_thrust * (t / ramp_time) if ramp_time > 0 else peak_thrust
        progress = (t - ramp_time) / decay_duration
        return peak_thrust * (1.0 - np.log1p(progress * (np.e - 1.0)))

    return _thrust


def _unit_norm(v: np.ndarray) -> np.ndarray:
    """Return the unit vector of v, or v itself if its magnitude is zero."""
    n = np.linalg.norm(v)
    return v / n if n > 0 else v
