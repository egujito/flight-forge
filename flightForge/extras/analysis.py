from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse

from .results import CampaignResults

_METRIC_COLUMNS = (
    "apogee_m",
    "apogee_t",
    "max_speed_ms",
    "max_mach",
    "max_accel_ms2",
    "final_t",
    "final_x",
    "final_y",
    "final_z",
    "final_range_m",
)


def landing_scatter(
    results: CampaignResults,
    ax: Optional[plt.Axes] = None,
    n_sigma: tuple[float, ...] = (1.0, 2.0),
    show_centroid: bool = True,
) -> plt.Axes:
    """Scatter plot of landing positions with Gaussian dispersion ellipses.

    The ellipses are drawn from the eigendecomposition of the (x, y) landing
    covariance and represent constant-Mahalanobis-distance contours.
    """
    df = results.summary()
    x = df["final_x"].to_numpy()
    y = df["final_y"].to_numpy()

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(x, y, s=20, alpha=0.6, color="tab:blue", label="impact")

    if len(x) >= 2:
        cov = np.cov(x, y)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = eigvals.argsort()[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        angle = float(np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0])))
        cx, cy = float(np.mean(x)), float(np.mean(y))
        for k in n_sigma:
            w = 2 * k * float(np.sqrt(max(eigvals[0], 0.0)))
            h = 2 * k * float(np.sqrt(max(eigvals[1], 0.0)))
            ax.add_patch(
                Ellipse(
                    (cx, cy), width=w, height=h, angle=angle,
                    fill=False, edgecolor="firebrick", linewidth=1.3,
                    label=f"{k:g}σ",
                )
            )
        if show_centroid:
            ax.scatter([cx], [cy], marker="x", color="black", s=60, label="centroid")

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("Impact X (m)")
    ax.set_ylabel("Impact Y (m)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    return ax


def param_correlation(
    results: CampaignResults,
    metrics: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Return Pearson correlations between override parameters and flight metrics.

    Rows are override paths; columns are metric names. Useful for ranking which
    inputs most strongly drive each output.
    """
    df = results.summary()
    param_cols = [c for c in df.columns if c not in _METRIC_COLUMNS and c != "label"]
    metric_cols = metrics if metrics is not None else list(_METRIC_COLUMNS)

    if not param_cols:
        raise RuntimeError("No override parameters found in results to correlate.")

    numeric_params = [c for c in param_cols if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_params:
        raise RuntimeError("No numeric override parameters to correlate.")

    return df[numeric_params + metric_cols].corr().loc[numeric_params, metric_cols]


def sensitivity_tornado(
    results: CampaignResults,
    metric: str = "apogee_m",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Tornado chart ranking each override parameter by its correlation to ``metric``.

    Bars show Pearson r; the chart is sorted by |r| descending so the strongest
    drivers appear at the top.
    """
    corr = param_correlation(results, metrics=[metric])[metric]
    corr = corr.reindex(corr.abs().sort_values(ascending=True).index)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, max(3.0, 0.5 * len(corr))))

    colors = ["tab:red" if v < 0 else "tab:blue" for v in corr.values]
    ax.barh(corr.index, corr.values, color=colors, edgecolor="black", linewidth=0.5)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel(f"Pearson r vs {metric}")
    ax.set_xlim(-1.05, 1.05)
    ax.grid(True, axis="x", alpha=0.3)
    return ax


def apogee_histogram(
    results: CampaignResults,
    bins: int = 30,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Histogram of apogee altitude across all runs."""
    df = results.summary()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["apogee_m"], bins=bins, edgecolor="black", linewidth=0.5, color="tab:blue", alpha=0.7)
    p5, p50, p95 = np.percentile(df["apogee_m"], [5, 50, 95])
    for x, label in [(p5, "P5"), (p50, "P50"), (p95, "P95")]:
        ax.axvline(x, linestyle="--", color="black", linewidth=1.0)
        ax.text(x, ax.get_ylim()[1] * 0.95, f" {label}={x:.0f}", va="top", fontsize=9)
    ax.set_xlabel("Apogee (m)")
    ax.set_ylabel("Runs")
    ax.grid(True, alpha=0.3)
    return ax
