from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Union

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class FixedParam:
    """A single deterministic value applied to every run."""

    value: Any


@dataclass(frozen=True)
class SweepParam:
    """A discrete array of values, one per run."""

    values: tuple


@dataclass(frozen=True)
class StochasticParam:
    """A distribution sampled once per run.

    Attributes:
        sampler:     Callable ``(rng) -> value`` used for random Monte Carlo.
        ppf:         Optional inverse CDF ``(u in [0, 1]) -> value`` enabling
                     Latin Hypercube sampling. ``None`` if the distribution
                     does not expose a usable quantile function.
        description: Short human-readable summary used in reports.

    Resolution happens at spec-build time so worker processes never see the
    sampler or ppf themselves — only the concrete values they produce.
    """

    sampler: Callable[[np.random.Generator], Any]
    ppf: Optional[Callable[[float], Any]] = None
    description: str = ""


ParamLike = Union[FixedParam, SweepParam, StochasticParam, Any]


class Param:
    """Factory for parameter specifications used by :class:`Campaign`.

    Use the class methods below to build a parameter spec; never instantiate
    :class:`Param` directly.

    Examples:
        >>> Param.fixed(4.5)
        >>> Param.sweep(np.linspace(3000, 5000, 10))
        >>> Param.normal(mu=4.2, sigma=0.1)
        >>> Param.uniform(lo=0.0, hi=10.0)
        >>> Param.from_dist(scipy.stats.truncnorm(...))
    """

    @staticmethod
    def fixed(value: Any) -> FixedParam:
        """Return a fixed-value parameter applied identically to every run."""
        return FixedParam(value)

    @staticmethod
    def sweep(values: Iterable) -> SweepParam:
        """Return a deterministic sweep over the provided iterable of values."""
        seq = tuple(values)
        if len(seq) == 0:
            raise ValueError("sweep values must be non-empty.")
        return SweepParam(seq)

    @staticmethod
    def normal(mu: float, sigma: float) -> StochasticParam:
        """Return a gaussian-distributed parameter with mean ``mu`` and std ``sigma``."""
        if sigma < 0:
            raise ValueError("sigma must be non-negative.")
        return StochasticParam(
            sampler=lambda rng: float(rng.normal(mu, sigma)),
            ppf=lambda u: float(stats.norm.ppf(u, loc=mu, scale=sigma)),
            description=f"N({mu}, {sigma})",
        )

    @staticmethod
    def uniform(lo: float, hi: float) -> StochasticParam:
        """Return a uniform-distributed parameter on the half-open interval [lo, hi)."""
        if hi <= lo:
            raise ValueError("hi must be greater than lo.")
        span = hi - lo
        return StochasticParam(
            sampler=lambda rng: float(rng.uniform(lo, hi)),
            ppf=lambda u: float(lo + span * u),
            description=f"U[{lo}, {hi})",
        )

    @staticmethod
    def from_dist(dist: Any) -> StochasticParam:
        """Wrap any object exposing an ``rvs(random_state=...)`` method.

        Compatible with ``scipy.stats`` frozen distributions. If ``dist`` also
        exposes a ``ppf`` method, the parameter becomes usable with Latin
        Hypercube sampling.
        """
        if not hasattr(dist, "rvs"):
            raise TypeError("dist must expose an 'rvs' method (e.g. scipy.stats frozen rv).")
        ppf = (lambda u: float(dist.ppf(u))) if hasattr(dist, "ppf") else None
        return StochasticParam(
            sampler=lambda rng: float(dist.rvs(random_state=rng)),
            ppf=ppf,
            description=str(dist),
        )
