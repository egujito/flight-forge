from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Union

import numpy as np


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

    The sampler is a callable taking a numpy ``Generator`` and returning a
    concrete value. Resolution happens at spec-build time so the worker
    processes never see the sampler itself.
    """

    sampler: Callable[[np.random.Generator], Any]
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
            description=f"N({mu}, {sigma})",
        )

    @staticmethod
    def uniform(lo: float, hi: float) -> StochasticParam:
        """Return a uniform-distributed parameter on the half-open interval [lo, hi)."""
        if hi <= lo:
            raise ValueError("hi must be greater than lo.")
        return StochasticParam(
            sampler=lambda rng: float(rng.uniform(lo, hi)),
            description=f"U[{lo}, {hi})",
        )

    @staticmethod
    def from_dist(dist: Any) -> StochasticParam:
        """Wrap any object exposing an ``rvs(random_state=...)`` method.

        Compatible with ``scipy.stats`` frozen distributions.
        """
        if not hasattr(dist, "rvs"):
            raise TypeError("dist must expose an 'rvs' method (e.g. scipy.stats frozen rv).")
        return StochasticParam(
            sampler=lambda rng: float(dist.rvs(random_state=rng)),
            description=str(dist),
        )
