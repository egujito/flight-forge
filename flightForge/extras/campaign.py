from __future__ import annotations

import itertools
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Iterable, Optional

import numpy as np
from scipy.stats import qmc

from ..flight import FlightData
from .param import StochasticParam, SweepParam
from .runner import BaseObjects, RunSpec, execute_run


class Campaign:
    """Batch driver for running many simulations against a shared base setup.

    A campaign captures a base :class:`Environment` and :class:`Rocket`, plus the
    constructor and run keyword arguments common to all simulations. Use
    :meth:`sweep` or :meth:`add_run` to enqueue individual runs, then call
    :meth:`run` to execute them in parallel.
    """

    def __init__(
        self,
        environment: Any,
        rocket: Any,
        sim_kwargs: dict[str, Any],
        run_kwargs: Optional[dict[str, Any]] = None,
        label: str = "campaign",
    ) -> None:
        """Build a campaign description.

        Args:
            environment: Base :class:`Environment`. Deep-copied per run.
            rocket:      Base :class:`Rocket`. Deep-copied per run.
            sim_kwargs:  Keyword arguments forwarded to :class:`Simulation`
                         (``rail_length``, ``inclination``, ``heading``).
            run_kwargs:  Keyword arguments forwarded to :meth:`Simulation.run`.
            label:       Human-readable name for the campaign.
        """
        self.environment = environment
        self.rocket = rocket
        self.sim_kwargs = dict(sim_kwargs)
        self.run_kwargs = dict(run_kwargs) if run_kwargs else {}
        self.label = label
        self.specs: list[RunSpec] = []

    def sweep(
        self,
        path: str,
        values: Iterable,
        label_fmt: str = "{path}={value:.3g}",
    ) -> "Campaign":
        """Enqueue one run per value, overriding ``path`` with that value.

        Args:
            path:      Dotted attribute path beginning with ``"env."`` or ``"rocket."``.
            values:    Iterable of concrete values, one per run.
            label_fmt: ``str.format`` template with ``path``, ``value`` and ``i`` keys.

        Returns:
            ``self`` to allow chaining.
        """
        seq = list(values)
        if not seq:
            raise ValueError("sweep values must be non-empty.")
        for i, v in enumerate(seq):
            self.specs.append(
                RunSpec(
                    label=_format_label(label_fmt, path, v, i),
                    overrides={path: v},
                    sim_kwargs=self.sim_kwargs,
                    run_kwargs=self.run_kwargs,
                )
            )
        return self

    def sweep_multiple(
        self,
        params: dict[str, Any],
        mode: str = "grid",
        n: Optional[int] = None,
        seed: Optional[int] = None,
        label_fmt: str = "run{i:04d}",
    ) -> "Campaign":
        """Enqueue runs that vary several attributes at once.

        Args:
            params: Mapping from dotted attribute path to one of:

                - an iterable of concrete values (treated as a deterministic
                  sweep over those values);
                - a :class:`SweepParam` (same semantics, explicit form);
                - a :class:`StochasticParam` (sampled per run when
                  ``mode`` is ``"random"`` or ``"lhs"``).

            mode: How values are combined across parameters:

                - ``"zip"``  — paired; all entries must have equal length.
                - ``"grid"`` — full Cartesian product across all entries.
                - ``"random"`` — independent random sampling; every entry
                  must be a :class:`StochasticParam`. ``n`` runs are produced.
                - ``"lhs"`` — Latin Hypercube; every entry must be a
                  :class:`StochasticParam` with a ``ppf`` defined.
                  ``n`` runs are produced.

            n:         Number of runs for ``"random"`` and ``"lhs"`` modes.
            seed:      Seed for the random generator (reproducible sampling).
            label_fmt: ``str.format`` template with key ``i`` (run index).

        Returns:
            ``self`` to allow chaining.
        """
        if not params:
            raise ValueError("params must contain at least one entry.")
        if mode not in ("zip", "grid", "random", "lhs"):
            raise ValueError(f"Invalid mode '{mode}'. Use 'zip', 'grid', 'random', or 'lhs'.")

        rng = np.random.default_rng(seed)

        if mode in ("zip", "grid"):
            value_lists = {p: _as_sweep_values(v) for p, v in params.items()}
            if mode == "zip":
                rows = _zip_rows(value_lists)
            else:
                rows = _grid_rows(value_lists)
        else:
            if n is None or n <= 0:
                raise ValueError(f"mode '{mode}' requires a positive integer 'n'.")
            stoch = {p: _as_stochastic(v, p) for p, v in params.items()}
            if mode == "random":
                rows = _random_rows(stoch, n, rng)
            else:
                rows = _lhs_rows(stoch, n, rng)

        for i, overrides in enumerate(rows):
            self.specs.append(
                RunSpec(
                    label=label_fmt.format(i=i),
                    overrides=overrides,
                    sim_kwargs=self.sim_kwargs,
                    run_kwargs=self.run_kwargs,
                )
            )
        return self

    def add_run(self, overrides: dict[str, Any], label: str = "") -> "Campaign":
        """Enqueue a single run with the given attribute overrides.

        Args:
            overrides: Mapping of dotted attribute paths to override values.
            label:     Optional label; auto-generated from ``overrides`` if empty.

        Returns:
            ``self`` to allow chaining.
        """
        if not label:
            label = ", ".join(f"{k}={v!r}" for k, v in overrides.items()) or "run"
        self.specs.append(
            RunSpec(
                label=label,
                overrides=dict(overrides),
                sim_kwargs=self.sim_kwargs,
                run_kwargs=self.run_kwargs,
            )
        )
        return self

    def clear(self) -> "Campaign":
        """Drop all enqueued specs without affecting the base setup."""
        self.specs.clear()
        return self

    def run(
        self,
        n_workers: int = 4,
        show_progress: bool = True,
    ) -> list[tuple[RunSpec, FlightData]]:
        """Execute every enqueued spec and return the raw ``(spec, flight)`` pairs.

        Runs are executed concurrently via :class:`ProcessPoolExecutor` when
        ``n_workers > 1``. Pass ``n_workers=1`` to run sequentially (useful when
        overrides contain non-picklable callables, or for debugging).
        """
        if not self.specs:
            raise RuntimeError("No runs enqueued. Use .sweep() or .add_run() first.")

        base = BaseObjects(env=self.environment, rocket=self.rocket)
        total = len(self.specs)
        results: list[tuple[RunSpec, FlightData]] = []

        if n_workers <= 1:
            for i, spec in enumerate(self.specs):
                results.append(execute_run(base, spec))
                if show_progress:
                    _print_progress(i + 1, total)
        else:
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futures = [pool.submit(execute_run, base, s) for s in self.specs]
                for i, future in enumerate(as_completed(futures)):
                    results.append(future.result())
                    if show_progress:
                        _print_progress(i + 1, total)

        if show_progress:
            sys.stdout.write("\n")
            sys.stdout.flush()

        results.sort(key=lambda r: self.specs.index(r[0]))
        return results


def _as_sweep_values(spec: Any) -> list:
    if isinstance(spec, SweepParam):
        return list(spec.values)
    if isinstance(spec, StochasticParam):
        raise TypeError(
            "StochasticParam is not valid in 'zip'/'grid' mode. "
            "Use mode='random' or mode='lhs', or convert to a deterministic sweep."
        )
    seq = list(spec)
    if not seq:
        raise ValueError("sweep entry must contain at least one value.")
    return seq


def _as_stochastic(spec: Any, path: str) -> StochasticParam:
    if isinstance(spec, StochasticParam):
        return spec
    raise TypeError(
        f"Param '{path}' must be a StochasticParam (Param.normal, Param.uniform, "
        f"Param.from_dist) when using mode='random' or mode='lhs'."
    )


def _zip_rows(value_lists: dict[str, list]) -> list[dict[str, Any]]:
    lengths = {p: len(v) for p, v in value_lists.items()}
    n = next(iter(lengths.values()))
    if any(L != n for L in lengths.values()):
        raise ValueError(f"zip mode requires equal lengths, got: {lengths}")
    paths = list(value_lists.keys())
    return [{p: value_lists[p][i] for p in paths} for i in range(n)]


def _grid_rows(value_lists: dict[str, list]) -> list[dict[str, Any]]:
    paths = list(value_lists.keys())
    combos = itertools.product(*(value_lists[p] for p in paths))
    return [dict(zip(paths, combo)) for combo in combos]


def _random_rows(
    stoch: dict[str, StochasticParam],
    n: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    return [{p: param.sampler(rng) for p, param in stoch.items()} for _ in range(n)]


def _lhs_rows(
    stoch: dict[str, StochasticParam],
    n: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    missing = [p for p, param in stoch.items() if param.ppf is None]
    if missing:
        raise ValueError(
            f"LHS requires a 'ppf' on every StochasticParam; missing on: {missing}"
        )
    paths = list(stoch.keys())
    sampler = qmc.LatinHypercube(d=len(paths), rng=rng)
    quantiles = sampler.random(n=n)
    return [
        {
            p: stoch[p].ppf(float(quantiles[i, j]))  # type: ignore[misc]
            for j, p in enumerate(paths)
        }
        for i in range(n)
    ]


def _format_label(fmt: str, path: str, value: Any, i: int) -> str:
    try:
        return fmt.format(path=path, value=value, i=i)
    except (ValueError, TypeError):
        return f"{path}=#{i}"


def _print_progress(done: int, total: int) -> None:
    bar_len = 24
    filled = int(bar_len * done / total)
    sys.stdout.write(
        f"\r[{'#' * filled}{'-' * (bar_len - filled)}] {done}/{total}"
    )
    sys.stdout.flush()
