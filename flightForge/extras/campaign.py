from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Iterable, Optional

from ..flight import FlightData
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
