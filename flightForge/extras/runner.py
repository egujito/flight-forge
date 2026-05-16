from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from ..flight import FlightData
from ..simulation import Simulation


@dataclass
class BaseObjects:
    """Container holding the unmodified simulation inputs.

    Deep-copied per run so overrides applied in one run cannot leak into another.
    """

    env: Any
    rocket: Any


@dataclass
class RunSpec:
    """Description of a single simulation run.

    Attributes:
        label:      Human-readable identifier used in tables and plots.
        overrides:  Mapping of dotted attribute paths (e.g. ``"rocket.dry_mass"``)
                    to the value the attribute should take for this run.
        sim_kwargs: Keyword arguments forwarded to :class:`Simulation`.
        run_kwargs: Keyword arguments forwarded to :meth:`Simulation.run`.
    """

    label: str
    overrides: dict[str, Any] = field(default_factory=dict)
    sim_kwargs: dict[str, Any] = field(default_factory=dict)
    run_kwargs: dict[str, Any] = field(default_factory=dict)


def deep_set(obj: Any, path: str, value: Any) -> None:
    """Set a nested attribute on ``obj`` using a dotted path.

    Example:
        >>> deep_set(rocket, "motor.burn_time", 4.5)
    """
    parts = path.split(".")
    target = obj
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], value)


def deep_get(obj: Any, path: str) -> Any:
    """Return a nested attribute from ``obj`` using a dotted path."""
    target = obj
    for part in path.split("."):
        target = getattr(target, part)
    return target


def _resolve_root(base: BaseObjects, path: str) -> tuple[Any, str]:
    head, _, tail = path.partition(".")
    if head == "env":
        return base.env, tail
    if head == "rocket":
        return base.rocket, tail
    raise KeyError(
        f"Unknown override root '{head}' in path '{path}'. "
        "Paths must start with 'env.' or 'rocket.' (motor is 'rocket.motor.*')."
    )


def execute_run(base: BaseObjects, spec: RunSpec) -> tuple[RunSpec, FlightData]:
    """Apply ``spec.overrides`` to a deep copy of ``base`` and run a simulation.

    This function is module-level (not a method) so it can be pickled and
    dispatched by :class:`concurrent.futures.ProcessPoolExecutor`.
    """
    local = BaseObjects(env=copy.deepcopy(base.env), rocket=copy.deepcopy(base.rocket))

    for path, value in spec.overrides.items():
        root, tail = _resolve_root(local, path)
        if not tail:
            raise KeyError(f"Override path '{path}' must include an attribute after the root.")
        deep_set(root, tail, value)

    sim = Simulation(local.env, local.rocket, **spec.sim_kwargs)
    flight = sim.run(**spec.run_kwargs)
    return spec, flight
