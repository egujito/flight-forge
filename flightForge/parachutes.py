from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union


@dataclass
class Parachute:
    """Parachute deployment model with signal and opening events."""

    name: str
    cd_s: float
    lag: float
    trigger: Union[str, float]
    signal_t: Optional[float] = field(default=None, init=False)
    opening_t: Optional[float] = field(default=None, init=False)
    logged: bool = field(default=False, init=False)
