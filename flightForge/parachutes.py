from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union


@dataclass
class Parachute:
    name: str
    cd_s: float
    lag: float
    trigger: Union[str, float]
    deploy_t: Optional[float] = field(default=None, init=False)
    logged: bool = field(default=False, init=False)
