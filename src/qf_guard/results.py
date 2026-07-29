from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True, slots=True)
class SolverResult:
    objective: float
    runtime: float
    feasible: bool
    selected_indices: tuple[int, ...]
    sample: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.objective, bool) or not isinstance(self.objective, Real):
            raise ValueError("objective must be numeric")
        if isinstance(self.runtime, bool) or not isinstance(self.runtime, Real):
            raise ValueError("runtime must be numeric")
        objective = float(self.objective)
        runtime = float(self.runtime)
        sample = np.asarray(self.sample, dtype=np.int8)
        if not np.isfinite(objective) or not np.isfinite(runtime) or runtime < 0:
            raise ValueError("invalid solver result")
        if sample.ndim != 1 or not np.all(np.isin(sample, (0, 1))):
            raise ValueError("sample must be a binary vector")
        indices = tuple(int(index) for index in np.flatnonzero(sample))
        if tuple(self.selected_indices) != indices:
            raise ValueError("selected_indices do not match sample")
        sample.setflags(write=False)
        object.__setattr__(self, "objective", objective)
        object.__setattr__(self, "runtime", runtime)
        object.__setattr__(self, "selected_indices", indices)
        object.__setattr__(self, "sample", sample)
        object.__setattr__(self, "metadata", dict(self.metadata))
