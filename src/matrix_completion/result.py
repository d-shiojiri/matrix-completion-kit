"""Public immutable result returned by every completion method."""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class CompletionResult:
    """Low-rank reconstruction, completed data, factors, and fit diagnostics."""

    reconstruction: np.ndarray
    completed: np.ndarray
    left_factor: np.ndarray
    right_factor: np.ndarray
    n_iter: int
    converged: bool
    objective: float
    method: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
