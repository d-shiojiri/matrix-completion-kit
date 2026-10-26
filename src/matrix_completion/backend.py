"""Result contract for pluggable matrix-completion backends."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class BackendResult:
    """Fields a registered backend must return to ``MatrixCompleter``."""

    reconstruction: np.ndarray
    left_factor: np.ndarray
    right_factor: np.ndarray
    n_iter: int
    converged: bool
    objective: float
    metadata: dict[str, Any] = field(default_factory=dict)
