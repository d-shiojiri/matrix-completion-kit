"""Input normalization shared by all matrix-completion methods."""

import numpy as np


def validate_input(matrix, observed=None):
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("matrix must be a nonempty two-dimensional array")
    mask = np.isfinite(values) if observed is None else np.asarray(observed, dtype=bool)
    if mask.shape != values.shape:
        raise ValueError("observed mask must have the same shape as matrix")
    if not mask.any():
        raise ValueError("at least one entry must be observed")
    if np.any(~np.isfinite(values[mask])):
        raise ValueError("entries marked observed must be finite")
    return values, mask


def validate_rank(rank, shape):
    if not isinstance(rank, (int, np.integer)) or not 1 <= rank <= min(shape):
        raise ValueError(f"rank must be an integer from 1 to {min(shape)}")
