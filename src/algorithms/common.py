"""Initialization and convergence helpers for built-in algorithms."""

import numpy as np


def converged(previous, current, tolerance):
    if previous is None or not np.isfinite(previous) or not np.isfinite(current):
        return False
    return abs(previous - current) <= tolerance * max(1.0, abs(previous))


def mean_fill(matrix, observed):
    filled = np.asarray(matrix, dtype=np.float64).copy()
    global_mean = float(np.mean(filled[observed]))
    for row in range(filled.shape[0]):
        row_values = filled[row, observed[row]]
        filled[row, ~observed[row]] = row_values.mean() if row_values.size else global_mean
    return filled


def nndsvd(matrix, rank, epsilon):
    u, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    left = np.zeros((matrix.shape[0], rank))
    right = np.zeros((rank, matrix.shape[1]))
    left[:, 0] = np.sqrt(singular[0]) * np.abs(u[:, 0])
    right[0] = np.sqrt(singular[0]) * np.abs(vt[0])
    for component in range(1, rank):
        up, un = np.maximum(u[:, component], 0), np.maximum(-u[:, component], 0)
        vp, vn = np.maximum(vt[component], 0), np.maximum(-vt[component], 0)
        positive = np.linalg.norm(up) * np.linalg.norm(vp)
        negative = np.linalg.norm(un) * np.linalg.norm(vn)
        lobe_left, lobe_right, norm = ((up, vp, positive) if positive >= negative
                                       else (un, vn, negative))
        if norm > epsilon:
            scale = np.sqrt(singular[component] * norm)
            left[:, component] = scale * lobe_left / max(np.linalg.norm(lobe_left), epsilon)
            right[component] = scale * lobe_right / max(np.linalg.norm(lobe_right), epsilon)
    return np.maximum(left, epsilon), np.maximum(right, epsilon)
