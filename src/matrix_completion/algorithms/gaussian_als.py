"""Gaussian-prior alternating least-squares matrix completion."""

import numpy as np

from ..backend import BackendResult
from .common import converged, mean_fill


def fit_gaussian_als(matrix, observed, rank, max_iter, tolerance, random_state,
                     ridge=1e-3, **_):
    """Fit row-centered low-rank factors with L2-regularized ALS updates."""
    del random_state
    x = np.asarray(matrix, dtype=np.float64)
    filled = mean_fill(x, observed)
    counts = observed.sum(axis=1)
    row_mean = np.divide(np.sum(np.where(observed, x, 0), axis=1), counts,
                         out=np.full(x.shape[0], np.mean(x[observed])), where=counts > 0)
    u, singular, vt = np.linalg.svd(filled - row_mean[:, None], full_matrices=False)
    root = np.sqrt(np.maximum(singular[:rank], 1e-12))
    left, right = u[:, :rank] * root, root[:, None] * vt[:rank]
    identity = np.eye(rank); previous = None; did_converge = False
    for iteration in range(1, max_iter + 1):
        for row in range(x.shape[0]):
            use = observed[row]
            if use.any():
                design = right[:, use].T
                left[row] = np.linalg.solve(design.T @ design + ridge * identity,
                                            design.T @ (x[row, use] - row_mean[row]))
        for column in range(x.shape[1]):
            use = observed[:, column]
            if use.any():
                design = left[use]
                right[:, column] = np.linalg.solve(
                    design.T @ design + ridge * identity,
                    design.T @ (x[use, column] - row_mean[use]))
        reconstruction = row_mean[:, None] + left @ right
        residual = reconstruction[observed] - x[observed]
        objective = float(np.mean(residual ** 2) + ridge *
                          (np.mean(left ** 2) + np.mean(right ** 2)))
        if converged(previous, objective, tolerance):
            did_converge = True; break
        previous = objective
    return BackendResult(reconstruction, left, right, iteration, did_converge,
                         objective, {"row_mean": row_mean})
