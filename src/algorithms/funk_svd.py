"""Biased Funk-SVD trained only on observed entries."""

import numpy as np

from ..result import _Factorization
from .common import converged


def fit_funk_svd(matrix, observed, rank, max_iter, tolerance, random_state,
                 learning_rate=0.005, regularization=0.02, **_):
    x = np.asarray(matrix, dtype=np.float64)
    rows, columns = np.nonzero(observed)
    ratings = x[rows, columns]
    rng = np.random.default_rng(random_state)
    order = np.arange(ratings.size)
    row_bias = np.zeros(x.shape[0]); column_bias = np.zeros(x.shape[1])
    left = rng.normal(0, .05, (x.shape[0], rank))
    right = rng.normal(0, .05, (rank, x.shape[1]))
    global_mean = float(ratings.mean()); previous = None; did_converge = False
    for iteration in range(1, max_iter + 1):
        rng.shuffle(order)
        for position in order:
            row, column = rows[position], columns[position]
            prediction = (global_mean + row_bias[row] + column_bias[column]
                          + left[row] @ right[:, column])
            error = ratings[position] - prediction
            row_bias[row] += learning_rate * (error - regularization * row_bias[row])
            column_bias[column] += learning_rate * (
                error - regularization * column_bias[column])
            old_left = left[row].copy()
            left[row] += learning_rate * (error * right[:, column]
                                          - regularization * left[row])
            right[:, column] += learning_rate * (error * old_left
                                                  - regularization * right[:, column])
        fitted = (global_mean + row_bias[rows] + column_bias[columns]
                  + np.sum(left[rows] * right[:, columns].T, axis=1))
        objective = float(np.mean((ratings - fitted) ** 2))
        if converged(previous, objective, tolerance):
            did_converge = True; break
        previous = objective
    reconstruction = (global_mean + row_bias[:, None] + column_bias[None, :]
                      + left @ right)
    return _Factorization(reconstruction, left, right, iteration, did_converge,
                          objective, {"global_mean": global_mean,
                                     "row_bias": row_bias,
                                     "column_bias": column_bias})
