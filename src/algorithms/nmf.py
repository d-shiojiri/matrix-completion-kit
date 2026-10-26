"""Mask-aware NMF with Frobenius, KL, and Gamma-Poisson objectives."""

import numpy as np

from ..result import _Factorization
from .common import converged, mean_fill, nndsvd


def fit_nmf_frobenius(matrix, observed, rank, max_iter, tolerance, random_state, **kwargs):
    return _fit_multiplicative(matrix, observed, rank, max_iter, tolerance,
                               random_state, divergence="frobenius", **kwargs)


def fit_nmf_kl(matrix, observed, rank, max_iter, tolerance, random_state, **kwargs):
    return _fit_multiplicative(matrix, observed, rank, max_iter, tolerance,
                               random_state, divergence="kl", **kwargs)


def _fit_multiplicative(matrix, observed, rank, max_iter, tolerance, random_state,
                        divergence, epsilon=1e-8, **_):
    del random_state
    x, mask = _nonnegative_data(matrix, observed)
    numeric_mask = mask.astype(np.float64)
    left, right = nndsvd(np.maximum(mean_fill(x, mask), 0) + epsilon, rank, epsilon)
    previous = None; did_converge = False
    for iteration in range(1, max_iter + 1):
        estimate = np.maximum(left @ right, epsilon)
        if divergence == "frobenius":
            left *= ((numeric_mask * x) @ right.T) / np.maximum(
                (numeric_mask * estimate) @ right.T, epsilon)
        else:
            left *= ((numeric_mask * x / estimate) @ right.T) / np.maximum(
                numeric_mask @ right.T, epsilon)
        left = np.maximum(left, epsilon)
        estimate = np.maximum(left @ right, epsilon)
        if divergence == "frobenius":
            right *= (left.T @ (numeric_mask * x)) / np.maximum(
                left.T @ (numeric_mask * estimate), epsilon)
        else:
            right *= (left.T @ (numeric_mask * x / estimate)) / np.maximum(
                left.T @ numeric_mask, epsilon)
        right = np.maximum(right, epsilon)
        reconstruction = left @ right
        objective = _objective(x, mask, reconstruction, divergence, epsilon)
        if converged(previous, objective, tolerance):
            did_converge = True; break
        previous = objective
    return _Factorization(reconstruction, left, right, iteration, did_converge,
                          objective, {"divergence": divergence})


def fit_gamma_poisson_nmf(matrix, observed, rank, max_iter, tolerance,
                          random_state, epsilon=1e-8, gamma_shape=1.0,
                          gamma_rate=1.0, **_):
    del random_state
    x, mask = _nonnegative_data(matrix, observed)
    numeric_mask = mask.astype(np.float64)
    left, right = nndsvd(np.maximum(mean_fill(x, mask), 0) + epsilon, rank, epsilon)
    previous = None; did_converge = False
    for iteration in range(1, max_iter + 1):
        estimate = np.maximum(left @ right, epsilon)
        left = np.maximum((gamma_shape + left * ((numeric_mask * x / estimate) @ right.T)) /
                          np.maximum(gamma_rate + numeric_mask @ right.T, epsilon), epsilon)
        estimate = np.maximum(left @ right, epsilon)
        right = np.maximum((gamma_shape + right * (left.T @ (numeric_mask * x / estimate))) /
                           np.maximum(gamma_rate + left.T @ numeric_mask, epsilon), epsilon)
        reconstruction = left @ right
        objective = _objective(x, mask, reconstruction, "kl", epsilon)
        if converged(previous, objective, tolerance):
            did_converge = True; break
        previous = objective
    return _Factorization(reconstruction, left, right, iteration, did_converge,
                          objective, {"divergence": "kl", "gamma_shape": gamma_shape,
                                      "gamma_rate": gamma_rate})


def _nonnegative_data(matrix, observed):
    if np.any(np.asarray(matrix)[observed] < 0):
        raise ValueError("NMF methods require nonnegative observed values")
    return np.where(observed, np.asarray(matrix, dtype=np.float64), 0), observed


def _objective(x, observed, reconstruction, divergence, epsilon):
    truth, estimate = x[observed], reconstruction[observed]
    if divergence == "frobenius":
        return float(np.mean((truth - estimate) ** 2))
    terms = np.where(truth > 0, truth * np.log(np.maximum(truth, epsilon) / estimate)
                     - truth + estimate, estimate)
    return float(np.mean(terms))
