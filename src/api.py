"""Single public interface for the bundled matrix-completion algorithms."""

import numpy as np

from .algorithms import (fit_funk_svd, fit_gamma_poisson_nmf, fit_gaussian_als,
                         fit_nmf_frobenius, fit_nmf_kl)
from .result import CompletionResult
from .validation import validate_input, validate_rank

_METHODS = {
    "gaussian_als": fit_gaussian_als,
    "funk_svd": fit_funk_svd,
    "nmf_frobenius": fit_nmf_frobenius,
    "nmf_kl": fit_nmf_kl,
    "gamma_poisson_nmf": fit_gamma_poisson_nmf,
}
_ALIASES = {"bayesian_pca": "gaussian_als", "bayesian_nmf": "gamma_poisson_nmf"}


class MatrixCompleter:
    """Fit one bundled low-rank method and fill a partially observed matrix."""

    def __init__(self, method="gaussian_als", rank=2, *, max_iter=200,
                 tolerance=1e-5, random_state=42, preserve_observed=True,
                 ridge=1e-3, learning_rate=0.005, regularization=0.02,
                 epsilon=1e-8, gamma_shape=1.0, gamma_rate=1.0):
        self.method = method
        self.rank = rank
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.random_state = random_state
        self.preserve_observed = preserve_observed
        self.ridge = ridge
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.epsilon = epsilon
        self.gamma_shape = gamma_shape
        self.gamma_rate = gamma_rate

    def fit(self, matrix, observed=None):
        values, mask = validate_input(matrix, observed)
        validate_rank(self.rank, values.shape)
        if self.max_iter < 1 or self.tolerance < 0:
            raise ValueError("max_iter must be positive and tolerance nonnegative")
        if self.ridge < 0 or self.regularization < 0:
            raise ValueError("ridge and regularization must be nonnegative")
        if self.learning_rate <= 0 or self.epsilon <= 0:
            raise ValueError("learning_rate and epsilon must be positive")
        if self.gamma_shape <= 0 or self.gamma_rate <= 0:
            raise ValueError("gamma_shape and gamma_rate must be positive")
        canonical = _ALIASES.get(str(self.method).lower(), str(self.method).lower())
        if canonical not in _METHODS:
            raise ValueError(f"unknown method {self.method!r}; choose from {tuple(_METHODS)}")
        fitted = _METHODS[canonical](
            values, mask, self.rank, self.max_iter, self.tolerance, self.random_state,
            ridge=self.ridge, learning_rate=self.learning_rate,
            regularization=self.regularization, epsilon=self.epsilon,
            gamma_shape=self.gamma_shape, gamma_rate=self.gamma_rate)
        reconstruction = np.asarray(fitted.reconstruction, dtype=np.float64)
        if reconstruction.shape != values.shape or not np.all(np.isfinite(reconstruction)):
            raise RuntimeError("algorithm returned an invalid reconstruction")
        completed = reconstruction.copy()
        if self.preserve_observed:
            completed[mask] = values[mask]
        self.result_ = CompletionResult(
            reconstruction=reconstruction,
            completed=completed,
            left_factor=np.asarray(fitted.left_factor),
            right_factor=np.asarray(fitted.right_factor),
            n_iter=fitted.n_iter,
            converged=fitted.converged,
            objective=fitted.objective,
            method=canonical,
            metadata=fitted.metadata)
        return self

    def fit_transform(self, matrix, observed=None):
        """Fit the selected method and return a copy of the completed matrix."""
        return self.fit(matrix, observed).result_.completed.copy()
