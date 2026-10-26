"""Scikit-learn-like facade shared by every completion backend."""

import numpy as np

from .algorithms import (fit_funk_svd, fit_gamma_poisson_nmf, fit_gaussian_als,
                         fit_nmf_frobenius, fit_nmf_kl)
from .registry import register_method, resolve_method
from .result import CompletionResult
from .validation import validate_input, validate_rank


class MatrixCompleter:
    """Fit and apply a mask-aware low-rank matrix-completion model.

    Missing entries may be represented by ``NaN`` or by ``observed=False``.
    ``fit_transform`` returns a completed matrix that keeps observed values
    unchanged; ``result_.reconstruction`` contains the pure low-rank estimate.
    """

    def __init__(self, method="gaussian_als", rank=2, *, max_iter=200,
                 tolerance=1e-5, random_state=42, preserve_observed=True,
                 **method_options):
        self.method = method
        self.rank = rank
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.random_state = random_state
        self.preserve_observed = preserve_observed
        self.method_options = dict(method_options)

    def fit(self, matrix, observed=None):
        values, mask = validate_input(matrix, observed)
        validate_rank(self.rank, values.shape)
        if self.max_iter < 1 or self.tolerance < 0:
            raise ValueError("max_iter must be positive and tolerance nonnegative")
        canonical, backend = resolve_method(self.method)
        fitted = backend(values, mask, self.rank, self.max_iter, self.tolerance,
                         self.random_state, **self.method_options)
        reconstruction = np.asarray(fitted.reconstruction, dtype=np.float64)
        if reconstruction.shape != values.shape or not np.all(np.isfinite(reconstruction)):
            raise RuntimeError("completion backend returned an invalid reconstruction")
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
            metadata=fitted.metadata,
        )
        self.observed_mask_ = mask.copy()
        self.shape_ = values.shape
        return self

    def fit_transform(self, matrix, observed=None):
        """Fit the requested method and return the completed matrix."""
        return self.fit(matrix, observed).result_.completed.copy()

    def get_result(self):
        if not hasattr(self, "result_"):
            raise RuntimeError("fit must be called before get_result")
        return self.result_


def _register_builtins():
    register_method("gaussian_als", fit_gaussian_als, aliases=("bayesian_pca",))
    register_method("funk_svd", fit_funk_svd)
    register_method("nmf_frobenius", fit_nmf_frobenius)
    register_method("nmf_kl", fit_nmf_kl)
    register_method("gamma_poisson_nmf", fit_gamma_poisson_nmf,
                    aliases=("bayesian_nmf",))


_register_builtins()
