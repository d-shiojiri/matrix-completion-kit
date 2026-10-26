# matrix-completion-kit

A small, domain-independent Python library for completing partially observed
two-dimensional matrices with a unified estimator interface.

## Installation

```bash
python -m pip install -e .
python -m pip install -e '.[test]'
```

## Quick start

```python
import numpy as np
from matrix_completion import MatrixCompleter

x = np.array([
    [1.0, np.nan, 3.0],
    [2.0, 4.0, 6.0],
    [3.0, 6.0, np.nan],
])

model = MatrixCompleter(
    method="gaussian_als",
    rank=2,
    max_iter=300,
    tolerance=1e-6,
    random_state=42,
)
completed = model.fit_transform(x)
result = model.get_result()

print(completed)
print(result.objective, result.converged)
print(result.left_factor.shape, result.right_factor.shape)
```

An explicit Boolean mask can be supplied when missing entries contain a finite
placeholder:

```python
completed = model.fit_transform(x_with_placeholder, observed=quality_mask)
```

By default, `completed` preserves every observed value and replaces only
missing entries. `result.reconstruction` is the unconstrained low-rank estimate
at both observed and missing positions.

## Built-in methods

| `method` | Model/objective | Data restriction |
|---|---|---|
| `gaussian_als` | Row-centered, L2-regularized alternating least squares | Signed values allowed |
| `funk_svd` | Biased latent factors fitted by SGD | Signed values allowed |
| `nmf_frobenius` | Mask-aware NMF with squared-error objective | Observed values ≥ 0 |
| `nmf_kl` | Mask-aware NMF with generalized KL divergence | Observed values ≥ 0 |
| `gamma_poisson_nmf` | Gamma-regularized Poisson/NMF coordinate updates | Observed values ≥ 0 |

For compatibility with the source experiment, `bayesian_pca` aliases
`gaussian_als`, and `bayesian_nmf` aliases `gamma_poisson_nmf`. These algorithms
return point estimates only; they do not provide posterior samples or credible
intervals. The canonical names avoid overstating their Bayesian capabilities.

Method-specific options are passed as keyword arguments:

```python
MatrixCompleter(method="gaussian_als", rank=5, ridge=1e-2)
MatrixCompleter(method="funk_svd", rank=5, learning_rate=0.003,
                regularization=0.05)
MatrixCompleter(method="nmf_kl", rank=5, epsilon=1e-9)
MatrixCompleter(method="gamma_poisson_nmf", rank=5,
                gamma_shape=1.0, gamma_rate=1.0)
```

## Extending the library

New methods can be registered without changing `MatrixCompleter`:

```python
from matrix_completion import BackendResult, register_method

register_method("my_method", my_backend)
```

A backend receives `(matrix, observed, rank, max_iter, tolerance,
random_state, **options)` and returns a `BackendResult` containing the
reconstruction, left/right factors, iteration count, convergence flag,
objective, and optional metadata.

## Important modeling notes

- Rank must be selected with held-out observed entries or another criterion
  independent of the final scientific outcome.
- Random masking can overestimate performance when real gaps are spatially or
  temporally clustered. Evaluation masks should mimic the intended use case.
- NMF methods do not shift or clip data automatically. Domain-specific
  transforms belong in the calling application and should be inverted there.
- `gaussian_als` and `gamma_poisson_nmf` are MAP/coordinate-update style point
  estimators, not full Bayesian uncertainty quantification.
