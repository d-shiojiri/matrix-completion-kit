import numpy as np
import pytest

from matrix_completion import MatrixCompleter, available_methods


@pytest.fixture
def nonnegative_matrix():
    left = np.array([[1., 2.], [2., .5], [.5, 1.5], [1.2, .8]])
    right = np.array([[1., 2., .5, 1.5, .8], [.7, .2, 1.8, .4, 1.1]])
    complete = left @ right
    observed = np.ones(complete.shape, dtype=bool)
    observed[0, 1] = observed[1, 3] = observed[3, 0] = False
    matrix = complete.copy(); matrix[~observed] = np.nan
    return matrix, observed


@pytest.mark.parametrize("method", available_methods())
def test_every_method_uses_the_same_interface(method, nonnegative_matrix):
    matrix, observed = nonnegative_matrix
    model = MatrixCompleter(method=method, rank=2, max_iter=30, random_state=7)
    completed = model.fit_transform(matrix)
    result = model.get_result()
    assert completed.shape == matrix.shape
    assert np.all(np.isfinite(completed))
    assert np.allclose(completed[observed], matrix[observed])
    assert result.left_factor.shape == (matrix.shape[0], 2)
    assert result.right_factor.shape == (2, matrix.shape[1])


def test_explicit_mask_can_hide_finite_values(nonnegative_matrix):
    matrix, observed = nonnegative_matrix
    finite = np.nan_to_num(matrix, nan=99.0)
    completed = MatrixCompleter(rank=2).fit_transform(finite, observed)
    assert completed[0, 1] != 99.0


def test_nmf_rejects_negative_observations():
    with pytest.raises(ValueError, match="nonnegative"):
        MatrixCompleter(method="nmf_kl").fit_transform([[1.0, np.nan], [-1.0, 2.0]])


def test_aliases_resolve_to_canonical_method(nonnegative_matrix):
    matrix, _ = nonnegative_matrix
    result = MatrixCompleter(method="bayesian_pca", max_iter=2).fit(matrix).get_result()
    assert result.method == "gaussian_als"
