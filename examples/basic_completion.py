"""Complete a small matrix with the unified estimator interface."""

import numpy as np

from matrix_completion import MatrixCompleter, available_methods


matrix = np.array([
    [1.0, np.nan, 3.0, 4.0],
    [2.0, 4.0, 6.0, np.nan],
    [3.0, 6.0, np.nan, 12.0],
])

print("Available methods:", available_methods())
completed = MatrixCompleter(method="gaussian_als", rank=2).fit_transform(matrix)
print(completed)
