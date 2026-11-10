from collections.abc import Callable
from typing import Any, cast

import numba
import numpy as np
import scipy.sparse

from .types import FloatArray, IntArray


def njit[F: Callable[..., Any]](f: F) -> F:
    """`numba.njit` that keeps the wrapped function's signature visible to type checkers."""
    return cast(F, numba.njit(f))


@njit
def center_values(vals: FloatArray, mu: FloatArray, var: FloatArray) -> FloatArray:
    """Standardize `vals` by `mu` and `var`, mapping zero-variance entries to 0."""
    out = np.zeros_like(vals)

    for i in range(len(vals)):
        std = var[i] ** 0.5
        if std == 0:
            out[i] = 0
        else:
            out[i] = (vals[i] - mu[i]) / std

    return out


@njit
def neighbor_smoothing_row(
    vals: FloatArray, neighbors: IntArray, weights: FloatArray, _lambda: float = 0.9
) -> FloatArray:
    """

    Output is (neighborhood average) * _lambda + self * (1-_lambda)

    vals: expression matrix (genes x cells)
    neighbors: neighbor indices (cells x K)
    weights: neighbor weights (cells x K)
    _lambda: ratio controlling self vs. neighborhood
    """
    out = np.zeros_like(vals, dtype=np.float64)
    out_denom = np.zeros_like(vals, dtype=np.float64)

    N = neighbors.shape[0]  # Cells
    K = neighbors.shape[1]  # Neighbors

    for i in range(N):
        xi = vals[i]

        for k in range(K):
            j = neighbors[i, k]

            wij = weights[i, k]
            xj = vals[j]

            out[i] += xj * wij
            out[j] += xi * wij

            out_denom[i] += wij
            out_denom[j] += wij

    out /= out_denom

    out = (out * _lambda) + (1 - _lambda) * vals

    return out


def _build_sparse_weight_matrix(
    neighbors: IntArray, weights: FloatArray, shape: tuple[int, int], square: bool = False, use_gpu: bool = False
) -> Any:
    """Build a sparse CSR matrix (scipy, or CuPy if use_gpu) from neighbor/weight arrays.

    W[i, neighbors[i,k]] = weights[i,k]  for all i, k where weights[i,k] != 0.
    If square=True, uses weights^2 instead (for moment computations).
    """
    N, K = neighbors.shape
    rows = np.repeat(np.arange(N, dtype=np.int32), K)
    cols = neighbors.ravel().astype(np.int32)
    vals = weights.ravel().astype(np.float64)
    if square:
        vals = vals**2

    mask = vals != 0
    rows, cols, vals = rows[mask], cols[mask], vals[mask]

    W = scipy.sparse.csr_matrix((vals, (rows, cols)), shape=shape)
    if not use_gpu:
        return W
    import cupyx.scipy.sparse as cp_sparse  # pragma: no cover

    return cp_sparse.csr_matrix(W)  # pragma: no cover
