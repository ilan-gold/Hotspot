import numpy as np
from numba import njit


@njit
def neighbor_smoothing_row(vals, neighbors, weights, _lambda=.9):
    """

    output is (neighborhood average) * _lambda + self * (1-_lambda)


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

            out[i] += xj*wij
            out[j] += xi*wij

            out_denom[i] += wij
            out_denom[j] += wij

    out /= out_denom

    out = (out * _lambda) + (1 - _lambda) * vals

    return out
