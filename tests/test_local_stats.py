import numpy as np
from numba import jit
import pandas as pd
from hotspot import sim_data
from hotspot.knn import neighbors_and_weights, make_weights_non_redundant
from hotspot.local_stats import _compute_moments_weights
from hotspot.knn import _build_sparse_weight_matrix
import math


@jit(nopython=True)
def compute_moments_weights_slow(mu, x2, neighbors, weights):
    """
    This version exaustively iterates over all |E|^2 terms
    to compute the expected moments exactly.  Used to test
    the more optimized formulations that follow
    """

    N = neighbors.shape[0]
    K = neighbors.shape[1]

    # Calculate E[G]
    EG = 0
    for i in range(N):
        for k in range(K):
            j = neighbors[i, k]
            wij = weights[i, k]

            EG += wij * mu[i] * mu[j]

    # Calculate E[G^2]
    EG2 = 0
    for i in range(N):

        EG2_i = 0

        for k in range(K):
            j = neighbors[i, k]
            wij = weights[i, k]

            for x in range(N):
                for z in range(K):
                    y = neighbors[x, z]
                    wxy = weights[x, z]

                    s = wij * wxy
                    if s == 0:
                        continue

                    if i == x:
                        if j == y:
                            t1 = x2[i] * x2[j]
                        else:
                            t1 = x2[i] * mu[j] * mu[y]
                    elif i == y:
                        if j == x:
                            t1 = x2[i] * x2[j]
                        else:
                            t1 = x2[i] * mu[j] * mu[x]
                    else:  # i is unique since i can't equal j

                        if j == x:
                            t1 = mu[i] * x2[j] * mu[y]
                        elif j == y:
                            t1 = mu[i] * x2[j] * mu[x]
                        else:  # i and j are unique, no shared nodes
                            t1 = mu[i] * mu[j] * mu[x] * mu[y]

                    EG2_i += s * t1

        EG2 += EG2_i

    return EG, EG2


def test_moments_fast():
    """
    Output of _compute_moments_weights should be the same as
    compute_moments_weights_slow
    """

    # Simulate some data
    N_CELLS = 100
    N_DIM = 10

    latent = sim_data.sim_latent(N_CELLS, N_DIM)
    latent = pd.DataFrame(latent)

    def _compute(neighbors, weights, rel_tol=1e-12):
        neighbors = neighbors.values
        weights = weights.values

        weights = make_weights_non_redundant(neighbors, weights)

        # Just generate random muX, muY, x2, y2
        muX = np.random.rand(N_CELLS) * 2 + 2
        x2 = (np.random.rand(N_CELLS) * 2 + 1)**2 + muX**2

        EG, EG2 = compute_moments_weights_slow(
            muX, x2, neighbors, weights
        )
        W = _build_sparse_weight_matrix(neighbors, weights, (N_CELLS, N_CELLS))
        W_sq = _build_sparse_weight_matrix(
            neighbors, weights, (N_CELLS, N_CELLS), square=True)
        EG_fast, EG2_fast = _compute_moments_weights(
            muX[None], x2[None], W, W_sq, W + W.T, W_sq + W_sq.T
        )

        assert math.isclose(EG, EG_fast[0], rel_tol=rel_tol)
        assert math.isclose(EG2, EG2_fast[0], rel_tol=rel_tol)

    neighbors, weights = neighbors_and_weights(
        latent, n_neighbors=30, neighborhood_factor=3, approx_neighbors=False)

    _compute(neighbors, weights)


    neighbors, weights = neighbors_and_weights(
        latent, n_neighbors=30, neighborhood_factor=3, approx_neighbors=True)

    _compute(neighbors, weights, rel_tol=0.005)
