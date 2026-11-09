import numpy as np
import pandas as pd
from hotspot import sim_data
from hotspot.knn import neighbors_and_weights, make_weights_non_redundant
from numba import jit
import math

from hotspot.knn import compute_node_degree, _build_sparse_weight_matrix
from hotspot.local_stats_pairs import _hs_pairs_centered_cond_matrix


# Ground truth: the loop implementations master ran for local correlations
# (compute_hs_pairs_centered_cond); the library now computes them as matrix
# products (_hs_pairs_centered_cond_matrix).


@jit(nopython=True)
def conditional_eg2(x, neighbors, weights):
    """
    Computes EG2 for the conditional correlation
    """
    N = neighbors.shape[0]
    K = neighbors.shape[1]

    t1x = np.zeros(N)

    for i in range(N):
        for k in range(K):
            j = neighbors[i, k]

            wij = weights[i, k]
            if wij == 0:
                continue

            t1x[i] += wij*x[j]
            t1x[j] += wij*x[i]

    out_eg2 = (t1x**2).sum()

    return out_eg2


@jit(nopython=True)
def local_cov_pair(x, y, neighbors, weights):
    """Test statistic for local pair-wise autocorrelation"""
    out = 0

    for i in range(len(x)):
        xi = x[i]
        yi = y[i]
        if xi == 0 and yi == 0:
            continue
        for k in range(neighbors.shape[1]):

            j = neighbors[i, k]
            w_ij = weights[i, k]

            xj = x[j]
            yj = y[j]

            out += w_ij*(xi*yj + yi*xj)/2

    return out


def test_matrix_matches_loops():
    """_hs_pairs_centered_cond_matrix matches the loop ground truth."""
    N_CELLS, N_GENES = 300, 6

    latent = pd.DataFrame(sim_data.sim_latent(N_CELLS, 10))
    neighbors, weights = neighbors_and_weights(
        latent, n_neighbors=30, neighborhood_factor=3, approx_neighbors=False)
    neighbors = neighbors.values
    weights = make_weights_non_redundant(neighbors, weights.values)

    X = np.random.randn(N_GENES, N_CELLS)
    W = _build_sparse_weight_matrix(neighbors, weights, (N_CELLS, N_CELLS))
    D = compute_node_degree(neighbors, weights)
    lcs, lc_zs = _hs_pairs_centered_cond_matrix(X, (W + W.T).tocsr(), D, np)

    for a in range(N_GENES):
        for b in range(N_GENES):
            if a == b:
                continue
            g = local_cov_pair(X[a], X[b], neighbors, weights)
            z_ab = 2 * g / conditional_eg2(X[a], neighbors, weights) ** 0.5
            z_ba = 2 * g / conditional_eg2(X[b], neighbors, weights) ** 0.5
            z = z_ab if abs(z_ab) < abs(z_ba) else z_ba
            assert math.isclose(lc_zs[a, b], z, rel_tol=1e-10)
