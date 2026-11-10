import numpy as np
import pandas as pd

from hotspot import sim_data
from hotspot.knn import make_weights_non_redundant, neighbors_and_weights
from hotspot.local_stats_pairs import compute_hs_pairs_centered_cond

from .numba_reference import conditional_eg2, local_cov_pair


def test_local_correlation_cond() -> None:
    """
    Batched compute_hs_pairs_centered_cond should match the per-pair
    reference loops (local_cov_pair, conditional_eg2) on centered counts
    """
    N_CELLS = 1000
    N_GENES = 10

    latent = pd.DataFrame(sim_data.sim_latent(N_CELLS, 10))

    neighbors, weights = neighbors_and_weights(latent, n_neighbors=30, neighborhood_factor=3)
    weights = pd.DataFrame(
        make_weights_non_redundant(neighbors.values, weights.values),
        index=neighbors.index,
        columns=neighbors.columns,
    )
    nb, w = neighbors.values, weights.values

    # model="none" leaves counts unchanged, so they are their own centered values
    X = np.random.randn(N_GENES, N_CELLS)
    counts = pd.DataFrame(X, index=[f"g{i}" for i in range(N_GENES)])
    umi = pd.Series(np.ones(N_CELLS))

    lcs, lc_zs = compute_hs_pairs_centered_cond(counts, neighbors, weights, umi, "none")

    eg2s = np.array([conditional_eg2(x, nb, w) for x in X])
    for i in range(N_GENES):
        for j in range(N_GENES):
            if i == j:
                continue
            lc = local_cov_pair(X[i], X[j], nb, w) * 2
            z_xy, z_yx = lc / eg2s[i] ** 0.5, lc / eg2s[j] ** 0.5
            z = z_xy if abs(z_xy) < abs(z_yx) else z_yx
            np.testing.assert_allclose(lc_zs.values[i, j], z, rtol=1e-10)
