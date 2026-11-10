import math

import numpy as np
import pandas as pd

from hotspot import sim_data
from hotspot.knn import make_weights_non_redundant, neighbors_and_weights
from hotspot.types import FloatArray, GeneModelFit
from hotspot.utils import center_values

from .numba_reference import local_cov_pair, local_cov_weights


def true_params_scaled(gene_p: float, umi_counts: FloatArray) -> GeneModelFit:

    detect_p = 1 - (1 - gene_p / 10000) ** umi_counts

    mu = detect_p
    var = detect_p * (1 - detect_p)
    x2 = detect_p

    return mu, var, x2


def test_local_autocorrelation_centered() -> None:
    """
    Test if the expected moment calculation is correct
    """

    # Simulate some data
    N_CELLS = 1000
    N_DIM = 10

    latent = pd.DataFrame(sim_data.sim_latent(N_CELLS, N_DIM))

    umi_counts = sim_data.sim_umi_counts(N_CELLS, 2000, 200)

    neighbors_df, weights_df = neighbors_and_weights(
        latent, n_neighbors=30, neighborhood_factor=3, approx_neighbors=False
    )
    neighbors = neighbors_df.to_numpy()
    weights = make_weights_non_redundant(neighbors, weights_df.to_numpy())

    Wtot2 = float((weights**2).sum())

    # Simulate counts
    for gene_p in [0.2, 1, 5, 10]:
        N_REPS = 10000

        mu, var, x2 = true_params_scaled(gene_p, umi_counts)

        Gs_list = []
        for _i in range(N_REPS):
            counts_i = sim_data.sim_counts_bernoulli(N_CELLS, umi_counts, gene_p)

            counts_i = center_values(counts_i, mu, var)

            g = local_cov_weights(counts_i, neighbors, weights)
            Gs_list.append(g)

        Gs = np.array(Gs_list)

        EG = 0
        EG2 = Wtot2
        EstdG = (EG2 - EG**2) ** 0.5

        Gmean = Gs.mean()
        Gstd = Gs.std()

        assert math.isclose(0, abs(Gmean / Gstd), abs_tol=5e-2), (
            f"EG is off for gene_p={gene_p}, Actual={Gmean:.2f}, Expected={EG:.2f}"
        )
        assert math.isclose(EstdG, Gstd, rel_tol=5e-2), (
            f"stdG is off for gene_p={gene_p}, Actual={Gstd:.2f}, Expected={EstdG:.2f}"
        )


def test_local_correlation_centered() -> None:
    """
    Test if the expected moment calculation is correct
    For pair-wise function
    """

    # Simulate some data
    N_CELLS = 1000
    N_DIM = 10

    latent = pd.DataFrame(sim_data.sim_latent(N_CELLS, N_DIM))

    umi_counts = sim_data.sim_umi_counts(N_CELLS, 2000, 200)

    neighbors_df, weights_df = neighbors_and_weights(latent, n_neighbors=30, neighborhood_factor=3)
    neighbors = neighbors_df.to_numpy()
    weights = make_weights_non_redundant(neighbors, weights_df.to_numpy())

    Wtot2 = float((weights**2).sum())

    gene_p2 = 1

    # Simulate counts
    for gene_p in [0.2, 1, 5, 10]:
        N_REPS = 10000

        mu, var, x2 = true_params_scaled(gene_p, umi_counts)

        mu_j, var_j, x2_j = true_params_scaled(gene_p2, umi_counts)

        Gs_list = []
        for _i in range(N_REPS):
            counts_i = sim_data.sim_counts_bernoulli(N_CELLS, umi_counts, gene_p)

            counts_i = center_values(counts_i, mu, var)

            counts_j = sim_data.sim_counts_bernoulli(N_CELLS, umi_counts, gene_p2)

            counts_j = center_values(counts_j, mu_j, var_j)

            g = local_cov_pair(counts_i, counts_j, neighbors, weights)
            Gs_list.append(g)

        Gs = np.array(Gs_list)

        EG = 0
        EG2 = Wtot2 / 2
        EstdG = (EG2 - EG**2) ** 0.5

        Gmean = Gs.mean()
        Gstd = Gs.std()

        assert math.isclose(0, abs(Gmean / Gstd), abs_tol=5e-2), (
            f"EG is off for gene_p={gene_p}, Actual={Gmean:.2f}, Expected={EG:.2f}"
        )
        assert math.isclose(EstdG, Gstd, rel_tol=1e-1), (
            f"stdG is off for gene_p={gene_p}, Actual={Gstd:.2f}, Expected={EstdG:.2f}"
        )
