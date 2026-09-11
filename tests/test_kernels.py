"""The parallel numba kernels must agree with a plain sequential reference."""
import numpy as np
import pandas as pd

from hotspot import sim_data
from hotspot.knn import (
    neighbors_and_weights,
    make_weights_non_redundant,
    compute_node_degree,
)
from hotspot.local_stats import (
    compute_hs,
    compute_local_cov_max,
    compute_moments_weights,
    local_cov_weights,
    _fit_genes,
)
from hotspot.local_stats_pairs import (
    _compute_hs_pairs_inner_centered_cond_sym,
    compute_hs_pairs_centered_cond,
    create_centered_counts,
    conditional_eg2,
)

N_CELLS = 300
N_GENES = 12
MODEL = "danb"


def _data():
    np.random.seed(0)
    latent = pd.DataFrame(sim_data.sim_latent(N_CELLS, 5))
    umi = pd.Series(sim_data.sim_umi_counts(N_CELLS, 2000, 200))
    neighbors, weights = neighbors_and_weights(latent, n_neighbors=10)
    weights = pd.DataFrame(
        make_weights_non_redundant(neighbors.values, weights.values),
        index=weights.index,
        columns=weights.columns,
    )
    counts = np.random.negative_binomial(5, 0.5, size=(N_GENES, N_CELLS))
    counts = counts.astype("double")
    genes = pd.Index(["g{}".format(i) for i in range(N_GENES)])
    return umi, neighbors, weights, counts, genes


def test_compute_hs_matches_sequential():
    umi, neighbors, weights, counts, genes = _data()

    result = compute_hs(counts, neighbors, weights, umi, MODEL, genes, jobs=4)

    D = compute_node_degree(neighbors.values, weights.values)
    vals, mu, x2 = _fit_genes(
        counts, range(N_GENES), umi.values, MODEL, False
    )

    for i, gene in enumerate(genes):
        G = local_cov_weights(vals[i], neighbors.values, weights.values)
        EG, EG2 = compute_moments_weights(
            mu[i], x2[i], neighbors.values, weights.values
        )
        Z = (G - EG) / (EG2 - EG * EG) ** 0.5
        C = (G - EG) / compute_local_cov_max(D, vals[i])

        assert np.isclose(result.loc[gene, "Z"], Z)
        assert np.isclose(result.loc[gene, "C"], C)


def test_compute_hs_pairs_cond_matches_sequential():
    umi, neighbors, weights, counts, genes = _data()
    counts_df = pd.DataFrame(counts, index=genes)

    lcs, lc_zs = compute_hs_pairs_centered_cond(
        counts_df, neighbors, weights, umi, MODEL, jobs=4
    )

    nb, w = neighbors.values, weights.values
    centered = create_centered_counts(counts, MODEL, umi.values)
    eg2s = np.array([conditional_eg2(centered[i], nb, w) for i in range(N_GENES)])
    D = compute_node_degree(nb, w)
    gene_maxs = np.array(
        [compute_local_cov_max(D, centered[i]) for i in range(N_GENES)]
    )

    for i in range(N_GENES):
        for j in range(i + 1, N_GENES):
            lc, Z = _compute_hs_pairs_inner_centered_cond_sym(
                (i, j), centered, nb, w, eg2s
            )
            lc_max = (gene_maxs[i] + gene_maxs[j]) / 2

            assert np.isclose(lcs.iloc[i, j], lc / lc_max)
            assert np.isclose(lcs.iloc[j, i], lc / lc_max)
            assert np.isclose(lc_zs.iloc[i, j], Z)

    assert (np.diag(lcs.values) == 0).all()


def test_jobs_does_not_change_results():
    umi, neighbors, weights, counts, genes = _data()
    counts_df = pd.DataFrame(counts, index=genes)

    a = compute_hs_pairs_centered_cond(counts_df, neighbors, weights, umi, MODEL, jobs=1)
    b = compute_hs_pairs_centered_cond(counts_df, neighbors, weights, umi, MODEL, jobs=8)

    np.testing.assert_allclose(a[0].values, b[0].values)
    np.testing.assert_allclose(a[1].values, b[1].values)
