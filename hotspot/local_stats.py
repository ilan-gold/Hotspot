from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tqdm import tqdm
import pandas as pd
from scipy.stats import norm
from scipy.sparse import issparse, csr_matrix
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits

from . import danb_model
from . import bernoulli_model
from . import normal_model
from . import none_model

from .knn import compute_node_degree, _build_sparse_weight_matrix
from .gpu import backend, to_host

# ponytail: genes per block = this // cells, ~10 live (genes x cells) arrays
# per block and one block per thread; raise it if the GPU is underused.
_BLOCK_DOUBLES = {False: 2**21, True: 2**25}


def compute_hs(
    counts, neighbors, weights, num_umi, model, genes, centered=False, jobs=1,
    use_gpu=False
):
    """Local autocorrelation per gene; counts is genes x cells (dense or sparse).

    Genes are processed in blocks by the same code on CPU (numpy +
    scipy.sparse, `jobs` threads over blocks) and GPU (cupy + cupyx).
    """
    xp, sp = backend(use_gpu)
    if use_gpu:
        jobs = 1

    neighbors = neighbors.values
    weights = weights.values
    num_umi = num_umi.values

    D = compute_node_degree(neighbors, weights)
    Wtot2 = (weights**2).sum()

    N_cells = len(num_umi)
    W = _build_sparse_weight_matrix(
        neighbors, weights, shape=(N_cells, N_cells), use_gpu=use_gpu
    )
    if not centered:
        W_sq = _build_sparse_weight_matrix(
            neighbors, weights, shape=(N_cells, N_cells), square=True, use_gpu=use_gpu
        )
        W_sym, W_sq_sym = W + W.T, W_sq + W_sq.T  # once, not per block

    D = xp.asarray(D)
    num_umi = xp.asarray(num_umi, dtype="double")
    bins = bernoulli_model.umi_bins(num_umi) if model == "bernoulli" else None

    if issparse(counts):
        counts = csr_matrix(counts)  # slicing a CSC row is O(nnz) per gene
    step = max(1, _BLOCK_DOUBLES[use_gpu] // N_cells)

    def block(start):
        raw = counts[start:start + step]
        if issparse(raw):
            raw = sp.csr_matrix(raw).toarray()
        vals, mu, var, x2 = _fit_gene(
            xp.asarray(raw, dtype="double"), model, num_umi, bins
        )
        N_genes = vals.shape[0]

        if centered:
            std = var ** 0.5
            vals = xp.where(std == 0, 0, (vals - mu) / xp.where(std == 0, 1, std))

        G_stats = _local_cov_weights(vals, W)

        if centered:
            EG = xp.zeros(N_genes, dtype="double")
            EG2 = xp.full(N_genes, Wtot2, dtype="double")
        else:
            EG, EG2 = _compute_moments_weights(mu, x2, W, W_sq, W_sym, W_sq_sym)

        stdG = (EG2 - EG * EG) ** 0.5
        Z = (G_stats - EG) / stdG

        G_max = _compute_local_cov_max(D, vals)
        C = (G_stats - EG) / G_max

        return to_host(xp.stack([G_stats, EG, stdG, Z, C], axis=1))

    starts = range(0, counts.shape[0], step)
    with threadpool_limits(1), ThreadPoolExecutor(jobs) as pool:
        results = list(tqdm(pool.map(block, starts), total=len(starts)))

    results = pd.DataFrame(
        np.concatenate(results), index=genes, columns=["G", "EG", "stdG", "Z", "C"]
    )

    return _postprocess_results(results)


def _postprocess_results(results):
    results["Pval"] = norm.sf(results["Z"].values)
    results["FDR"] = multipletests(results["Pval"], method="fdr_bh")[1]
    results = results.sort_values("Z", ascending=False)
    results.index.name = "Gene"
    results = results[["C", "Z", "Pval", "FDR"]]
    return results


def _fit_gene(vals, model, num_umi, bins=None):
    """Fit a gene model and return (vals, mu, var, x2).

    For the bernoulli model, vals is binarized before fitting.
    """
    if model == "bernoulli":
        vals = (vals > 0).astype("double")
        mu, var, x2 = bernoulli_model.fit_gene_model(vals, num_umi, bins)
    elif model == "danb":
        mu, var, x2 = danb_model.fit_gene_model(vals, num_umi)
    elif model == "normal":
        mu, var, x2 = normal_model.fit_gene_model(vals, num_umi)
    elif model == "none":
        mu, var, x2 = none_model.fit_gene_model(vals, num_umi)
    else:
        raise ValueError("Invalid Model: {}".format(model))
    return vals, mu, var, x2


def _local_cov_weights(vals_gpu, W):
    """Batch of local_cov_weights: G[g] = vals[g] . (W @ vals[g]) for all genes."""
    smoothed_T = W @ vals_gpu.T
    return (vals_gpu * smoothed_T.T).sum(axis=1)


def _compute_moments_weights(mu_gpu, x2_gpu, W, W_sq, W_sym, W_sq_sym):
    """Batch of compute_moments_weights; W_sym = W + W.T, W_sq_sym = W_sq + W_sq.T."""
    # EG[g] = mu[g] . (W @ mu[g])
    EG = (mu_gpu * (W @ mu_gpu.T).T).sum(axis=1)

    # t1[g] = (W + W.T) @ mu[g],  t2[g] = (W_sq + W_sq.T) @ mu[g]^2
    mu2_gpu = mu_gpu ** 2

    t1_T = W_sym @ mu_gpu.T
    t2_T = W_sq_sym @ mu2_gpu.T

    # Contribution 1: sum_i (x2[i] - mu[i]^2) * (t1[i]^2 - t2[i])
    diff_var = (x2_gpu - mu2_gpu).T
    eg2_c1 = (diff_var * (t1_T ** 2 - t2_T)).sum(axis=0)

    # Contribution 2: sum_{edges} w^2 * (x2[i]*x2[j] - mu[i]^2*mu[j]^2)
    eg2_c2 = (x2_gpu.T * (W_sq @ x2_gpu.T)).sum(axis=0)
    eg2_c2 -= (mu2_gpu.T * (W_sq @ mu2_gpu.T)).sum(axis=0)

    EG2 = eg2_c1 + eg2_c2 + EG ** 2
    return EG, EG2


def _compute_local_cov_max(D_gpu, vals_gpu):
    """Batch of compute_local_cov_max: G_max[g] = sum_i D[i]*vals[g,i]^2 / 2."""
    return (D_gpu * vals_gpu ** 2).sum(axis=1) / 2
