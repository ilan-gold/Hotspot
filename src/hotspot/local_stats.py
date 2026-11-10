from concurrent.futures import ThreadPoolExecutor
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from . import bernoulli_model, danb_model, none_model, normal_model
from .gpu import asnumpy, get_backend
from .knn import compute_node_degree
from .types import Counts, FloatArray, Model
from .utils import _build_sparse_weight_matrix, center_values

# Genes per block are chosen so each (genes x cells) dense array holds ~this many doubles
_BLOCK_ELEMS = 2**22


def compute_hs(
    counts: Counts,
    neighbors: pd.DataFrame,
    weights: pd.DataFrame,
    num_umi: pd.Series,
    model: Model,
    genes: pd.Index,
    centered: bool = False,
    jobs: int = 1,
    use_gpu: bool = False,
) -> pd.DataFrame:
    """Local autocorrelation statistics per gene.

    Batched over genes via sparse matrix multiplication on CPU (numpy/scipy) or GPU (cupy, one block).
    CPU gene blocks run on `jobs` threads, BLAS limited to 1 thread each.
    """
    xp = get_backend(use_gpu)

    neighbors_np = neighbors.to_numpy()
    weights_np = weights.to_numpy()
    num_umi_np = num_umi.to_numpy()

    N_genes, N_cells = counts.shape

    D = xp.asarray(compute_node_degree(neighbors_np, weights_np))
    Wtot2 = float((weights_np**2).sum())
    W = _build_sparse_weight_matrix(neighbors_np, weights_np, shape=(N_cells, N_cells), use_gpu=use_gpu)
    W_sq = (
        None
        if centered
        else _build_sparse_weight_matrix(
            neighbors_np, weights_np, shape=(N_cells, N_cells), square=True, use_gpu=use_gpu
        )
    )

    block = N_genes if use_gpu else max(1, _BLOCK_ELEMS // N_cells)

    def _block(i: int) -> FloatArray:
        return _compute_hs_block(counts[i : i + block], W, W_sq, D, Wtot2, num_umi_np, model, centered, xp)

    starts = range(0, N_genes, block)
    with threadpool_limits(1), ThreadPoolExecutor(jobs) as pool:
        blocks = list(tqdm(pool.map(_block, starts), total=len(starts)))

    results = pd.DataFrame(np.vstack(blocks), index=genes, columns=["G", "EG", "stdG", "Z", "C"])

    return _postprocess_results(results)


def _postprocess_results(results: pd.DataFrame) -> pd.DataFrame:
    results["Pval"] = norm.sf(results["Z"].to_numpy(dtype=np.float64))
    results["FDR"] = multipletests(results["Pval"], method="fdr_bh")[1]
    results = results.sort_values("Z", ascending=False)
    results.index.name = "Gene"
    results = results[["C", "Z", "Pval", "FDR"]]
    return results


def _fit_gene(
    vals: FloatArray, model: Model, num_umi: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Fit a gene model and return (vals, mu, var, x2).

    For the bernoulli model, vals is binarized before fitting.
    """
    if model == "bernoulli":
        vals = (vals > 0).astype("double")
        mu, var, x2 = bernoulli_model.fit_gene_model(vals, num_umi)
    elif model == "danb":
        mu, var, x2 = danb_model.fit_gene_model(vals, num_umi)
    elif model == "normal":
        mu, var, x2 = normal_model.fit_gene_model(vals, num_umi)
    elif model == "none":
        mu, var, x2 = none_model.fit_gene_model(vals, num_umi)
    else:
        raise ValueError(f"Invalid Model: {model}")
    return vals, mu, var, x2


def _local_cov_weights(vals: Any, W: Any) -> Any:
    """Batch of local_cov_weights: G[g] = vals[g] . (W @ vals[g]) for all genes."""
    smoothed_T = W @ vals.T
    return (vals * smoothed_T.T).sum(axis=1)


def _compute_moments_weights(mu: Any, x2: Any, W: Any, W_sq: Any) -> tuple[Any, Any]:
    """Batch of compute_moments_weights for all genes at once."""
    # EG[g] = mu[g] . (W @ mu[g])
    EG = (mu * (W @ mu.T).T).sum(axis=1)

    # t1[g] = (W + W.T) @ mu[g],  t2[g] = (W_sq + W_sq.T) @ mu[g]^2
    W_sym = W + W.T
    W_sq_sym = W_sq + W_sq.T
    mu2 = mu**2

    t1_T = W_sym @ mu.T
    t2_T = W_sq_sym @ mu2.T

    # Contribution 1: sum_i (x2[i] - mu[i]^2) * (t1[i]^2 - t2[i])
    diff_var = (x2 - mu2).T
    eg2_c1 = (diff_var * (t1_T**2 - t2_T)).sum(axis=0)

    # Contribution 2: sum_{edges} w^2 * (x2[i]*x2[j] - mu[i]^2*mu[j]^2)
    eg2_c2 = (x2.T * (W_sq @ x2.T)).sum(axis=0)
    eg2_c2 -= (mu2.T * (W_sq @ mu2.T)).sum(axis=0)

    EG2 = eg2_c1 + eg2_c2 + EG**2
    return EG, EG2


def _compute_local_cov_max(D: Any, vals: Any) -> Any:
    """Batch of compute_local_cov_max: G_max[g] = sum_i D[i]*vals[g,i]^2 / 2."""
    return (D * vals**2).sum(axis=1) / 2


def _compute_hs_block(
    counts: Counts,
    W: Any,
    W_sq: Any,
    D: Any,
    Wtot2: float,
    num_umi: FloatArray,
    model: Model,
    centered: bool,
    xp: ModuleType,
) -> FloatArray:
    """Statistics for one block of genes, returned as a (genes x 5) numpy array."""
    dense = np.asarray(counts if isinstance(counts, np.ndarray) else counts.toarray(), dtype="double")

    N_genes, N_cells = dense.shape
    all_vals = np.zeros((N_genes, N_cells), dtype="double")
    if not centered:
        all_mu = np.zeros((N_genes, N_cells), dtype="double")
        all_x2 = np.zeros((N_genes, N_cells), dtype="double")

    for i in range(N_genes):
        vals, mu, var, x2 = _fit_gene(dense[i], model, num_umi)
        if centered:
            vals = center_values(vals, mu, var)
        else:
            all_mu[i] = mu
            all_x2[i] = x2
        all_vals[i] = vals

    vals_xp = xp.asarray(all_vals)

    G_stats = _local_cov_weights(vals_xp, W)

    if centered:
        EG = xp.zeros(N_genes, dtype="double")
        EG2 = xp.full(N_genes, Wtot2, dtype="double")
    else:
        EG, EG2 = _compute_moments_weights(xp.asarray(all_mu), xp.asarray(all_x2), W, W_sq)

    stdG = (EG2 - EG * EG) ** 0.5
    Z = (G_stats - EG) / stdG

    G_max = _compute_local_cov_max(D, vals_xp)
    C = (G_stats - EG) / G_max

    return asnumpy(xp.stack([G_stats, EG, stdG, Z, C], axis=1))
