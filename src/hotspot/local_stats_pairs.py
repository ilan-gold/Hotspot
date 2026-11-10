from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from .gpu import asnumpy, get_backend
from .knn import compute_node_degree
from .local_stats import _fit_gene
from .types import FloatArray, Model
from .utils import _build_sparse_weight_matrix


def create_centered_counts(counts: FloatArray, model: Model, num_umi: FloatArray) -> FloatArray:
    """Creates a matrix of centered/standardized counts given the selected statistical model."""
    out = np.zeros_like(counts, dtype="double")

    for i in tqdm(range(out.shape[0])):
        vals_x = counts[i]

        out_x = create_centered_counts_row(vals_x, model, num_umi)

        out[i] = out_x

    return out


def create_centered_counts_row(vals_x: FloatArray, model: Model, num_umi: FloatArray) -> FloatArray:
    """Center and standardize one gene's values under the selected null model."""
    vals_x, mu_x, var_x, x2_x = _fit_gene(vals_x, model, num_umi)

    var_x[var_x == 0] = 1
    out_x = (vals_x - mu_x) / (var_x**0.5)
    out_x[out_x == 0] = 0

    return out_x


def compute_hs_pairs_centered_cond(
    counts: pd.DataFrame,
    neighbors: pd.DataFrame,
    weights: pd.DataFrame,
    num_umi: pd.Series,
    model: Model,
    jobs: int = 1,
    use_gpu: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pair-wise local correlations (and Z-scores) between genes.

    Batched over all gene pairs via dense matmul: M = X @ (W @ X.T), lc = M + M.T
    on CPU (numpy/scipy, BLAS limited to `jobs` threads) or GPU (cupy).
    """
    xp = get_backend(use_gpu)

    genes = counts.index
    counts_np = counts.to_numpy()
    neighbors_np = neighbors.to_numpy()
    weights_np = weights.to_numpy()
    num_umi_np = num_umi.to_numpy()
    N_cells = counts_np.shape[1]

    centered_counts = create_centered_counts(counts_np, model, num_umi_np)

    X = xp.asarray(centered_counts)
    D = xp.asarray(compute_node_degree(neighbors_np, weights_np))
    W = _build_sparse_weight_matrix(neighbors_np, weights_np, shape=(N_cells, N_cells), use_gpu=use_gpu)
    W_sym = W + W.T

    with threadpool_limits(jobs):
        eg2s = _conditional_eg2(X, W_sym)
        lc_matrix = _local_cov_pair_all(X, W, xp)

    std_genes = eg2s**0.5
    Z_xy = lc_matrix / std_genes[:, None]
    Z_yx = lc_matrix / std_genes[None, :]
    lc_zs = xp.where(xp.abs(Z_xy) < xp.abs(Z_yx), Z_xy, Z_yx)

    gene_maxs = (D * X**2).sum(axis=1) / 2
    lc_maxs = (gene_maxs[:, None] + gene_maxs[None, :]) / 2
    lcs = lc_matrix / lc_maxs

    lcs_df = pd.DataFrame(asnumpy(lcs), index=genes, columns=genes)
    lc_zs_df = pd.DataFrame(asnumpy(lc_zs), index=genes, columns=genes)

    return lcs_df, lc_zs_df


def _conditional_eg2(X: Any, W_sym: Any) -> Any:
    """Batch of conditional_eg2: eg2[g] = ||(W + W.T) @ x[g]||^2."""
    t1x_T = W_sym @ X.T
    return (t1x_T**2).sum(axis=0)


def _local_cov_pair_all(X: Any, W: Any, xp: ModuleType) -> Any:
    """Batch of local_cov_pair for ALL gene pairs via dense matmul.

    Returns the full G x G matrix of lc values (= local_cov_pair * 2).
    Diagonal is zeroed (no self-pairs).
    """
    smoothed_T = W @ X.T  # (N, G)
    M = X @ smoothed_T  # (G, G):  M[a,b] = x_a . (W @ x_b)
    lc_matrix = M + M.T  # symmetrize: lc[a,b] = x_a.(Wx_b) + x_b.(Wx_a)
    xp.fill_diagonal(lc_matrix, 0)
    return lc_matrix
