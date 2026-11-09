import numpy as np
from tqdm import tqdm
import pandas as pd
from scipy.sparse import csr_matrix, issparse
from threadpoolctl import threadpool_limits

from . import danb_model
from . import bernoulli_model
from . import normal_model
from . import none_model
from .knn import compute_node_degree, _build_sparse_weight_matrix
from .gpu import backend, to_host


def create_centered_counts(counts, model, num_umi):
    """
    Creates a matrix of centered/standardized counts given
    the selected statistical model
    """
    if issparse(counts):
        counts = csr_matrix(counts)  # slicing a CSC row is O(nnz) per gene

    out = np.zeros(counts.shape, dtype='double')

    for i in tqdm(range(out.shape[0])):

        vals_x = counts[i]
        if issparse(vals_x):
            vals_x = vals_x.toarray().ravel()

        out_x = create_centered_counts_row(
            vals_x, model, num_umi)

        out[i] = out_x

    return out


def create_centered_counts_row(vals_x, model, num_umi):

    if model == 'bernoulli':
        vals_x = (vals_x > 0).astype('double')
        mu_x, var_x, x2_x = bernoulli_model.fit_gene_model(
            vals_x, num_umi)

    elif model == 'danb':
        mu_x, var_x, x2_x = danb_model.fit_gene_model(
            vals_x, num_umi)

    elif model == 'normal':
        mu_x, var_x, x2_x = normal_model.fit_gene_model(
            vals_x, num_umi)

    elif model == 'none':
        mu_x, var_x, x2_x = none_model.fit_gene_model(
            vals_x, num_umi)

    else:
        raise Exception("Invalid Model: {}".format(model))

    var_x[var_x == 0] = 1
    out_x = (vals_x-mu_x)/(var_x**0.5)
    out_x[out_x == 0] = 0

    return out_x


def _hs_pairs_centered_cond_matrix(X, S, D, xp):
    """All-pairs conditional local correlation; X is genes x cells, S = W + W^T.

    Backend-agnostic: xp/S from gpu.backend (numpy + scipy.sparse, or cupy + cupyx).
    """
    # SX = S @ X.T (cells x genes), in gene blocks: a single product would
    # first copy all of X.T into C order (a full extra genes x cells array).
    SX = xp.empty((X.shape[1], X.shape[0]))
    for i in range(0, X.shape[0], 64):
        SX[:, i:i + 64] = S @ X[i:i + 64].T
    eg2s = xp.einsum("ij,ij->j", SX, SX)  # conditional_eg2 per gene, no SX**2 temp since eg2s = (SX**2).sum(0)
    lc = X @ SX                       # x_a.W.x_b + x_b.W.x_a == local_cov_pair * 2
    del SX
    lc = (lc + lc.T) / 2
    xp.fill_diagonal(lc, 0)

    std = eg2s ** 0.5
    Z_xy = lc / std[:, None]
    Z_yx = lc / std[None, :]
    lc_zs = xp.where(abs(Z_xy) < abs(Z_yx), Z_xy, Z_yx)

    lcs = lc / compute_local_cov_pairs_max(D, X, xp)
    return lcs, lc_zs


def compute_hs_pairs_centered_cond(counts, neighbors, weights,
                                   num_umi, model, jobs=1, use_gpu=False,
                                   genes=None):
    """counts: genes x cells DataFrame, or a (sparse) matrix plus `genes`."""

    xp, _ = backend(use_gpu)

    if isinstance(counts, pd.DataFrame):
        genes, counts = counts.index, counts.values
    neighbors = neighbors.values
    weights = weights.values
    num_umi = num_umi.values

    D = compute_node_degree(neighbors, weights)

    counts = create_centered_counts(counts, model, num_umi)

    N_cells = len(num_umi)
    W = _build_sparse_weight_matrix(
        neighbors, weights, shape=(N_cells, N_cells), use_gpu=use_gpu)
    with threadpool_limits(jobs):
        lcs, lc_zs = _hs_pairs_centered_cond_matrix(
            xp.asarray(counts), (W + W.T).tocsr(), xp.asarray(D), xp)
    lcs, lc_zs = to_host(lcs), to_host(lc_zs)

    lcs = pd.DataFrame(lcs, index=genes, columns=genes)
    lc_zs = pd.DataFrame(lc_zs, index=genes, columns=genes)

    return lcs, lc_zs


def compute_local_cov_pairs_max(node_degrees, counts, xp):
    """
    For a Genes x Cells count matrix, compute the maximal pair-wise correlation
    between any two genes
    """
    # sum_i D[i] * counts[g, i]**2 without materializing counts**2
    gene_maxs = xp.einsum("ij,ij,j->i", counts, counts, node_degrees.astype("double")) / 2
    return (gene_maxs[:, None] + gene_maxs[None, :]) / 2
