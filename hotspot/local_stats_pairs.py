import numpy as np
from numba import jit, njit, prange
from tqdm import tqdm
import pandas as pd

from . import danb_model
from . import bernoulli_model
from . import normal_model
from . import none_model
from .local_stats import compute_local_cov_max, _fit_genes
from .knn import compute_node_degree
from .utils import numba_threads


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


def conditional_eg2_slow(x, neighbors, weights):
    """
    Computes EG2 for the conditional correlation
    This version is slower and more explicit for debugging purposes
    """
    N = neighbors.shape[0]
    K = neighbors.shape[1]

    # enumerate ALL neighbors of each node and the edge weight

    neighbors_all = [[] for _ in range(N)]
    weights_all = [[] for _ in range(N)]

    for i in range(N):
        neighbors_i = neighbors_all[i]
        weights_i = weights_all[i]

        for k in range(K):
            j = neighbors[i, k]
            wij = weights[i, k]
            if wij == 0:
                continue

            neighbors_j = neighbors_all[j]
            weights_j = weights_all[j]

            neighbors_i.append(j)
            weights_i.append(wij)

            neighbors_j.append(i)
            weights_j.append(wij)

    node_totals = np.zeros(N)
    node_totals_diag = np.zeros(N)

    for i in range(N):
        neighbors_i = neighbors_all[i]
        weights_i = weights_all[i]

        for j, wij in zip(neighbors_i, weights_i):
            node_totals[i] += x[j]*wij
            node_totals_diag[i] += (x[j]*wij)**2

    node_totals = node_totals**2

    dup = 0

    for i in range(N):
        for k in range(K):
            j = neighbors[i, k]
            wij = weights[i, k]
            if wij == 0:
                continue

            dup += wij**2*(x[i]**2+x[j]**2)

    return node_totals.sum() - node_totals_diag.sum() + dup


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


@jit(nopython=True)
def compute_moments_weights_pairs_slow(muA, a2, muB, b2, neighbors, weights):
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

            EG += wij * (muA[i] * muB[j] + muB[i] * muA[j]) / 2

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
                            t1 = (
                                a2[i] * b2[j]
                                + muA[i] * muB[j] * muA[j] * muB[i]
                                + muA[j] * muB[i] * muA[i] * muB[j]
                                + a2[j] * b2[i]
                            )
                        else:
                            t1 = (
                                a2[i] * muB[j] * muB[y]
                                + muA[i] * muB[j] * muA[y] * muB[i]
                                + muA[j] * muB[i] * muA[i] * muB[y]
                                + muA[j] * b2[i] * muA[y]
                            )
                    elif i == y:
                        if j == x:
                            t1 = (
                                muA[i] * muB[j] * muA[j] * muB[i]
                                + a2[i] * b2[j]
                                + a2[j] * b2[i]
                                + muA[j] * muB[i] * muA[i] * muB[j]
                            )
                        else:
                            t1 = (
                                muA[i] * muB[j] * muA[x] * muB[i]
                                + a2[i] * muB[j] * muB[x]
                                + muA[j] * b2[i] * muA[x]
                                + muA[j] * muB[i] * muA[i] * muB[x]
                            )
                    else:  # i is unique since i can't equal j

                        if j == x:
                            t1 = (
                                muA[i] * muB[j] * muA[j] * muB[y]
                                + muA[i] * b2[j] * muA[y]
                                + a2[j] * muB[i] * muB[y]
                                + muA[j] * muB[i] * muA[y] * muB[j]
                            )
                        elif j == y:
                            t1 = (
                                muA[i] * b2[j] * muA[x]
                                + muA[i] * muB[j] * muA[j] * muB[x]
                                + muA[j] * muB[i] * muA[x] * muB[j]
                                + a2[j] * muB[i] * muB[x]
                            )
                        else:  # i and j are unique, no shared nodes
                            t1 = (
                                muA[i] * muB[j] * muA[x] * muB[y]
                                + muA[i] * muB[j] * muA[y] * muB[x]
                                + muA[j] * muB[i] * muA[x] * muB[y]
                                + muA[j] * muB[i] * muA[y] * muB[x]
                            )

                    EG2_i += s * t1 / 4

        EG2 += EG2_i

    return EG, EG2


@jit(nopython=True)
def compute_moments_weights_pairs(
        muX, x2,
        muY, y2,
        neighbors, weights):
    """Computes the expectations of the local pair-wise test statistic"""

    N = neighbors.shape[0]
    K = neighbors.shape[1]

    # Calculate E[G]
    EG = 0
    for i in range(N):
        for k in range(K):
            j = neighbors[i, k]
            wij = weights[i, k]

            EG += wij*(muX[i]*muY[j] + muY[i]*muX[j])/2

    # Calculate E[G^2]
    EG2 = 0
    EG2 += (EG**2)

    #   Get the x^2*y*z terms
    t1x = np.zeros(N)
    t2x = np.zeros(N)

    t1y = np.zeros(N)
    t2y = np.zeros(N)

    for i in range(N):
        for k in range(K):
            j = neighbors[i, k]

            wij = weights[i, k]
            if wij == 0:
                continue

            t1x[i] += wij*muX[j]
            t2x[i] += wij**2*muX[j]**2

            t1x[j] += wij*muX[i]
            t2x[j] += wij**2*muX[i]**2

            t1y[i] += wij*muY[j]
            t2y[i] += wij**2*muY[j]**2

            t1y[j] += wij*muY[i]
            t2y[j] += wij**2*muY[i]**2

    t1x = t1x**2
    t1y = t1y**2

    for i in range(N):
        EG2 += (y2[i] - muY[i]**2)*(t1x[i] - t2x[i])/4
        EG2 += (x2[i] - muX[i]**2)*(t1y[i] - t2y[i])/4

    #  Get the x^2*y^2 terms
    for i in range(N):
        for k in range(K):
            j = neighbors[i, k]

            wij = weights[i, k]

            EG2 += wij**2*(
                x2[i]*y2[j] - (muX[i]**2)*(muY[j]**2) +
                y2[i]*x2[j] - (muY[i]**2)*(muX[j]**2)
            )/4

    return EG, EG2


@jit(nopython=True)
def compute_moments_weights_pairs_fast(
        muX, x2,
        muY, y2,
        neighbors, weights):
    """
    Computes the expectations of the local pair-wise test statistic

    About 2x as fast as compute_moments_weights_pairs
    """

    N = neighbors.shape[0]
    K = neighbors.shape[1]

    #   Get the x^2*y*z terms
    t1x = np.zeros(N)
    t2x = np.zeros(N)

    t1y = np.zeros(N)
    t2y = np.zeros(N)

    EG = 0
    EG2 = 0

    for i in range(N):
        for k in range(K):
            j = neighbors[i, k]
            wij = weights[i, k]
            wij_2 = wij**2

            if wij == 0:
                continue

            muX_i2 = muX[i]**2
            muX_j2 = muX[j]**2
            muY_i2 = muY[i]**2
            muY_j2 = muY[j]**2

            EG += wij*(muX[i]*muY[j] + muY[i]*muX[j])/2

            t1x[i] += wij*muX[j]
            t2x[i] += wij_2*muX_j2

            t1x[j] += wij*muX[i]
            t2x[j] += wij_2*muX_i2

            t1y[i] += wij*muY[j]
            t2y[i] += wij_2*muY_j2

            t1y[j] += wij*muY[i]
            t2y[j] += wij_2*muY_i2

            EG2 += wij_2*(
                x2[i]*y2[j] - (muX_i2)*(muY_j2) +
                y2[i]*x2[j] - (muY_i2)*(muX_j2)
            )/4

    # Calculate E[G^2]
    t1x = t1x**2
    t1y = t1y**2

    for i in range(N):
        EG2 += (y2[i] - muY[i]**2)*(t1x[i] - t2x[i])/4
        EG2 += (x2[i] - muX[i]**2)*(t1y[i] - t2y[i])/4

    EG2 += (EG**2)

    return EG, EG2


@jit(nopython=True)
def compute_moments_weights_pairs_std(neighbors, weights):
    """
    Computes the expectations of the local pair-wise test statistic

    This version assumes variables are standardized,
    and so the moments are actually the same for all
    pairs of variables
    """

    N = neighbors.shape[0]
    K = neighbors.shape[1]

    # Calculate E[G]
    EG = 0

    # Calculate E[G^2]
    EG2 = 0

    #  Get the x^2*y^2 terms
    for i in range(N):
        for k in range(K):
            wij = weights[i, k]

            EG2 += (wij**2)/2

    return EG, EG2


def create_centered_counts(counts, model, num_umi):
    """
    Creates a matrix of centered/standardized counts given
    the selected statistical model
    """
    out = np.zeros_like(counts, dtype='double')

    for i in tqdm(range(out.shape[0])):

        vals_x = counts[i]

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


@jit(nopython=True)
def _compute_hs_pairs_inner_centered_cond_sym(
    rowpair, counts, neighbors, weights, eg2s
):
    """
    This version assumes that the counts have already been modeled
    and centered
    """
    row_i, row_j = rowpair

    vals_x = counts[row_i]
    vals_y = counts[row_j]

    lc = local_cov_pair(vals_x, vals_y, neighbors, weights)*2

    # Compute xy
    EG, EG2 = 0, eg2s[row_i]

    stdG = (EG2 - EG ** 2) ** 0.5

    Zxy = (lc - EG) / stdG

    # Compute yx
    EG, EG2 = 0, eg2s[row_j]

    stdG = (EG2 - EG ** 2) ** 0.5

    Zyx = (lc - EG) / stdG

    if abs(Zxy) < abs(Zyx):
        Z = Zxy
    else:
        Z = Zyx

    return (lc, Z)


@njit(parallel=True)
def _hs_pairs_kernel(vals, mu, x2, neighbors, weights, centered, Wtot2,
                     pair_i, pair_j):
    """All-pairs version of the per-pair statistic, one thread per pair."""
    n_genes = vals.shape[0]
    lcs = np.zeros((n_genes, n_genes))
    lc_zs = np.zeros((n_genes, n_genes))

    for p in prange(len(pair_i)):
        i = pair_i[p]
        j = pair_j[p]

        if centered:
            EG, EG2 = 0.0, Wtot2 / 2
        else:
            EG, EG2 = compute_moments_weights_pairs_fast(
                mu[i], x2[i], mu[j], x2[j], neighbors, weights)

        lc = local_cov_pair(vals[i], vals[j], neighbors, weights)
        Z = (lc - EG) / (EG2 - EG**2)**0.5

        lcs[i, j] = lc
        lcs[j, i] = lc
        lc_zs[i, j] = Z
        lc_zs[j, i] = Z

    return lcs, lc_zs


@njit(parallel=True)
def _hs_pairs_cond_kernel(counts, neighbors, weights, eg2s, pair_i, pair_j):
    """All-pairs version of the conditional statistic, one thread per pair."""
    n_genes = counts.shape[0]
    lcs = np.zeros((n_genes, n_genes))
    lc_zs = np.zeros((n_genes, n_genes))

    for p in prange(len(pair_i)):
        i = pair_i[p]
        j = pair_j[p]

        lc, Z = _compute_hs_pairs_inner_centered_cond_sym(
            (i, j), counts, neighbors, weights, eg2s)

        lcs[i, j] = lc
        lcs[j, i] = lc
        lc_zs[i, j] = Z
        lc_zs[j, i] = Z

    return lcs, lc_zs


def _pair_index(n_genes, include_diagonal):
    """Flat (i, j) upper-triangle index -- one balanced prange slice per pair."""
    pair_i, pair_j = np.triu_indices(n_genes, k=0 if include_diagonal else 1)
    return pair_i.astype("int32"), pair_j.astype("int32")


def compute_hs_pairs(counts, neighbors, weights,
                     num_umi, model, centered=False, jobs=1):

    genes = counts.index

    counts = counts.values
    neighbors = neighbors.values
    weights = weights.values
    num_umi = num_umi.values

    D = compute_node_degree(neighbors, weights)
    Wtot2 = (weights**2).sum()

    vals, mu, x2 = _fit_genes(
        counts, range(counts.shape[0]), num_umi, model, centered)

    with numba_threads(jobs):
        lcs, lc_zs = _hs_pairs_kernel(
            vals, mu, x2, neighbors, weights, centered, Wtot2,
            *_pair_index(counts.shape[0], include_diagonal=True))

    lc_maxs = compute_local_cov_pairs_max(D, counts)
    lcs = lcs / lc_maxs

    lcs = pd.DataFrame(lcs, index=genes, columns=genes)
    lc_zs = pd.DataFrame(lc_zs, index=genes, columns=genes)

    return lcs, lc_zs


def compute_hs_pairs_centered(counts, neighbors, weights,
                              num_umi, model, jobs=1):

    genes = counts.index

    counts = counts.values
    neighbors = neighbors.values
    weights = weights.values
    num_umi = num_umi.values

    D = compute_node_degree(neighbors, weights)
    Wtot2 = (weights**2).sum()

    counts = create_centered_counts(counts, model, num_umi)

    with numba_threads(jobs):
        # mu/x2 are unread on the centered branch; pass counts to keep types simple
        lcs, lc_zs = _hs_pairs_kernel(
            counts, counts, counts, neighbors, weights, True, Wtot2,
            *_pair_index(counts.shape[0], include_diagonal=False))

    lc_maxs = compute_local_cov_pairs_max(D, counts)
    lcs = lcs / lc_maxs

    lcs = pd.DataFrame(lcs, index=genes, columns=genes)
    lc_zs = pd.DataFrame(lc_zs, index=genes, columns=genes)

    return lcs, lc_zs


def compute_hs_pairs_centered_cond(counts, neighbors, weights,
                                   num_umi, model, jobs=1, use_gpu=False):

    if use_gpu:
        return _compute_hs_pairs_centered_cond_gpu(
            counts, neighbors, weights, num_umi, model
        )

    genes = counts.index

    counts = counts.values
    neighbors = neighbors.values
    weights = weights.values
    num_umi = num_umi.values

    D = compute_node_degree(neighbors, weights)

    counts = create_centered_counts(counts, model, num_umi)

    eg2s = np.array(
        [
            conditional_eg2(counts[i], neighbors, weights)
            for i in range(counts.shape[0])
        ]
    )

    with numba_threads(jobs):
        lcs, lc_zs = _hs_pairs_cond_kernel(
            counts, neighbors, weights, eg2s,
            *_pair_index(counts.shape[0], include_diagonal=False))

    lc_maxs = compute_local_cov_pairs_max(D, counts)
    lcs = lcs / lc_maxs

    lcs = pd.DataFrame(lcs, index=genes, columns=genes)
    lc_zs = pd.DataFrame(lc_zs, index=genes, columns=genes)

    return lcs, lc_zs


def compute_local_cov_pairs_max(node_degrees, counts):
    """
    For a Genes x Cells count matrix, compute the maximal pair-wise correlation
    between any two genes
    """

    N_GENES = counts.shape[0]

    gene_maxs = np.zeros(N_GENES)
    for i in range(N_GENES):
        gene_maxs[i] = compute_local_cov_max(node_degrees, counts[i])

    result = gene_maxs.reshape((-1, 1)) + gene_maxs.reshape((1, -1))
    result = result / 2
    return result


def _conditional_eg2_gpu(X, W_sym):
    """GPU batch of conditional_eg2: eg2[g] = ||(W + W.T) @ x[g]||^2."""
    t1x_T = W_sym @ X.T
    return (t1x_T ** 2).sum(axis=0)


def _local_cov_pair_all_gpu(X, W):
    """GPU batch of local_cov_pair for ALL gene pairs via dense matmul.

    Returns the full G x G matrix of lc values (= local_cov_pair * 2).
    Diagonal is zeroed (no self-pairs).
    """
    import cupy as cp
    smoothed_T = W @ X.T                # (N, G)
    M = X @ smoothed_T                  # (G, G):  M[a,b] = x_a . (W @ x_b)
    lc_matrix = M + M.T                 # symmetrize: lc[a,b] = x_a.(Wx_b) + x_b.(Wx_a)
    cp.fill_diagonal(lc_matrix, 0)
    return lc_matrix


def _compute_hs_pairs_centered_cond_gpu(counts, neighbors, weights, num_umi, model):
    """
    GPU-accelerated version of compute_hs_pairs_centered_cond, batched
    over all gene pairs via dense matmul: M = X @ (W @ X.T), lc = M + M.T
    """
    import cupy as cp
    from .gpu import _require_gpu, _build_sparse_weight_matrix

    _require_gpu()

    genes = counts.index
    counts_np = counts.values
    neighbors_np = neighbors.values
    weights_np = weights.values
    num_umi_np = num_umi.values
    N_cells = counts_np.shape[1]

    D = compute_node_degree(neighbors_np, weights_np)

    centered_counts = create_centered_counts(counts_np, model, num_umi_np)

    X = cp.asarray(centered_counts)
    D_gpu = cp.asarray(D)
    W = _build_sparse_weight_matrix(neighbors_np, weights_np, shape=(N_cells, N_cells))
    W_sym = W + W.T

    eg2s = _conditional_eg2_gpu(X, W_sym)

    lc_matrix = _local_cov_pair_all_gpu(X, W)

    std_genes = eg2s ** 0.5
    Z_xy = lc_matrix / std_genes[:, None]
    Z_yx = lc_matrix / std_genes[None, :]
    lc_zs = cp.where(cp.abs(Z_xy) < cp.abs(Z_yx), Z_xy, Z_yx)

    gene_maxs = (D_gpu * X ** 2).sum(axis=1) / 2
    lc_maxs = (gene_maxs[:, None] + gene_maxs[None, :]) / 2
    lcs = lc_matrix / lc_maxs

    lcs_df = pd.DataFrame(cp.asnumpy(lcs), index=genes, columns=genes)
    lc_zs_df = pd.DataFrame(cp.asnumpy(lc_zs), index=genes, columns=genes)

    return lcs_df, lc_zs_df
