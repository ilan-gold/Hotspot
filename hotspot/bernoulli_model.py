import pandas as pd
import numpy as np
from .gpu import array_module, to_host

N_BIN_TARGET = 30


def umi_bins(umi_counts):
    """(cells x bins one-hot, bin centers) for log10 UMI quantile bins.

    Depends only on umi_counts, so compute it once and pass it to
    fit_gene_model_linear for every block of genes.
    """
    xp = array_module(umi_counts)
    log_umi = np.log10(to_host(umi_counts))
    idx, bins = pd.qcut(
        log_umi, N_BIN_TARGET, labels=False, retbins=True, duplicates='drop'
    )
    centers = bins[:-1] / 2 + bins[1:] / 2
    onehot = idx[:, None] == np.arange(len(centers))
    return xp.asarray(onehot, dtype=xp.float64), xp.asarray(centers)


def fit_gene_model_linear(gene_detects, umi_counts, bins=None):
    """Fit one gene (cells,) or a block of genes (genes, cells) at once."""
    xp = array_module(gene_detects)
    onehot, bin_centers = umi_bins(umi_counts) if bins is None else bins

    # Need to account for 0% detects
    #    Add 1 to numerator and denominator
    # Need to account for 100% detects
    #    Add 1 to denominator
    bin_detects = (gene_detects @ onehot + 1) / (xp.sum(onehot, axis=0) + 2)

    lbin_detects = xp.log(bin_detects / (1 - bin_detects))

    X = xp.stack((xp.ones_like(bin_centers), bin_centers), axis=1)
    b = lbin_detects @ X @ xp.linalg.inv(X.T @ X)  # least squares, per gene
    q = b[..., 0:1] + b[..., 1:2] * xp.log10(umi_counts)
    detect_p = xp.exp(q) / (1 + xp.exp(q))

    mu = detect_p
    var = detect_p * (1 - detect_p)
    x2 = detect_p

    return mu, var, x2


fit_gene_model = fit_gene_model_linear
