"""
Simplest Model - just assumes expression data is normal
UMI counts are regressed out
"""
from .gpu import array_module


def fit_gene_model(gene_counts, umi_counts):
    """Fit one gene (cells,) or a block of genes (genes, cells) at once."""
    xp = array_module(gene_counts)

    if xp.var(umi_counts) == 0:
        mu = xp.mean(gene_counts, axis=-1, keepdims=True) + xp.zeros_like(gene_counts)
    else:
        X = xp.stack((xp.ones_like(umi_counts), umi_counts), axis=1)
        B = gene_counts @ X @ xp.linalg.inv(X.T @ X)  # least squares, per gene
        mu = B @ X.T

    var = xp.var(gene_counts - mu, axis=-1, keepdims=True) + xp.zeros_like(mu)
    x2 = mu**2 + var

    return mu, var, x2
