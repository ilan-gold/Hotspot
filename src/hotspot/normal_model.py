"""Simplest Model - just assumes expression data is normal.

UMI counts are regressed out.
"""

import numpy as np

from .types import FloatArray, GeneModelFit


def fit_gene_model(gene_counts: FloatArray, umi_counts: FloatArray) -> GeneModelFit:
    """Fit a normal null model whose mean is linear in UMI count."""
    X = np.vstack((np.ones(len(umi_counts)), umi_counts)).T
    y = gene_counts.reshape((-1, 1))

    if umi_counts.var() == 0:
        mu = np.repeat(gene_counts.mean(), len(umi_counts))
        var = np.repeat(gene_counts.var(), len(umi_counts))
        return mu, var, mu**2 + var

    B = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    mu_col = X.dot(B)

    var = np.repeat((y - mu_col).var(), len(umi_counts))

    mu = mu_col.ravel()

    return mu, var, mu**2 + var
