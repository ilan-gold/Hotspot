import numpy as np

from .types import FloatArray, GeneModelFit


def fit_gene_model(gene_counts: FloatArray, umi_counts: FloatArray) -> GeneModelFit:
    """Null model for pre-standardized data: zero mean, unit variance."""
    N = gene_counts.size

    mu = np.zeros(N)
    var = np.ones(N)
    x2 = np.ones(N)

    return mu, var, x2
