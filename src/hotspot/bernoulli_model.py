import numpy as np
import pandas as pd

from .types import FloatArray, GeneModelFit, IntArray
from .utils import njit

N_BIN_TARGET = 30


def fit_gene_model_linear(gene_detects: FloatArray, umi_counts: FloatArray) -> GeneModelFit:
    """Fit detection probability per cell by linear regression of binned logit detection rate on log UMI count."""
    umi_count_bins, bins = pd.qcut(np.log10(umi_counts), N_BIN_TARGET, labels=False, retbins=True, duplicates="drop")
    bin_centers = np.array([bins[i] / 2 + bins[i + 1] / 2 for i in range(len(bins) - 1)])

    N_BIN = len(bin_centers)

    bin_detects = bin_gene_detection(gene_detects, umi_count_bins.astype(np.intp), N_BIN)

    lbin_detects = logit(bin_detects)

    X = np.ones((N_BIN, 2))
    X[:, 1] = bin_centers
    Y = lbin_detects

    b = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
    detect_p = ilogit(b[0] + b[1] * np.log10(umi_counts))

    mu = detect_p
    var = detect_p * (1 - detect_p)
    x2 = detect_p

    return mu, var, x2


fit_gene_model = fit_gene_model_linear


@njit
def logit(p: FloatArray) -> FloatArray:
    """Log-odds of `p`."""
    return np.log(p / (1 - p))


@njit
def ilogit(q: FloatArray) -> FloatArray:
    """Inverse of `logit`."""
    e: FloatArray = np.exp(q)
    return e / (1 + e)


@njit
def bin_gene_detection(gene_detects: FloatArray, umi_count_bins: IntArray, N_BIN: int) -> FloatArray:
    """Smoothed detection rate per UMI-count bin (pseudo-counts avoid 0% and 100%)."""
    bin_detects = np.zeros(N_BIN)
    bin_totals = np.zeros(N_BIN)

    for i in range(len(gene_detects)):
        x = gene_detects[i]
        bin_i = umi_count_bins[i]
        bin_detects[bin_i] += x
        bin_totals[bin_i] += 1

    # Need to account for 0% detects
    #    Add 1 to numerator and denominator
    # Need to account for 100% detects
    #    Add 1 to denominator

    return (bin_detects + 1) / (bin_totals + 2)
