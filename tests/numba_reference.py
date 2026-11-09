"""
Reference loop implementations removed from the library when the CPU path
switched to batched matrix code, copied verbatim from master as ground truth.
"""
import numpy as np
from numba import jit


@jit(nopython=True)
def local_cov_weights(x, neighbors, weights):
    out = 0

    for i in range(len(x)):
        for k in range(neighbors.shape[1]):

            j = neighbors[i, k]
            w_ij = weights[i, k]

            xi = x[i]
            xj = x[j]
            if xi == 0 or xj == 0 or w_ij == 0:
                out += 0
            else:
                out += xi * xj * w_ij

    return out


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


@jit(nopython=True)
def compute_moments_weights_slow(mu, x2, neighbors, weights):
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

            EG += wij * mu[i] * mu[j]

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
                            t1 = x2[i] * x2[j]
                        else:
                            t1 = x2[i] * mu[j] * mu[y]
                    elif i == y:
                        if j == x:
                            t1 = x2[i] * x2[j]
                        else:
                            t1 = x2[i] * mu[j] * mu[x]
                    else:  # i is unique since i can't equal j

                        if j == x:
                            t1 = mu[i] * x2[j] * mu[y]
                        elif j == y:
                            t1 = mu[i] * x2[j] * mu[x]
                        else:  # i and j are unique, no shared nodes
                            t1 = mu[i] * mu[j] * mu[x] * mu[y]

                    EG2_i += s * t1

        EG2 += EG2_i

    return EG, EG2


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
