from .types import FloatArray, GeneModelFit


def fit_gene_model(gene_counts: FloatArray, umi_counts: FloatArray) -> GeneModelFit:
    """Fit the depth-adjusted negative binomial (DANB) null model for one gene."""
    tj = gene_counts.sum()
    tis = umi_counts
    total = tis.sum()

    N = gene_counts.size

    min_size = 10 ** (-10)

    mu = tj * tis / total
    vv = (gene_counts - mu).var() * (N / (N - 1))
    my_rowvar = vv

    # size = (tj**2) * (tis**2).sum()/total**2 / ((N-1)*my_rowvar-tj)
    # regroup terms to protect against overflow errors
    size = float(((tj**2) / total) * ((tis**2).sum() / total) / ((N - 1) * my_rowvar - tj))

    if size < 0:  # Can't have negative dispersion
        size = 1e9

    if size < min_size and size >= 0:
        size = min_size

    var = mu * (1 + mu / size)
    x2 = var + mu**2

    return mu, var, x2
