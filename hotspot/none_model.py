from .gpu import array_module


def fit_gene_model(gene_counts, umi_counts):
    xp = array_module(gene_counts)

    mu = xp.zeros_like(gene_counts)
    var = xp.ones_like(gene_counts)
    x2 = xp.ones_like(gene_counts)

    return mu, var, x2
