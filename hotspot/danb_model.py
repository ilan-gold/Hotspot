

def fit_gene_model(gene_counts, umi_counts):
    """Fit one gene (cells,) or a block of genes (genes, cells); numpy or cupy."""

    tj = gene_counts.sum(axis=-1, keepdims=True)
    tis = umi_counts
    total = tis.sum()

    N = gene_counts.shape[-1]

    min_size = 10**(-10)

    mu = tj*tis/total
    vv = (gene_counts - mu).var(axis=-1, keepdims=True)*(N/(N-1))
    my_rowvar = vv

    # size = (tj**2) * (tis**2).sum()/total**2 / ((N-1)*my_rowvar-tj)
    # regroup terms to protect against overflow errors
    size = ((tj**2) / total) * ((tis**2).sum() / total) / ((N-1)*my_rowvar-tj)

    size[(size < min_size) & (size >= 0)] = min_size
    size[size < 0] = 1e9    # Can't have negative dispersion

    var = mu*(1+mu/size)
    x2 = var+mu**2

    return mu, var, x2
