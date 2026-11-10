import warnings
from collections.abc import Sequence
from typing import cast

import anndata
import numpy as np
import pandas as pd
from matplotlib.colors import Colormap
from scipy.sparse import csr_array, csr_matrix, issparse
from tqdm import tqdm

from . import modules
from .gpu import is_gpu_available
from .knn import (
    TreeNode,
    make_weights_non_redundant,
    neighbors_and_weights,
    neighbors_and_weights_from_distances,
    tree_neighbors_and_weights,
)
from .local_stats import compute_hs
from .local_stats_pairs import compute_hs_pairs_centered_cond
from .plots import local_correlation_plot
from .types import Counts, FloatArray, Model


class Hotspot:
    """Gene autocorrelation, local correlation and module analysis over a cell-similarity graph."""

    neighbors: pd.DataFrame
    """Set by :meth:`create_knn_graph`"""
    weights: pd.DataFrame
    """Set by :meth:`create_knn_graph`"""
    results: pd.DataFrame
    """Set by :meth:`compute_autocorrelations`"""
    local_correlation_c: pd.DataFrame
    """Set by :meth:`compute_local_correlations`"""

    def __init__(
        self,
        adata: anndata.AnnData,
        layer_key: str | None = None,
        model: Model = "danb",
        latent_obsm_key: str | None = None,
        distances_obsp_key: str | None = None,
        tree: TreeNode | None = None,
        umi_counts_obs_key: str | None = None,
        use_gpu: bool = False,
    ) -> None:
        """Initialize a Hotspot object for analysis

        Either `latent` or `tree` or `distances` is required.

        Parameters
        ----------
        adata : anndata.AnnData
            Count matrix (shape is cells by genes)
        layer_key: str
            Key in adata.layers with count data, uses adata.X if None.
        model : string, optional
            Specifies the null model to use for gene expression.
            Valid choices are:

                - 'danb': Depth-Adjusted Negative Binomial
                - 'bernoulli': Models probability of detection
                - 'normal': Depth-Adjusted Normal
                - 'none': Assumes data has been pre-standardized

        latent_obsm_key : string, optional
            Latent space encoding cell-cell similarities with euclidean
            distances.  Shape is (cells x dims). Input is key in adata.obsm
        distances_obsp_key : pandas.DataFrame, optional
            Distances encoding cell-cell similarities directly
            Shape is (cells x cells). Input is key in adata.obsp
        tree : ete3.coretype.tree.TreeNode
            Root tree node.  Can be created using ete3.Tree
        umi_counts_obs_key : str
            Total umi count per cell.  Used as a size factor.
            If omitted, the sum over genes in the counts matrix is used
        use_gpu : bool, optional
            Whether to use GPU acceleration via CuPy for embarrassingly
            parallel operations. Requires CuPy and a CUDA-capable GPU.
            Default is False.
        """
        counts = self._counts_from_anndata(adata, layer_key)
        distances = adata.obsp[distances_obsp_key] if distances_obsp_key is not None else None
        latent_arr = adata.obsm[latent_obsm_key] if latent_obsm_key is not None else None
        umi_counts_obs = adata.obs[umi_counts_obs_key] if umi_counts_obs_key is not None else None

        if latent_arr is None and distances is None and tree is None:
            raise ValueError(
                "Neither `latent_obsm_key` or `tree` or `distances_obsp_key` arguments were supplied.  One of these is required"
            )

        if latent_arr is not None and distances is not None:
            raise ValueError(
                "Both `latent_obsm_key` and `distances_obsp_key` provided - only one of these should be provided."
            )

        if latent_arr is not None and tree is not None:
            raise ValueError("Both `latent_obsm_key` and `tree` provided - only one of these should be provided.")

        if distances is not None and tree is not None:
            raise ValueError("Both `distances_obsp_key` and `tree` provided - only one of these should be provided.")

        latent = pd.DataFrame(np.asarray(latent_arr), index=adata.obs_names) if latent_arr is not None else None

        # because of transpose we check if its csr
        if issparse(counts) and not isinstance(counts, csr_matrix | csr_array):
            warnings.warn("Hotspot will work faster when counts are a csr sparse matrix.", stacklevel=2)

        if tree is not None:
            try:
                all_leaves = []
                for x in tree:
                    if x.is_leaf():
                        all_leaves.append(x.name)
            except Exception as e:
                raise ValueError("Can't parse supplied tree") from e

            if len(all_leaves) != counts.shape[1] or len(set(all_leaves) & set(adata.obs_names)) != len(all_leaves):
                raise ValueError("Tree leaf labels don't match columns in supplied counts matrix")

        if umi_counts_obs is None:
            # asarray/ravel handles sparse matrix outputs of sum
            umi_counts = pd.Series(np.asarray(counts.sum(axis=0)).ravel(), index=adata.obs_names)
        else:
            if umi_counts_obs.size != counts.shape[1]:  # pragma: no cover
                raise ValueError(
                    f"`adata.obs[{umi_counts_obs_key!r}]` has {umi_counts_obs.size} entries, expected {counts.shape[1]} (one per cell)."
                )
            umi_counts = umi_counts_obs

        valid_models = {"danb", "bernoulli", "normal", "none"}
        if model not in valid_models:
            raise ValueError(f"Input `model` should be one of {valid_models}")

        if isinstance(counts, np.ndarray):
            # For a dense matrix, check if all values in each row are identical
            valid_genes = ~(np.all(counts == counts[:, [0]], axis=1))
        else:
            # For a sparse matrix, check if all values in each row are identical
            # A row (gene) is considered valid if it has more than one unique value.
            row_min = counts.min(axis=1).toarray().flatten()
            row_max = counts.max(axis=1).toarray().flatten()
            valid_genes = row_min != row_max  # Valid if min and max are not equal, indicating variation

        # valid_genes is now a boolean array indicating which rows (genes) have non-identical values.

        n_invalid = counts.shape[0] - valid_genes.sum()
        if n_invalid > 0:
            raise ValueError("\nDetected genes with zero variance. Please filter adata and reinitialize.")

        self.adata = adata
        self.layer_key = layer_key

        self.counts = counts
        self.latent = latent
        self.distances = distances
        self.tree = tree
        self.model = model

        self.umi_counts = umi_counts

        self.graph = None
        self.modules: pd.Series | None = None
        self.local_correlation_z: pd.DataFrame | None = None
        self.linkage: FloatArray | None = None
        self.module_scores: pd.DataFrame | None = None

        if use_gpu:
            if not is_gpu_available():
                raise RuntimeError(
                    "use_gpu=True but GPU is not available. Ensure CuPy is installed and a CUDA GPU is present."
                )
        self.use_gpu = use_gpu

    @classmethod
    def legacy_init(
        cls,
        counts: pd.DataFrame,
        model: Model = "danb",
        latent: pd.DataFrame | None = None,
        distances: pd.DataFrame | None = None,
        tree: TreeNode | None = None,
        umi_counts: pd.Series | FloatArray | None = None,
    ) -> "Hotspot":
        """
        Initialize a Hotspot object for analysis using legacy method

        Either `latent` or `tree` or `distances` is required.


        Parameters
        ----------
        counts : pandas.DataFrame
            Count matrix (shape is genes x cells)
        model : string, optional
            Specifies the null model to use for gene expression.
            Valid choices are:

                - 'danb': Depth-Adjusted Negative Binomial
                - 'bernoulli': Models probability of detection
                - 'normal': Depth-Adjusted Normal
                - 'none': Assumes data has been pre-standardized

        latent : pandas.DataFrame, optional
            Latent space encoding cell-cell similarities with euclidean
            distances.  Shape is (cells x dims)
        distances : pandas.DataFrame, optional
            Distances encoding cell-cell similarities directly
            Shape is (cells x cells)
        tree : ete3.coretype.tree.TreeNode
            Root tree node.  Can be created using ete3.Tree
        umi_counts : pandas.Series, optional
            Total umi count per cell.  Used as a size factor.
            If omitted, the sum over genes in the counts matrix is used

        Examples
        --------
        >>> gene_exp = pd.read_csv(path, index_col=0)  # genes by cells
        >>> latent = pd.read_csv(latent_path, index_col=0)  # cells by dims
        >>> hs = hotspot.Hotspot.legacy_init(gene_exp, model="normal", latent=latent)
        """
        if latent is None and distances is None and tree is None:
            raise ValueError(
                "Neither `latent` or `tree` or `distance` arguments were supplied.  One of these is required"
            )

        if latent is not None and distances is not None:
            raise ValueError("Both `latent` and `distances` provided - only one of these should be provided.")

        if latent is not None and tree is not None:
            raise ValueError("Both `latent` and `tree` provided - only one of these should be provided.")

        if distances is not None and tree is not None:
            raise ValueError("Both `distances` and `tree` provided - only one of these should be provided.")

        if latent is not None:
            if counts.shape[1] != latent.shape[0]:
                if counts.shape[0] == latent.shape[0]:
                    raise ValueError("`counts` input should be a Genes x Cells dataframe.  Maybe needs transpose?")
                raise ValueError("Size mismatch counts/latent. Columns of `counts` should match rows of `latent`.")

        if distances is not None:
            if distances.shape != (counts.shape[1], counts.shape[1]):  # pragma: no cover
                raise ValueError(
                    f"`distances` has shape {distances.shape}, expected (cells x cells) = {(counts.shape[1], counts.shape[1])}."
                )

        if umi_counts is None:
            umi_counts = counts.sum(axis=0)
        else:
            if umi_counts.size != counts.shape[1]:  # pragma: no cover
                raise ValueError(
                    f"`umi_counts` has {umi_counts.size} entries, expected {counts.shape[1]} (one per cell)."
                )

        if not isinstance(umi_counts, pd.Series):
            umi_counts = pd.Series(umi_counts)

        valid_genes = counts.var(axis=1) > 0
        n_invalid = counts.shape[0] - valid_genes.sum()
        if n_invalid > 0:
            counts = counts.loc[valid_genes]
            print(f"\nRemoving {n_invalid} undetected/non-varying genes")

        input_adata = anndata.AnnData(counts)
        input_adata = input_adata.transpose()
        tc_key = "total_counts"
        input_adata.obs[tc_key] = umi_counts.values
        dkey = "distances"
        if distances is not None:
            input_adata.obsp[dkey] = distances
            dist_input = True
        else:
            dist_input = False
        lkey = "latent"
        if latent is not None:
            input_adata.obsm[lkey] = np.asarray(latent)
            latent_input = True
        else:
            latent_input = False

        return cls(
            input_adata,
            model=model,
            latent_obsm_key=lkey if latent_input else None,
            distances_obsp_key=dkey if dist_input else None,
            umi_counts_obs_key=tc_key,
            tree=tree,
        )

    @staticmethod
    def _counts_from_anndata(adata: anndata.AnnData, layer_key: str | None) -> Counts:
        """Counts as a genes x cells matrix (dense or sparse, as stored)."""
        counts = adata.layers[layer_key] if layer_key is not None else adata.X
        # handles adata view
        # as sparse matrix in view is just a sparse matrix, while dense is ArrayView
        # (cast: AnnData admits more array types than Hotspot supports)
        stored = cast(Counts, counts if issparse(counts) else np.asarray(counts))
        return stored.transpose()

    @staticmethod
    def _dense_counts_from_anndata(adata: anndata.AnnData, layer_key: str | None) -> FloatArray:
        """Counts as a dense genes x cells array."""
        counts = Hotspot._counts_from_anndata(adata, layer_key)
        return counts if isinstance(counts, np.ndarray) else counts.toarray()

    def create_knn_graph(
        self,
        weighted_graph: bool = False,
        n_neighbors: int = 30,
        neighborhood_factor: float = 3,
        approx_neighbors: bool = True,
    ) -> None:
        """Create's the KNN graph and graph weights

        The resulting matrices containing the neighbors and weights are
        stored in the object at `self.neighbors` and `self.weights`

        Parameters
        ----------
        weighted_graph: bool
            Whether or not to create a weighted graph
        n_neighbors: int
            Neighborhood size
        neighborhood_factor: float
            Used when creating a weighted graph.  Sets how quickly weights decay
            relative to the distances within the neighborhood.  The weight for
            a cell with a distance d will decay as exp(-d/D) where D is the distance
            to the `n_neighbors`/`neighborhood_factor`-th neighbor.
        approx_neighbors: bool
            Use approximate nearest neighbors or exact scikit-learn neighbors. Only
            when hotspot initialized with `latent`.
        """
        if self.latent is not None:
            neighbors, weights = neighbors_and_weights(
                self.latent,
                n_neighbors=n_neighbors,
                neighborhood_factor=neighborhood_factor,
                approx_neighbors=approx_neighbors,
            )
        elif self.tree is not None:
            if weighted_graph:
                raise ValueError("When using `tree` as the metric space, `weighted_graph=True` is not supported")
            neighbors, weights = tree_neighbors_and_weights(
                self.tree, n_neighbors=n_neighbors, cell_labels=self.adata.obs_names
            )
        else:
            if self.distances is None:  # pragma: no cover
                raise ValueError("No `latent`, `tree` or `distances` set; this is guaranteed by `__init__`.")
            neighbors, weights = neighbors_and_weights_from_distances(
                self.distances,
                cell_index=self.adata.obs_names,
                n_neighbors=n_neighbors,
                neighborhood_factor=neighborhood_factor,
            )

        neighbors = neighbors.loc[self.adata.obs_names]
        weights = weights.loc[self.adata.obs_names]

        self.neighbors = neighbors

        if not weighted_graph:
            weights = pd.DataFrame(
                np.ones_like(weights.values),
                index=weights.index,
                columns=weights.columns,
            )

        self.weights = pd.DataFrame(
            make_weights_non_redundant(neighbors.values, weights.values),
            index=neighbors.index,
            columns=neighbors.columns,
        )

    def _compute_hotspot(self, jobs: int = 1) -> pd.DataFrame:
        """Perform feature selection using local autocorrelation

        In addition to returning output, this also stores the output
        in `self.results`.

        Alias for `self.compute_autocorrelations`

        Parameters
        ----------
        jobs: int
            Number of parallel jobs to run

        Returns
        -------
        results : pandas.DataFrame
            A dataframe with four columns:

              - C: Scaled -1:1 autocorrelation coeficients
              - Z: Z-score for autocorrelation
              - Pval:  P-values computed from Z-scores
              - FDR:  Q-values using the Benjamini-Hochberg procedure

            Gene ids are in the index

        """
        results = compute_hs(
            self.counts,
            self.neighbors,
            self.weights,
            self.umi_counts,
            self.model,
            genes=self.adata.var_names,
            centered=True,
            jobs=jobs,
            use_gpu=self.use_gpu,
        )

        self.results = results

        return self.results

    def compute_autocorrelations(self, jobs: int = 1) -> pd.DataFrame:
        """Perform feature selection using local autocorrelation

        In addition to returning output, this also stores the output
        in `self.results`

        Parameters
        ----------
        jobs: int
            Number of parallel jobs to run

        Returns
        -------
        results : pandas.DataFrame
            A dataframe with four columns:

              - C: Scaled -1:1 autocorrelation coeficients
              - Z: Z-score for autocorrelation
              - Pval:  P-values computed from Z-scores
              - FDR:  Q-values using the Benjamini-Hochberg procedure

            Gene ids are in the index

        """
        return self._compute_hotspot(jobs)

    def compute_local_correlations(self, genes: Sequence[str] | pd.Index, jobs: int = 1) -> pd.DataFrame:
        """Define gene-gene relationships with pair-wise local correlations

        In addition to returning output, this method stores its result
        in `self.local_correlation_z`

        Parameters
        ----------
        genes: iterable of str
            gene identifies to compute local correlations on
            should be a smaller subset of all genes
        jobs: int
            Number of parallel jobs to run

        Returns
        -------
        local_correlation_z : pandas.DataFrame
                local correlation Z-scores between genes
                shape is genes x genes
        """
        print(f"Computing pair-wise local correlation on {len(genes)} features...")
        adata_genes = self.adata[:, genes]
        counts_dense = pd.DataFrame(
            self._dense_counts_from_anndata(adata_genes, self.layer_key),
            index=adata_genes.var_names,
            columns=adata_genes.obs_names,
        )

        lc, lcz = compute_hs_pairs_centered_cond(
            counts_dense,
            self.neighbors,
            self.weights,
            self.umi_counts,
            self.model,
            jobs=jobs,
            use_gpu=self.use_gpu,
        )

        self.local_correlation_c = lc
        self.local_correlation_z = lcz

        return lcz

    def create_modules(
        self, min_gene_threshold: int = 20, core_only: bool = True, fdr_threshold: float = 0.05
    ) -> pd.Series:
        """Groups genes into modules

        In addition to being returned, the results of this method are retained
        in the object at `self.modules`.  Additionally, the linkage matrix
        (in the same form as that of scipy.cluster.hierarchy.linkage) is saved
        in `self.linkage` for plotting or manual clustering.

        Parameters
        ----------
        min_gene_threshold: int
            Controls how small modules can be.  Increase if there are too many
            modules being formed.  Decrease if substructre is not being captured
        core_only: bool
            Whether or not to assign ambiguous genes to a module or leave unassigned
        fdr_threshold: float
            Correlation theshold at which to stop assigning genes to modules

        Returns
        -------
        modules: pandas.Series
            Maps gene to module number.  Unassigned genes are indicated with -1


        """
        if self.local_correlation_z is None:
            raise RuntimeError("Run `compute_local_correlations` before `create_modules`.")
        gene_modules, Z = modules.compute_modules(
            self.local_correlation_z,
            min_gene_threshold=min_gene_threshold,
            fdr_threshold=fdr_threshold,
            core_only=core_only,
        )

        self.modules = gene_modules
        self.linkage = Z

        return gene_modules

    def calculate_module_scores(self) -> pd.DataFrame:
        """Calculate Module Scores

        In addition to returning its result, this method stores
        its output in the object at `self.module_scores`

        Returns
        -------
        module_scores: pandas.DataFrame
            Scores for each module for each gene
            Dimensions are genes x modules

        """
        if self.modules is None:
            raise RuntimeError("Run `create_modules` before `calculate_module_scores`.")
        gene_modules = self.modules
        modules_to_compute = sorted([x for x in gene_modules.unique() if x != -1])

        print(f"Computing scores for {len(modules_to_compute)} modules...")

        scores_per_module = {}
        for module in tqdm(modules_to_compute):
            module_genes = gene_modules.index[gene_modules == module]

            counts_dense = self._dense_counts_from_anndata(self.adata[:, module_genes], self.layer_key)

            scores = modules.compute_scores(
                counts_dense,
                self.model,
                self.umi_counts.to_numpy(),
                self.neighbors.to_numpy(),
                self.weights.to_numpy(),
            )

            scores_per_module[module] = scores

        module_scores = pd.DataFrame(scores_per_module)
        module_scores.index = self.adata.obs_names

        self.module_scores = module_scores

        return module_scores

    def plot_local_correlations(
        self,
        mod_cmap: str | Colormap = "tab10",
        vmin: float = -8,
        vmax: float = 8,
        z_cmap: str | Colormap = "RdBu_r",
        yticklabels: bool = False,
    ) -> None:
        """Plots a clustergrid of the local correlation values

        Parameters
        ----------
        mod_cmap: valid matplotlib colormap str or object
            discrete colormap for module assignments on the left side
        vmin: float
            minimum value for colorscale for Z-scores
        vmax: float
            maximum value for colorscale for Z-scores
        z_cmap: valid matplotlib colormap str or object
            continuous colormap for correlation Z-scores
        yticklabels: bool
            Whether or not to plot all gene labels
            Default is false as there are too many.  However
            if using this plot interactively you may with to set
            to true so you can zoom in and read gene names
        """
        if self.local_correlation_z is None or self.modules is None or self.linkage is None:
            raise RuntimeError("Run `compute_local_correlations` and `create_modules` before plotting.")
        return local_correlation_plot(
            self.local_correlation_z,
            self.modules,
            self.linkage,
            mod_cmap=mod_cmap,
            vmin=vmin,
            vmax=vmax,
            z_cmap=z_cmap,
            yticklabels=yticklabels,
        )
