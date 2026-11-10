from __future__ import annotations

import warnings
from collections.abc import Iterator
from math import ceil
from typing import Protocol

import numpy as np
import pandas as pd
from pynndescent import NNDescent
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from .types import Counts, FloatArray, IntArray
from .utils import njit


class TreeNode(Protocol):
    """The subset of `ete3.TreeNode` that Hotspot uses."""

    name: str
    children: list[TreeNode]
    up: TreeNode

    def __iter__(self) -> Iterator[TreeNode]: ...
    def is_leaf(self) -> bool:
        """Whether this node has no children."""

    def is_root(self) -> bool:
        """Whether this node has no parent."""


def neighbors_and_weights(
    data: pd.DataFrame, n_neighbors: int = 30, neighborhood_factor: float = 3, approx_neighbors: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Computes nearest neighbors and associated weights for data.

    Uses euclidean distance between rows of `data`.

    Parameters
    ----------
    data: pandas.Dataframe num_cells x num_features

    Returns
    -------
    neighbors:      pandas.Dataframe num_cells x n_neighbors
    weights:  pandas.Dataframe num_cells x n_neighbors

    """
    coords = data.values

    if approx_neighbors:
        # pynndescent first neighbor is self, unlike sklearn
        index = NNDescent(coords, n_neighbors=n_neighbors + 1)
        ind, dist = index.neighbor_graph
        ind, dist = ind[:, 1:], dist[:, 1:]
    else:
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="ball_tree").fit(coords)
        dist, ind = nbrs.kneighbors()

    weights = compute_weights(dist, neighborhood_factor=neighborhood_factor)

    neighbors = pd.DataFrame(ind, index=data.index)
    return neighbors, pd.DataFrame(weights, index=neighbors.index, columns=neighbors.columns)


def neighbors_and_weights_from_distances(
    distances: pd.DataFrame | Counts,
    cell_index: pd.Index,
    n_neighbors: int = 30,
    neighborhood_factor: float = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Computes nearest neighbors and associated weights using provided distance matrix directly.

    Parameters
    ----------
    distances: pandas.Dataframe num_cells x num_cells

    Returns
    -------
    neighbors:      pandas.Dataframe num_cells x n_neighbors
    weights:  pandas.Dataframe num_cells x n_neighbors

    """
    if isinstance(distances, pd.DataFrame):
        distances = distances.values

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="brute", metric="precomputed").fit(distances)
    try:
        dist, ind = nbrs.kneighbors()
    # already is a neighbors graph
    except ValueError:
        nn = int((distances[[0]] > 0).sum())
        warnings.warn(
            f"Provided cell-cell distance graph is likely a {nn}-neighbors graph. Using {nn} precomputed neighbors.",
            stacklevel=2,
        )
        dist, ind = nbrs.kneighbors(n_neighbors=nn - 1)

    weights = compute_weights(dist, neighborhood_factor=neighborhood_factor)

    neighbors = pd.DataFrame(ind, index=cell_index)
    return neighbors, pd.DataFrame(weights, index=neighbors.index, columns=neighbors.columns)


def compute_weights(distances: FloatArray, neighborhood_factor: float = 3) -> FloatArray:
    """Computes weights on the nearest neighbors based on a gaussian kernel and their distances.

    Kernel width is set to the num_neighbors / neighborhood_factor's distance

    distances:  cells x neighbors ndarray
    neighborhood_factor: float

    returns weights:  cells x neighbors ndarray

    """
    radius_ii = ceil(distances.shape[1] / neighborhood_factor)

    sigma = distances[:, [radius_ii - 1]]
    sigma[sigma == 0] = 1

    weights = np.exp(-1 * distances**2 / sigma**2)

    wnorm = weights.sum(axis=1, keepdims=True)
    wnorm[wnorm == 0] = 1.0
    return weights / wnorm


@njit
def compute_node_degree(neighbors: IntArray, weights: FloatArray) -> FloatArray:
    """Weighted degree of every cell in the (undirected) KNN graph."""
    D = np.zeros(neighbors.shape[0])

    for i in range(neighbors.shape[0]):
        for k in range(neighbors.shape[1]):
            j = neighbors[i, k]
            w_ij = weights[i, k]

            D[i] += w_ij
            D[j] += w_ij

    return D


@njit
def make_weights_non_redundant(neighbors: IntArray, weights: FloatArray) -> FloatArray:
    """Merge each reciprocal edge's weight into one direction so every edge is counted once."""
    w_no_redundant = weights.copy()

    for i in range(neighbors.shape[0]):
        for k in range(neighbors.shape[1]):
            j = neighbors[i, k]

            if j < i:
                continue

            for k2 in range(neighbors.shape[1]):
                if neighbors[j, k2] == i:
                    w_ji = w_no_redundant[j, k2]
                    w_no_redundant[j, k2] = 0
                    w_no_redundant[i, k] += w_ji

    return w_no_redundant


# Neighbors and weights given an ete3 tree instead


def _search(current_node: TreeNode, previous_node: TreeNode | None, distance: int) -> dict[str, int]:
    if current_node.is_root():
        nodes_to_search = current_node.children
    else:
        nodes_to_search = [*current_node.children, current_node.up]
    nodes_to_search = [x for x in nodes_to_search if x != previous_node]

    if len(nodes_to_search) == 0:
        return {current_node.name: distance}

    result = {}
    for new_node in nodes_to_search:
        res = _search(new_node, current_node, distance + 1)
        for k, v in res.items():
            result[k] = v

    return result


def _knn(leaf: TreeNode, K: int) -> list[str]:
    dists = pd.Series(_search(leaf, None, 0), dtype=float)
    dists = dists + np.random.rand(len(dists)) * 0.9  # to break ties randomly

    return dists.sort_values().index[0:K].tolist()


def tree_neighbors_and_weights(
    tree: TreeNode, n_neighbors: int, cell_labels: pd.Index
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Computes nearest neighbors and associated weights for data.

    Uses distance along the tree object.

    Names of the leaves of the tree must match the columns in counts

    Parameters
    ----------
    tree: ete3.TreeNode
        The root of the tree
    n_neighbors: int
        Number of neighbors to find
    cell_labels
        Labels of cells (barcodes)

    Returns
    -------
    neighbors:      pandas.Dataframe num_cells x n_neighbors
    weights:  pandas.Dataframe num_cells x n_neighbors

    """
    K = n_neighbors

    all_leaves = []
    for x in tree:
        if x.is_leaf():
            all_leaves.append(x)

    all_neighbors = {}

    for leaf in tqdm(all_leaves):
        neighbors = _knn(leaf, K)
        all_neighbors[leaf.name] = neighbors

    cell_ix = {c: i for i, c in enumerate(cell_labels)}

    knn_ix = np.zeros((len(all_neighbors), K), dtype="int64")
    for cell in all_neighbors:
        row = cell_ix[cell]
        nn_ix = [cell_ix[x] for x in all_neighbors[cell]]
        knn_ix[row, :] = nn_ix

    neighbors_df = pd.DataFrame(knn_ix, index=cell_labels)
    weights = pd.DataFrame(np.ones_like(neighbors_df, dtype="float64"), index=cell_labels)

    return neighbors_df, weights
