"""Type aliases used across Hotspot."""

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csc_array, csc_matrix, csr_array, csr_matrix

type FloatArray = NDArray[np.float64]
"""Dense float64 array."""
type IntArray = NDArray[np.integer]
"""Dense integer array, e.g. neighbor indices."""
type Counts = FloatArray | csr_matrix | csc_matrix | csr_array | csc_array
"""Count matrix: dense or CSR/CSC sparse."""
type Model = Literal["danb", "bernoulli", "normal", "none"]
"""Null model of gene expression."""
type GeneModelFit = tuple[FloatArray, FloatArray, FloatArray]
"""Per-cell ``(mu, var, x2)`` from fitting a gene's null model."""
