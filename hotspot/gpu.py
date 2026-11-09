"""
CPU/GPU backend helpers for Hotspot: CuPy availability checks and the
backend choice used by local_stats.py and local_stats_pairs.py.
"""

import numpy as np

try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sparse

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


def is_gpu_available():
    """Check whether GPU acceleration is available (CuPy installed + CUDA device present)."""
    if not HAS_CUPY:
        return False
    try:
        cp.cuda.Device(0).compute_capability
        return True
    except Exception:
        return False


def _require_gpu():
    """Raise an informative error if GPU is not available."""
    if not HAS_CUPY:
        raise ImportError(
            "CuPy is required for GPU acceleration. "
            "Install with: pip install hotspotsc[gpu]  "
            "(or install cupy separately for your CUDA version, "
            "e.g. pip install cupy-cuda12x)"
        )
    try:
        cp.cuda.Device(0).compute_capability
    except Exception as e:
        raise RuntimeError(
            "No CUDA-capable GPU device found. "
            "GPU acceleration requires an NVIDIA GPU with CUDA support."
        ) from e


def backend(use_gpu):
    """(numpy or cupy, scipy.sparse or cupyx.scipy.sparse) for CPU or GPU.

    Code written against these runs on both: cupy mirrors the numpy API and
    cupyx.scipy.sparse mirrors scipy.sparse (csr_matrix, @, toarray).
    """
    if use_gpu:
        _require_gpu()
        return cp, cp_sparse
    import scipy.sparse as sp
    return np, sp


def array_module(x):
    """numpy or cupy, whichever holds x."""
    return cp if HAS_CUPY and isinstance(x, cp.ndarray) else np


def to_host(x):
    """numpy copy of a numpy or cupy array."""
    return x.get() if array_module(x) is not np else np.asarray(x)
