"""Backend utilities for Hotspot.

The batched paths in local_stats.py and local_stats_pairs.py run on numpy/scipy (CPU) or cupy (GPU).
"""

from types import ModuleType
from typing import Any, cast

import numpy as np

from .types import FloatArray

try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


def is_gpu_available() -> bool:
    """Check whether GPU acceleration is available (CuPy installed + CUDA device present)."""
    if not HAS_CUPY:
        return False
    try:
        _ = cp.cuda.Device(0).compute_capability
        return True
    except Exception:  # noqa: BLE001 - any CUDA failure means no usable GPU
        return False


def _require_gpu() -> None:
    """Raise an informative error if GPU is not available."""
    if not HAS_CUPY:
        raise ImportError(
            "CuPy is required for GPU acceleration. "
            "Install with: pip install hotspotsc[gpu]  "
            "(or install cupy separately for your CUDA version, "
            "e.g. pip install cupy-cuda12x)"
        )
    try:
        _ = cp.cuda.Device(0).compute_capability
    except Exception as e:
        raise RuntimeError(
            "No CUDA-capable GPU device found. GPU acceleration requires an NVIDIA GPU with CUDA support."
        ) from e


def get_backend(use_gpu: bool) -> ModuleType:
    """Return the array module (numpy or cupy) for the requested device."""
    if use_gpu:
        _require_gpu()
        return cast(ModuleType, cp)
    return np


def asnumpy(x: Any) -> FloatArray:
    """Move a numpy or cupy array to host memory as numpy."""
    return cp.asnumpy(x) if HAS_CUPY else np.asarray(x)
