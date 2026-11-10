"""
GPU equivalence and performance tests for Hotspot.

These tests verify that GPU-accelerated computations produce results
numerically equivalent to their CPU counterparts, and benchmark speedup.

All tests are marked with @pytest.mark.gpu and are excluded from CI by
default (no GPU resources). Run explicitly with:

    pytest -m gpu -v

or to include GPU tests alongside regular tests:

    pytest -m '' -v

Tolerance rationale:
    GPU sparse matmul and CPU sequential Numba loops accumulate
    floating-point sums in different order, yielding ~1e-7 relative
    differences at float64 precision. We use rtol=1e-6 (10x margin).

Note:
    All tests share a single KNN graph between CPU and GPU runs to
    isolate the GPU computation from pynndescent's stochastic KNN
    construction.
"""

import time

import anndata
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csc_matrix

from hotspot import Hotspot
from hotspot.gpu import is_gpu_available
from hotspot.types import Model

# Skip the entire module if no GPU is available
pytestmark = pytest.mark.gpu

GPU_AVAILABLE = is_gpu_available()
skip_no_gpu = pytest.mark.skipif(not GPU_AVAILABLE, reason="No GPU available")

# Tolerance for GPU vs CPU comparison (see module docstring)
RTOL = 1e-6
ATOL = 1e-10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_test_data(n_cells: int, n_genes: int, n_dim: int = 10, seed: int = 42) -> tuple[anndata.AnnData, list[str]]:
    """Create reproducible test data for GPU equivalence tests."""
    rng = np.random.RandomState(seed)

    latent = rng.normal(size=(n_cells, n_dim))
    latent_df = pd.DataFrame(
        latent,
        index=[f"Cell{i + 1}" for i in range(n_cells)],
    )

    umi_counts = np.floor(np.exp(rng.normal(loc=np.log(2000), scale=0.3, size=n_cells)))

    gene_exp = rng.rand(n_genes, n_cells) * 10 + 0.1
    gene_names = [f"Gene{i + 1}" for i in range(n_genes)]
    gene_exp_df = pd.DataFrame(gene_exp, index=gene_names, columns=latent_df.index)

    adata = anndata.AnnData(gene_exp_df.T)
    adata.layers["sparse"] = csc_matrix(gene_exp.T)
    adata.obsm["latent"] = latent
    adata.obs["umi_counts"] = umi_counts

    return adata, gene_names


def _make_cpu_gpu_pair(
    adata: anndata.AnnData, model: Model, layer_key: str | None = None, n_neighbors: int = 30
) -> tuple[Hotspot, Hotspot]:
    """Create CPU and GPU Hotspot objects sharing the same KNN graph."""
    hs_cpu = Hotspot(
        adata,
        model=model,
        latent_obsm_key="latent",
        umi_counts_obs_key="umi_counts",
        layer_key=layer_key,
        use_gpu=False,
    )
    hs_cpu.create_knn_graph(False, n_neighbors=n_neighbors)

    hs_gpu = Hotspot(
        adata,
        model=model,
        latent_obsm_key="latent",
        umi_counts_obs_key="umi_counts",
        layer_key=layer_key,
        use_gpu=True,
    )
    # Share the same KNN graph so we compare GPU vs CPU computation only
    hs_gpu.neighbors = hs_cpu.neighbors
    hs_gpu.weights = hs_cpu.weights

    return hs_cpu, hs_gpu


# ---------------------------------------------------------------------------
# compute_autocorrelations: GPU vs CPU equivalence (per model)
# ---------------------------------------------------------------------------


@skip_no_gpu
@pytest.mark.parametrize("model", ["danb", "bernoulli", "normal", "none"])
def test_compute_autocorrelations_equivalence(model: Model) -> None:
    """GPU and CPU compute_autocorrelations produce equivalent results."""
    adata, genes = _make_test_data(n_cells=200, n_genes=20)
    hs_cpu, hs_gpu = _make_cpu_gpu_pair(adata, model)

    results_cpu = hs_cpu.compute_autocorrelations()
    results_gpu = hs_gpu.compute_autocorrelations()

    # Align index ordering
    results_cpu = results_cpu.loc[sorted(results_cpu.index)]
    results_gpu = results_gpu.loc[sorted(results_gpu.index)]

    assert results_cpu.shape == results_gpu.shape

    np.testing.assert_allclose(
        results_gpu["Z"].to_numpy(),
        results_cpu["Z"].to_numpy(),
        rtol=RTOL,
        atol=ATOL,
        err_msg=f"Z-score mismatch for model={model}",
    )

    np.testing.assert_allclose(
        results_gpu["C"].to_numpy(),
        results_cpu["C"].to_numpy(),
        rtol=RTOL,
        atol=ATOL,
        err_msg=f"C value mismatch for model={model}",
    )

    np.testing.assert_allclose(
        results_gpu["Pval"].to_numpy(),
        results_cpu["Pval"].to_numpy(),
        rtol=1e-4,
        atol=1e-15,
        err_msg=f"P-value mismatch for model={model}",
    )

    np.testing.assert_allclose(
        results_gpu["FDR"].to_numpy(),
        results_cpu["FDR"].to_numpy(),
        rtol=1e-4,
        atol=1e-15,
        err_msg=f"FDR mismatch for model={model}",
    )


# ---------------------------------------------------------------------------
# compute_local_correlations: GPU vs CPU equivalence (per model)
# ---------------------------------------------------------------------------


@skip_no_gpu
@pytest.mark.parametrize("model", ["danb", "bernoulli", "normal", "none"])
def test_compute_local_correlations_equivalence(model: Model) -> None:
    """GPU and CPU compute_local_correlations produce equivalent results."""
    adata, genes = _make_test_data(n_cells=200, n_genes=15)
    hs_cpu, hs_gpu = _make_cpu_gpu_pair(adata, model)

    hs_cpu.compute_autocorrelations()
    lcz_cpu = hs_cpu.compute_local_correlations(genes)

    hs_gpu.compute_autocorrelations()
    lcz_gpu = hs_gpu.compute_local_correlations(genes)

    assert lcz_cpu.shape == lcz_gpu.shape

    np.testing.assert_allclose(
        lcz_gpu.values,
        lcz_cpu.values,
        rtol=RTOL,
        atol=ATOL,
        err_msg=f"Local correlation Z mismatch for model={model}",
    )

    lc_cpu = hs_cpu.local_correlation_c
    lc_gpu = hs_gpu.local_correlation_c
    np.testing.assert_allclose(
        lc_gpu.values,
        lc_cpu.values,
        rtol=RTOL,
        atol=ATOL,
        err_msg=f"Local correlation C mismatch for model={model}",
    )

    # Verify symmetry of GPU output
    np.testing.assert_allclose(
        lcz_gpu.values,
        lcz_gpu.values.T,
        rtol=1e-12,
        atol=1e-12,
        err_msg="GPU Z-score matrix should be symmetric",
    )
    np.testing.assert_allclose(
        lc_gpu.values,
        lc_gpu.values.T,
        rtol=1e-12,
        atol=1e-12,
        err_msg="GPU correlation matrix should be symmetric",
    )


# ---------------------------------------------------------------------------
# Sparse input equivalence
# ---------------------------------------------------------------------------


@skip_no_gpu
@pytest.mark.parametrize("model", ["danb", "normal"])
def test_sparse_input_equivalence(model: Model) -> None:
    """GPU produces identical results with sparse vs dense input."""
    adata, genes = _make_test_data(n_cells=150, n_genes=12)

    # Dense
    hs_dense = Hotspot(
        adata,
        model=model,
        latent_obsm_key="latent",
        umi_counts_obs_key="umi_counts",
        layer_key=None,
        use_gpu=True,
    )
    hs_dense.create_knn_graph(False, n_neighbors=30)
    results_dense = hs_dense.compute_autocorrelations()

    # Sparse, sharing the same KNN graph
    hs_sparse = Hotspot(
        adata,
        model=model,
        latent_obsm_key="latent",
        umi_counts_obs_key="umi_counts",
        layer_key="sparse",
        use_gpu=True,
    )
    hs_sparse.neighbors = hs_dense.neighbors
    hs_sparse.weights = hs_dense.weights
    results_sparse = hs_sparse.compute_autocorrelations()

    results_dense = results_dense.loc[sorted(results_dense.index)]
    results_sparse = results_sparse.loc[sorted(results_sparse.index)]

    np.testing.assert_allclose(
        results_sparse["Z"].to_numpy(),
        results_dense["Z"].to_numpy(),
        rtol=RTOL,
        atol=ATOL,
        err_msg="Sparse vs dense GPU results differ",
    )


# ---------------------------------------------------------------------------
# Different data sizes
# ---------------------------------------------------------------------------


@skip_no_gpu
@pytest.mark.parametrize(
    "n_cells,n_genes,n_neighbors",
    [(50, 5, 10), (100, 10, 20), (300, 25, 30)],
)
def test_varying_sizes(n_cells: int, n_genes: int, n_neighbors: int) -> None:
    """GPU matches CPU across different dataset sizes."""
    adata, genes = _make_test_data(n_cells=n_cells, n_genes=n_genes)
    hs_cpu, hs_gpu = _make_cpu_gpu_pair(adata, "normal", n_neighbors=n_neighbors)

    results_cpu = hs_cpu.compute_autocorrelations()
    results_gpu = hs_gpu.compute_autocorrelations()

    results_cpu = results_cpu.loc[sorted(results_cpu.index)]
    results_gpu = results_gpu.loc[sorted(results_gpu.index)]

    np.testing.assert_allclose(
        results_gpu["Z"].to_numpy(),
        results_cpu["Z"].to_numpy(),
        rtol=RTOL,
        atol=ATOL,
        err_msg=f"Size {n_cells}x{n_genes} k={n_neighbors}: Z mismatch",
    )


# ---------------------------------------------------------------------------
# GPU determinism
# ---------------------------------------------------------------------------


@skip_no_gpu
def test_gpu_determinism() -> None:
    """Running GPU computation twice on the same graph produces identical results."""
    adata, genes = _make_test_data(n_cells=150, n_genes=15)

    hs = Hotspot(
        adata,
        model="normal",
        latent_obsm_key="latent",
        umi_counts_obs_key="umi_counts",
        use_gpu=True,
    )
    hs.create_knn_graph(False, n_neighbors=30)

    # First run
    r1 = hs.compute_autocorrelations()
    lcz1 = hs.compute_local_correlations(genes)

    # Second run (same object, same graph)
    r2 = hs.compute_autocorrelations()
    lcz2 = hs.compute_local_correlations(genes)

    r1 = r1.loc[sorted(r1.index)]
    r2 = r2.loc[sorted(r2.index)]

    # Allow 1 ULP of float64 (~2.2e-16) from GPU parallel reduction non-determinism
    np.testing.assert_allclose(
        r1.values, r2.values, rtol=0, atol=1e-15, err_msg="GPU autocorrelation results are not deterministic"
    )
    np.testing.assert_allclose(
        lcz1.values, lcz2.values, rtol=0, atol=1e-15, err_msg="GPU local correlation results are not deterministic"
    )


# ---------------------------------------------------------------------------
# Full pipeline equivalence (end-to-end including modules)
# ---------------------------------------------------------------------------


@skip_no_gpu
@pytest.mark.parametrize("model", ["danb", "bernoulli", "normal", "none"])
def test_full_pipeline_equivalence(model: Model) -> None:
    """Full Hotspot pipeline produces equivalent results with GPU and CPU."""
    adata, genes = _make_test_data(n_cells=200, n_genes=20)
    hs_cpu, hs_gpu = _make_cpu_gpu_pair(adata, model)

    # CPU pipeline
    hs_cpu.compute_autocorrelations()
    lcz_cpu = hs_cpu.compute_local_correlations(genes)
    modules_cpu = hs_cpu.create_modules(min_gene_threshold=2, fdr_threshold=1)
    scores_cpu = hs_cpu.calculate_module_scores()

    # GPU pipeline
    hs_gpu.compute_autocorrelations()
    lcz_gpu = hs_gpu.compute_local_correlations(genes)
    modules_gpu = hs_gpu.create_modules(min_gene_threshold=2, fdr_threshold=1)
    scores_gpu = hs_gpu.calculate_module_scores()

    r_cpu = hs_cpu.results.loc[sorted(hs_cpu.results.index)]
    r_gpu = hs_gpu.results.loc[sorted(hs_gpu.results.index)]
    np.testing.assert_allclose(r_gpu["Z"].to_numpy(), r_cpu["Z"].to_numpy(), rtol=RTOL, atol=ATOL)

    np.testing.assert_allclose(
        lcz_gpu.to_numpy(),
        lcz_cpu.to_numpy(),
        rtol=RTOL,
        atol=ATOL,
    )

    pd.testing.assert_series_equal(
        modules_gpu.sort_index(),
        modules_cpu.sort_index(),
        check_names=True,
    )

    np.testing.assert_allclose(
        scores_gpu.to_numpy(),
        scores_cpu.to_numpy(),
        rtol=1e-6,
        atol=1e-6,
    )


# ---------------------------------------------------------------------------
# Diagonal of local correlation matrix should be zero
# ---------------------------------------------------------------------------


@skip_no_gpu
def test_local_correlation_diagonal_zero() -> None:
    """GPU local correlation matrix diagonal should be zero."""
    adata, genes = _make_test_data(n_cells=100, n_genes=10)
    hs = Hotspot(
        adata,
        model="normal",
        latent_obsm_key="latent",
        umi_counts_obs_key="umi_counts",
        use_gpu=True,
    )
    hs.create_knn_graph(False, n_neighbors=30)
    hs.compute_autocorrelations()
    lcz = hs.compute_local_correlations(genes)

    np.testing.assert_array_equal(np.diag(lcz.values), np.zeros(len(genes)))
    np.testing.assert_array_equal(np.diag(hs.local_correlation_c.values), np.zeros(len(genes)))


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_use_gpu_without_gpu() -> None:
    """Setting use_gpu=True without GPU raises RuntimeError."""
    if GPU_AVAILABLE:
        pytest.skip("GPU is available, cannot test missing-GPU error")

    adata, genes = _make_test_data(n_cells=50, n_genes=5)
    with pytest.raises(RuntimeError, match="GPU is not available"):
        Hotspot(
            adata,
            model="normal",
            latent_obsm_key="latent",
            umi_counts_obs_key="umi_counts",
            use_gpu=True,
        )


# ---------------------------------------------------------------------------
# Benchmarks: quantify GPU speedup
# ---------------------------------------------------------------------------


@skip_no_gpu
def test_benchmark_compute_autocorrelations() -> None:
    """Benchmark GPU vs CPU speed for compute_autocorrelations."""
    adata, genes = _make_test_data(n_cells=500, n_genes=100, seed=123)

    # Warmup GPU
    hs_w = Hotspot(adata, model="normal", latent_obsm_key="latent", umi_counts_obs_key="umi_counts", use_gpu=True)
    hs_w.create_knn_graph(False, n_neighbors=30)
    hs_w.compute_autocorrelations()

    # Shared graph for fair comparison
    hs_cpu, hs_gpu = _make_cpu_gpu_pair(adata, "normal")

    t0 = time.perf_counter()
    results_cpu = hs_cpu.compute_autocorrelations()
    cpu_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    results_gpu = hs_gpu.compute_autocorrelations()
    gpu_time = time.perf_counter() - t0

    speedup = cpu_time / gpu_time if gpu_time > 0 else float("inf")
    print("\n--- compute_autocorrelations benchmark (500 cells, 100 genes) ---")
    print(f"  CPU time: {cpu_time:.3f}s")
    print(f"  GPU time: {gpu_time:.3f}s")
    print(f"  Speedup:  {speedup:.1f}x")

    results_cpu = results_cpu.loc[sorted(results_cpu.index)]
    results_gpu = results_gpu.loc[sorted(results_gpu.index)]
    np.testing.assert_allclose(
        results_gpu["Z"].to_numpy(),
        results_cpu["Z"].to_numpy(),
        rtol=RTOL,
        atol=ATOL,
    )


@skip_no_gpu
def test_benchmark_compute_local_correlations() -> None:
    """Benchmark GPU vs CPU speed for compute_local_correlations."""
    adata, genes = _make_test_data(n_cells=500, n_genes=50, seed=456)

    # Warmup
    hs_w = Hotspot(adata, model="normal", latent_obsm_key="latent", umi_counts_obs_key="umi_counts", use_gpu=True)
    hs_w.create_knn_graph(False, n_neighbors=30)
    hs_w.compute_autocorrelations()
    hs_w.compute_local_correlations(genes)

    hs_cpu, hs_gpu = _make_cpu_gpu_pair(adata, "normal")
    hs_cpu.compute_autocorrelations()
    hs_gpu.compute_autocorrelations()

    t0 = time.perf_counter()
    lcz_cpu = hs_cpu.compute_local_correlations(genes)
    cpu_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    lcz_gpu = hs_gpu.compute_local_correlations(genes)
    gpu_time = time.perf_counter() - t0

    speedup = cpu_time / gpu_time if gpu_time > 0 else float("inf")
    print("\n--- compute_local_correlations benchmark (500 cells, 50 genes) ---")
    print(f"  CPU time: {cpu_time:.3f}s")
    print(f"  GPU time: {gpu_time:.3f}s")
    print(f"  Speedup:  {speedup:.1f}x")

    np.testing.assert_allclose(lcz_gpu.values, lcz_cpu.values, rtol=RTOL, atol=ATOL)


@skip_no_gpu
def test_benchmark_large_scale() -> None:
    """Benchmark at realistic scale: 1000 cells, 200 genes (danb model)."""
    adata, genes = _make_test_data(n_cells=1000, n_genes=200, seed=789)

    # Warmup
    hs_w = Hotspot(adata, model="danb", latent_obsm_key="latent", umi_counts_obs_key="umi_counts", use_gpu=True)
    hs_w.create_knn_graph(False, n_neighbors=30)
    hs_w.compute_autocorrelations()

    hs_cpu, hs_gpu = _make_cpu_gpu_pair(adata, "danb")

    t0 = time.perf_counter()
    results_cpu = hs_cpu.compute_autocorrelations()
    cpu_auto_time = time.perf_counter() - t0

    subset_genes = genes[:80]
    t0 = time.perf_counter()
    lcz_cpu = hs_cpu.compute_local_correlations(subset_genes)
    cpu_lc_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    results_gpu = hs_gpu.compute_autocorrelations()
    gpu_auto_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    lcz_gpu = hs_gpu.compute_local_correlations(subset_genes)
    gpu_lc_time = time.perf_counter() - t0

    auto_speedup = cpu_auto_time / gpu_auto_time if gpu_auto_time > 0 else float("inf")
    lc_speedup = cpu_lc_time / gpu_lc_time if gpu_lc_time > 0 else float("inf")

    print("\n--- Large-scale benchmark (1000 cells, 200/80 genes, danb model) ---")
    print("  compute_autocorrelations:")
    print(f"    CPU: {cpu_auto_time:.3f}s | GPU: {gpu_auto_time:.3f}s | Speedup: {auto_speedup:.1f}x")
    print(f"  compute_local_correlations (80 genes, {80 * 79 // 2} pairs):")
    print(f"    CPU: {cpu_lc_time:.3f}s | GPU: {gpu_lc_time:.3f}s | Speedup: {lc_speedup:.1f}x")

    results_cpu = results_cpu.loc[sorted(results_cpu.index)]
    results_gpu = results_gpu.loc[sorted(results_gpu.index)]
    np.testing.assert_allclose(
        results_gpu["Z"].to_numpy(),
        results_cpu["Z"].to_numpy(),
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_allclose(lcz_gpu.values, lcz_cpu.values, rtol=RTOL, atol=ATOL)
