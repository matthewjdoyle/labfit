# LabFit benchmark harness

This directory contains the reproducible benchmark script for LabFit:

- `benchmark_labfit.py` — runs fit/render benchmarks and renders publication-ready plots
- `benchmark_optimizations.py` — compares correlated fitting and confidence/prediction
  band computation with their pre-R12 algorithms, validating numerical agreement

## What it measures

The harness times:

- `fit_curve()` on synthetic linear and Gaussian datasets across multiple sizes
- `plot_fit()` on the same fitted results, both with and without residual panels

The script records raw per-iteration timings, aggregates them into summary statistics, and exports three figures:

- scaling curve
- latency distribution
- comparison bar chart

## Run it

From the repository root:

```bash
python benchmarks/benchmark_labfit.py
```

Optional arguments:

```bash
python benchmarks/benchmark_labfit.py --help
```

## Output layout

Each invocation creates a timestamped directory under `benchmarks/results/`:

- `benchmark_runs.csv` — raw per-iteration timing data
- `benchmark_results.json` — environment metadata + aggregate statistics
- `summary.md` — concise human summary
- `scaling_curve.{png,svg,pdf}`
- `latency_distribution.{png,svg,pdf}`
- `comparison_bar_chart.{png,svg,pdf}`

The plots are saved at 300 DPI in PNG form plus vector PDF/SVG exports for reports and papers.

## R12 optimization comparisons

Run the targeted comparison separately from other CPU-heavy work:

```bash
python benchmarks/benchmark_optimizations.py --sizes 32 128 512 --grids 200 2000 --repeats 7 --warmup 2
```

For reproducible single-thread BLAS measurements on PowerShell, set the limits
before starting Python:

```powershell
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
python benchmarks/benchmark_optimizations.py --sizes 32 128 512 --grids 200 2000 --repeats 7 --warmup 2
```

The workloads isolate four operations:

- **Whitening:** general `numpy.linalg.solve` versus `scipy.linalg.solve_triangular`
  on the same precomputed lower Cholesky factor. Factorization is excluded.
- **Full correlated fits:** identical linear/Gaussian data, guesses and covariance
  policies. The reference substitutes the old dense whitening solver; patch setup
  is outside the timer. Validation, factorization and optimization are included.
- **Confidence bands:** the old scalar finite differences at each grid point versus
  parameter perturbations evaluated across the whole grid.
- **Prediction bands:** the same comparison, with identical heterogeneous future
  observation errors. These two workloads exclude Matplotlib rendering and I/O.

The reference and optimized timing order alternates. Both receive warmups; inputs
are seeded. Parameters, covariance, fitted values, statistics, whitened residuals
and band endpoints must agree at `rtol=2e-6, atol=1e-9` before timing proceeds.
Timing thresholds are deliberately absent from tests: hardware and BLAS affect
latency. Regression tests instead check analytical results and ensure band model
calls remain `2 * parameter_count + 1`, independent of grid length.

Each run creates a timestamped folder in `benchmarks/results/r12/` containing:

- `samples.csv` — every timing sample for each variant
- `results.json` — medians, speedups, numerical differences, exact source hashes,
  git SHA/status, package versions, configuration and requested thread limits
- `summary.md` — the comparison table and scope of the measurements

See [R12_RESULTS.md](R12_RESULTS.md) for the verified measurements from this change.
The original fit/render harness remains available for end-to-end plotting costs.
