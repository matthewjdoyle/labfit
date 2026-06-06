# LabFit benchmark harness

This directory contains the reproducible benchmark script for LabFit:

- `benchmark_labfit.py` — runs the suite benchmarks and renders publication-ready plots

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
