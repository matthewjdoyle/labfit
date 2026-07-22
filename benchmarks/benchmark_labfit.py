from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from time import perf_counter_ns

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from labfit import DataSeries, fit_curve, plot_fit  # noqa: E402

OKABE_ITO = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#F0E442",
    "#56B4E9",
    "#E69F00",
    "#000000",
]

MODEL_SPECS = {
    "linear": {
        "x": lambda n: np.linspace(-1.0, 1.0, n),
        "y": lambda x, rng: 1.8 * x + 0.75 + rng.normal(0.0, 0.02, size=x.size),
        "sigma": lambda x: np.full_like(x, 0.02, dtype=float),
    },
    "gaussian": {
        "x": lambda n: np.linspace(-3.0, 3.0, n),
        "y": lambda x, rng: (
            3.2 * np.exp(-0.5 * ((x - 0.35) / 0.65) ** 2) + 0.35 + rng.normal(0.0, 0.015, size=x.size)
        ),
        "sigma": lambda x: np.full_like(x, 0.015, dtype=float),
    },
}

DEFAULT_SIZES = (32, 128, 512, 2048)
DEFAULT_REPEATS = 18
DEFAULT_WARMUP = 4


@dataclass(frozen=True)
class BenchmarkRow:
    run_id: str
    benchmark_version: str
    timestamp_utc: str
    operation: str
    model: str
    n_points: int
    show_residuals: bool
    repetition: int
    duration_ms: float
    short_git_sha: str
    python_version: str
    platform: str
    numpy_version: str
    matplotlib_version: str


@dataclass(frozen=True)
class AggregateRow:
    operation: str
    model: str
    n_points: int
    show_residuals: bool
    count: int
    mean_ms: float
    median_ms: float
    stdev_ms: float
    ci95_low_ms: float
    ci95_high_ms: float
    p25_ms: float
    p75_ms: float


def _git_sha() -> str:
    repo = Path(__file__).resolve().parents[1]
    try:
        value = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return value or "unknown"
    except Exception:
        return "unknown"


def _short_git_sha() -> str:
    sha = _git_sha()
    return sha[:8] if sha != "unknown" else sha


def _version_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _make_dataset(
    model: str, n_points: int, *, seed: int
) -> tuple[DataSeries, np.ndarray, np.ndarray, np.ndarray]:
    spec = MODEL_SPECS[model]
    rng = np.random.default_rng(seed)
    x = np.asarray(spec["x"](n_points), dtype=float)
    y = np.asarray(spec["y"](x, rng), dtype=float)
    sigma = np.asarray(spec["sigma"](x), dtype=float)
    series = DataSeries(x=x, y=y, sigma=sigma, label=f"{model} ({n_points} pts)")
    return series, x, y, sigma


def _time_ms(fn) -> float:
    start = perf_counter_ns()
    fn()
    end = perf_counter_ns()
    return (end - start) / 1_000_000.0


def _stable_stats(values: Iterable[float]) -> tuple[float, float, float, float, float, float, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        raise ValueError("cannot summarize an empty sample")
    med = float(np.median(arr))
    avg = float(mean(arr.tolist()))
    if arr.size > 1:
        spread = float(stdev(arr.tolist()))
        sem = spread / math.sqrt(arr.size)
        margin = 1.96 * sem
    else:
        spread = 0.0
        margin = 0.0
    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    return avg, med, spread, max(0.0, med - margin), med + margin, q1, q3


def _record_rows(
    rows: list[BenchmarkRow],
    *,
    operation: str,
    model: str,
    n_points: int,
    show_residuals: bool,
    timings: list[float],
    run_id: str,
    timestamp_utc: str,
    short_git_sha: str,
) -> None:
    for repetition, duration_ms in enumerate(timings, start=1):
        rows.append(
            BenchmarkRow(
                run_id=run_id,
                benchmark_version="1",
                timestamp_utc=timestamp_utc,
                operation=operation,
                model=model,
                n_points=n_points,
                show_residuals=show_residuals,
                repetition=repetition,
                duration_ms=duration_ms,
                short_git_sha=short_git_sha,
                python_version=platform.python_version(),
                platform=platform.platform(),
                numpy_version=np.__version__,
                matplotlib_version=matplotlib.__version__,
            )
        )


def _benchmark_fit(
    model: str,
    n_points: int,
    *,
    repeats: int,
    warmup: int,
    seed: int,
    run_id: str,
    timestamp_utc: str,
    short_git_sha: str,
    rows: list[BenchmarkRow],
) -> tuple[DataSeries, np.ndarray, np.ndarray, np.ndarray, list[float]]:
    series, x, y, sigma = _make_dataset(model, n_points, seed=seed)
    for _ in range(warmup):
        fit_curve(model, x, y, sigma)

    timings: list[float] = []
    for _ in range(repeats):
        duration_ms = _time_ms(lambda: fit_curve(model, x, y, sigma))
        timings.append(duration_ms)
    _record_rows(
        rows,
        operation="fit_curve",
        model=model,
        n_points=n_points,
        show_residuals=False,
        timings=timings,
        run_id=run_id,
        timestamp_utc=timestamp_utc,
        short_git_sha=short_git_sha,
    )
    return series, x, y, sigma, timings


def _benchmark_plot(
    result,
    *,
    show_residuals: bool,
    repeats: int,
    warmup: int,
    scratch_dir: Path,
    run_id: str,
    timestamp_utc: str,
    short_git_sha: str,
    model: str,
    n_points: int,
    rows: list[BenchmarkRow],
) -> list[float]:
    tmp_path = scratch_dir / f"{model}-{n_points}-{int(show_residuals)}.png"
    for _ in range(warmup):
        plot = plot_fit(result=result, show_residuals=show_residuals, title=f"{model} ({n_points} pts)")
        plot.save(tmp_path, dpi=300, bbox_inches="tight")
        plt.close(plot.figure)

    timings: list[float] = []
    for _ in range(repeats):

        def _render() -> None:
            plot = plot_fit(result=result, show_residuals=show_residuals, title=f"{model} ({n_points} pts)")
            plot.save(tmp_path, dpi=300, bbox_inches="tight")
            plt.close(plot.figure)

        duration_ms = _time_ms(_render)
        timings.append(duration_ms)
    _record_rows(
        rows,
        operation="plot_fit",
        model=model,
        n_points=n_points,
        show_residuals=show_residuals,
        timings=timings,
        run_id=run_id,
        timestamp_utc=timestamp_utc,
        short_git_sha=short_git_sha,
    )
    return timings


def _aggregate(rows: list[BenchmarkRow]) -> list[AggregateRow]:
    grouped: dict[tuple[str, str, int, bool], list[float]] = {}
    for row in rows:
        key = (row.operation, row.model, row.n_points, row.show_residuals)
        grouped.setdefault(key, []).append(row.duration_ms)

    aggregates: list[AggregateRow] = []
    for (operation, model, n_points, show_residuals), values in sorted(grouped.items()):
        avg, med, spread, low, high, q1, q3 = _stable_stats(values)
        aggregates.append(
            AggregateRow(
                operation=operation,
                model=model,
                n_points=n_points,
                show_residuals=show_residuals,
                count=len(values),
                mean_ms=avg,
                median_ms=med,
                stdev_ms=spread,
                ci95_low_ms=low,
                ci95_high_ms=high,
                p25_ms=q1,
                p75_ms=q3,
            )
        )
    return aggregates


def _figure_style() -> None:
    plt.rcParams.update(
        {
            # Font — sans-serif (clean benchmark look)
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 10,
            "mathtext.fontset": "dejavusans",
            # Figure
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            # Axes — all four spines visible (fully enclosed box)
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.linewidth": 1.2,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "axes.prop_cycle": plt.cycler(color=OKABE_ITO),
            "axes.grid": False,
            # Ticks — inward on all four sides for integrated look
            "xtick.direction": "in",
            "xtick.major.size": 4,
            "xtick.major.width": 0.7,
            "xtick.labelsize": 9.5,
            "xtick.top": True,
            "ytick.direction": "in",
            "ytick.major.size": 4,
            "ytick.major.width": 0.7,
            "ytick.labelsize": 9.5,
            "ytick.right": True,
            # Legend
            "legend.frameon": False,
            "legend.fontsize": 9,
            "legend.handlelength": 1.2,
            # Lines
            "lines.linewidth": 1.5,
            "lines.markersize": 6,
        }
    )


def _save_figure(fig: plt.Figure, stem: Path) -> dict[str, str]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = {
        "png": str(stem.with_suffix(".png")),
        "svg": str(stem.with_suffix(".svg")),
        "pdf": str(stem.with_suffix(".pdf")),
    }
    fig.savefig(outputs["png"], bbox_inches="tight", dpi=300)
    fig.savefig(outputs["svg"], bbox_inches="tight")
    fig.savefig(outputs["pdf"], bbox_inches="tight")
    return outputs


def _plot_scaling(aggregates: list[AggregateRow], output_dir: Path) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for model in MODEL_SPECS:
        rows = [row for row in aggregates if row.operation == "fit_curve" and row.model == model]
        rows.sort(key=lambda row: row.n_points)
        xs = [row.n_points for row in rows]
        medians = [row.median_ms for row in rows]
        yerr = [
            [max(0.0, row.median_ms - row.ci95_low_ms) for row in rows],
            [max(0.0, row.ci95_high_ms - row.median_ms) for row in rows],
        ]
        ax.errorbar(xs, medians, yerr=yerr, marker="o", capsize=4, lw=2.0, label=f"{model} fit")

    ax.set_xlabel("dataset size (points)")
    ax.set_ylabel("median runtime (ms)")
    ax.set_title("LabFit fit_curve scaling across dataset sizes")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.legend(loc="best")
    fig.tight_layout()
    return _save_figure(fig, output_dir / "scaling_curve")


def _plot_distribution(
    aggregates: list[AggregateRow], rows: list[BenchmarkRow], output_dir: Path
) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    largest = max(row.n_points for row in rows)
    labels: list[str] = []
    samples: list[list[float]] = []
    for model in MODEL_SPECS:
        for operation, show_residuals in (("fit_curve", False), ("plot_fit", False), ("plot_fit", True)):
            values = [
                row.duration_ms
                for row in rows
                if row.operation == operation
                and row.model == model
                and row.n_points == largest
                and row.show_residuals == show_residuals
            ]
            if values:
                suffix = " + residuals" if show_residuals else ""
                labels.append(f"{model}\n{operation}{suffix}")
                samples.append(values)

    bp = ax.boxplot(samples, patch_artist=True, showmeans=True)
    ax.set_xticks(range(1, len(labels) + 1), labels)
    for patch, color in zip(bp["boxes"], OKABE_ITO, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.28)
    for median_line in bp["medians"]:
        median_line.set_color("black")
        median_line.set_linewidth(1.5)
    ax.set_ylabel("runtime (ms)")
    ax.set_title(f"LabFit latency distribution at {largest} points")
    ax.tick_params(axis="x", rotation=20)
    ax.set_yscale("log")
    fig.tight_layout()
    return _save_figure(fig, output_dir / "latency_distribution")


def _plot_comparison(aggregates: list[AggregateRow], output_dir: Path) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    sizes = sorted({row.n_points for row in aggregates if row.operation == "fit_curve"})
    if not sizes:
        raise ValueError("comparison plot needs fit_curve aggregates")
    target_size = sizes[len(sizes) // 2]
    categories: list[str] = []
    medians: list[float] = []
    lowers: list[float] = []
    uppers: list[float] = []

    selected = [
        ("plot_fit", "linear", False),
        ("plot_fit", "linear", True),
        ("plot_fit", "gaussian", False),
        ("plot_fit", "gaussian", True),
    ]
    for operation, model, show_residuals in selected:
        match = next(
            row
            for row in aggregates
            if row.operation == operation
            and row.model == model
            and row.n_points == target_size
            and row.show_residuals == show_residuals
        )
        label = f"{operation}\n{model}" + ("\n+ residuals" if show_residuals else "")
        categories.append(label)
        medians.append(match.median_ms)
        lowers.append(max(0.0, match.median_ms - match.ci95_low_ms))
        uppers.append(max(0.0, match.ci95_high_ms - match.median_ms))

    x = np.arange(len(categories))
    ax.bar(x, medians, color=OKABE_ITO[: len(categories)], alpha=0.9)
    ax.errorbar(x, medians, yerr=[lowers, uppers], fmt="none", ecolor="black", capsize=4, lw=1.2)
    ax.set_xticks(x, categories)
    ax.set_ylabel("median runtime (ms)")
    ax.set_title(f"LabFit operation comparison at {target_size} points")
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="x", rotation=18)
    fig.tight_layout()
    return _save_figure(fig, output_dir / "comparison_bar_chart")


def _write_csv(path: Path, rows: list[BenchmarkRow]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_summary_markdown(
    path: Path,
    *,
    run_id: str,
    sha: str,
    rows: list[BenchmarkRow],
    aggregates: list[AggregateRow],
    figure_paths: dict[str, dict[str, str]],
) -> None:
    fit_rows = [row for row in aggregates if row.operation == "fit_curve"]
    plot_rows = [row for row in aggregates if row.operation == "plot_fit"]
    fastest_fit = min(fit_rows, key=lambda row: row.median_ms)
    slowest_plot = max(plot_rows, key=lambda row: row.median_ms)

    lines = [
        f"# LabFit benchmark run {run_id}",
        "",
        f"- Git SHA: `{sha}`",
        f"- Raw measurements: `{len(rows)}`",
        f"- Aggregates: `{len(aggregates)}`",
        "",
        "## Highlights",
        (
            f"- Fastest fit_curve median: `{fastest_fit.model}` at `{fastest_fit.n_points}` points"
            f" → `{fastest_fit.median_ms:.3f} ms`"
        ),
        (
            f"- Slowest plot_fit median: `{slowest_plot.model}` at `{slowest_plot.n_points}` points"
            f" `{'+ residuals' if slowest_plot.show_residuals else ''}`"
            f" → `{slowest_plot.median_ms:.3f} ms`"
        ),
        "",
        "## Figure files",
    ]
    for name, outputs in figure_paths.items():
        lines.append(f"- {name}: `{outputs['png']}`, `{outputs['svg']}`, `{outputs['pdf']}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_benchmarks(*, output_root: Path, sizes: Iterable[int], repeats: int, warmup: int, seed: int) -> Path:
    _figure_style()
    run_id = f"{_version_stamp()}_{_short_git_sha()}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir = Path(tempfile.mkdtemp(prefix="scratch-", dir=run_dir))

    timestamp_utc = datetime.now(timezone.utc).isoformat()
    short_git_sha = _short_git_sha()
    rows: list[BenchmarkRow] = []

    sizes = tuple(int(size) for size in sizes)
    for model_index, model in enumerate(MODEL_SPECS):
        for size_index, n_points in enumerate(sizes):
            _series, x, y, sigma, _fit_timings = _benchmark_fit(
                model,
                n_points,
                repeats=repeats,
                warmup=warmup,
                seed=seed + model_index * 1000 + size_index,
                run_id=run_id,
                timestamp_utc=timestamp_utc,
                short_git_sha=short_git_sha,
                rows=rows,
            )
            result = fit_curve(model, x, y, sigma)
            for show_residuals in (False, True):
                _benchmark_plot(
                    result,
                    show_residuals=show_residuals,
                    repeats=repeats,
                    warmup=warmup,
                    scratch_dir=scratch_dir,
                    run_id=run_id,
                    timestamp_utc=timestamp_utc,
                    short_git_sha=short_git_sha,
                    model=model,
                    n_points=n_points,
                    rows=rows,
                )

    aggregates = _aggregate(rows)
    # Re-apply benchmark style — plot_fit() may have changed rcParams
    _figure_style()
    csv_path = run_dir / "benchmark_runs.csv"
    json_path = run_dir / "benchmark_results.json"
    summary_md = run_dir / "summary.md"
    _write_csv(csv_path, rows)

    figure_paths: dict[str, dict[str, str]] = {}
    figure_paths["scaling_curve"] = _plot_scaling(aggregates, run_dir)
    figure_paths["latency_distribution"] = _plot_distribution(aggregates, rows, run_dir)
    figure_paths["comparison_bar_chart"] = _plot_comparison(aggregates, run_dir)
    _write_summary_markdown(
        summary_md,
        run_id=run_id,
        sha=short_git_sha,
        rows=rows,
        aggregates=aggregates,
        figure_paths=figure_paths,
    )

    payload = {
        "benchmark_version": "1",
        "run_id": run_id,
        "generated_at_utc": timestamp_utc,
        "git_sha": short_git_sha,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "numpy": np.__version__,
            "matplotlib": plt.matplotlib.__version__,
            "cpu_count": os.cpu_count(),
        },
        "configuration": {
            "sizes": list(sizes),
            "repeats": repeats,
            "warmup": warmup,
            "seed": seed,
        },
        "raw_csv": str(csv_path),
        "summary_markdown": str(summary_md),
        "figures": figure_paths,
        "aggregates": [asdict(row) for row in aggregates],
    }
    _write_json(json_path, payload)

    # Persist a compact index for humans skimming the latest benchmark results.
    index_path = run_dir / "README.txt"
    index_path.write_text(
        "\n".join(
            [
                f"LabFit benchmark run: {run_id}",
                f"Git SHA: {short_git_sha}",
                f"Raw CSV: {csv_path.name}",
                f"JSON summary: {json_path.name}",
                f"Markdown summary: {summary_md.name}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark the LabFit Python suite and render publication-ready plots."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/results"),
        help="Directory that will receive a timestamped benchmark run folder.",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_SIZES),
        help="Dataset sizes (in points) to benchmark.",
    )
    parser.add_argument(
        "--repeats", type=int, default=DEFAULT_REPEATS, help="Timed repetitions per workload."
    )
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP, help="Warmup iterations per workload.")
    parser.add_argument(
        "--seed", type=int, default=20260606, help="Deterministic random seed for synthetic data."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_dir = run_benchmarks(
        output_root=args.output_dir,
        sizes=args.sizes,
        repeats=args.repeats,
        warmup=args.warmup,
        seed=args.seed,
    )
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
