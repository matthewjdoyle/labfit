"""Compare R12 against its previous algorithms on identical deterministic inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from contextlib import nullcontext
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from time import perf_counter_ns
from unittest.mock import patch

import numpy as np
import scipy
from scipy.linalg import solve_triangular
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from labfit import fit  # noqa: E402
from labfit import fitter_impl as fitting  # noqa: E402
from labfit.plot import _confidence_band, _prediction_variance  # noqa: E402

RTOL = 2e-6
ATOL = 1e-9


def _dense_solve(factor, values, **kwargs):
    return np.linalg.solve(factor, values)


def _reference_band(result, xs, *, prediction=False, prediction_sigma=None):
    """The pre-R12 point-by-point finite-difference algorithm."""
    names = result.param_names or tuple(result.params)
    params = np.array([result.params[name] for name in names])
    mean = result.predict(xs)
    sigma = np.zeros_like(mean)
    for index, value in enumerate(xs):
        gradient = np.empty(len(params))
        for parameter in range(len(params)):
            step = max(abs(params[parameter]) * 1e-8, 1e-8)
            up, down = params.copy(), params.copy()
            up[parameter] += step
            down[parameter] -= step
            gradient[parameter] = (
                float(result.model(np.array([value]), *up)[0])
                - float(result.model(np.array([value]), *down)[0])
            ) / (2 * step)
        sigma[index] = np.sqrt(max(gradient @ result.covariance @ gradient, 0.0))
    if prediction:
        sigma = np.sqrt(sigma**2 + _prediction_variance(result, xs, prediction_sigma))
    z = norm.isf(0.025)
    return mean - z * sigma, mean + z * sigma


def _make_data(model, n_points, seed):
    rng = np.random.default_rng(seed)
    x = np.linspace(-3, 3, n_points)
    if model == "linear":
        parameters = [1.8, 0.75]
        mean = parameters[0] * x + parameters[1]
    else:
        parameters = [3.2, 0.35, 0.65]
        mean = parameters[0] * np.exp(-0.5 * ((x - parameters[1]) / parameters[2]) ** 2)
    sigma = 0.05 * (1 + 0.1 * np.abs(x))
    distance = np.abs(np.subtract.outer(np.arange(n_points), np.arange(n_points)))
    covariance = np.outer(sigma, sigma) * 0.4**distance
    factor = np.linalg.cholesky(covariance)
    y = mean + factor @ rng.normal(size=n_points)
    return x, y, covariance, factor, parameters


def _comparison(reference, optimized, *, is_fit):
    if is_fit:
        if not (
            reference.success and optimized.success and reference.identifiable and optimized.identifiable
        ):
            raise AssertionError("Benchmark fit must converge with identifiable parameters")
        arrays = {
            "parameters": (list(reference), list(optimized)),
            "covariance": (reference.covariance, optimized.covariance),
            "fitted_values": (reference.y_fit, optimized.y_fit),
            "statistics": (
                [reference.reduced_chi2, reference.p_value, reference.residual_variance],
                [optimized.reduced_chi2, optimized.p_value, optimized.residual_variance],
            ),
        }
    else:
        arrays = {"values": (reference, optimized)}
    errors = {}
    for name, (before, after) in arrays.items():
        before, after = np.asarray(before), np.asarray(after)
        np.testing.assert_allclose(after, before, rtol=RTOL, atol=ATOL)
        errors[name] = float(np.max(np.abs(after - before)))
    return errors


def _benchmark_pair(operation, model, n_points, reference_fn, optimized_fn, *, repeats, warmup):
    """Alternate measurement order; patch setup and fixture construction are untimed."""

    def reference_context():
        if operation == "correlated_fit":
            return patch.object(fitting, "solve_triangular", _dense_solve)
        return nullcontext()

    with reference_context():
        reference = reference_fn()
    optimized = optimized_fn()
    errors = _comparison(reference, optimized, is_fit=operation == "correlated_fit")
    timings = {"reference": [], "optimized": []}
    for iteration in range(warmup + repeats):
        variants = ("reference", "optimized") if iteration % 2 == 0 else ("optimized", "reference")
        for variant in variants:
            fn = reference_fn if variant == "reference" else optimized_fn
            context = reference_context() if variant == "reference" else nullcontext()
            with context:
                start = perf_counter_ns()
                fn()
                elapsed = (perf_counter_ns() - start) / 1e6
            if iteration >= warmup:
                timings[variant].append(elapsed)
    before = float(np.median(timings["reference"]))
    after = float(np.median(timings["optimized"]))
    return {
        "operation": operation,
        "model": model,
        "n_points": n_points,
        "reference_median_ms": before,
        "optimized_median_ms": after,
        "speedup": before / after,
        "max_abs_differences": errors,
        "reference_samples_ms": timings["reference"],
        "optimized_samples_ms": timings["optimized"],
    }


def _git_metadata():
    def git(*args):
        try:
            return subprocess.run(
                ["git", "-C", str(ROOT), *args], check=True, text=True, capture_output=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"

    return {"sha": git("rev-parse", "HEAD"), "status": git("status", "--porcelain")}


def run_benchmarks(
    *, output_root: Path, sizes: list[int], grids: list[int], repeats: int, warmup: int, seed: int
) -> Path:
    """Validate numerical equivalence and save raw timings, environment and a comparison report."""
    if not sizes or any(size < 8 for size in sizes):
        raise ValueError("sizes must contain dataset lengths of at least 8")
    if not grids or any(grid < 1 for grid in grids):
        raise ValueError("grids must contain positive grid lengths")
    if repeats < 1 or warmup < 0:
        raise ValueError("repeats must be positive and warmup nonnegative")
    rows = []
    configuration = dict(sizes=sizes, grids=grids, repeats=repeats, warmup=warmup, seed=seed)

    def record(operation, model, n_points, reference_fn, optimized_fn):
        row = _benchmark_pair(
            operation, model, n_points, reference_fn, optimized_fn, repeats=repeats, warmup=warmup
        )
        rows.append(row)
        print(
            f"{operation:18} {model:8} {n_points:5}: "
            f"{row['reference_median_ms']:.3f} -> {row['optimized_median_ms']:.3f} ms "
            f"({row['speedup']:.2f}x)",
            flush=True,
        )

    for size in sizes:
        x, y, covariance, factor, p0 = _make_data("linear", size, seed + size)
        values = y - (p0[0] * x + p0[1])
        record(
            "whitening",
            "shared",
            size,
            partial(np.linalg.solve, factor, values),
            partial(solve_triangular, factor, values, lower=True),
        )
        for index, model in enumerate(("linear", "gaussian")):
            x, y, covariance, factor, p0 = _make_data(model, size, seed + size + 1000 * index)

            fitted = partial(fit, x, y, model=model, p0=p0, sigma_cov=covariance, absolute_sigma=True)

            record("correlated_fit", model, size, fitted, fitted)

    for index, model in enumerate(("linear", "gaussian")):
        x, y, covariance, factor, p0 = _make_data(model, 64, seed + 2000 + index)
        result = fit(x, y, model=model, p0=p0, sigma_cov=covariance, absolute_sigma=True)
        for grid in grids:
            xs = np.linspace(-3, 3, grid)
            for prediction in (False, True):
                noise = 0.05 + 0.01 * np.abs(xs) if prediction else None
                operation = "prediction_band" if prediction else "confidence_band"
                record(
                    operation,
                    model,
                    grid,
                    partial(_reference_band, result, xs, prediction=prediction, prediction_sigma=noise),
                    partial(_confidence_band, result, xs, 0.95, prediction, noise),
                )

    run_dir = output_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir.mkdir(parents=True, exist_ok=True)
    source_hashes = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in ("labfit/fitter_impl.py", "labfit/plot.py", "benchmarks/benchmark_optimizations.py")
    }
    payload = {
        "benchmark_version": "r12-1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": _git_metadata(),
        "source_sha256": source_hashes,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "cpu_count": os.cpu_count(),
            "requested_thread_limits": {
                key: os.environ.get(key)
                for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")
            },
        },
        "configuration": configuration,
        "reference": (
            "Dense Cholesky-factor solves and the pre-R12 pointwise finite-difference band algorithm"
        ),
        "validation": {"rtol": RTOL, "atol": ATOL, "all_passed": True},
        "results": rows,
    }
    (run_dir / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with (run_dir / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["operation", "model", "n_points", "variant", "repetition", "duration_ms"])
        for row in rows:
            for variant in ("reference", "optimized"):
                for repetition, duration in enumerate(row[f"{variant}_samples_ms"], 1):
                    writer.writerow(
                        [row["operation"], row["model"], row["n_points"], variant, repetition, duration]
                    )
    summary = [
        "# R12 optimization benchmark",
        "",
        f"Environment: Python {platform.python_version()}, "
        f"NumPy {np.__version__}, SciPy {scipy.__version__}, {platform.platform()}.",
        "",
        f"Configuration: `{configuration}`.",
        "",
        "Reference: general dense solves on Cholesky factors and pointwise band derivatives. "
        "Full-fit comparisons replace only the whitening solver; patch setup is outside the timer. "
        "Both variants include the same covariance validation, factorization and optimization.",
        "",
        "All numerical comparisons passed (rtol=2e-6, atol=1e-9). Raw samples, exact source hashes, "
        "git status and requested thread limits are recorded in results.json and samples.csv.",
        "",
        "| Operation | Model | Points | Reference ms | Optimized ms | Speedup |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        summary.append(
            f"| {row['operation']} | {row['model']} | {row['n_points']} | "
            f"{row['reference_median_ms']:.4f} | {row['optimized_median_ms']:.4f} | "
            f"{row['speedup']:.2f}x |"
        )
    summary += [
        "",
        "Times are medians from this machine. Band workloads time computation only; "
        "they do not include Matplotlib rendering or file output. Whitening excludes the "
        "one-time Cholesky factorization. Timings vary with hardware, BLAS and thread settings.",
        "",
    ]
    (run_dir / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    return run_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/results/r12"))
    parser.add_argument("--sizes", type=int, nargs="+", default=[32, 128, 512])
    parser.add_argument("--grids", type=int, nargs="+", default=[200, 2000])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260930)
    args = parser.parse_args(argv)
    try:
        result = run_benchmarks(
            output_root=args.output_dir,
            sizes=args.sizes,
            grids=args.grids,
            repeats=args.repeats,
            warmup=args.warmup,
            seed=args.seed,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
