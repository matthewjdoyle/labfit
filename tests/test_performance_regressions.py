"""Independent numerical references and work-count checks for R12."""

import math

import numpy as np
import pytest
from scipy.stats import chi2, norm

from labfit import FitResult, fit
from labfit.models import gaussian
from labfit.plot import _confidence_band


@pytest.mark.parametrize("absolute_sigma", [False, True])
def test_correlated_fit_matches_generalized_linear_reference_without_dense_solves(
    absolute_sigma, monkeypatch
):
    x = np.array([-2, -1.5, -0.5, 0, 0.25, 1.5, 2, 3.5, 4.0])
    y = np.array([-1.6, -1.2, -0.4, 0.7, 0.5, 1.6, 2.7, 4.2, 4.0])
    sigma = np.linspace(0.3, 0.7, len(x))
    covariance = np.outer(sigma, sigma) * 0.6 ** np.abs(np.subtract.outer(np.arange(9), np.arange(9)))
    design = np.column_stack((x, np.ones_like(x)))
    precision = np.linalg.inv(covariance)
    reference_cov = np.linalg.inv(design.T @ precision @ design)
    reference_params = reference_cov @ design.T @ precision @ y
    residual = y - design @ reference_params
    reference_chi2 = float(residual @ precision @ residual)
    if not absolute_sigma:
        reference_cov *= reference_chi2 / 7

    def forbidden_dense_solve(*args, **kwargs):
        pytest.fail("Fitting factored correlated errors must not use a general dense solve")

    monkeypatch.setattr(np.linalg, "solve", forbidden_dense_solve)
    result = fit(x, y, sigma_cov=covariance, absolute_sigma=absolute_sigma)
    np.testing.assert_allclose(list(result), reference_params, rtol=2e-6, atol=1e-8)
    np.testing.assert_allclose(result.covariance, reference_cov, rtol=2e-6, atol=1e-9)
    assert result.reduced_chi2 == pytest.approx(reference_chi2 / 7)
    assert result.p_value == pytest.approx(chi2.sf(reference_chi2, 7))
    assert result.dof == 7
    assert result.success and result.identifiable


@pytest.mark.parametrize("n_points", [1, 17, 400, 2000])
@pytest.mark.parametrize("prediction", [False, True])
def test_band_model_calls_do_not_scale_with_grid_length(n_points, prediction):
    calls = []

    def polynomial(x, a, b, c):
        calls.append(x.copy())
        return a * x**2 + b * x + c

    covariance = np.array([[0.04, 0.01, -0.005], [0.01, 0.09, 0.01], [-0.005, 0.01, 0.16]])
    result = FitResult(
        reduced_chi2=1.0, params={"a": 0.0, "b": -2.0, "c": 3.0}, covariance=covariance, model=polynomial
    )
    xs = np.linspace(-2, 3, n_points)
    one_sigma = math.erf(1 / math.sqrt(2))
    noise = 0.2 + 0.1 * np.abs(xs)
    lower, upper = _confidence_band(result, xs, one_sigma, prediction, noise if prediction else None)
    assert len(calls) == 7  # one mean evaluation plus two per parameter
    for supplied_grid in calls:
        np.testing.assert_array_equal(supplied_grid, xs)
    design = np.column_stack((xs**2, xs, np.ones_like(xs)))
    variance = np.einsum("ij,jk,ik->i", design, covariance, design)
    if prediction:
        variance += noise**2
    mean = -2 * xs + 3
    np.testing.assert_allclose(lower, mean - np.sqrt(variance), rtol=2e-6, atol=1e-7)
    np.testing.assert_allclose(upper, mean + np.sqrt(variance), rtol=2e-6, atol=1e-7)


@pytest.mark.parametrize("prediction", [False, True])
def test_nonlinear_gaussian_bands_match_analytical_derivatives(prediction):
    amplitude, mean, sigma = 2.5, 0.3, 0.8
    factor = np.array([[0.1, 0, 0], [0.03, 0.04, 0], [-0.02, 0.01, 0.05]])
    covariance = factor @ factor.T
    result = FitResult(
        reduced_chi2=1.0,
        params={"amplitude": amplitude, "mean": mean, "sigma": sigma},
        param_names=("amplitude", "mean", "sigma"),
        covariance=covariance,
        model=gaussian,
    )
    xs = np.linspace(-3, 3, 401)
    exponential = np.exp(-0.5 * ((xs - mean) / sigma) ** 2)
    ys = amplitude * exponential
    gradients = np.column_stack((exponential, ys * (xs - mean) / sigma**2, ys * (xs - mean) ** 2 / sigma**3))
    variance = np.einsum("ij,jk,ik->i", gradients, covariance, gradients)
    if prediction:
        variance += 0.15**2
    width = norm.ppf(0.975) * np.sqrt(variance)
    lower, upper = _confidence_band(result, xs, 0.95, prediction, 0.15 if prediction else None)
    np.testing.assert_allclose(lower, ys - width, rtol=2e-6, atol=1e-8)
    np.testing.assert_allclose(upper, ys + width, rtol=2e-6, atol=1e-8)


@pytest.mark.parametrize("n_points", [0, 1, 25])
def test_constant_bands_support_empty_and_single_point_grids(n_points):
    result = FitResult(
        reduced_chi2=1.0,
        params={"level": 2.0},
        covariance=np.array([[0.25]]),
        model=lambda x, level: np.full_like(x, level),
    )
    xs = np.linspace(0, 1, n_points)
    lower, upper = _confidence_band(result, xs, math.erf(1 / math.sqrt(2)))
    assert lower.shape == upper.shape == xs.shape
    np.testing.assert_allclose(lower, np.full(n_points, 1.5), atol=1e-8)
    np.testing.assert_allclose(upper, np.full(n_points, 2.5), atol=1e-8)


@pytest.fixture
def benchmark_module():
    import importlib.util
    from pathlib import Path

    source = Path(__file__).resolve().parents[1] / "benchmarks" / "benchmark_optimizations.py"
    spec = importlib.util.spec_from_file_location("benchmark_optimizations", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmark_report_preserves_samples_and_validates_all_workloads(benchmark_module, tmp_path):
    import csv
    import json

    from labfit import fitter_impl

    original_solver = fitter_impl.solve_triangular
    directory = benchmark_module.run_benchmarks(
        output_root=tmp_path, sizes=[8], grids=[3], repeats=1, warmup=0, seed=11
    )
    assert fitter_impl.solve_triangular is original_solver
    report = json.loads((directory / "results.json").read_text(encoding="utf-8"))
    assert report["validation"]["all_passed"]
    assert len(report["results"]) == 7
    assert {row["operation"] for row in report["results"]} == {
        "whitening",
        "correlated_fit",
        "confidence_band",
        "prediction_band",
    }
    for row in report["results"]:
        assert len(row["reference_samples_ms"]) == len(row["optimized_samples_ms"]) == 1
        assert row["reference_median_ms"] > 0 and row["optimized_median_ms"] > 0
        assert row["max_abs_differences"]
    with (directory / "samples.csv").open(newline="", encoding="utf-8") as handle:
        samples = list(csv.DictReader(handle))
    assert len(samples) == 14
    assert {row["variant"] for row in samples} == {"reference", "optimized"}
    assert "All numerical comparisons passed" in (directory / "summary.md").read_text(encoding="utf-8")
    assert len(report["source_sha256"]) == 3


@pytest.mark.parametrize(
    "invalid", [{"sizes": []}, {"sizes": [1]}, {"grids": []}, {"grids": [0]}, {"repeats": 0}, {"warmup": -1}]
)
def test_benchmark_rejects_invalid_configuration(benchmark_module, tmp_path, invalid):
    options = dict(output_root=tmp_path, sizes=[8], grids=[3], repeats=1, warmup=0, seed=11)
    options.update(invalid)
    with pytest.raises(ValueError):
        benchmark_module.run_benchmarks(**options)
    assert not list(tmp_path.iterdir())
