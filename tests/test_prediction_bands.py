"""Independent analytical references for pointwise confidence/prediction bands."""

import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from labfit import FitResult, fit, plot_fit
from labfit.plot import _confidence_band

ONE_SIGMA_LEVEL = math.erf(1.0 / math.sqrt(2.0))
X = np.array([-1.0, 0.0, 1.0, 2.0])
Y = np.array([0.0, 1.0, 1.0, 4.0])


def _fit_unweighted(**kwargs):
    with pytest.warns(UserWarning, match="No uncertainties provided"):
        return fit(X, Y, **kwargs)


@pytest.mark.parametrize("absolute_sigma", [False, True])
@pytest.mark.parametrize("error_kind", ["sigma", "weights", "unweighted"])
def test_default_bands_match_analytical_linear_regression(absolute_sigma, error_kind):
    if error_kind == "unweighted":
        result = _fit_unweighted(absolute_sigma=absolute_sigma)
        covariance_scale = 1.0 if absolute_sigma else 0.9
        noise_variance = covariance_scale
    else:
        errors = {"sigma": 0.5} if error_kind == "sigma" else {"weights": np.full(4, 4.0)}
        result = fit(X, Y, absolute_sigma=absolute_sigma, **errors)
        covariance_scale = 0.25 if absolute_sigma else 0.9
        noise_variance = covariance_scale
    xs = np.array([-1.0, 0.25, 1.0, 2.0])
    # (D.T @ D)^-1 = [[0.2, -0.1], [-0.1, 0.3]], SSR = 1.8, dof = 2.
    mean = 1.2 * xs + 0.9
    mean_variance = covariance_scale * (0.2 * xs**2 - 0.2 * xs + 0.3)
    for prediction in (False, True):
        half_width = np.sqrt(mean_variance + (noise_variance if prediction else 0.0))
        lower, upper = _confidence_band(result, xs, ONE_SIGMA_LEVEL, prediction=prediction)
        np.testing.assert_allclose(lower, mean - half_width, rtol=2e-6, atol=1e-7)
        np.testing.assert_allclose(upper, mean + half_width, rtol=2e-6, atol=1e-7)


@pytest.mark.parametrize("absolute_sigma", [False, True])
@pytest.mark.parametrize("error_kind", ["heterogeneous", "correlated", "override"])
def test_heterogeneous_future_noise_matches_generalized_linear_reference(absolute_sigma, error_kind):
    sigma = np.array([0.3, 0.8, 0.4, 0.6])
    measurement_cov = np.diag(sigma**2)
    if error_kind == "correlated":
        measurement_cov += 0.05 * np.ones((4, 4))
    design = np.column_stack((X, np.ones(4)))
    precision = np.linalg.inv(measurement_cov)
    covariance = np.linalg.inv(design.T @ precision @ design)
    parameters = covariance @ design.T @ precision @ Y
    residual = Y - design @ parameters
    if not absolute_sigma:
        covariance *= (residual @ precision @ residual) / 2
    if error_kind == "override":
        from labfit import DataSeries

        result = fit(DataSeries(X, Y, sigma=5.0), sigma=sigma, absolute_sigma=absolute_sigma)
    else:
        errors = {"sigma_cov": measurement_cov} if error_kind == "correlated" else {"sigma": sigma}
        result = fit(X, Y, absolute_sigma=absolute_sigma, **errors)
    xs = np.array([-1.5, 0.0, 0.75, 2.5])
    future_sigma = 0.2 + 0.1 * np.abs(xs)
    future_design = np.column_stack((xs, np.ones_like(xs)))
    variance = np.einsum("ij,jk,ik->i", future_design, covariance, future_design)
    expected_width = np.sqrt(variance + future_sigma**2)
    expected_mean = future_design @ parameters
    for noise in (future_sigma, lambda grid: 0.2 + 0.1 * np.abs(grid)):
        lower, upper = _confidence_band(result, xs, ONE_SIGMA_LEVEL, True, noise)
        np.testing.assert_allclose(lower, expected_mean - expected_width, rtol=2e-6, atol=1e-7)
        np.testing.assert_allclose(upper, expected_mean + expected_width, rtol=2e-6, atol=1e-7)
    with pytest.raises(ValueError, match="provide prediction_sigma"):
        _confidence_band(result, xs, prediction=True)


@pytest.mark.parametrize("absolute_sigma", [False, True])
def test_explicit_future_sigma_is_absolute_and_can_be_zero(absolute_sigma):
    result = fit(X, Y, sigma=0.5, absolute_sigma=absolute_sigma)
    xs = np.array([0.0, 1.0])
    confidence = _confidence_band(result, xs, ONE_SIGMA_LEVEL)
    zero_noise = _confidence_band(result, xs, ONE_SIGMA_LEVEL, True, 0.0)
    np.testing.assert_allclose(zero_noise, confidence)
    for noise in (2.0, lambda grid: 2.0):
        lower, upper = _confidence_band(result, xs, ONE_SIGMA_LEVEL, True, noise)
        mean_variance = ((confidence[1] - confidence[0]) / 2) ** 2
        np.testing.assert_allclose(((upper - lower) / 2) ** 2 - mean_variance, 4.0, rtol=2e-6)


@pytest.mark.parametrize("absolute_sigma", [False, True])
@pytest.mark.parametrize("error_kind", ["sigma", "weights", "covariance"])
@pytest.mark.parametrize("explicit", [False, True])
def test_bands_scale_with_x_and_y_unit_changes(absolute_sigma, error_kind, explicit):
    bands = []
    for x_unit, y_unit in ((1.0, 1.0), (1000.0, 1000.0)):
        errors = {
            "sigma": {"sigma": y_unit * 0.5},
            "weights": {"weights": np.full(4, 1.0 / (y_unit * 0.5) ** 2)},
            "covariance": {"sigma_cov": y_unit**2 * (0.25 * np.eye(4) + 0.05 * np.ones((4, 4)))},
        }[error_kind]
        result = fit(x_unit * X, y_unit * Y, absolute_sigma=absolute_sigma, **errors)
        xs = x_unit * np.array([-1.0, 0.0, 1.0, 2.0])
        # Correlated training data always needs an explicit future error model.
        noise = (
            (lambda grid, yu=y_unit, xu=x_unit: yu * (0.2 + 0.1 * np.abs(grid / xu))) if explicit else None
        )
        if error_kind == "covariance" and not explicit:
            noise = y_unit * 0.5
        bands.append(np.asarray(_confidence_band(result, xs, 0.95, True, noise)) / y_unit)
    np.testing.assert_allclose(bands[0], bands[1], rtol=4e-6, atol=1e-7)


def test_unweighted_default_prediction_scales_with_y_units():
    bands = []
    for scale in (1.0, 1000.0):
        with pytest.warns(UserWarning, match="No uncertainties provided"):
            result = fit(X, scale * Y)
        bands.append(np.asarray(_confidence_band(result, X, 0.95, prediction=True)) / scale)
    np.testing.assert_allclose(bands[0], bands[1], rtol=3e-6)


@pytest.mark.parametrize("level", [0.0, 1.0, -0.1, 1.1, np.nan, np.inf, -np.inf, True, "bad", [0.95]])
def test_invalid_confidence_levels_fail_before_missing_covariance_or_plotting(level):
    result = FitResult(reduced_chi2=1.0, params={"a": 1.0})
    with pytest.raises(ValueError, match="ci_level"):
        _confidence_band(result, X, ci_level=level)
    with pytest.raises(ValueError, match="ci_level"):
        plot_fit(result, ci_level=level, show_ci=True)


def test_largest_valid_confidence_level_still_has_finite_endpoints():
    result = fit(X, Y, sigma=0.5, absolute_sigma=True)
    band = _confidence_band(result, X, np.nextafter(1.0, 0.0))
    assert np.all(np.isfinite(band))


@pytest.mark.parametrize(
    "noise", [-0.1, np.nan, np.inf, [[1.0, 2.0]], [1.0], "bad", 1e300, lambda xs: np.full_like(xs, -1.0)]
)
def test_invalid_future_sigmas_are_rejected(noise):
    result = fit(X, Y, sigma=0.5, absolute_sigma=True)
    with pytest.raises(ValueError, match="prediction_sigma"):
        _confidence_band(result, X, prediction=True, prediction_sigma=noise)


def test_missing_residual_scale_requires_explicit_future_noise():
    with pytest.warns(UserWarning, match="degrees of freedom"):
        result = fit(X[:2], Y[:2], sigma=0.5, absolute_sigma=True)
    result.absolute_sigma = False  # Covariance is present, but no relative noise scale exists.
    with pytest.raises(ValueError, match="provide prediction_sigma"):
        _confidence_band(result, X, prediction=True)
    assert _confidence_band(result, X, prediction=True, prediction_sigma=0.5) is not None
    unweighted = _fit_unweighted()
    unweighted.residual_variance = float("nan")
    with pytest.raises(ValueError, match="provide prediction_sigma"):
        _confidence_band(unweighted, X, prediction=True)


def test_public_plot_renders_analytical_prediction_endpoints():
    result = fit(X, Y, sigma=0.5, absolute_sigma=True)
    xs = np.linspace(X.min(), X.max(), 200)
    plotter = plot_fit(
        result,
        show_ci=True,
        prediction=True,
        ci_level=ONE_SIGMA_LEVEL,
        prediction_sigma=lambda grid: 0.2 + 0.1 * np.abs(grid),
        show_residuals=False,
    )
    try:
        band = next(c for c in plotter.axes.collections if c.get_label() == "68% prediction")
        vertices = band.get_paths()[0].vertices
        expected_mean = 1.2 * xs + 0.9
        expected_width = np.sqrt(0.25 * (0.2 * xs**2 - 0.2 * xs + 0.3) + (0.2 + 0.1 * np.abs(xs)) ** 2)
        for idx in (0, 30, 100, 199):
            endpoints = vertices[np.isclose(vertices[:, 0], xs[idx]), 1]
            assert endpoints.min() == pytest.approx(expected_mean[idx] - expected_width[idx], abs=1e-7)
            assert endpoints.max() == pytest.approx(expected_mean[idx] + expected_width[idx], abs=1e-7)
    finally:
        plt.close(plotter.figure)


@pytest.mark.parametrize("options", [{}, {"show_ci": True}, {"prediction": True}])
def test_public_future_noise_option_cannot_be_silently_ignored(options):
    result = fit(X, Y, sigma=0.5)
    with pytest.raises(ValueError, match="requires show_ci=True and prediction=True"):
        plot_fit(result, prediction_sigma=0.5, **options)
    with pytest.raises(ValueError, match="single FitResult"):
        plot_fit([result, result], show_ci=True, prediction=True, prediction_sigma=0.5)


def test_documented_prediction_example_runs(tmp_path, monkeypatch):
    document = (Path(__file__).resolve().parents[1] / "docs" / "concepts.rst").read_text(encoding="utf-8")
    section = document.split("Confidence and prediction bands", 1)[1]
    block = section.split(".. code-block:: python", 1)[1].lstrip("\n")
    lines = []
    for line in block.splitlines():
        if line and not line.startswith("   "):
            break
        lines.append(line)
    monkeypatch.chdir(tmp_path)
    namespace = {}
    try:
        exec(textwrap.dedent("\n".join(lines)), namespace)
        output = tmp_path / "prediction-band.png"
        assert output.exists() and output.stat().st_size > 0
        labels = namespace["plot"].axes[0].get_legend_handles_labels()[1]
        assert "95% prediction" in labels
    finally:
        if "plot" in namespace:
            plt.close(namespace["plot"].figure)


def test_public_invalid_future_noise_does_not_create_or_modify_a_figure():
    result = fit(X, Y, sigma=[0.3, 0.8, 0.4, 0.6])
    before = plt.get_fignums()
    with pytest.raises(ValueError, match="provide prediction_sigma"):
        plot_fit(result, show_ci=True, prediction=True)
    assert plt.get_fignums() == before
    fig, ax = plt.subplots()
    try:
        original_axes = fig.axes.copy()
        with pytest.raises(ValueError, match="prediction_sigma"):
            plot_fit(result, ax=ax, show_ci=True, prediction=True, prediction_sigma=-1.0)
        assert fig.axes == original_axes
        assert not ax.collections and not ax.lines
    finally:
        plt.close(fig)
