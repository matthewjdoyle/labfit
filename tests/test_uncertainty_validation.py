"""Validate errors before optimization and check zero weights against independent references."""

import itertools

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.stats import chi2

from labfit import AsymmetricError, DataSeries, Fitter, fit, fit_curve, fit_multi, plot_fit
from labfit.utils import effective_sigma

X = np.arange(4.0)
Y = np.array([1.0, 2.4, 5.1, 6.7])

INVALID_ERRORS = [
    ({"sigma": 0.0}, "strictly positive"),
    ({"sigma": [-1.0, 1.0, 1.0, 1.0]}, "strictly positive"),
    ({"sigma": np.nan}, "finite"),
    ({"sigma": np.inf}, "finite"),
    ({"sigma": np.ones((2, 2))}, "one-dimensional"),
    ({"sigma": np.ones((4, 1))}, "one-dimensional"),
    ({"sigma": [1.0]}, "same length"),
    ({"sigma": AsymmetricError(np.ones(2), np.ones(2))}, "same length"),
    ({"sigma_low": 1.0}, "supplied together"),
    ({"sigma_high": 1.0}, "supplied together"),
    ({"sigma_low": 0.0, "sigma_high": 1.0}, "strictly positive"),
    ({"sigma_low": 1.0, "sigma_high": -1.0}, "strictly positive"),
    ({"sigma_low": np.inf, "sigma_high": 1.0}, "finite"),
    ({"sigma_low": 1.0, "sigma_high": np.nan}, "finite"),
    ({"sigma_low": np.ones((2, 2)), "sigma_high": 1.0}, "one-dimensional"),
    ({"sigma_low": 1.0, "sigma_high": [1.0, 2.0]}, "same length"),
    ({"sigma_cov": np.eye(3)}, "square covariance matrix"),
    ({"sigma_cov": np.ones((4, 1))}, "square covariance matrix"),
    ({"sigma_cov": np.ones(4)}, "square covariance matrix"),
    ({"sigma_cov": np.diag([1.0, np.inf, 1.0, 1.0])}, "finite"),
    ({"sigma_cov": np.diag([1.0, np.nan, 1.0, 1.0])}, "finite"),
    ({"sigma_cov": np.eye(4) + np.diag(np.ones(3), 1)}, "symmetric"),
    ({"sigma_cov": np.ones((4, 4))}, "positive-definite"),
    ({"sigma_cov": np.diag([1.0, -1.0, 1.0, 1.0])}, "positive-definite"),
    ({"sigma_cov": np.diag([1.0, 0.0, 1.0, 1.0])}, "positive-definite"),
    ({"sigma_cov": np.array([[1, 2, 0, 0], [2, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])}, "positive-definite"),
]

INVALID_WEIGHTS = [
    ({"weights": -1.0}, "non-negative"),
    ({"weights": [-1.0, 1.0, 1.0, 1.0]}, "non-negative"),
    ({"weights": 0.0}, "at least one positive"),
    ({"weights": np.zeros(4)}, "at least one positive"),
    ({"weights": np.nan}, "finite"),
    ({"weights": np.inf}, "finite"),
    ({"weights": np.ones((2, 2))}, "one-dimensional"),
    ({"weights": [1.0, 1.0]}, "same length"),
]


def _entrypoint(kind, errors):
    if kind == "arrays":
        return fit(X, Y, **errors)
    if kind == "series":
        return fit(DataSeries(X, Y, sigma=2.0), **errors)
    if kind == "fitter":
        return Fitter().fit(DataSeries(X, Y, sigma_cov=np.eye(4)), **errors)
    if kind == "multi":
        return fit_multi([DataSeries(X, Y, sigma_low=1.0, sigma_high=2.0)], **errors)
    if kind == "fitter_multi":
        return Fitter().fit_multi([DataSeries(X, Y)], **errors)
    raise AssertionError(kind)


@pytest.mark.parametrize("kind", ["arrays", "series", "fitter", "multi", "fitter_multi"])
@pytest.mark.parametrize("errors, message", INVALID_ERRORS + INVALID_WEIGHTS)
def test_invalid_errors_are_rejected_before_optimization(kind, errors, message, monkeypatch):
    def unexpected_optimization(*args, **kwargs):
        pytest.fail("Invalid errors reached the optimizer")

    monkeypatch.setattr("labfit.fitter_impl.least_squares", unexpected_optimization)
    with pytest.raises(ValueError, match=message):
        _entrypoint(kind, errors)


@pytest.mark.parametrize("errors, message", INVALID_ERRORS)
def test_data_series_rejects_invalid_errors(errors, message):
    with pytest.raises(ValueError, match=message):
        DataSeries(X, Y, **errors)


SPECIFICATIONS = [
    {"sigma": 1.0},
    {"weights": 1.0},
    {"sigma_low": 1.0, "sigma_high": 2.0},
    {"sigma_cov": np.eye(4)},
]


@pytest.mark.parametrize("first, second", list(itertools.combinations(SPECIFICATIONS, 2)))
@pytest.mark.parametrize("kind", ["arrays", "series", "multi"])
def test_competing_error_specifications_are_rejected(first, second, kind):
    with pytest.raises(ValueError, match="exactly one"):
        _entrypoint(kind, first | second)
    if "weights" not in first | second:
        with pytest.raises(ValueError, match="exactly one"):
            DataSeries(X, Y, **(first | second))
        with pytest.raises(ValueError, match="exactly one"):
            effective_sigma(**(first | second))


@pytest.mark.parametrize(
    "lower, upper, message",
    [
        (0.0, 1.0, "strictly positive"),
        (1.0, -1.0, "strictly positive"),
        (np.nan, 1.0, "finite"),
        (1.0, np.inf, "finite"),
        ([[1.0, 2.0]], [[2.0, 3.0]], "one-dimensional"),
        ([1.0], [1.0, 2.0], "same shape"),
        (1.0, [1.0], "same shape"),
    ],
)
def test_asymmetric_errors_validate_both_sides(lower, upper, message):
    with pytest.raises(ValueError, match=message):
        AsymmetricError(lower, upper)


@pytest.mark.parametrize("sigma", [0.0, -1.0, np.nan, np.inf, np.ones((2, 2))])
def test_alias_and_curve_entrypoint_validate_errors(sigma):
    with pytest.raises(ValueError):
        DataSeries(X, Y, y_err=sigma)
    with pytest.raises(ValueError):
        fit_curve("linear", X, Y, sigma)


def test_alias_equivalence_preserves_asymmetric_sides():
    first = AsymmetricError([1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0])
    same_rms = AsymmetricError(first.upper, first.lower)
    with pytest.raises(ValueError, match="sigma and y_err"):
        DataSeries(X, Y, sigma=first, y_err=same_rms)
    with pytest.raises(ValueError, match="sigma and y_err"):
        DataSeries(X, Y, sigma=first, y_err=first.effective)
    series = DataSeries(X, Y, sigma=1.0, y_err=np.ones(4))
    assert series.sigma is series.y_err
    assert DataSeries(X, Y, sigma=first, y_err=first).sigma is first


@pytest.mark.parametrize(
    "errors, message",
    [
        ({"sigma": 0.0}, "strictly positive"),
        ({"sigma": np.ones((2, 2))}, "one-dimensional"),
        ({"sigma_low": 1.0}, "supplied together"),
        ({"sigma_low": [1.0, 2.0], "sigma_high": [1.0]}, "same length"),
        ({"sigma_cov": [[1.0, 1.0], [0.0, 1.0]]}, "symmetric"),
        ({"sigma_cov": np.ones((2, 2))}, "positive-definite"),
    ],
)
def test_effective_sigma_uses_shared_validation(errors, message):
    with pytest.raises(ValueError, match=message):
        effective_sigma(**errors)


@pytest.mark.parametrize("kind", ["sigma", "asymmetric", "pair", "covariance", "data"])
def test_mutated_series_is_revalidated_at_fit_time(kind):
    if kind == "sigma":
        series = DataSeries(X, Y, sigma=np.ones(4))
        series.sigma[0] = -1.0
    elif kind == "asymmetric":
        series = DataSeries(X, Y, sigma=AsymmetricError(np.ones(4), np.ones(4)))
        series.sigma.lower[0] = 0.0
    elif kind == "pair":
        series = DataSeries(X, Y, sigma_low=np.ones(4), sigma_high=np.ones(4))
        series.sigma_high = None
    elif kind == "covariance":
        series = DataSeries(X, Y, sigma_cov=np.eye(4))
        series.sigma_cov[0, 1] = 1.0
    else:
        series = DataSeries(X.copy(), Y, sigma=1.0)
        series.x[0] = np.nan
    with pytest.raises(ValueError):
        fit(series)
    if kind != "data":
        with pytest.raises(ValueError):
            _ = series.effective_sigma


@pytest.mark.parametrize(
    "stored",
    [
        {"sigma": 5.0},
        {"sigma": AsymmetricError(1.0, 2.0)},
        {"sigma_low": 1.0, "sigma_high": 2.0},
        {"sigma_cov": np.eye(4)},
    ],
)
@pytest.mark.parametrize("override", SPECIFICATIONS)
def test_complete_overrides_replace_the_stored_specification(stored, override):
    series = DataSeries(X, Y, label="original", **stored)
    result = fit(series, absolute_sigma=True, **override)
    reference = fit(X, Y, absolute_sigma=True, **override)
    np.testing.assert_allclose(list(result), list(reference), rtol=1e-8)
    np.testing.assert_allclose(result.covariance, reference.covariance, rtol=1e-8)
    assert result.series.label == "original"
    # The data container used by plotting must describe the errors actually fitted.
    if "sigma_cov" in override:
        assert result.series.sigma is None
        np.testing.assert_allclose(result.series.sigma_cov, override["sigma_cov"])
    else:
        assert result.series.sigma_cov is None
        np.testing.assert_allclose(result.series.effective_sigma, result.sigma)
    assert series.label == "original"


@pytest.mark.parametrize("absolute_sigma", [False, True])
def test_zero_weights_match_an_independent_weighted_linear_solution(absolute_sigma):
    x = np.arange(5.0)
    y = np.array([1.0, 2.4, 1e12, 5.1, 6.7])
    weights = np.array([4.0, 1.0, 0.0, 9.0, 2.0])
    keep = weights > 0
    design = np.column_stack([x[keep], np.ones(4)])
    precision = np.diag(weights[keep])
    absolute_cov = np.linalg.inv(design.T @ precision @ design)
    parameters = absolute_cov @ design.T @ precision @ y[keep]
    residual = y[keep] - design @ parameters
    chi_squared = float(residual @ precision @ residual)
    expected_cov = absolute_cov if absolute_sigma else absolute_cov * chi_squared / 2

    def model(points, slope, intercept):
        assert np.all(points != 2.0), "The excluded sample was evaluated"
        return slope * points + intercept

    with np.errstate(divide="raise", invalid="raise"):
        result = fit(x, y, model=model, p0=[1.0, 1.0], weights=weights, absolute_sigma=absolute_sigma)
    np.testing.assert_allclose(list(result), parameters, rtol=1e-6)
    np.testing.assert_allclose(result.covariance, expected_cov, rtol=1e-6)
    assert result.dof == 2
    assert result.reduced_chi2 == pytest.approx(chi_squared / 2)
    assert result.p_value == pytest.approx(chi2.sf(chi_squared, 2))
    assert result.residual_variance == pytest.approx(float(residual @ residual) / 2)
    np.testing.assert_array_equal(result.x, x[keep])
    np.testing.assert_array_equal(result.y, y[keep])
    np.testing.assert_allclose(result.residuals, residual, atol=1e-7)
    np.testing.assert_allclose(result.sigma, 1 / np.sqrt(weights[keep]))
    np.testing.assert_array_equal(result.series.x, result.x)


def test_zero_weights_are_excluded_from_default_plot():
    result = fit(np.arange(5.0), [1.0, 2.4, 1e12, 5.1, 6.7], weights=[4, 1, 0, 9, 2])
    plotted = plot_fit(result)
    try:
        ax = plotted.axes[0] if isinstance(plotted.axes, tuple) else plotted.axes
        data = ax.containers[0].lines[0]
        np.testing.assert_array_equal(data.get_xdata(), [0, 1, 3, 4])
    finally:
        plt.close(plotted.figure)


def test_zero_weights_reduce_degrees_of_freedom():
    with pytest.warns(UserWarning, match="0 degrees of freedom"):
        result = fit(X, Y, weights=[1, 0, 0, 1])
    assert result.dof == 0
    assert np.isnan(result.reduced_chi2)
    assert np.isnan(result.residual_variance)
    assert np.isnan(result.p_value)
    assert np.all(np.isnan(result.covariance))


@pytest.mark.parametrize(
    "errors",
    [
        {"sigma": 0.5},
        {"weights": 4.0},
        {"sigma": AsymmetricError(0.5, 0.5)},
        {"sigma_low": 0.5, "sigma_high": np.full(4, 0.5)},
    ],
)
def test_scalar_and_broadcast_errors_match_analytical_covariance(errors):
    design = np.column_stack([X, np.ones(4)])
    reference = 0.25 * np.linalg.inv(design.T @ design)
    result = fit(X, Y, absolute_sigma=True, **errors)
    np.testing.assert_allclose(result.covariance, reference, rtol=1e-6)


@pytest.mark.parametrize("scale", [1e-12, 1.0, 1e12])
def test_correlated_covariance_matches_generalized_linear_reference(scale):
    covariance = scale * (np.eye(4) + 0.1 * np.ones((4, 4)))
    design = np.column_stack([X, np.ones(4)])
    precision = np.linalg.inv(covariance)
    reference = np.linalg.inv(design.T @ precision @ design)
    result = fit(X, Y, sigma_cov=covariance, absolute_sigma=True)
    np.testing.assert_allclose(result.covariance, reference, rtol=1e-6)


def test_covariance_symmetry_tolerance_is_relative_and_canonicalized():
    covariance = np.eye(4) + 0.1 * np.ones((4, 4))
    covariance[0, 1] += 5e-14
    series = DataSeries(X, Y, sigma_cov=covariance)
    np.testing.assert_array_equal(series.sigma_cov, series.sigma_cov.T)
    assert effective_sigma(sigma_cov=covariance) is None
    covariance[0, 1] += 1e-8
    with pytest.raises(ValueError, match="symmetric"):
        DataSeries(X, Y, sigma_cov=covariance)


def test_asymmetric_rms_remains_finite_for_large_errors():
    maximum = np.finfo(float).max
    with np.errstate(over="raise", invalid="raise"):
        error = AsymmetricError(maximum, maximum)
        assert np.isfinite(error.effective)
        assert error.effective == pytest.approx(maximum)
        np.testing.assert_allclose(effective_sigma(sigma_low=maximum, sigma_high=maximum), maximum)


@pytest.mark.parametrize("loader_name", ["load_csv", "load_txt"])
def test_asymmetric_files_do_not_infer_a_competing_sigma(tmp_path, loader_name):
    from labfit import io

    separator = "," if loader_name == "load_csv" else " "
    path = tmp_path / "asymmetric.txt"
    rows = ["x y sigma_low sigma_high", "0 1 0.2 0.3", "1 3 0.3 0.4", "2 5 0.4 0.5"]
    path.write_text("\n".join(row.replace(" ", separator) for row in rows))
    series = getattr(io, loader_name)(path)
    assert series.sigma is None
    assert series.y_err is None
    np.testing.assert_allclose(series.effective_sigma, np.sqrt([0.065, 0.125, 0.205]))
    if loader_name == "load_csv":
        result = fit(path, sigma_cov=np.eye(3), label="override")
        assert result.series.label == "override"
        assert result.series.sigma_low is None
        np.testing.assert_array_equal(result.series.sigma_cov, np.eye(3))


@pytest.mark.parametrize("columns", ["sigma,sigma_low,sigma_high", "sigma_low", "sigma_high"])
def test_files_reject_ambiguous_or_incomplete_errors(tmp_path, columns):
    from labfit.io import load_csv

    path = tmp_path / "invalid.csv"
    values = ",".join("0.5" for _ in columns.split(","))
    path.write_text(f"x,y,{columns}\n0,1,{values}\n1,3,{values}\n")
    with pytest.raises(ValueError):
        load_csv(path)


@pytest.mark.parametrize("field", ["x", "y"])
def test_weights_cannot_hide_mutated_data_dimensions(field):
    series = DataSeries(X, Y, sigma=1.0)
    setattr(series, field, np.ones((2, 2)))
    with pytest.raises(ValueError, match="one-dimensional"):
        fit(series, weights=1.0)


def test_weights_must_retain_a_data_point_even_for_empty_input():
    with pytest.raises(ValueError, match="retain at least one"):
        fit([], [], weights=1.0)


def test_documented_zero_weight_example():
    import textwrap
    from pathlib import Path

    document = (Path(__file__).resolve().parents[1] / "docs" / "concepts.rst").read_text(encoding="utf-8")
    section = document.split("Valid uncertainty specifications", 1)[1]
    block = section.split(".. code-block:: python", 1)[1].lstrip("\n")
    lines = []
    for line in block.splitlines():
        if line and not line.startswith("   "):
            break
        lines.append(line)
    namespace = {}
    exec(textwrap.dedent("\n".join(lines)), namespace)
    assert namespace["result"].params["slope"] == pytest.approx(2.0)
    assert namespace["result"].params["intercept"] == pytest.approx(1.0)


@pytest.mark.parametrize("scale", [1e-15, 1.0, 1e15])
@pytest.mark.parametrize("asymmetric", [False, True])
def test_alias_agreement_is_independent_of_measurement_units(scale, asymmetric):
    sigma = AsymmetricError(scale, 2 * scale) if asymmetric else scale
    alias = AsymmetricError(2 * scale, scale) if asymmetric else 2 * scale
    with pytest.raises(ValueError, match="sigma and y_err"):
        DataSeries(X, Y, sigma=sigma, y_err=alias)
