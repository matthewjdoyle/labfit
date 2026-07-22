"""Tests for Phase 3 scientific-correctness features.

Covers:
- ``is_weighted`` flag and unweighted-chi² warning
- confidence/prediction band rendering
- numerical Jacobian fallback in ``propagate_errors``
- singular covariance warning
- new ``__repr__`` format
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest

from labfit import FitResult, fit, plot_fit
from labfit._fit import fit_curve
from labfit.plot import _confidence_band
from labfit.utils import propagate_errors


def test_unweighted_fit_sets_flag_and_warns():
    x = np.linspace(0.0, 10.0, 20)
    y = 2.0 * x + 1.0 + np.random.default_rng(42).normal(0, 0.5, size=x.size)

    with pytest.warns(UserWarning, match="No uncertainties provided"):
        result = fit(x, y, model="linear")

    assert result.is_weighted is False
    assert math.isfinite(result.reduced_chi2)


def test_weighted_fit_sets_flag_and_no_warning():
    x = np.linspace(0.0, 10.0, 20)
    y = 2.0 * x + 1.0 + np.random.default_rng(42).normal(0, 0.5, size=x.size)
    sigma = np.full_like(x, 0.5)

    result = fit(x, y, model="linear", sigma=sigma)
    assert result.is_weighted is True


def test_fit_curve_is_weighted():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = 2.0 * x + 1.0
    y_err = np.array([1.0, 1.0, 1.0, 1.0])

    result = fit_curve("linear", x, y, y_err)
    assert result.is_weighted is True


def test_confidence_band_returns_none_without_covariance():
    result = FitResult(
        reduced_chi2=1.0,
        params={"slope": 2.0, "intercept": 1.0},
        param_names=("slope", "intercept"),
        x=np.array([0.0, 1.0]),
        y=np.array([1.0, 3.0]),
        model=lambda t, m, b: m * t + b,
    )
    xs = np.linspace(0.0, 1.0, 10)
    band = _confidence_band(result, xs)
    assert band is None


def test_confidence_band_with_covariance():
    x = np.linspace(0.0, 10.0, 50)
    y = 2.0 * x + 1.0 + np.random.default_rng(100).normal(0, 0.2, size=x.size)
    sigma = np.full_like(x, 0.2)

    result = fit(x, y, model="linear", sigma=sigma)
    xs = np.linspace(0.0, 10.0, 20)
    band = _confidence_band(result, xs, ci_level=0.68)
    assert band is not None
    lower, upper = band
    assert lower.shape == xs.shape
    assert upper.shape == xs.shape
    assert np.all(upper >= lower)


def test_plot_fit_with_ci_band(tmp_path):
    x = np.linspace(0.0, 10.0, 30)
    y = 2.0 * x + 1.0 + np.random.default_rng(200).normal(0, 0.3, size=x.size)
    sigma = np.full_like(x, 0.3)

    result = fit(x, y, model="linear", sigma=sigma)
    plotter = plot_fit(result, show_ci=True, ci_level=0.95)
    assert plotter.figure is not None

    path = tmp_path / "ci_band.png"
    plotter.save(path)
    assert path.exists() and path.stat().st_size > 0


def test_propagate_errors_numerical_jacobian():
    value, uncertainty = propagate_errors(
        lambda x: x**2,
        x=3.0,
        x_error=0.5,
    )
    assert math.isclose(value, 9.0)
    # d/dx(x^2) = 2x = 6, so sigma = 6 * 0.5 = 3.0
    assert math.isclose(uncertainty, 3.0, rel_tol=0.01, abs_tol=0.01)


def test_propagate_errors_numerical_jacobian_with_covariance():
    covariance = np.array([[4.0]])
    value, uncertainty = propagate_errors(
        lambda x: x + 1.0,
        covariance=covariance,
        x=2.0,
    )
    assert value == 3.0
    # d/dx(x+1) = 1, so sigma = sqrt(1^2 * 4) = 2.0
    assert math.isclose(uncertainty, 2.0, rel_tol=0.01, abs_tol=0.01)


def test_fit_result_repr_concise():
    x = np.linspace(0.0, 5.0, 10)
    y = 2.0 * x + 1.0
    sigma = np.full_like(x, 0.1)

    result = fit(x, y, model="linear", sigma=sigma)
    r = repr(result)
    s = str(result)

    assert r != s
    assert "FitResult(" in r
    assert "model_name=" in r
    assert "reduced_chi2=" in r
    assert "success=" in r
    assert "\n" not in r  # repr should be single-line


def test_new_initial_guess_models():
    """Verify completed initial guesses for previously-missing models."""
    from labfit.fitter_impl import _initial_guess

    x = np.linspace(-5.0, 5.0, 50)
    y = np.sin(x)

    assert len(_initial_guess("sinc", "sinc", x, y)) == 3
    assert len(_initial_guess("exponential_rise", "exponential_rise", x, y)) == 3
    assert len(_initial_guess("double_exponential", "double_exponential", x, y)) == 4
    assert len(_initial_guess("moffat", "moffat", x, y)) == 4
    assert len(_initial_guess("gaussian_baseline", "gaussian_baseline", x, y)) == 5
    assert len(_initial_guess("bimodal_gaussian", "bimodal_gaussian", x, y)) == 6
