"""Statistics regressions with analytical references, including invalid dof."""

import warnings

import numpy as np
import pytest

from labfit import FitResult, fit, fitter_impl


@pytest.mark.parametrize("error_kind", ["sigma", "weights", "covariance"])
def test_weighted_linear_statistics_match_analytical_reference(error_kind):
    x = np.array([-1.0, 0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 1.0, 4.0])
    errors = {
        "sigma": {"sigma": 0.5},
        "weights": {"weights": np.full(4, 4.0)},
        "covariance": {"sigma_cov": 0.25 * np.eye(4) + 0.1 * np.ones((4, 4))},
    }[error_kind]
    result = fit(x, y, **errors)
    # Least-squares line: 1.2*x + 0.9, residuals [0.3, 0.1, -1.1, 0.7].
    # A common correlated offset changes intercept covariance, not these residuals.
    np.testing.assert_allclose(list(result.params.values()), [1.2, 0.9], rtol=1e-6)
    assert result.dof == 2
    assert result.residual_variance == pytest.approx(1.8 / 2)
    assert result.reduced_chi2 == pytest.approx(7.2 / 2)
    # For two degrees of freedom, the chi-square survival function is exp(-chi2/2).
    assert result.p_value == pytest.approx(np.exp(-7.2 / 2))


@pytest.mark.parametrize("absolute_sigma", [False, True])
@pytest.mark.parametrize("scale", [0.001, 1.0, 1000.0])
def test_unweighted_statistics_and_summary_respect_y_units(absolute_sigma, scale):
    x = [-1.0, 0.0, 1.0, 2.0]
    y = scale * np.array([0.0, 1.0, 1.0, 4.0])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fit(x, y, absolute_sigma=absolute_sigma)
    assert result.dof == 2
    assert result.residual_variance == pytest.approx(scale**2 * 0.9)
    assert np.isnan(result.reduced_chi2)
    assert np.isnan(result.p_value)
    assert len(caught) == 1
    assert "No uncertainties provided" in str(caught[0].message)
    summary = str(result)
    assert "residual variance" in summary
    assert "y units squared" in summary
    assert "degrees of freedom = 2" in summary
    assert "reduced chi2" not in summary
    assert "p =" not in summary and "p ~" not in summary
    assert "errors may be" not in summary and "errors underestimated" not in summary


@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("absolute_sigma", [False, True])
def test_nonpositive_dof_is_preserved(n, weighted, absolute_sigma):
    errors = {"sigma": 2.0} if weighted else {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fit(
            np.arange(n), 1.0 + 2.0 * np.arange(n), p0=[1.0, 1.0], absolute_sigma=absolute_sigma, **errors
        )
    assert result.dof == n - 2
    assert result.success
    assert np.isnan(result.residual_variance)
    assert np.isnan(result.reduced_chi2)
    assert np.isnan(result.p_value)
    assert any(f"{n - 2} degrees of freedom" in str(w.message) for w in caught)
    summary = str(result)
    assert f"degrees of freedom = {n - 2}" in summary
    assert "Nonpositive degrees of freedom" in summary
    assert "reduced chi2 =" not in summary and "p =" not in summary
    if n == 2 and absolute_sigma:
        assert result.identifiable
        variance = 4.0 if weighted else 1.0
        np.testing.assert_allclose(result.covariance, variance * np.array([[2, -1], [-1, 1]]), atol=1e-6)
    else:
        assert np.all(np.isnan(result.covariance))
        assert all(np.isnan(u) for u in result.uncertainties.values())
        if n == 2:
            assert result.identifiable  # Missing residual scale is not a rank failure.
            assert "Parameter uncertainties are unavailable" in summary


@pytest.mark.parametrize("options", [{"is_weighted": False}, {"dof": 0}, {"dof": -1}])
def test_summary_gates_manually_constructed_statistics(options):
    result = FitResult(reduced_chi2=100.0, p_value=0.00001, params={"a": 1.0}, **options)
    summary = str(result)
    assert "reduced chi2 =" not in summary
    assert "p ~" not in summary
    assert "errors underestimated" not in summary


def test_rank_deficient_fit_has_no_chi_square_p_value():
    def redundant(x, a, b):
        return (a + b) * x

    with pytest.warns(UserWarning, match="not identifiable"):
        result = fit([0, 1, 2, 3], [0, 2, 4, 6], model=redundant, sigma=0.5, p0=[1, 1])
    assert result.dof == 2
    assert not result.identifiable
    assert np.isnan(result.p_value)
    assert "errors may be overestimated" not in str(result)


def test_failed_optimization_has_no_chi_square_p_value(monkeypatch):
    original = fitter_impl.least_squares

    def failed(*args, **kwargs):
        opt = original(*args, **kwargs)
        opt.success = False
        opt.message = "Test optimizer failure"
        return opt

    monkeypatch.setattr(fitter_impl, "least_squares", failed)
    with pytest.warns(UserWarning, match="did NOT converge"):
        result = fit([-1, 0, 1, 2], [0, 1, 1, 4], sigma=0.5)
    assert not result.success
    assert np.isnan(result.p_value)
    assert "did NOT converge" in str(result)
    assert "p =" not in str(result)
