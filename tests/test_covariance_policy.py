"""Covariance checks against independent generalized linear regression."""

import numpy as np
import pytest

from labfit import DataSeries, Fitter, fit, fit_curve, fit_multi


@pytest.mark.parametrize("absolute_sigma", [False, True])
@pytest.mark.parametrize("error_kind", ["sigma", "weights", "covariance"])
def test_linear_covariance_and_error_rescaling(absolute_sigma, error_kind):
    x = np.array([-2.0, -0.5, 0.0, 1.0, 2.5, 4.0])
    y = np.array([-2.8, -0.9, 0.4, 1.1, 3.7, 5.0])
    sigma = np.array([0.4, 0.7, 0.5, 0.8, 0.6, 0.9])
    measurement_cov = np.diag(sigma**2)
    if error_kind == "covariance":
        measurement_cov += 0.06 * np.ones((x.size, x.size))
    design = np.column_stack((x, np.ones_like(x)))
    precision = np.linalg.inv(measurement_cov)
    expected_cov = np.linalg.inv(design.T @ precision @ design)
    expected_params = expected_cov @ design.T @ precision @ y
    residual = y - design @ expected_params
    expected_chi2 = residual @ precision @ residual
    if not absolute_sigma:
        expected_cov *= expected_chi2 / (x.size - 2)

    results = []
    for scale in (1.0, 10.0):
        errors = {
            "sigma": {"sigma": scale * sigma},
            "weights": {"weights": 1.0 / (scale * sigma) ** 2},
            "covariance": {"sigma_cov": scale**2 * measurement_cov},
        }[error_kind]
        result = fit(x, y, absolute_sigma=absolute_sigma, **errors)
        results.append(result)
        cov_scale = scale**2 if absolute_sigma else 1.0
        np.testing.assert_allclose(list(result.params.values()), expected_params, rtol=2e-6)
        np.testing.assert_allclose(
            result.covariance, cov_scale * expected_cov, rtol=2e-6, atol=cov_scale * 1e-9
        )
        np.testing.assert_allclose(
            list(result.uncertainties.values()), np.sqrt(np.diag(cov_scale * expected_cov)), rtol=2e-6
        )
        assert result.reduced_chi2 == pytest.approx(expected_chi2 / (scale**2 * (x.size - 2)))
        assert result.absolute_sigma is absolute_sigma
    expected_ratio = 10.0 if absolute_sigma else 1.0
    np.testing.assert_allclose(
        list(results[1].uncertainties.values()),
        expected_ratio * np.array(list(results[0].uncertainties.values())),
        rtol=2e-6,
    )


def test_default_policy_preserves_relative_scaling():
    x = np.arange(6.0)
    y = np.array([0.3, 1.0, 2.5, 2.7, 4.1, 5.4])
    default = fit(x, y, sigma=0.5)
    relative = fit(x, y, sigma=0.5, absolute_sigma=False)
    absolute = fit(x, y, sigma=0.5, absolute_sigma=True)
    assert default.absolute_sigma is False
    np.testing.assert_allclose(default.covariance, relative.covariance)
    np.testing.assert_allclose(relative.covariance, absolute.covariance * relative.reduced_chi2)
    np.testing.assert_allclose(list(relative.params.values()), list(absolute.params.values()))
    assert relative.p_value == absolute.p_value


@pytest.mark.parametrize("absolute_sigma", [False, True])
def test_unweighted_covariance_policy(absolute_sigma):
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = np.array([-1.2, 0.1, 0.7, 2.3, 2.8])
    design = np.column_stack((x, np.ones_like(x)))
    parameters = np.linalg.lstsq(design, y, rcond=None)[0]
    expected = np.linalg.inv(design.T @ design)
    if not absolute_sigma:
        expected *= np.sum((y - design @ parameters) ** 2) / (len(x) - 2)
    with pytest.warns(UserWarning, match="No uncertainties provided"):
        result = fit(x, y, absolute_sigma=absolute_sigma)
    np.testing.assert_allclose(result.covariance, expected, rtol=2e-6, atol=1e-9)


@pytest.mark.parametrize("entrypoint", ["fit", "fit_curve", "fit_multi", "fitter", "fitter_multi"])
def test_absolute_policy_reaches_all_public_entrypoints(entrypoint):
    # Exact data must still give nonzero errors when measurement sigma is known.
    series = DataSeries([-1, 0, 1], [-1, 1, 3], sigma=2.0)
    if entrypoint == "fit":
        result = fit(series, absolute_sigma=True)
    elif entrypoint == "fit_curve":
        result = fit_curve("linear", series.x, series.y, series.sigma, absolute_sigma=True)
    elif entrypoint == "fit_multi":
        result = fit_multi([series], absolute_sigma=True)[0]
    elif entrypoint == "fitter":
        result = Fitter().fit(series, absolute_sigma=True)
    else:
        result = Fitter().fit_multi([series], absolute_sigma=True)[0]
    np.testing.assert_allclose(result.covariance, np.diag([2.0, 4.0 / 3.0]), atol=1e-7)
    assert result.absolute_sigma is True
