"""Independent identifiability checks for fitted parameters."""

import numpy as np
import pytest

from labfit import fit


def redundant_line(x, a, b):
    return (a + b) * x


def almost_redundant_line(x, a, b):
    return a * x + b * (x + 1e-9)


@pytest.mark.parametrize("absolute_sigma", [False, True])
def test_converged_redundant_parameters_are_not_reported_as_known(absolute_sigma):
    x = np.arange(1.0, 7.0)
    y = 3.0 * x + np.array([0.1, -0.1, 0.1, -0.1, 0.1, -0.1])
    with pytest.warns(UserWarning, match="parameters are not identifiable"):
        result = fit(x, y, model=redundant_line, p0=[1.0, 1.0], sigma=0.2, absolute_sigma=absolute_sigma)
    assert result.success
    assert not result.identifiable
    assert result.jacobian_condition > 1e8
    assert np.isnan(result.covariance).all()
    assert all(np.isnan(value) for value in result.uncertainties.values())
    assert "not identifiable" in str(result)
    assert result.predict(x) == pytest.approx(3.0 * x, abs=0.1)


def test_nearly_redundant_parameters_are_flagged_by_conditioning():
    x = np.arange(1.0, 7.0)
    y = 3.0 * x + np.array([0.1, -0.1, 0.1, -0.1, 0.1, -0.1])
    with pytest.warns(UserWarning, match="condition"):
        result = fit(x, y, model=almost_redundant_line, p0=[1.0, 1.0], sigma=0.2)
    assert result.success
    assert not result.identifiable
    assert np.isnan(result.covariance).all()


def test_zero_sensitivity_parameter_is_unidentifiable():
    def flat_direction(x, slope, invisible):
        return slope * x + 0.0 * invisible

    x = np.arange(1.0, 7.0)
    with pytest.warns(UserWarning, match="rank 1/2"):
        result = fit(x, 2.0 * x, model=flat_direction, p0=[1.0, 5.0], sigma=1.0)
    assert result.success
    assert result.jacobian_rank == 1
    assert not result.identifiable
    assert np.isnan(result.covariance).all()


def test_well_identified_parameters_retain_finite_covariance():
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    y = 2.0 * x + 1.0 + np.array([0.05, -0.02, 0.02, -0.04, 0.01, 0.03])
    result = fit(x, y, model="linear", sigma=0.1, absolute_sigma=True)
    assert result.success and result.identifiable
    assert result.jacobian_rank == 2
    assert np.isfinite(result.jacobian_condition)
    assert np.isfinite(result.covariance).all()
    assert all(np.isfinite(value) and value > 0 for value in result.uncertainties.values())
    assert "not identifiable" not in str(result)
