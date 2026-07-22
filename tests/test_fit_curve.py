import math

import numpy as np

from labfit._fit import fit_curve


def test_fit_curve_linear_reports_chi2_pvalue_and_uncertainties():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = 2.0 * x + 1.0
    y_err = np.array([1.0, 1.0, 1.0, 1.0])

    result = fit_curve("linear", x, y, y_err)

    assert math.isclose(result.reduced_chi2, 0.0, abs_tol=1e-12)
    assert math.isclose(result.p_value, 1.0, rel_tol=0.0, abs_tol=1e-12)
    assert set(result.uncertainties) == {"slope", "intercept"}
    assert np.allclose(
        [result.uncertainties["slope"], result.uncertainties["intercept"]],
        np.sqrt(np.diag(result.covariance)),
    )
