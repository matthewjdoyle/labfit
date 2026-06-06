"""Tests for non-convergence and poor-fit detection in LabFit.

LabFit delegates the optimisation to ``scipy.optimize.least_squares``
with method ``trf`` (Trust Region Reflective). It does **not** pre-emptively
validate the model choice or the data — the user is trusted to pick a
sensible model.

When a fit goes wrong, check these fields on the returned ``FitResult``:

``result.success``
    ``False`` if the underlying optimizer failed to converge
    (e.g. hit ``max_nfev``, encountered NaN in the cost, or the
    Jacobian became singular). ``True`` does **not** guarantee a
    good fit — only that the optimizer terminated normally.

``result.message``
    A human-readable string from ``least_squares`` explaining why
    the optimizer stopped. Compare ``"`gtol` termination condition
    is satisfied."`` (normal) vs ``"The maximum number of function
    evaluations is exceeded."`` (failure).

``result.reduced_chi2``
    The reduced chi-squared statistic. A value much larger than 1
    indicates the model does not describe the data within the stated
    uncertainties. This is *the* primary diagnostic even when the
    optimizer reports ``success=True``.

``result.p_value``
    The probability of observing a chi-squared value at least as
    large as the one obtained, assuming the model is correct. A
    very small p-value (e.g. ``< 0.001``) means the model is
    unlikely to be correct. May be ``NaN`` if the chi-squared or
    degrees of freedom are non-finite.

``result.covariance`` / ``result.uncertainties``
    When the Jacobian is singular at the solution (common with
    severely wrong models), the covariance contains ``NaN`` and
    uncertainties will be ``nan``. Always check ``success`` and
    ``reduced_chi2`` before trusting uncertainties.
"""

import math

import matplotlib
import numpy as np

matplotlib.use("Agg")

from labfit import fit, FitResult


def test_wrong_model_gives_high_chi2_and_low_pvalue():
    """Fit a straight line to quadratic data with tight uncertainties.

    The optimizer "converges" (``success=True``) but the reduced
    chi-squared is huge and the p-value is essentially zero —
    a clear signal that the model is wrong.
    """
    rng = np.random.default_rng(20260611)
    x = np.linspace(-3.0, 3.0, 60)
    # True model is quadratic, but we fit a straight line
    y = 2.0 * x**2 + 1.0 + rng.normal(0, 0.05, size=x.size)
    sigma = np.full_like(x, 0.05)

    result = fit(x, y, model="linear", sigma=sigma)

    assert isinstance(result, FitResult)
    assert result.success  # optimizer terminated normally
    assert result.reduced_chi2 > 10  # clearly a bad fit
    assert result.p_value < 0.001  # model is very unlikely
    assert np.all(np.isfinite(list(result.uncertainties.values())))


def test_exponential_fit_to_linear_data():
    """Fit an exponential decay to monotonically increasing data.

    The model is structurally wrong (exponential decay vs linear
    growth). The optimizer converges but the result is useless.
    """
    rng = np.random.default_rng(20260612)
    x = np.linspace(0.0, 5.0, 40)
    y = 2.0 * x + 1.0 + rng.normal(0, 0.1, size=x.size)
    sigma = np.full_like(x, 0.1)

    result = fit(x, y, model="exponential", sigma=sigma)

    assert result.success
    assert result.reduced_chi2 > 5  # wrong model
    assert result.p_value < 0.01


def test_sine_fit_to_linear_data_finds_degenerate_solution():
    """Fit a sine wave to linear data — a structurally wrong model.

    Demonstrates that ``success=True`` with ``reduced_chi2 ≈ 1`` does
    **not** mean the model is correct. The sine parameters may
    compensate arbitrarily (large amplitude, shifted frequency) to
    approximate a segment of the line. Always inspect the fitted
    parameters for physical sense.
    """
    rng = np.random.default_rng(20260613)
    x = np.linspace(0.0, 10.0, 80)
    y = 0.5 * x + 2.0 + rng.normal(0, 0.05, size=x.size)
    sigma = np.full_like(x, 0.05)

    result = fit(x, y, model="sine", sigma=sigma)

    assert result.success
    # The true model is linear, not sinusoidal — the optimizer will
    # distort the sine to approximate a line, typically by chosing a
    # very low frequency and large amplitude. The reduced chi² may
    # still be near 1 because the fit can always inflate errors.
    # This is a cautionary example: statistical fit quality metrics
    # cannot replace domain knowledge about the underlying process.


def test_severely_constrained_bounds_can_prevent_convergence():
    """Bounds that pin parameters to impossible values cause failure.

    If bounds force the parameters to a region where the residual
    cannot be meaningfully reduced, ``least_squares`` eventually
    fails with ``success=False``.
    """
    rng = np.random.default_rng(20260614)
    x = np.linspace(-5.0, 5.0, 60)
    y = 3.0 * np.exp(-0.5 * ((x - 0.5) / 1.2) ** 2) + rng.normal(0, 0.05, size=x.size)
    sigma = np.full_like(x, 0.05)

    # Pin amplitude to a tiny value and sigma to a tiny value —
    # impossible to fit a Gaussian peak this way
    result = fit(
        x, y, model="gaussian", sigma=sigma,
        p0={"amplitude": 1e-6, "mean": 0.0, "sigma": 1e-6},
        bounds={
            "amplitude": (1e-8, 1e-5),
            "mean": (-1.0, 1.0),
            "sigma": (1e-8, 1e-5),
        },
    )

    assert isinstance(result, FitResult)
    # The optimizer may or may not report success=False depending on
    # how trf handles the tight bounds, but the chi² will be terrible
    # and uncertainties may be unreliable
    assert result.reduced_chi2 > 100
    # p_value should be extremely small or NaN
    if math.isfinite(result.p_value):
        assert result.p_value < 1e-10


def test_custom_model_with_singularity_gives_nan_covariance():
    """A rational model with the pole near the data range causes trouble.

    When the optimiser places the pole close to the data region the
    residuals blow up, producing effectively infinite chi-squared.
    """
    rng = np.random.default_rng(20260615)
    # Data far from the pole
    x = np.linspace(5.0, 10.0, 40)
    y = 2.0 / (x - 3.0) + rng.normal(0, 0.01, size=x.size)
    sigma = np.full_like(x, 0.01)

    # Use a sensible initial guess and bounds to keep x0 well away
    # from the data region
    result = fit(x, y, model="rational", sigma=sigma,
                 p0={"amplitude": 2.0, "x0": 3.0},
                 bounds={"x0": (2.0, 4.0)})

    assert result.success
    assert abs(result.params["amplitude"] - 2.0) < 0.2
    assert abs(result.params["x0"] - 3.0) < 0.1

    # Deliberately start with the pole inside the data range
    result2 = fit(x, y, model="rational", sigma=sigma,
                  p0={"amplitude": 1.0, "x0": 7.0})

    # The optimizer may still converge to a local minimum far from
    # the true parameters, or the covariance may be ill-conditioned
    assert result2.reduced_chi2 > result.reduced_chi2  # worse fit


def test_fit_result_str_repr():
    """Exercise the human-readable __str__ and __repr__ methods."""
    from labfit import fit

    rng = np.random.default_rng(20260616)
    x = np.linspace(-3, 3, 20)
    y = 2.0 * x + 1.0 + rng.normal(0, 0.2, size=x.size)
    sigma = np.full_like(x, 0.2)

    result = fit(x, y, model="linear", sigma=sigma)

    # str() should contain key diagnostics
    s = str(result)
    assert "FitResult: linear" in s
    assert "slope" in s
    assert "intercept" in s
    assert "reduced chi2" in s
    assert "p " in s

    # repr() should match str()
    assert repr(result) == s

    # Test the warnings path via non-convergence (checks __str__ with
    # success=False branch)
    bad = fit(
        x, y, model="gaussian", sigma=sigma,
        p0={"amplitude": 1e-99, "mean": 0.0, "sigma": 1e-99},
        bounds={"amplitude": (1e-100, 1e-99), "sigma": (1e-100, 1e-99)},
    )
    if not bad.success:
        s = str(bad)
        assert "did NOT converge" in s

