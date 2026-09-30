"""Independent numerical and public API regressions from the repository review."""

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from labfit import AsymmetricError, DataSeries, fit
from labfit.models import exgaussian
from labfit.plot import _plot_series


@pytest.mark.parametrize("x", [-5.0, -1.0, 0.0, 2.0, 8.0, 30.0])
def test_exgaussian_matches_independent_convolution(x):
    amplitude, mu, sigma, tau = 2.5, 0.2, 0.8, 1.5
    expected, _ = quad(
        lambda t: amplitude * norm.pdf(x - t, loc=mu, scale=sigma) * np.exp(-t / tau) / tau,
        0.0,
        np.inf,
        epsabs=1e-12,
    )
    assert exgaussian(x, amplitude, mu, sigma, tau) == pytest.approx(expected, rel=1e-7, abs=1e-12)


def test_exgaussian_area_and_tails():
    area, _ = quad(lambda x: exgaussian(x, 2.5, 0.2, 0.8, 1.5), -np.inf, np.inf)
    assert area == pytest.approx(2.5)
    tails = exgaussian([-100.0, 100.0], 1.0, 0.0, 1.0, 1.0)
    assert np.all(np.isfinite(tails))
    assert np.all(tails < 1e-20)
    assert exgaussian(10.0, 1.0, 0.0, 1.0, 1.0) > exgaussian(-10.0, 1.0, 0.0, 1.0, 1.0)


@pytest.mark.parametrize("argument", ["p0", "bounds"])
def test_parameter_typos_are_rejected(argument):
    value = 1.0 if argument == "p0" else (0.0, 0.1)
    with pytest.raises(ValueError, match=f"Unknown {argument} parameters: slpoe.*slope, intercept"):
        fit([0, 1, 2], [1, 3, 5], sigma=1.0, **{argument: {"slpoe": value}})


@pytest.mark.parametrize("p0", [[], [1.0], [1.0, 2.0, 3.0], [[1.0, 2.0]], [[1.0], [2.0]]])
def test_malformed_initial_guesses_are_rejected(p0):
    with pytest.raises(ValueError, match="p0 must contain exactly 2 values in a one-dimensional array"):
        fit([0, 1, 2], [1, 3, 5], sigma=1.0, p0=p0)


def test_partial_parameter_dictionaries_and_scalar_constant_guess():
    result = fit([0, 1, 2], [1, 3, 5], sigma=1.0, p0={"slope": 1.0}, bounds={"slope": (0, 3)})
    assert result.params == pytest.approx({"slope": 2.0, "intercept": 1.0})
    constant = fit([0, 1, 2], [3, 3, 3], model="constant", sigma=1.0, p0=2.0)
    assert constant.params["level"] == pytest.approx(3.0)


@pytest.mark.parametrize(
    "errors, lower, upper",
    [
        ({"sigma_low": [1, 2, 3], "sigma_high": [3, 4, 5]}, [1, 2, 3], [3, 4, 5]),
        ({"sigma": AsymmetricError(np.array([1, 2, 3]), np.array([3, 4, 5]))}, [1, 2, 3], [3, 4, 5]),
        ({"sigma": AsymmetricError(np.array(1.0), np.array(3.0))}, [1, 1, 1], [3, 3, 3]),
        ({"sigma_low": 1.0, "sigma_high": 3.0}, [1, 1, 1], [3, 3, 3]),
        ({"sigma_cov": [[1, 0.5, 0], [0.5, 4, 0], [0, 0, 9]]}, [1, 2, 3], [1, 2, 3]),
    ],
)
def test_error_bar_endpoints_preserve_measurement_errors(errors, lower, upper):
    series = DataSeries([0, 1, 2], [1, 3, 5], **errors)
    fig, ax = plt.subplots()
    try:
        _plot_series(ax, series, 0)
        segments = np.asarray(ax.containers[0].lines[2][0].get_segments())
        np.testing.assert_allclose(segments[:, 0, 1], series.y - lower)
        np.testing.assert_allclose(segments[:, 1, 1], series.y + upper)
    finally:
        plt.close(fig)


def test_documented_half_life_example():
    document = (Path(__file__).resolve().parents[1] / "docs" / "concepts.rst").read_text(encoding="utf-8")
    section = document.split("Errors on derived quantities", 1)[1]
    block = section.split(".. code-block:: python", 1)[1].lstrip("\n")
    lines = []
    for line in block.splitlines():
        if line and not line.startswith("   "):
            break
        lines.append(line)
    namespace = {}
    exec(textwrap.dedent("\n".join(lines)), namespace)
    assert namespace["half_life"] == pytest.approx(np.log(2.0) / 0.5)
    assert namespace["half_life_error"] == pytest.approx(np.log(2.0) / 0.5**2 * 0.02)
