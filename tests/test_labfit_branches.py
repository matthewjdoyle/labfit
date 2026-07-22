from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from labfit import AsymmetricError, DataSeries, Dataset, FitResult, Plotter, fit
from labfit import models as m
from labfit.fitter_impl import (
    _coerce_bounds,
    _coerce_series_input,
    _fit_single,
    _initial_guess,
    _normalize_sigma,
    fit_multi,
)
from labfit.io import load_csv as _load_csv
from labfit.plot import _add_result_axes, _as_results, plot_result
from labfit.utils import effective_sigma, propagate_errors


def test_models_registry_and_builtin_functions():
    fn, name = m.get_model("gaussian")
    assert name == "gaussian"
    assert fn is m.gaussian
    assert m.model_param_names("linear") == ("slope", "intercept")
    assert len(m.MODEL_NAMES) >= 10

    x = np.array([0.0, 1.0, 2.0])
    assert np.allclose(m.constant(x, 3.0), 3.0)
    assert np.allclose(m.linear(x, 2.0, 1.0), [1.0, 3.0, 5.0])
    assert np.allclose(m.quadratic(x, 1.0, 0.0, 0.0), x**2)
    assert np.allclose(m.cubic(x, 1.0, 0.0, 0.0, 0.0), x**3)
    assert m.gaussian(x, 1.0, 1.0, 1.0).shape == x.shape
    assert m.lorentzian(x, 1.0, 1.0, 1.0).shape == x.shape
    assert m.exponential(x, 1.0, 1.0).shape == x.shape
    assert m.power_law(np.array([1.0, 2.0, 3.0]), 2.0, 1.0).shape == (3,)
    assert m.logistic(x, 1.0, 1.0, 1.0).shape == x.shape
    assert m.sine(x, 1.0, 1.0, 0.0).shape == x.shape
    assert m.cosine(x, 1.0, 1.0, 0.0).shape == x.shape
    assert m.damped_oscillator(x, 1.0, 0.1, 1.0, 0.0).shape == x.shape
    assert m.damped_sine(x, 1.0, 0.1, 1.0, 0.0).shape == x.shape


def test_newer_model_shapes_and_param_names():
    x = np.array([0.0, 1.0, 2.0])
    assert m.sinc(x, 1.0, 1.0, 1.0).shape == x.shape
    assert m.exponential_rise(x, 1.0, 1.0).shape == x.shape
    assert m.double_exponential(x, 1.0, 1.0, 1.0, 1.0).shape == x.shape
    assert m.moffat(x, 1.0, 0.0, 1.0, 1.0).shape == x.shape
    assert m.gaussian_baseline(x, 1.0, 0.0, 1.0, 0.5, 0.1).shape == x.shape
    assert m.bimodal_gaussian(x, 1.0, 0.0, 1.0, 1.5, 0.5, 1.0).shape == x.shape
    assert m.model_param_names("sinc") == ("amplitude", "center", "width")
    assert m.model_param_names("exponential_rise") == ("amplitude", "tau", "offset")
    assert m.model_param_names("double_exponential") == ("amplitude1", "tau1", "amplitude2", "tau2")
    assert m.model_param_names("moffat") == ("amplitude", "x0", "alpha", "beta")
    assert m.model_param_names("gaussian_baseline") == ("amplitude", "mean", "sigma", "m", "b")
    assert m.model_param_names("bimodal_gaussian") == (
        "amplitude1",
        "mean1",
        "sigma1",
        "amplitude2",
        "mean2",
        "sigma2",
    )


def test_newer_model_end_to_end_fits():
    rng = np.random.default_rng(20260608)
    # sinc:  amplitude=2.0, center=0.0, width=1.5
    x = np.linspace(-5.0, 5.0, 120)
    y = 2.0 * np.sinc(x / 1.5) + rng.normal(0, 0.02, size=x.size)
    res = fit(x, y, model="sinc", p0={"amplitude": 1.5, "center": 0.0, "width": 1.0})
    assert abs(res.params["amplitude"] - 2.0) < 0.3
    assert abs(res.params["center"]) < 0.2

    # exponential_rise:  amplitude=4.0, tau=2.0
    xr = np.linspace(0.0, 10.0, 80)
    yr = 4.0 * (1.0 - np.exp(-xr / 2.0)) + rng.normal(0, 0.03, size=xr.size)
    res = fit(xr, yr, model="exponential_rise", p0={"amplitude": 3.0, "tau": 1.5})
    assert abs(res.params["amplitude"] - 4.0) < 0.4
    assert abs(res.params["tau"] - 2.0) < 0.5

    # double_exponential:  amplitude1=3.0, tau1=0.5, amplitude2=1.5, tau2=3.0
    xd = np.linspace(0.0, 6.0, 100)
    yd = 3.0 * np.exp(-xd / 0.5) + 1.5 * np.exp(-xd / 3.0) + rng.normal(0, 0.02, size=xd.size)
    res = fit(
        xd,
        yd,
        model="double_exponential",
        p0={"amplitude1": 2.0, "tau1": 0.4, "amplitude2": 1.0, "tau2": 2.0},
    )
    assert abs(res.params["amplitude1"] - 3.0) < 0.4
    assert abs(res.params["tau1"] - 0.5) < 0.15

    # moffat:  amplitude=5.0, x0=1.0, alpha=1.5, beta=2.0
    xm = np.linspace(-4.0, 6.0, 100)
    ym = 5.0 * (1.0 + ((xm - 1.0) / 1.5) ** 2) ** (-2.0) + rng.normal(0, 0.03, size=xm.size)
    res = fit(xm, ym, model="moffat", p0={"amplitude": 4.0, "x0": 0.5, "alpha": 1.0, "beta": 1.5})
    assert abs(res.params["amplitude"] - 5.0) < 0.5
    assert abs(res.params["x0"] - 1.0) < 0.15

    # gaussian_baseline:  amplitude=3.0, mean=0.0, sigma=1.2, m=0.5, b=1.0
    xg = np.linspace(-4.0, 4.0, 100)
    yg = 3.0 * np.exp(-0.5 * (xg / 1.2) ** 2) + 0.5 * xg + 1.0 + rng.normal(0, 0.03, size=xg.size)
    res = fit(
        xg,
        yg,
        model="gaussian_baseline",
        p0={"amplitude": 2.0, "mean": 0.0, "sigma": 1.0, "m": 0.0, "b": 0.0},
    )
    assert abs(res.params["amplitude"] - 3.0) < 0.4


def test_new_models_param_names_and_shapes():
    """Verify param_names and basic evaluation for all 12 new models."""
    x = np.array([0.0, 1.0, 2.0])
    cases = [
        ("voigt", ("amplitude", "center", "sigma", "gamma"), [1.0, 0.0, 1.0, 1.0]),
        ("skew_normal", ("amplitude", "location", "scale", "alpha"), [1.0, 0.0, 1.0, 0.0]),
        ("gaussian_fwhm", ("amplitude", "center", "fwhm"), [1.0, 0.0, 2.355]),
        ("lorentzian_fwhm", ("amplitude", "center", "fwhm"), [1.0, 0.0, 2.0]),
        ("exgaussian", ("amplitude", "mu", "sigma", "tau"), [1.0, 0.0, 1.0, 2.0]),
        ("stretched_exponential", ("amplitude", "tau", "beta", "offset"), [1.0, 1.0, 1.0, 0.0]),
        ("tanh", ("amplitude", "center", "width", "offset"), [1.0, 0.0, 1.0, 0.0]),
        ("arctan", ("amplitude", "center", "width", "offset"), [1.0, 0.0, 1.0, 0.0]),
        ("beat", ("amplitude", "frequency1", "frequency2", "phase", "offset"), [1.0, 0.5, 1.0, 0.0, 0.0]),
        ("rational", ("amplitude", "x0"), [1.0, -1.0]),
        ("quartic", ("a", "b", "c", "d", "e"), [1.0, 0.0, 0.0, 0.0, 0.0]),
        ("quintic", ("a", "b", "c", "d", "e", "f"), [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
    ]
    for name, expected_params, args in cases:
        assert m.model_param_names(name) == expected_params, f"{name} param_names"
        fn = m.MODEL_REGISTRY[name]
        y = fn(x, *args)
        assert y.shape == x.shape, f"{name} shape"
        assert np.all(np.isfinite(y)), f"{name} non-finite"


def test_new_models_end_to_end_fits():
    """End-to-end fits for all 12 new models."""
    rng = np.random.default_rng(20260609)

    # voigt:  amplitude=2.0, center=0.0, sigma=1.0, gamma=0.5
    x = np.linspace(-8.0, 8.0, 120)
    y = m.voigt(x, 2.0, 0.0, 1.0, 0.5) + rng.normal(0, 0.01, size=x.size)
    res = fit(x, y, model="voigt", p0={"amplitude": 1.5, "center": 0.0, "sigma": 0.8, "gamma": 0.3})
    assert abs(res.params["amplitude"] - 2.0) < 0.5
    assert abs(res.params["center"]) < 0.15

    # skew_normal:  amplitude=3.0, location=0.0, scale=1.2, alpha=3.0
    y = m.skew_normal(x, 3.0, 0.0, 1.2, 3.0) + rng.normal(0, 0.02, size=x.size)
    res = fit(x, y, model="skew_normal", p0={"amplitude": 2.0, "location": 0.0, "scale": 1.0, "alpha": 2.0})
    assert abs(res.params["amplitude"] - 3.0) < 0.5

    # gaussian_fwhm:  amplitude=4.0, center=0.5, fwhm=3.0
    y = m.gaussian_fwhm(x, 4.0, 0.5, 3.0) + rng.normal(0, 0.02, size=x.size)
    res = fit(x, y, model="gaussian_fwhm", p0={"amplitude": 3.0, "center": 0.0, "fwhm": 2.0})
    assert abs(res.params["amplitude"] - 4.0) < 0.5
    assert abs(res.params["center"] - 0.5) < 0.15
    assert abs(res.params["fwhm"] - 3.0) < 0.4

    # lorentzian_fwhm:  amplitude=3.0, center=-0.5, fwhm=2.0
    y = m.lorentzian_fwhm(x, 3.0, -0.5, 2.0) + rng.normal(0, 0.02, size=x.size)
    res = fit(x, y, model="lorentzian_fwhm", p0={"amplitude": 2.0, "center": 0.0, "fwhm": 1.5})
    assert abs(res.params["amplitude"] - 3.0) < 0.5
    assert abs(res.params["center"] + 0.5) < 0.15

    # exgaussian:  amplitude=2.5, mu=0.0, sigma=0.8, tau=1.5
    x_tail = np.linspace(-4.0, 8.0, 150)
    y = m.exgaussian(x_tail, 2.5, 0.0, 0.8, 1.5) + rng.normal(0, 0.01, size=x_tail.size)
    res = fit(x_tail, y, model="exgaussian", p0={"amplitude": 2.0, "mu": 0.0, "sigma": 0.6, "tau": 1.0})
    assert abs(res.params["amplitude"] - 2.5) < 0.5

    # stretched_exponential:  amplitude=4.0, tau=2.0, beta=0.7
    xd = np.linspace(0.2, 6.0, 80)
    y = m.stretched_exponential(xd, 4.0, 2.0, 0.7) + rng.normal(0, 0.03, size=xd.size)
    res = fit(xd, y, model="stretched_exponential", p0={"amplitude": 3.0, "tau": 1.5, "beta": 0.8})
    assert abs(res.params["amplitude"] - 4.0) < 0.6

    # tanh:  amplitude=2.0, center=1.0, width=0.5, offset=1.0
    y = m.tanh(x, 2.0, 1.0, 0.5, 1.0) + rng.normal(0, 0.02, size=x.size)
    res = fit(x, y, model="tanh", p0={"amplitude": 1.5, "center": 0.5, "width": 0.4, "offset": 0.5})
    assert abs(res.params["center"] - 1.0) < 0.15

    # arctan:  amplitude=1.5, center=0.0, width=1.0, offset=0.0
    y = m.arctan(x, 1.5, 0.0, 1.0, 0.0) + rng.normal(0, 0.02, size=x.size)
    res = fit(x, y, model="arctan", p0={"amplitude": 1.0, "center": 0.0, "width": 0.8, "offset": 0.0})
    assert abs(res.params["amplitude"] - 1.5) < 0.4

    # beat:  amplitude=2.0, f1=0.5, f2=0.7, phase=0.2
    xb = np.linspace(0.0, 20.0, 300)
    y = m.beat(xb, 2.0, 0.5, 0.7, 0.2) + rng.normal(0, 0.03, size=xb.size)
    res = fit(xb, y, model="beat", p0={"amplitude": 2.0, "frequency1": 0.5, "frequency2": 0.7, "phase": 0.0})
    assert abs(res.params["amplitude"] - 2.0) < 0.4
    assert abs(res.params["frequency1"] - 0.5) < 0.05

    # rational:  amplitude=2.0, x0=3.0
    xr = np.linspace(3.5, 8.0, 80)
    y = m.rational(xr, 2.0, 3.0) + rng.normal(0, 0.02, size=xr.size)
    res = fit(xr, y, model="rational", p0={"amplitude": 1.5, "x0": 2.5})
    assert abs(res.params["amplitude"] - 2.0) < 0.3

    # quartic:  y = x^4 - 2x^2 + 0.5
    xp = np.linspace(-2.0, 2.0, 50)
    y = m.quartic(xp, 1.0, 0.0, -2.0, 0.0, 0.5) + rng.normal(0, 0.05, size=xp.size)
    res = fit(xp, y, model="quartic", p0={"a": 0.5, "b": 0.0, "c": -1.0, "d": 0.0, "e": 0.0})
    assert abs(res.params["a"] - 1.0) < 0.2
    assert abs(res.params["c"] + 2.0) < 0.5

    # quintic:  y = 0.5x^5 - 2x^3 + x
    y = m.quintic(xp, 0.5, 0.0, -2.0, 0.0, 1.0, 0.0) + rng.normal(0, 0.05, size=xp.size)
    res = fit(xp, y, model="quintic", p0={"a": 0.0, "b": 0.0, "c": -1.0, "d": 0.0, "e": 0.5, "f": 0.0})
    assert abs(res.params["a"] - 0.5) < 0.2


def test_new_models_auto_initial_guesses():
    """Fits without explicit p0 to exercise automatic _initial_guess branches."""
    rng = np.random.default_rng(20260610)

    # gaussian_fwhm with automatic guess
    x = np.linspace(-5.0, 5.0, 80)
    y = m.gaussian_fwhm(x, 3.0, 0.5, 4.0) + rng.normal(0, 0.03, size=x.size)
    res = fit(x, y, model="gaussian_fwhm")
    assert abs(res.params["amplitude"] - 3.0) < 0.5
    assert abs(res.params["center"] - 0.5) < 0.2

    # tanh with automatic guess
    y = m.tanh(x, 2.0, 0.5, 1.0, 0.5) + rng.normal(0, 0.03, size=x.size)
    res = fit(x, y, model="tanh")
    assert abs(res.params["center"] - 0.5) < 0.3


def test_get_model_edge_cases():
    import pytest

    fn, name = m.get_model(None)
    assert fn is m.linear
    assert name == "linear"
    with pytest.raises(KeyError):
        m.get_model("nonexistent_model")
    with pytest.raises(TypeError):
        m.get_model(42)
    fn2, name2 = m.get_model(m.quadratic)
    assert name2 == "quadratic"
    assert m.model_param_names(m.constant) == ("level",)


def test_types_utils_and_dataset_helpers(tmp_path: Path):
    asym = AsymmetricError(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    assert np.allclose(asym.effective, np.sqrt(np.array([5.0, 10.0])))
    assert np.allclose(effective_sigma(sigma=asym), asym.effective)
    assert np.allclose(effective_sigma(sigma_low=[1, 2], sigma_high=[3, 4]), asym.effective)

    series = DataSeries(x=[0, 1], y=[2, 3], sigma=[0.1, 0.2], label="orig")
    assert np.allclose(series.error(), [0.1, 0.2])
    assert series.with_label("new").label == "new"

    asym_series = DataSeries(x=[0, 1], y=[2, 3], sigma=asym)
    assert np.allclose(asym_series.effective_sigma, asym.effective)
    cov_series = DataSeries(x=[0, 1], y=[2, 3], sigma_cov=np.eye(2))
    assert cov_series.effective_sigma is None

    dataset = Dataset([series])
    dataset.append(asym_series)
    dataset.extend([cov_series])
    assert len(dataset) == 3
    assert dataset[1].label == ""
    assert list(dataset)[0].label == "orig"

    result = FitResult(
        reduced_chi2=1.23,
        params={"slope": 2.0, "intercept": 1.0},
        param_names=("slope", "intercept"),
        x=np.array([0.0, 1.0]),
        y=np.array([1.0, 3.0]),
        model=m.linear,
    )
    assert result["slope"] == 2.0
    assert tuple(result) == (2.0, 1.0)
    assert np.allclose(result.predict([2.0]), [5.0])
    assert np.allclose(result.residuals, [0.0, 0.0])
    assert dict(result.items()) == {"slope": 2.0, "intercept": 1.0}
    assert list(result.keys()) == ["slope", "intercept"]
    assert list(result.values()) == [2.0, 1.0]

    csv = tmp_path / "data.csv"
    csv.write_text("x,y,sigma,sigma_low,sigma_high\n0,1,0.1,0.2,0.3\n1,2,0.2,0.3,0.4\n")
    loaded = _load_csv(csv)
    assert len(loaded.x) == 2 and len(loaded.y) == 2
    series2 = _coerce_series_input(csv)
    assert np.allclose(series2.x, loaded.x)
    assert np.allclose(series2.y, loaded.y)
    assert np.allclose(series2.sigma, loaded.sigma)
    assert np.allclose(series2.sigma_low, loaded.sigma_low)
    assert np.allclose(series2.sigma_high, loaded.sigma_high)

    yhat, err = propagate_errors(
        lambda decay: decay / np.log(2.0),
        decay=1.5,
        decay_error=0.3,
        jacobian=lambda decay: 1.0 / np.log(2.0),
    )
    assert np.isclose(yhat, 1.5 / np.log(2.0))
    assert err > 0


def test_fitter_helpers_and_branches():
    x = np.linspace(0.0, 5.0, 40)
    y = 2.0 * x + 1.0
    series = DataSeries(x=x, y=y, label="line")

    assert np.allclose(_initial_guess("linear", "linear", x, y)[:2], [2.0, 1.0])
    assert len(_initial_guess("quadratic", "quadratic", x, y)) == 3
    assert len(_initial_guess("cubic", "cubic", x, y)) == 4
    assert len(_initial_guess("gaussian", "gaussian", x, y)) == 3
    assert len(_initial_guess("lorentzian", "lorentzian", x, y)) == 3
    assert len(_initial_guess("exponential", "exponential", x, y)) == 2
    assert len(_initial_guess("power_law", "power_law", x[1:], y[1:] + 1)) == 2
    assert len(_initial_guess("logistic", "logistic", x, y)) == 4
    assert len(_initial_guess("sine", "sine", x, y)) == 4
    assert len(_initial_guess("cosine", "cosine", x, y)) == 4
    assert len(_initial_guess("damped_oscillator", "damped_oscillator", x, y)) == 4
    assert len(_initial_guess("damped_sine", "damped_sine", x, y)) == 5

    # New-model initial guesses (length checks)
    assert len(_initial_guess("voigt", "voigt", x, y)) == 4
    assert len(_initial_guess("skew_normal", "skew_normal", x, y)) == 4
    assert len(_initial_guess("gaussian_fwhm", "gaussian_fwhm", x, y)) == 3
    assert len(_initial_guess("lorentzian_fwhm", "lorentzian_fwhm", x, y)) == 3
    assert len(_initial_guess("exgaussian", "exgaussian", x, y)) == 4
    assert len(_initial_guess("stretched_exponential", "stretched_exponential", x, y)) == 4
    assert len(_initial_guess("tanh", "tanh", x, y)) == 4
    assert len(_initial_guess("arctan", "arctan", x, y)) == 4
    assert len(_initial_guess("beat", "beat", x, y)) == 5
    assert len(_initial_guess("rational", "rational", x, y)) == 2
    assert len(_initial_guess("quartic", "quartic", x, y)) == 5
    assert len(_initial_guess("quintic", "quintic", x, y)) == 6

    lower, upper = _coerce_bounds({"slope": (0.0, 5.0), "intercept": (None, None)}, ("slope", "intercept"))
    assert np.allclose(lower, [0.0, -np.inf], equal_nan=True)
    assert np.allclose(upper, [5.0, np.inf], equal_nan=True)
    sigma, cov = _normalize_sigma(series, sigma=None, weights=np.ones_like(x))
    assert np.allclose(sigma, 1.0)
    assert cov is None

    res = _fit_single(series, model="linear", sigma=series.sigma)
    assert abs(res.params["slope"] - 2.0) < 1e-6
    assert abs(res.params["intercept"] - 1.0) < 1e-6

    res2 = fit(x, y, model="linear", sigma=0.1)
    assert abs(res2.params["slope"] - 2.0) < 1e-6

    cov = np.diag(np.full_like(x, 0.1**2, dtype=float))
    res3 = fit(x, y, model="linear", sigma_cov=cov)
    assert abs(res3.params["slope"] - 2.0) < 1e-6

    def custom(x, amplitude, damping, frequency, phase):
        return amplitude * np.exp(-damping * x) * np.cos(2 * np.pi * frequency * x + phase)

    y2 = custom(x, 2.0, 0.3, 1.0, 0.2)
    res4 = fit(
        x,
        y2,
        model=custom,
        p0={"amplitude": 1.0, "damping": 0.2, "frequency": 1.0, "phase": 0.0},
        bounds={"amplitude": (0.0, None)},
    )
    assert abs(res4.params["frequency"] - 1.0) < 0.2

    results = fit_multi(Dataset([series, DataSeries(x=x, y=3.0 * x + 2.0, sigma=0.1)]), model="linear")
    assert len(results) == 2
    assert abs(results[1].params["slope"] - 3.0) < 1e-6


def test_private_branch_helpers_and_error_paths():
    x = np.linspace(0.0, 5.0, 10)
    y = 2.0 * x + 1.0
    series = DataSeries(x=x, y=y)
    assert _coerce_series_input(series) is series
    with pytest.raises(TypeError):
        _coerce_series_input(1)

    assert np.allclose(_initial_guess("linear", "linear", np.array([0.0]), np.array([2.0])), [1.0, 2.0])
    assert np.allclose(_initial_guess("linear", "linear", x, y, p0=2.0), [2.0])
    assert np.allclose(_initial_guess("linear", "linear", x, y, p0=[1.0, 2.0]), [1.0, 2.0])
    assert np.allclose(_coerce_bounds(None, ("a", "b"))[0], [-np.inf, -np.inf])
    with pytest.raises(TypeError):
        _coerce_bounds(5, ("a",))
    with pytest.raises(ValueError):
        _normalize_sigma(series, sigma=0.1, weights=np.ones_like(x))

    assert effective_sigma(sigma_cov=np.eye(2)) is None
    assert _as_results(None) == []
    with pytest.raises(TypeError):
        _as_results("bad")

    result = fit(x, y, model="linear")
    fig, ax = plt.subplots()
    _add_result_axes(ax, result, show_residuals=True)
    fig2, ax2 = plt.subplots()
    existing = Plotter()
    existing.figure = fig2
    existing.axes = ax2
    plot_result(result, plotter=existing)

    assert len(fit_multi([series, DataSeries(x=x, y=3.0 * x + 2.0)], model="linear")) == 2
    with pytest.raises(TypeError):
        fit(x, y, model="linear", nonsense=1)
