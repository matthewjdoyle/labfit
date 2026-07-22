import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

import labfit
from labfit import DataSeries, Dataset, FitResult, Fitter, Plotter, fit, plot_fit, plot_residuals


def test_public_api_exports():
    expected = {
        "fit",
        "fit_curve",
        "fit_multi",
        "plot_fit",
        "plot_multi_fit",
        "plot_residuals",
        "DataSeries",
        "Dataset",
        "Series",
        "Fitter",
        "FitResult",
        "Plotter",
        "AsymmetricError",
    }
    for name in expected:
        assert hasattr(labfit, name), name


def test_linear_fit_with_weights_and_reduced_chi2():
    rng = np.random.default_rng(123)
    x = np.linspace(-2.0, 3.0, 50)
    y = 1.75 * x - 0.5 + rng.normal(0, 0.05, size=x.size)
    sigma = np.full_like(x, 0.05)

    result = fit(x, y, model="linear", sigma=sigma)

    assert isinstance(result, FitResult)
    assert math.isfinite(result.reduced_chi2)
    assert abs(result.params["slope"] - 1.75) < 0.08
    assert abs(result.params["intercept"] + 0.5) < 0.1
    assert result.reduced_chi2 < 3.0


def test_fit_gaussian():
    rng = np.random.default_rng(42)
    x = np.linspace(-5.0, 5.0, 300)
    y = 4.0 * np.exp(-0.5 * ((x - 0.8) / 1.2) ** 2) + rng.normal(0, 0.08, size=x.size)

    result = fit(x, y, model="gaussian")

    assert abs(result.params["mean"] - 0.8) < 0.15
    assert abs(result.params["sigma"] - 1.2) < 0.2
    assert abs(result.params["amplitude"] - 4.0) < 0.5


def test_csv_input_plot_and_residuals(tmp_path: Path):
    csv = tmp_path / "data.csv"
    csv.write_text("x,y,sigma\n0,1,0.1\n1,3,0.1\n2,5,0.1\n3,7,0.1\n")

    result = fit(csv, model="linear")
    fit_plot = plot_fit(result)
    res_plot = plot_residuals(result)

    fit_png = tmp_path / "fit.png"
    res_png = tmp_path / "residuals.png"
    fit_plot.save(fit_png)
    res_plot.save(res_png)

    assert fit_png.exists() and fit_png.stat().st_size > 0
    assert res_png.exists() and res_png.stat().st_size > 0
    assert abs(result.params["slope"] - 2.0) < 0.1


def test_dataseries_dataset_and_multi_fit():
    x = np.linspace(0, 1, 10)
    s1 = DataSeries(x=x, y=2 * x + 1, sigma=0.1, label="a")
    s2 = DataSeries(x=x, y=3 * x + 2, sigma=0.1, label="b")
    dataset = Dataset([s1, s2])

    assert len(dataset) == 2
    assert dataset[0].label == "a"

    results = Fitter(model="linear").fit_multi(dataset)
    assert len(results) == 2
    assert abs(results[0].params["slope"] - 2.0) < 0.1
    assert abs(results[1].params["slope"] - 3.0) < 0.1


def test_custom_callable_model_and_plotter():
    def damped_oscillator(t, amplitude, damping, frequency, phase):
        return amplitude * np.exp(-damping * t) * np.cos(2 * np.pi * frequency * t + phase)

    rng = np.random.default_rng(7)
    x = np.linspace(0.0, 4.0, 200)
    y = damped_oscillator(x, 1.5, 0.3, 1.2, 0.4) + rng.normal(0, 0.02, size=x.size)
    fitter = Fitter(
        model=damped_oscillator, p0={"amplitude": 1.0, "damping": 0.2, "frequency": 1.0, "phase": 0.0}
    )
    result = fitter.fit(x, y, sigma=0.02)

    assert abs(result.params["frequency"] - 1.2) < 0.1
    assert isinstance(plot_fit(result), Plotter)
    assert isinstance(plot_residuals(result), Plotter)
