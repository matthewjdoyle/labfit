"""Static consumer contract, checked by tests/test_typing_contract.py."""

from collections.abc import ItemsView, KeysView, ValuesView
from pathlib import Path

import numpy as np
from typing_extensions import assert_type

from labfit import (
    AsymmetricError,
    DataSeries,
    Dataset,
    FitResult,
    Fitter,
    Plotter,
    fit,
    fit_curve,
    fit_multi,
    plot_fit,
    plot_multi_fit,
    plot_residuals,
)
from labfit.api import fit as api_fit
from labfit.io import combine_series, load_csv, load_txt
from labfit.models import gaussian, get_model, linear, model_param_names
from labfit.plot import plot_result, use_publication_style
from labfit.utils import effective_sigma, propagate_errors


def custom_model(x: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    return slope * x + intercept


def derived_value(rate: float) -> float:
    return 1.0 / rate


x = [0.0, 1.0, 2.0]
y = [1.0, 3.0, 5.0]
errors = AsymmetricError(0.1, 0.2)
series = DataSeries(x, y, sigma=errors)
assert_type(series.x, np.ndarray)
assert_type(errors.lower, np.ndarray)
assert_type(series.effective_sigma, np.ndarray | None)
assert_type(series.with_label("data"), DataSeries)
assert_type(load_csv(Path("data.csv"), error_mode="unweighted"), DataSeries)
assert_type(load_txt("data.txt", x_col=0, y_col=1), DataSeries)
assert_type(combine_series(series, [series], Dataset([series])), Dataset)
result = fit(x, y, model=custom_model, sigma=0.1, p0={"slope": 2.0}, absolute_sigma=True)
assert_type(result, FitResult)
assert_type(api_fit(series), FitResult)
assert_type(fit(Path("data.csv"), model="linear"), FitResult)
assert_type(fit_curve(linear, x, y, errors, bounds={"slope": (0.0, None)}), FitResult)
assert_type(fit_multi(Dataset([series])), list[FitResult])
assert_type(Fitter("linear").fit(series), FitResult)
assert_type(Fitter("linear")(x, y), FitResult)
assert_type(Fitter("linear").fit_multi([series]), list[FitResult])
assert_type(result.predict(x), np.ndarray)
assert_type(result["slope"], float)
assert_type(result.items(), ItemsView[str, float])
assert_type(result.keys(), KeysView[str])
assert_type(result.values(), ValuesView[float])
assert_type(linear(x, slope=2.0, intercept=1.0), np.ndarray)
assert_type(linear(x=x, slope=2.0, intercept=1.0), np.ndarray)
assert_type(gaussian(x, amplitude=1.0, mean=0.0, sigma=1.0), np.ndarray)
assert_type(model_param_names("linear"), tuple[str, ...])
model, name = get_model("linear")
assert_type(name, str)
assert_type(plot_fit(result, show_ci=True, prediction=True, prediction_sigma=lambda grid: 0.2), Plotter)
assert_type(plot_multi_fit([result], [series]), Plotter)
assert_type(plot_residuals(result), Plotter)
assert_type(plot_result([result]), Plotter)
assert_type(use_publication_style(), None)
plotter = Plotter().add_series(x, y, sigma=0.1).add_series(series)
assert_type(plotter.plot(result), Plotter)
assert_type(plotter(result), Plotter)
assert_type(plotter.save(Path("result.png")), Path)
assert_type(effective_sigma(sigma=errors), np.ndarray | None)
assert_type(propagate_errors(derived_value, covariance=[[0.01]], rate=2.0), tuple[float, float])
