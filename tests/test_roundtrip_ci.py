from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from labfit import (
    DataSeries,
    Dataset,
    FitResult,
    Plotter,
    fit,
    fit_curve,
    plot_fit,
    plot_multi_fit,
    plot_residuals,
)
from labfit.io import combine_series, load_csv, load_txt
from labfit.plot import _as_results, _as_series, _fit_label, plot_result
from labfit.types import AsymmetricError
from labfit.utils import propagate_errors


def test_loader_variants_round_trip_and_series_combination(tmp_path: Path):
    weighted_csv = tmp_path / "weighted.csv"
    weighted_csv.write_text(
        "x,y,y_err,sigma_low,sigma_high\n0,1,0.1,0.2,0.3\n1,4,0.2,0.4,0.6\n2,9,0.3,0.6,0.8\n"
    )
    weighted = load_csv(weighted_csv, 0, 1, y_err_col="y_err", label="weighted")
    assert weighted.label == "weighted"
    assert np.allclose(weighted.x, [0.0, 1.0, 2.0])
    assert np.allclose(weighted.y, [1.0, 4.0, 9.0])
    assert np.allclose(weighted.y_err, [0.1, 0.2, 0.3])
    assert np.allclose(weighted.sigma_low, [0.2, 0.4, 0.6])
    assert np.allclose(weighted.sigma_high, [0.3, 0.6, 0.8])

    fractional_csv = tmp_path / "fractional.csv"
    fractional_csv.write_text("x,y,frac_err\n0,10,0.1\n1,12,0.25\n")
    fractional = load_csv(fractional_csv, "x", "y")
    assert np.allclose(fractional.y_err, [1.0, 3.0])

    counts_txt = tmp_path / "counts.txt"
    counts_txt.write_text("0 4\n1 9\n2 16\n")
    counts = load_txt(counts_txt, 0, 1)
    assert np.allclose(counts.y_err, np.sqrt([4.0, 9.0, 16.0]))

    combined = combine_series(weighted, Dataset([fractional]), [counts])
    assert len(combined) == 3
    assert combined[0].label == "weighted"
    assert np.allclose(combined[1].y, fractional.y)
    assert np.allclose(combined[2].x, counts.x)


def test_error_propagation_and_validation_edges():
    value, uncertainty = propagate_errors(
        lambda x: x**2,
        jacobian=lambda x: 2.0 * x,
        x=3.0,
        x_error=0.5,
    )
    assert value == 9.0
    assert math.isclose(uncertainty, 3.0, rel_tol=0.0, abs_tol=1e-12)

    covariance = np.array([[4.0]])
    value2, uncertainty2 = propagate_errors(
        lambda x: x + 1.0,
        jacobian=lambda x: np.array([1.0]),
        covariance=covariance,
        x=2.0,
        x_error=0.5,
    )
    assert value2 == 3.0
    assert math.isclose(uncertainty2, 2.0, rel_tol=0.0, abs_tol=1e-12)

    # Without jacobian, numerical fallback is used (returns zero uncertainty
    # since covariance is zero when no _error keywords are passed)
    val_num, err_num = propagate_errors(lambda x: x**2, x=3.0)
    assert val_num == 9.0
    assert err_num == 0.0

    with pytest.raises(ValueError, match="sigma and y_err must describe the same uncertainties"):
        DataSeries(x=[0, 1], y=[1, 2], sigma=[0.1, 0.2], y_err=[0.1, 0.3])

    with pytest.raises(ValueError, match="sigma must be non-negative"):
        DataSeries(x=[0, 1], y=[1, 2], sigma=[-0.1, 0.2])

    with pytest.raises(ValueError, match="sigma_cov must be a square covariance matrix"):
        DataSeries(x=[0, 1], y=[1, 2], sigma_cov=np.eye(3))

    asym = AsymmetricError(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    series = DataSeries(x=[0, 1], y=[2, 3], sigma=asym)
    assert np.allclose(series.effective_sigma, asym.effective)


def test_reduced_chi_squared_matches_manual_formula():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([1.0, 2.4, 5.1, 6.7])
    sigma = np.full_like(x, 0.5)

    result = fit_curve("linear", x, y, sigma)
    dof = x.size - len(result.param_names)
    expected = float(np.sum(((result.y - result.predict(result.x)) / sigma) ** 2) / dof)

    assert math.isclose(result.reduced_chi2, expected, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isfinite(result.p_value) or math.isnan(result.p_value)


def test_plotting_smoke_covers_single_and_multi_series(tmp_path: Path):
    x = np.linspace(0.0, 3.0, 5)
    series_a = DataSeries(x=x, y=2.0 * x + 1.0, sigma=0.1, label="series A")
    series_b = DataSeries(x=x, y=3.0 * x + 2.0, sigma=0.2, label="series B")

    result_a = fit(x, series_a.y, model="linear", sigma=series_a.sigma, label=series_a.label)
    result_b = fit(x, series_b.y, model="linear", sigma=series_b.sigma, label=series_b.label)

    assert _fit_label(FitResult(0.0, {"slope": 1.0}, model_name=""), 0) == "fit 1"
    assert _as_results((result_a, result_b)) == [result_a, result_b]
    assert _as_series((series_a, series_b)) == [series_a, series_b]

    plotter = Plotter()
    plotter.add_series(series_a)
    plotter.add_series(series_b.x, series_b.y, sigma=series_b.sigma, label=series_b.label)
    assert len(plotter.series) == 2

    with pytest.raises(TypeError):
        plotter.add_series()
    with pytest.raises(ValueError, match="Nothing has been plotted yet"):
        Plotter().save(tmp_path / "empty.png")
    with pytest.raises(TypeError, match="result must be a FitResult"):
        plot_fit(result="bad")
    with pytest.raises(TypeError, match="plot_fit expects a single DataSeries"):
        plot_fit(result=result_a, data_series=[series_a, series_b])
    with pytest.raises(TypeError, match="plot_fit expects a DataSeries"):
        plot_fit(result=result_a, data_series="bad")
    with pytest.raises(TypeError, match="series_list is required"):
        plot_multi_fit([result_a], None)
    with pytest.raises(ValueError, match="layout must be 'grid'"):
        plot_multi_fit([result_a], [series_a], layout="stacked")
    with pytest.raises(TypeError, match="result must be a FitResult"):
        plot_result(result="bad")
    with pytest.raises(TypeError, match="data_series must be a DataSeries"):
        _as_series("bad")
    with pytest.raises(TypeError, match="result must be a FitResult"):
        _as_results("bad")

    single = plot_result(result=None, plotter=plotter, title="series only")
    assert single.figure is not None
    assert len(single.figure.axes) == 1
    assert single.axes is not None

    fig, ax = plt.subplots()
    axr = ax.twinx()
    existing = Plotter(figure=fig, axes=(ax, axr))
    overlaid = plot_result(
        result=[result_a, result_b], plotter=existing, show_residuals=True, title="overlay"
    )
    assert overlaid.figure is fig
    assert len(overlaid.figure.axes) >= 2

    fitted = plot_fit(result=result_a, data_series=series_a, show_residuals=True)
    multi = plot_fit(result=[result_a, result_b], show_residuals=False)
    residuals = plot_residuals(result=result_a, data_series=series_a)
    grid = plot_multi_fit([result_a, result_b], [series_a, series_b])

    for name, plot in {
        "fitted": fitted,
        "multi": multi,
        "residuals": residuals,
        "grid": grid,
    }.items():
        path = tmp_path / f"{name}.png"
        saved = plot.save(path)
        assert Path(saved) == path
        assert path.exists() and path.stat().st_size > 0

    assert len(grid.figure.axes) == 4
    ax1, ax2 = grid.axes[0]
    ax1r, ax2r = grid.axes[1]
    assert ax1r.get_shared_x_axes().joined(ax1, ax1r)
    assert ax2r.get_shared_x_axes().joined(ax2, ax2r)
