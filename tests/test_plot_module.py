from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import labfit
from labfit import DataSeries, FitResult


def _linear_result(x, y, series, slope=2.0, intercept=1.0):
    return FitResult(
        reduced_chi2=0.0,
        params={"slope": slope, "intercept": intercept},
        param_names=("slope", "intercept"),
        x=x,
        y=y,
        sigma=np.full_like(x, 0.1, dtype=float),
        model=lambda t, m, b: m * t + b,
        series=series,
        model_name="linear",
    )


def test_plot_fit_supports_explicit_series_and_residual_axis(tmp_path: Path):
    x = np.linspace(0.0, 3.0, 5)
    y = 2.0 * x + 1.0
    series = DataSeries(x=x, y=y, sigma=0.1, label="sample A")
    result = _linear_result(x, y, series)

    assert hasattr(labfit, "plot_fit")
    assert hasattr(labfit, "plot_multi_fit")

    plotter = labfit.plot_fit(result, data_series=series, show_residuals=True)

    assert plotter.figure is not None
    assert plotter.axes is not None
    assert len(plotter.figure.axes) == 2
    ax, axr = plotter.axes
    assert axr.get_shared_x_axes().joined(ax, axr)

    legend_labels = [text.get_text() for text in ax.get_legend().texts]
    assert "sample A" in legend_labels
    assert any(label.startswith("fit:") for label in legend_labels)

    fit_png = tmp_path / "labfit-plot-fit.png"
    fit_pdf = tmp_path / "labfit-plot-fit.pdf"
    assert str(plotter.save(fit_png)).endswith(".png")
    assert str(plotter.save(fit_pdf)).endswith(".pdf")
    assert fit_png.exists() and fit_png.stat().st_size > 0
    assert fit_pdf.exists() and fit_pdf.stat().st_size > 0


def test_plot_multi_fit_creates_comparison_grid(tmp_path: Path):
    x = np.linspace(0.0, 3.0, 5)
    s1 = DataSeries(x=x, y=2.0 * x + 1.0, sigma=0.1, label="series 1")
    s2 = DataSeries(x=x, y=3.0 * x + 2.0, sigma=0.1, label="series 2")
    r1 = _linear_result(x, s1.y, s1, slope=2.0, intercept=1.0)
    r2 = _linear_result(x, s2.y, s2, slope=3.0, intercept=2.0)

    plotter = labfit.plot_multi_fit([r1, r2], [s1, s2], layout="grid")

    assert plotter.figure is not None
    assert len(plotter.figure.axes) == 4
    ax1, ax2 = plotter.axes[0]
    ax1r, ax2r = plotter.axes[1]
    assert ax1r.get_shared_x_axes().joined(ax1, ax1r)
    assert ax2r.get_shared_x_axes().joined(ax2, ax2r)

    labels1 = [text.get_text() for text in ax1.get_legend().texts]
    labels2 = [text.get_text() for text in ax2.get_legend().texts]
    assert "series 1" in labels1
    assert "series 2" in labels2

    multi_png = tmp_path / "labfit-plot-multi.png"
    multi_pdf = tmp_path / "labfit-plot-multi.pdf"
    assert str(plotter.save(multi_png)).endswith(".png")
    assert str(plotter.save(multi_pdf)).endswith(".pdf")
    assert multi_png.exists() and multi_png.stat().st_size > 0
    assert multi_pdf.exists() and multi_pdf.stat().st_size > 0
