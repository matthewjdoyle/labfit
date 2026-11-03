from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

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


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_overlay_groups_series_by_identity():
    x = np.arange(5.0)
    series = DataSeries(x=x, y=x, label="sample")
    other = DataSeries(x=x, y=x, label="sample")
    results = [_linear_result(x, x, series, slope=slope) for slope in range(1, 5)]
    results.extend([_linear_result(x, x, other), _linear_result(x, x, None)])

    ax, axr = labfit.plot_fit(results).axes

    markers = [line for line in ax.lines if line.get_label() == "sample"]
    fits = [line for line in ax.lines if line.get_label().startswith("fit:")]
    assert len(markers) == 2
    assert all(marker.get_markerfacecolor() == marker.get_color() for marker in markers)
    assert [line.get_linestyle() for line in fits[:4]] == ["-", "--", "-.", ":"]
    assert [line.get_color() for line in fits[:4]] == [markers[0].get_color()] * 4
    assert fits[4].get_color() == markers[1].get_color() != markers[0].get_color()
    assert fits[5].get_color() not in {marker.get_color() for marker in markers}
    zero, *residuals = axr.lines
    assert len(residuals) == len(fits)
    assert list(zero.get_ydata()) == [0.0, 0.0]
    assert [line.get_color() for line in residuals] == [line.get_color() for line in fits]
    assert zero.get_zorder() < residuals[0].get_zorder()


def test_fit_artist_layers_and_uncertainty_colour():
    x = np.arange(5.0)
    result = labfit.fit(x, 2 * x + [1, 1.1, 0.9, 1.2, 1], sigma=0.1, absolute_sigma=True)

    ax, axr = labfit.plot_fit(result, show_ci=True).axes

    marker, caps, (bars,) = ax.containers[0].lines
    curve = next(line for line in ax.lines if line.get_label() == "fit: linear")
    band = next(c for c in ax.collections if c.get_label() == "68% confidence interval for the mean")
    assert band.get_zorder() < bars.get_zorder() < curve.get_zorder() < marker.get_zorder()
    assert marker.get_markerfacecolor() == marker.get_color()
    assert curve.get_linewidth() == 1.7
    assert bars.get_linewidths()[0] == 0.8
    assert caps[0].get_markersize() == 4.0
    assert marker.get_color() == curve.get_color() == axr.lines[-1].get_color()
    assert tuple(band.get_facecolor()[0, :3]) == to_rgb(curve.get_color())


@pytest.mark.parametrize(
    ("errors", "expected"),
    [
        ({"sigma": 0.5}, 4.0),
        ({"sigma_low": 0.001, "sigma_high": 0.5}, 2.5),
        ({"sigma_cov": np.eye(3) * 1e-6}, 2.5),
    ],
)
def test_marker_size_uses_rendered_error_arms(errors, expected):
    x = np.arange(3.0)
    series = DataSeries(x=x, y=x, **errors)
    fig, ax = plt.subplots()
    labfit.plot_fit(_linear_result(x, x, series), ax=ax, show_residuals=False)
    fig.canvas.draw()

    marker, _, (bars,) = ax.containers[0].lines
    assert marker.get_markersize() == expected
    legend_dots = ax.get_legend().findobj(lambda artist: getattr(artist, "get_marker", lambda: None)() == "o")
    assert [dot.get_markersize() for dot in legend_dots] == [4.0]

    bars.set_segments(np.stack([marker.get_xydata()] * 2, axis=1))
    fig.canvas.draw()
    assert marker.get_markersize() == 4.0


def test_marker_size_uses_visible_spacing_across_series():
    x = np.array([0.0, 0.0, 1.0, 2.0])
    shifted = x + 0.001
    series = DataSeries(x=x, y=x, label="first")
    other = DataSeries(x=shifted, y=x, label="second")
    plot = labfit.plot_fit([_linear_result(x, x, series), _linear_result(shifted, x, other)])
    plot.figure.set_size_inches(7, 10)
    ax, axr = plot.axes
    markers = [line for line in ax.lines if line.get_label() in {"first", "second"}]
    markers[0].set_data(np.append(x, np.nan), np.append(x, 0.0))
    plot.figure.canvas.draw()
    assert [marker.get_markersize() for marker in markers] == [2.5, 2.5]
    assert all(line.get_markersize() == 2.5 for line in axr.lines[1:])

    ax.set_xlim(-0.0005, 0.0015)
    plot.figure.canvas.draw()
    assert [marker.get_markersize() for marker in markers] == [4.0, 4.0]
    assert all(line.get_markersize() == 3.5 for line in axr.lines[1:])


def test_marker_size_tracks_panel_size_and_is_dpi_independent():
    x = np.arange(3.0)
    series = DataSeries(x=x, y=x, label="sample")
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))
    labfit.plot_fit(_linear_result(x, x, series), ax=ax, show_residuals=False)
    marker = next(line for line in ax.lines if line.get_label() == "sample")
    fig.canvas.draw()
    assert marker.get_markersize() == 4.0

    fig.set_size_inches(1.5, 3)
    fig.canvas.draw()
    size = marker.get_markersize()
    assert 2.5 <= size < 4.0
    fig.set_dpi(fig.dpi * 2)
    fig.canvas.draw()
    assert marker.get_markersize() == pytest.approx(size)


@pytest.mark.parametrize("sharey", [False, True])
def test_comparison_titles_labels_and_limits(sharey):
    x = np.arange(5.0)
    low = DataSeries(x=x, y=x + 1, label="Low")
    high = DataSeries(x=x, y=x + 100)
    results = [
        _linear_result(x, low.y, low, slope=1),
        _linear_result(x, high.y, high, slope=1, intercept=90),
    ]

    plot = labfit.plot_multi_fit(
        results, [low, high], title="Comparison", xlabel="Time", ylabel="Signal", sharey=sharey
    )

    (ax1, ax2), (res1, res2) = plot.axes
    assert [text.get_text() for text in plot.figure.texts] == ["Comparison", "Time"]
    assert [ax1.get_title(), ax2.get_title()] == ["Low", "series 2"]
    assert [ax1.get_ylabel(), ax2.get_ylabel(), res1.get_ylabel(), res2.get_ylabel()] == [
        "Signal",
        "",
        "residuals",
        "",
    ]
    assert all(not ax.get_xlabel() for ax in plot.axes.flat)
    assert (ax1.get_ylim() == ax2.get_ylim()) == sharey
    assert (res1.get_ylim() == res2.get_ylim()) == sharey
    if sharey:
        assert ax1.get_ylim()[0] < min(low.y) < max(high.y) < ax1.get_ylim()[1]
        assert res1.get_ylim()[0] == 0 < 10 < res1.get_ylim()[1]
        assert not ax1.get_shared_y_axes().joined(ax1, res1)


def test_single_comparison_panel():
    x = np.arange(5.0)
    series = DataSeries(x=x, y=x, label="Trial")
    plot = labfit.plot_multi_fit([_linear_result(x, x, series)], [series], sharey=True)
    assert plot.axes.shape == (2, 1)
    assert plot.axes[0, 0].get_title() == "Trial"


def test_scoped_styles_and_save_defaults(tmp_path, monkeypatch):
    result = labfit.fit(np.arange(5.0), np.arange(5.0), sigma=0.1)
    with plt.rc_context({"axes.spines.top": True, "axes.spines.right": False, "axes.labelsize": 16}):
        before = plt.rcParams.copy()
        default = labfit.plot_fit(result, show_residuals=False)
        publication = labfit.plot_residuals(result, style="publication")
        themed = labfit.Plotter().plot(result, style=None)
        assert plt.rcParams == before
        assert not default.axes.spines["top"].get_visible()
        assert publication.axes[0].spines["right"].get_visible()
        assert themed.axes.spines["top"].get_visible()
        assert not themed.axes.spines["right"].get_visible()
        assert themed.axes.xaxis.label.get_fontsize() == 16

    for plotter, expected in [
        (default, {"dpi": 300, "bbox_inches": "tight"}),
        (publication, {"dpi": 300, "bbox_inches": "tight"}),
        (themed, {}),
        (labfit.Plotter().plot(result), {"dpi": 300, "bbox_inches": "tight"}),
    ]:
        calls = []
        monkeypatch.setattr(
            plotter.figure, "savefig", lambda path, calls=calls, **kwargs: calls.append(kwargs)
        )
        plotter.save(tmp_path / "default.png")
        plotter.save(tmp_path / "custom.png", dpi=96, bbox_inches=None)
        assert calls == [expected, {"dpi": 96, "bbox_inches": None}]


def test_supplied_subplot_retains_artists_and_style(tmp_path, monkeypatch):
    result = labfit.fit(np.arange(5.0), np.arange(5.0), sigma=0.1)
    fig, ax = plt.subplots()
    ax.spines["right"].set_visible(False)
    ax.xaxis.label.set_fontsize(18)
    (original,) = ax.plot([0, 1], [0, 1])
    position = ax.get_position().bounds

    with pytest.raises(ValueError, match="style must be"):
        labfit.plot_fit(result, ax=ax, style="unknown")
    assert fig.axes == [ax]
    assert ax.get_position().bounds == position

    plot = labfit.plot_fit(result, ax=ax, style="publication")
    main, residual = plot.axes
    assert plot.figure is fig and main is ax
    assert original in main.lines
    assert not main.spines["right"].get_visible()
    assert main.xaxis.label.get_fontsize() == 18
    assert main.get_shared_x_axes().joined(main, residual)
    calls = []
    monkeypatch.setattr(fig, "savefig", lambda path, **kwargs: calls.append(kwargs))
    plot.save(tmp_path / "existing.png")
    assert calls == [{}]


def test_rcparams_restored_if_drawing_fails(monkeypatch):
    result = labfit.fit(np.arange(5.0), np.arange(5.0), sigma=0.1)

    def fail(*args):
        assert not plt.rcParams["axes.spines.top"]
        raise RuntimeError("drawing failed")

    with plt.rc_context({"axes.spines.top": True}):
        before = plt.rcParams.copy()
        monkeypatch.setattr(result, "predict", fail)
        with pytest.raises(RuntimeError, match="drawing failed"):
            labfit.plot_fit(result)
        assert plt.rcParams == before


def test_default_axes_start_at_labeled_ticks_and_include_data():
    x = np.array([0, 1, 2, 3, 4.17])
    residuals = np.array([-0.27, 0, -0.19, 0.17, 0])
    y = x + residuals
    series = DataSeries(x=x, y=y, sigma=0.2)
    result = _linear_result(x, y, series, slope=1, intercept=0)
    for plot in (
        labfit.plot_fit(result),
        labfit.plot_multi_fit([result], [series]),
        labfit.Plotter().plot(result),
    ):
        main = plot.figure.axes[0]
        assert main.get_xlim()[0] == 0
        assert main.get_xlim()[1] >= x.max()
        assert main.get_ylim()[0] <= (y - 0.2).min()
        assert main.get_ylim()[1] >= (y + 0.2).max()
        for ax in plot.figure.axes:
            assert ax.get_xlim() == (ax.get_xticks()[0], ax.get_xticks()[-1])
            assert ax.get_ylim() == (ax.get_yticks()[0], ax.get_yticks()[-1])
        if len(plot.figure.axes) == 2:
            residual_ax = plot.figure.axes[1]
            assert residual_ax.get_ylim()[0] <= residuals.min()
            assert residual_ax.get_ylim()[1] >= residuals.max()


def test_positive_axes_can_start_at_the_origin_and_allow_explicit_limits():
    x = np.arange(5.0)
    result = _linear_result(x, x, DataSeries(x=x, y=x), slope=1, intercept=0)
    ax = labfit.plot_fit(result, show_residuals=False).axes
    assert ax.get_xlim()[0] == ax.get_ylim()[0] == 0
    ax.set_xlim(4.17, -0.23)
    assert ax.get_xlim() == (4.17, -0.23)
    assert -0.23 not in ax.get_xticks()


def test_tick_aligned_limits_are_specific_to_default_created_axes():
    x = np.arange(5.0)
    result = _linear_result(x, x, DataSeries(x=x, y=x))
    _, supplied = plt.subplots()
    for ax in (
        labfit.plot_fit(result, ax=supplied, show_residuals=False).axes,
        labfit.plot_fit(result, style="publication", show_residuals=False).axes,
        labfit.plot_fit(result, style=None, show_residuals=False).axes,
    ):
        assert ax.get_xlim()[0] < 0
        assert ax.get_xlim()[0] not in ax.get_xticks()
