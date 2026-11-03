from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from numbers import Real
from typing import Any, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.backend_bases import RendererBase
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoLocator
from scipy.spatial import cKDTree
from scipy.stats import norm

from .types import AsymmetricError, DataSeries, FitResult, MaybeArray, Plotter, ResultInput, SeriesInput

# ── Colour-blind friendly palette (Okabe-Ito) ──────────────────────
_OKABE_ITO = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#F0E442",
    "#56B4E9",
    "#E69F00",
    "#000000",
]


# ── Publication style ──────────────────────────────────────────────


_PUBLICATION_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 9.5,
    "mathtext.fontset": "dejavusans",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.linewidth": 0.8,
    "axes.labelsize": 10.5,
    "axes.titlesize": 11,
    "axes.prop_cycle": plt.cycler(color=_OKABE_ITO),
    "axes.grid": False,
    "xtick.direction": "in",
    "xtick.major.size": 4,
    "xtick.major.width": 0.7,
    "xtick.minor.size": 2,
    "xtick.minor.width": 0.5,
    "xtick.labelsize": 9,
    "xtick.top": True,
    "ytick.direction": "in",
    "ytick.major.size": 4,
    "ytick.major.width": 0.7,
    "ytick.minor.size": 2,
    "ytick.minor.width": 0.5,
    "ytick.labelsize": 9,
    "ytick.right": True,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "legend.handlelength": 1.2,
    "lines.linewidth": 1.4,
    "lines.markersize": 5.5,
}

_DEFAULT_RC = {
    **_PUBLICATION_RC,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "out",
    "xtick.top": False,
    "ytick.direction": "out",
    "ytick.right": False,
}
_Style = Literal["default", "publication"] | None
_FIT_LINESTYLES = ("-", "--", "-.", ":")
_RESIDUAL_ZERO_LABEL = "_labfit_residual_zero"
_style_applied = False


def _style_context(style: _Style, *, owns_figure: bool) -> AbstractContextManager[None]:
    if style not in ("default", "publication", None):
        raise ValueError("style must be 'default', 'publication', or None")
    if not owns_figure or style is None:
        return nullcontext()
    return plt.rc_context(_DEFAULT_RC if style == "default" else _PUBLICATION_RC)


class _AlignedLocator(AutoLocator):
    def __init__(self) -> None:
        super().__init__()
        self._aligned_ticks: list[float] = []

    def view_limits(self, dmin: float, dmax: float) -> tuple[float, float]:
        self._aligned_ticks = list(map(float, self.tick_values(dmin, dmax)))
        return self._aligned_ticks[0], self._aligned_ticks[-1]

    def __call__(self) -> list[float]:
        assert self.axis is not None
        lower, upper = sorted(map(float, self.axis.get_view_interval()))
        if self._aligned_ticks and (lower, upper) == (self._aligned_ticks[0], self._aligned_ticks[-1]):
            return self._aligned_ticks
        return list(map(float, super().__call__()))


def _align_axis_limits(*axes: Axes) -> None:
    for ax in axes:
        ax.xaxis.set_major_locator(_AlignedLocator())
        ax.yaxis.set_major_locator(_AlignedLocator())
        # Allow the autoscale margin to stop at zero for data starting there.
        cast(list[float], ax.patch.sticky_edges.x).append(0.0)
        cast(list[float], ax.patch.sticky_edges.y).append(0.0)


def _set_save_defaults(plotter: Plotter, style: _Style) -> None:
    plotter._save_defaults = (
        {} if style is None else {"dpi": _PUBLICATION_RC["savefig.dpi"], "bbox_inches": "tight"}
    )


def use_publication_style() -> None:
    """Apply matplotlib rcParams suitable for journal-quality figures.

    Uses a sans-serif font family (DejaVu Sans, built into matplotlib)
    for a clean modern appearance — no external TeX installation needed.
    Idempotent: only applies once per session.
    """
    global _style_applied
    if _style_applied:
        return
    plt.rcParams.update(_PUBLICATION_RC)
    _style_applied = True


def _as_results(result: ResultInput) -> list[FitResult]:
    if result is None:
        return []
    if isinstance(result, FitResult):
        return [result]
    if isinstance(result, list | tuple) and all(isinstance(item, FitResult) for item in result):
        return list(result)
    raise TypeError("result must be a FitResult or a sequence of FitResult objects")


def _as_series(data_series: SeriesInput) -> list[DataSeries] | None:
    if data_series is None:
        return None
    if isinstance(data_series, DataSeries):
        return [data_series]
    if isinstance(data_series, list | tuple) and all(isinstance(item, DataSeries) for item in data_series):
        return list(data_series)
    raise TypeError("data_series must be a DataSeries or a sequence of DataSeries objects")


def _series_label(series: DataSeries, index: int) -> str:
    label = (series.label or "").strip()
    return label if label else f"series {index + 1}"


def _fit_label(result: FitResult, index: int) -> str:
    name = (result.model_name or "").strip()
    if not name:
        return f"fit {index + 1}"
    return f"fit: {name}"


def _residuals(result: FitResult) -> np.ndarray | None:
    if result.x is None or result.y is None:
        return None
    return result.y - result.predict(result.x)


class _MarkerSizing(Artist):
    """Size LabFit markers before the other artists on this axes draw."""

    def __init__(self) -> None:
        super().__init__()
        self.set_zorder(float("-inf"))
        self.set_in_layout(False)
        self.markers: list[tuple[Line2D, float, LineCollection | None]] = []

    def draw(self, renderer: RendererBase) -> None:
        ax = self.axes
        assert ax is not None
        pixels_per_point = cast(float, renderer.points_to_pixels(1))
        box = ax.bbox
        panel_limit = min(box.width, box.height) / (25 * pixels_per_point)
        visible_markers = []
        for line, standard_size, bars in self.markers:
            if line.axes is not ax or not line.get_visible():
                continue
            centres = line.get_transform().transform(line.get_xydata())
            visible = (
                np.isfinite(centres).all(axis=1)
                & (centres[:, 0] >= box.xmin)
                & (centres[:, 0] <= box.xmax)
                & (centres[:, 1] >= box.ymin)
                & (centres[:, 1] <= box.ymax)
            )
            visible_markers.append((line, standard_size, bars, centres, visible))
        if not visible_markers:
            return

        visible_centres = np.concatenate([centres[visible] for _, _, _, centres, visible in visible_markers])
        distinct = np.unique(visible_centres, axis=0)
        outer_limit = panel_limit
        if len(distinct) > 1:
            distances, _ = cKDTree(distinct).query(distinct, k=2)
            spacing = float(np.quantile(distances[:, 1], 0.25)) / pixels_per_point
            outer_limit = min(outer_limit, 0.7 * spacing)

        for line, standard_size, bars, centres, visible in visible_markers:
            limit = outer_limit
            if bars is not None and bars.axes is ax and bars.get_visible():
                error_limit = _error_marker_limit(bars, centres, visible, pixels_per_point)
                limit = min(limit, error_limit)
            size = max(2.5, min(standard_size, limit - line.get_markeredgewidth()))
            if line.get_markersize() != size:
                line.set_markersize(size)
        self.stale = False


def _error_marker_limit(
    bars: LineCollection, centres: np.ndarray, visible: np.ndarray, pixels_per_point: float
) -> float:
    segments = [np.asarray(path.vertices) for path in bars.get_paths()]
    indices = [
        i
        for i, (segment, shown) in enumerate(zip(segments, visible, strict=False))
        if shown and segment.shape == (2, 2)
    ]
    if not indices:
        return float("inf")
    endpoints = np.stack([segments[i] for i in indices])
    display = bars.get_transform().transform(endpoints.reshape(-1, 2)).reshape(-1, 2, 2)
    lengths = np.linalg.norm(display - centres[indices, None, :], axis=2)
    lengths[~np.isfinite(lengths) | (lengths <= 0)] = np.inf
    shorter_arms = lengths.min(axis=1)
    valid_arms = shorter_arms[np.isfinite(shorter_arms)]
    if not len(valid_arms):
        return float("inf")
    return 2 * (float(np.quantile(valid_arms, 0.25)) / pixels_per_point - 1)


def _size_marker(ax: Axes, line: Line2D, bars: LineCollection | None = None) -> None:
    sizing = next((artist for artist in ax.artists if isinstance(artist, _MarkerSizing)), None)
    if sizing is None:
        sizing = _MarkerSizing()
        ax.add_artist(sizing)
    sizing.markers.append((line, line.get_markersize(), bars))


def _plot_legend(ax: Axes) -> None:
    # Matplotlib copies marker properties when it builds the legend.
    for artist in ax.artists:
        if isinstance(artist, _MarkerSizing):
            for line, standard_size, _ in artist.markers:
                if line.axes is ax:
                    line.set_markersize(standard_size)
    ax.legend(loc="best")


def _plot_series(ax: Axes, series: DataSeries, index: int, *, color: str | None = None) -> None:
    """Plot data series with error bars and caps using Okabe-Ito colours."""
    if series.sigma_cov is not None:
        yerr = np.sqrt(np.diag(series.sigma_cov))
    elif series.sigma_low is not None and series.sigma_high is not None:
        yerr = np.stack(np.broadcast_arrays(series.sigma_low, series.sigma_high, series.y)[:2])
    elif isinstance(series.sigma, AsymmetricError):
        yerr = np.stack(np.broadcast_arrays(series.sigma.lower, series.sigma.upper, series.y)[:2])
    else:
        yerr = series.effective_sigma
    label = _series_label(series, index)
    if color is None:
        color = _OKABE_ITO[index % len(_OKABE_ITO)]
    if yerr is None:
        (marker,) = ax.plot(series.x, series.y, "o", ms=4.0, mfc=color, color=color, zorder=4, label=label)
        _size_marker(ax, marker)
        return
    container = ax.errorbar(
        series.x,
        series.y,
        yerr=yerr,
        fmt="o",
        color=color,
        markerfacecolor=color,
        markersize=4.0,
        capsize=2.0,
        capthick=0.8,
        elinewidth=0.8,
        zorder=2,
        label=label,
    )
    marker_line, _, (bars,) = container.lines
    assert marker_line is not None
    marker_line.set_zorder(4)
    _size_marker(ax, marker_line, bars)


def _plot_fit_line(ax: Axes, result: FitResult, index: int, color: str, *, linestyle: str = "-") -> None:
    if result.x is None or result.y is None:
        return
    xs = np.linspace(float(np.min(result.x)), float(np.max(result.x)), 400)
    ax.plot(
        xs, result.predict(xs), color=color, ls=linestyle, lw=1.7, zorder=3, label=_fit_label(result, index)
    )


_PredictionSigma = MaybeArray | Callable[[np.ndarray], MaybeArray] | None


def _validate_ci_level(ci_level: float) -> float:
    if not isinstance(ci_level, Real) or isinstance(ci_level, bool):
        raise ValueError("ci_level must be a finite number strictly between 0 and 1")
    level = float(ci_level)
    if not np.isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError("ci_level must be a finite number strictly between 0 and 1")
    return level


def _prediction_variance(result: FitResult, xs: np.ndarray, prediction_sigma: _PredictionSigma) -> np.ndarray:
    """Variance of future observations independent of the fitted data."""
    if prediction_sigma is not None:
        supplied = prediction_sigma(xs) if callable(prediction_sigma) else prediction_sigma
        try:
            sigma = np.asarray(supplied, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("prediction_sigma must contain finite nonnegative standard deviations") from exc
        if sigma.ndim != 0 and sigma.shape != xs.shape:
            raise ValueError("prediction_sigma must be a scalar or match the prediction grid shape")
        if not np.all(np.isfinite(sigma)) or np.any(sigma < 0):
            raise ValueError("prediction_sigma must contain finite nonnegative standard deviations")
        # Explicit future sigmas are absolute errors; never rescale them using
        # the fitted residuals, even when training errors are relative weights.
        with np.errstate(over="ignore"):
            variance = np.square(sigma)
    elif result.is_weighted:
        training_sigma = None if result.sigma is None else np.asarray(result.sigma, dtype=float)
        if (
            training_sigma is None
            or training_sigma.size == 0
            or not np.all(training_sigma == training_sigma.flat[0])
        ):
            raise ValueError(
                "Heterogeneous or correlated training errors do not define future observation noise; "
                "provide prediction_sigma (a scalar, grid array, or function of x)."
            )
        if not np.all(np.isfinite(training_sigma)) or np.any(training_sigma <= 0):
            raise ValueError("Training sigmas cannot define future noise; provide prediction_sigma")
        scale = 1.0 if result.absolute_sigma else result.reduced_chi2
        with np.errstate(over="ignore"):
            variance = np.square(training_sigma.flat[0]) * scale
    else:
        # The absolute unweighted policy assumes unit variance; the relative
        # policy estimates homoscedastic noise in squared y units from SSR/dof.
        variance = np.asarray(1.0 if result.absolute_sigma else result.residual_variance)
    if not np.all(np.isfinite(variance)) or np.any(variance < 0):
        raise ValueError("Future observation variance is unavailable or invalid; provide prediction_sigma")
    return np.broadcast_to(variance, xs.shape)


def _confidence_band(
    result: FitResult,
    xs: np.ndarray,
    ci_level: float = 0.68,
    prediction: bool = False,
    prediction_sigma: _PredictionSigma = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Compute a confidence or prediction band for the fit line.

    These are pointwise normal approximations using the local parameter
    covariance. Prediction bands add variance of future observations independent
    of the fitted data, in squared y units. Explicit prediction_sigma values
    describe absolute future errors; see plot_fit for default noise policies.
    Model perturbations are evaluated across the complete prediction grid.
    Returns ``(lower, upper)`` arrays, or ``None`` if covariance is unavailable.
    """
    ci_level = _validate_ci_level(ci_level)
    xs = np.asarray(xs, dtype=float)
    if result.covariance is None or result.model is None:
        return None
    cov = result.covariance
    if cov.ndim != 2 or not np.all(np.isfinite(cov)):
        return None

    names = result.param_names or tuple(result.params.keys())
    p0 = np.array([result.params[n] for n in names], dtype=float)
    n_params = len(p0)

    ys = np.asarray(result.predict(xs), dtype=float)
    gradients = np.empty((xs.size, n_params), dtype=float)

    eps = 1e-8
    for j in range(n_params):
        step = max(abs(p0[j]) * eps, eps)
        up = p0.copy()
        up[j] += step
        down = p0.copy()
        down[j] -= step
        gradients[:, j] = (
            np.asarray(result.model(xs, *up), dtype=float) - np.asarray(result.model(xs, *down), dtype=float)
        ) / (2.0 * step)
    variance = np.einsum("ij,jk,ik->i", gradients, cov, gradients)
    sigma_fit = np.sqrt(np.maximum(variance, 0.0))

    if prediction:
        sigma_fit = np.sqrt(sigma_fit**2 + _prediction_variance(result, xs, prediction_sigma))

    z = norm.isf((1.0 - ci_level) / 2.0)
    return ys - z * sigma_fit, ys + z * sigma_fit


def _plot_ci_band(
    ax: Axes,
    xs: np.ndarray,
    band: tuple[np.ndarray, np.ndarray] | None,
    color: str,
    ci_level: float,
    prediction: bool,
) -> None:
    if band is None:
        return
    lower, upper = band
    interval = "prediction interval" if prediction else "confidence interval for the mean"
    label = f"{ci_level:.0%} {interval}"
    ax.fill_between(xs, lower, upper, color=color, alpha=0.18, zorder=1, label=label)


def _plot_residual_axis(axr: Axes, result: FitResult, color: str) -> None:
    residuals = _residuals(result)
    if residuals is None:
        return
    if not any(line.get_label() == _RESIDUAL_ZERO_LABEL for line in axr.lines):
        axr.axhline(0.0, color="0.35", lw=1.0, ls="--", zorder=1, label=_RESIDUAL_ZERO_LABEL)
    assert result.x is not None
    (marker,) = axr.plot(result.x, residuals, marker="o", ls="none", ms=3.5, color=color)
    _size_marker(axr, marker)
    axr.set_ylabel("residuals")


def _style_axes(
    ax: Axes, axr: Axes | None = None, *, xlabel: str = "x", ylabel: str = "y", title: str | None = None
) -> None:
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    # With sharex=True, only the bottom panel gets the x-label
    if axr is not None:
        axr.set_xlabel(xlabel)
    else:
        ax.set_xlabel(xlabel)


def _create_single_figure(
    *, show_residuals: bool, figsize: tuple[float, float], style: _Style
) -> tuple[Figure, Axes, Axes | None]:
    if show_residuals:
        fig, (ax, axr) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(figsize[0], figsize[1] * 1.35),
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06},
        )
        if style == "default":
            _align_axis_limits(ax, axr)
        return fig, ax, axr
    fig, ax = plt.subplots(figsize=figsize)
    if style == "default":
        _align_axis_limits(ax)
    return fig, ax, None


def _single_axes_from_existing(ax: Axes, *, show_residuals: bool) -> tuple[Figure, Axes, Axes | None]:
    fig = cast(Figure, ax.figure)
    if not show_residuals:
        return fig, ax, None
    spec = ax.get_subplotspec()
    assert spec is not None
    inner = spec.subgridspec(2, 1, height_ratios=[3, 1], hspace=0.06)
    ax.set_subplotspec(inner[0])
    axr = fig.add_subplot(inner[1], sharex=ax)
    ax.tick_params(labelbottom=False)
    return fig, ax, axr


def _plot_single_fit(
    result: FitResult,
    *,
    data_series: SeriesInput = None,
    ax: Axes | None = None,
    show_residuals: bool = True,
    show_ci: bool = False,
    ci_level: float = 0.68,
    prediction: bool = False,
    prediction_sigma: _PredictionSigma = None,
    title: str | None = None,
    xlabel: str = "x",
    ylabel: str = "y",
    figsize: tuple[float, float] = (7, 4),
    plotter: Plotter | None = None,
    style: _Style = "default",
) -> Plotter:
    series = data_series
    if series is None:
        series = result.series
    if series is None:
        raise ValueError("plot_fit needs a DataSeries either via data_series or result.series")
    if isinstance(series, list | tuple):
        if len(series) != 1:
            raise TypeError("plot_fit expects a single DataSeries")
        series = series[0]
    if not isinstance(series, DataSeries):
        raise TypeError("plot_fit expects a DataSeries")

    xs_band = None
    band = None
    if show_ci and result.x is not None:
        xs_band = np.linspace(float(np.min(result.x)), float(np.max(result.x)), 200)
        band = _confidence_band(result, xs_band, ci_level, prediction, prediction_sigma)

    owns_figure = ax is None or (show_residuals and ax.get_subplotspec() is None)
    with _style_context(style, owns_figure=owns_figure):
        if owns_figure:
            fig, ax_main, ax_res = _create_single_figure(
                show_residuals=show_residuals, figsize=figsize, style=style
            )
        else:
            assert ax is not None
            fig, ax_main, ax_res = _single_axes_from_existing(ax, show_residuals=show_residuals)

        plotter = plotter or Plotter()
        plotter.figure = fig
        plotter.axes = (ax_main, ax_res) if ax_res is not None else ax_main
        plotter.series = [series]
        _set_save_defaults(plotter, style if owns_figure else None)

        color = _OKABE_ITO[0]
        _plot_series(ax_main, series, 0)
        _plot_fit_line(ax_main, result, 0, color)
        if xs_band is not None:
            _plot_ci_band(ax_main, xs_band, band, color, ci_level, prediction)
        if ax_res is not None:
            _plot_residual_axis(ax_res, result, color)
        _style_axes(ax_main, ax_res, xlabel=xlabel, ylabel=ylabel, title=title)
        handles, labels = ax_main.get_legend_handles_labels()
        if handles:
            _plot_legend(ax_main)
        if owns_figure and ax_res is None:
            fig.tight_layout()
        return plotter


def _multi_layout(count: int, layout: str) -> tuple[int, int]:
    layout = (layout or "grid").lower()
    if layout != "grid":
        raise ValueError("layout must be 'grid'")
    return 2, max(count, 1)


def plot_multi_fit(
    results: ResultInput,
    series_list: SeriesInput,
    layout: str = "grid",
    *,
    figsize: tuple[float, float] = (5.0, 4.0),
    title: str | None = None,
    xlabel: str = "x",
    ylabel: str = "y",
    sharey: bool = False,
    style: _Style = "default",
) -> Plotter:
    """Compare multiple fits in a side-by-side grid.

    Each fit gets its own column with data and fit line in the top
    panel and residuals below. Requires exactly one
    :class:`~labfit.DataSeries` per fit result.

    Parameters
    ----------
    results : list of FitResult
        The fits to display.
    series_list : list of DataSeries
        The corresponding measured data (one per fit).
    layout : str, default ``"grid"``
        Layout style (currently only ``"grid"`` is supported).
    figsize : tuple, default ``(5, 4)``
        Dimensions per panel in inches.
    title : str, optional
        Overall figure title.
    xlabel, ylabel : str
        Axis labels shared by all panels.
    sharey : bool, default False
        Match y-limits within the fit row and separately within the residual row.
    style : {"default", "publication", None}, optional
        Theme for new figures. The publication preset uses a boxed frame.
        Pass None to use the active Matplotlib theme.

    Returns
    -------
    Plotter
    """
    results = _as_results(results)
    series_items = _as_series(series_list)
    if series_items is None:
        raise TypeError("series_list is required")
    if len(results) != len(series_items):
        raise ValueError("results and series_list must have the same length")

    with _style_context(style, owns_figure=True):
        nrows, ncols = _multi_layout(len(results), layout)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            sharex="col",
            sharey="row" if sharey else False,
            squeeze=False,
            layout="constrained",
            figsize=(figsize[0] * ncols, figsize[1] * nrows),
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08, "wspace": 0.25},
        )
        if style == "default":
            _align_axis_limits(*axes.flat)
        if title:
            fig.suptitle(title)
        fig.supxlabel(xlabel)

        dataset_indices: dict[int, int] = {}
        for idx, (result, series) in enumerate(zip(results, series_items, strict=True)):
            ax = axes[0, idx]
            axr = axes[1, idx]
            color_index = dataset_indices.setdefault(id(series), idx)
            color = _OKABE_ITO[color_index % len(_OKABE_ITO)]
            _plot_series(ax, series, idx, color=color)
            _plot_fit_line(ax, result, idx, color)
            _plot_residual_axis(axr, result, color)
            ax.set_title(_series_label(series, idx))
            ax.set_ylabel(ylabel if idx == 0 else "")
            axr.set_ylabel("residuals" if idx == 0 else "")
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                _plot_legend(ax)

        plotter = Plotter(series=series_items, figure=fig, axes=axes)
        _set_save_defaults(plotter, style)
        return plotter


def _add_result_axes(ax: Axes, result: FitResult, show_residuals: bool = False) -> Axes:
    color = _OKABE_ITO[0]
    if result.series is not None:
        _plot_series(ax, result.series, 0)
    _plot_fit_line(ax, result, 0, color)
    _plot_legend(ax)
    if show_residuals:
        ax2 = ax.twinx()
        _plot_residual_axis(ax2, result, color)
        return ax2
    return ax


def plot_result(
    result: ResultInput = None,
    *,
    plotter: Plotter | None = None,
    show_residuals: bool = False,
    title: str | None = None,
    xlabel: str = "x",
    ylabel: str = "y",
    figsize: tuple[float, float] = (7, 4),
    style: _Style = "default",
    **kwargs: Any,
) -> Plotter:
    """Plot fit results or raw data series on a single set of axes.

    Use this when you want to overlay several fits or plot raw series
    without a fit. Pass a sequence of :class:`~labfit.FitResult` objects
    to overlay them on the same axes. Results referencing the same DataSeries
    share one set of observation markers and use different fit line styles.
    Marker sizes adapt to the visible point spacing and error bars on each draw.

    Parameters
    ----------
    result : FitResult or list of FitResult, optional
        One or more fit results to overlay. If omitted, plots the
        series already registered in the provided ``plotter``.
    plotter : Plotter, optional
        An existing :class:`~labfit.Plotter` to draw into.
    show_residuals : bool, default ``False``
        Show a residual sub-panel below the main axes.
    title : str, optional
        Plot title.
    xlabel, ylabel : str, default ``"x"`` / ``"y"``
        Axis labels.
    figsize : tuple, default ``(7, 4)``
        Figure dimensions in inches.
    style : {"default", "publication", None}, optional
        Theme for new figures. The publication preset uses a boxed frame.
        Pass None to use the active Matplotlib theme.

    Returns
    -------
    Plotter
        Container with the figure, axes, and a ``.save(path)`` method.
    """
    results = _as_results(result)
    plotter = plotter or Plotter()

    owns_figure = plotter.figure is None
    with _style_context(style, owns_figure=owns_figure):
        if owns_figure:
            fig, ax, axr = _create_single_figure(show_residuals=show_residuals, figsize=figsize, style=style)
            plotter.figure = fig
            _set_save_defaults(plotter, style)
            plotter.axes = (ax, axr) if axr is not None else ax
        else:
            fig = plotter.figure
            axes = plotter.axes
            if isinstance(axes, tuple):
                ax = axes[0]
                axr = axes[1]
            elif isinstance(axes, np.ndarray) and axes.ndim >= 2:
                ax = axes[0, 0]
                axr = axes[1, 0] if axes.shape[0] > 1 else None
            else:
                ax = axes
                axr = None

        if not results and plotter.series:
            for idx, series_item in enumerate(plotter.series):
                _plot_series(ax, series_item, idx)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                _plot_legend(ax)
            _style_axes(ax, axr, xlabel=xlabel, ylabel=ylabel, title=title)
            if owns_figure and axr is None:
                fig.tight_layout()
            return plotter

        datasets: dict[int, tuple[int, int]] = {}
        for idx, result_item in enumerate(results):
            series = result_item.series
            if series is None and idx < len(plotter.series):
                series = plotter.series[idx]
            color_index, model_index = idx, 0
            if series is not None:
                key = id(series)
                if key in datasets:
                    color_index, model_index = datasets[key]
                else:
                    _plot_series(ax, series, color_index)
                datasets[key] = (color_index, model_index + 1)
            color = _OKABE_ITO[color_index % len(_OKABE_ITO)]
            linestyle = _FIT_LINESTYLES[model_index % len(_FIT_LINESTYLES)]
            _plot_fit_line(ax, result_item, idx, color, linestyle=linestyle)
            if show_residuals and axr is not None:
                _plot_residual_axis(axr, result_item, color)

        _style_axes(ax, axr if show_residuals else None, xlabel=xlabel, ylabel=ylabel, title=title)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            _plot_legend(ax)
        if owns_figure and not show_residuals:
            fig.tight_layout()
        return plotter


def plot_fit(
    result: ResultInput = None,
    *,
    data_series: SeriesInput = None,
    ax: Axes | None = None,
    show_residuals: bool = True,
    show_ci: bool = False,
    ci_level: float = 0.68,
    prediction: bool = False,
    prediction_sigma: _PredictionSigma = None,
    style: _Style = "default",
    **kwargs: Any,
) -> Plotter:
    """Plot a single fit result with its data and residuals.

    The simplest way to visualise a fitted curve. Data points with
    error bars, the best-fit line, and (by default) a residual panel
    underneath are all generated automatically.

    Parameters
    ----------
    result : FitResult
        The fit result to plot.
    data_series : DataSeries, optional
        The data to show. Falls back to ``result.series``.
    ax : matplotlib Axes, optional
        Existing axes to plot into.
    show_residuals : bool, default ``True``
        Include a residual sub-panel.
    show_ci : bool, default ``False``
        Draw a confidence band around the fit line.
    ci_level : float, default ``0.68``
        Finite confidence level strictly between 0 and 1 (e.g. 0.68 or 0.95).
        Bands are pointwise normal approximations, not simultaneous intervals.
    prediction : bool, default ``False``
        If ``True``, draw a prediction band (includes data scatter)
        instead of a confidence band for the mean. Requires ``show_ci=True``.
        Future observations are assumed independent of the fitted data.
    prediction_sigma : float, array-like or callable, optional
        Absolute future 1-σ errors in y units. A scalar applies everywhere;
        an array must match the 200-point grid spanning ``result.x``; a callable
        receives that grid and returns a scalar or matching array. Explicit
        values are never scaled by reduced chi-square and may be zero.
        Without this option, constant training sigmas are reused, scaled by
        sqrt(reduced_chi2) under the relative covariance policy. Heterogeneous
        or correlated training errors require an explicit future noise model.
        Unweighted relative fits use sqrt(residual_variance); unweighted
        absolute fits assume unit noise, matching their covariance policy.
    title : str, optional
        Plot title.
    xlabel, ylabel : str
        Axis labels.
    style : {"default", "publication", None}, optional
        Theme for new figures. The publication preset uses a boxed frame.
        Pass None to use the active Matplotlib theme.

    Returns
    -------
    Plotter
    """
    ci_level = _validate_ci_level(ci_level)
    if prediction_sigma is not None and not (show_ci and prediction):
        raise ValueError("prediction_sigma requires show_ci=True and prediction=True")
    results = _as_results(result)
    if len(results) != 1:
        if show_ci:
            raise ValueError("Confidence and prediction bands require a single FitResult")
        return plot_result(result=result, show_residuals=show_residuals, style=style, **kwargs)
    return _plot_single_fit(
        results[0],
        data_series=data_series,
        ax=ax,
        show_residuals=show_residuals,
        show_ci=show_ci,
        ci_level=ci_level,
        prediction=prediction,
        prediction_sigma=prediction_sigma,
        style=style,
        **kwargs,
    )


def plot_residuals(
    result: ResultInput = None, *, data_series: SeriesInput = None, ax: Axes | None = None, **kwargs: Any
) -> Plotter:
    """Plot a fit emphasising the residuals.

    Equivalent to ``plot_fit(…, show_residuals=True)``.

    Parameters
    ----------
    result : FitResult
        The fit result to plot.
    data_series : DataSeries, optional
        The data to show.
    ax : matplotlib Axes, optional
        Existing axes to plot into.

    Returns
    -------
    Plotter
    """
    kwargs.setdefault("show_residuals", True)
    return plot_fit(result=result, data_series=data_series, ax=ax, **kwargs)


__all__ = [
    "plot_fit",
    "plot_multi_fit",
    "plot_residuals",
    "plot_result",
]
