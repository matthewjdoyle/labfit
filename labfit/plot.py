from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from .types import DataSeries, FitResult, Plotter

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

_style_applied = False


def use_publication_style():
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


def _as_results(result):
    if result is None:
        return []
    if isinstance(result, FitResult):
        return [result]
    if isinstance(result, (list, tuple)) and all(isinstance(item, FitResult) for item in result):
        return list(result)
    raise TypeError("result must be a FitResult or a sequence of FitResult objects")


def _as_series(data_series):
    if data_series is None:
        return None
    if isinstance(data_series, DataSeries):
        return [data_series]
    if isinstance(data_series, (list, tuple)) and all(isinstance(item, DataSeries) for item in data_series):
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


def _plot_series(ax, series: DataSeries, index: int):
    """Plot data series with error bars and caps using Okabe-Ito colours."""
    yerr = series.effective_sigma
    label = _series_label(series, index)
    color = _OKABE_ITO[index % len(_OKABE_ITO)]
    if yerr is None:
        (line,) = ax.plot(series.x, series.y, "o", ms=4.0, color=color, zorder=2, label=label)
        return line.get_color()
    container = ax.errorbar(
        series.x,
        series.y,
        yerr=yerr,
        fmt="o",
        color=color,
        markersize=4.0,
        capsize=3.5,
        capthick=1.2,
        elinewidth=1.2,
        zorder=2,
        label=label,
    )
    # Fix error bar lines z-order — matplotlib internally offsets them
    # to zorder-0.1, which puts them behind the markers and caps
    if len(container.lines) > 2:
        barlinecol = container.lines[2]
        if hasattr(barlinecol, "set_zorder"):
            barlinecol.set_zorder(2)
    return container.lines[0].get_color()


def _plot_fit_line(ax, result: FitResult, index: int):
    if result.x is None or result.y is None:
        return None
    # Use a visibly darker line — offset palette by 1 so the fit line
    # contrasts with the data markers while staying colour-blind friendly
    color = _OKABE_ITO[(index + 1) % len(_OKABE_ITO)]
    xs = np.linspace(float(np.min(result.x)), float(np.max(result.x)), 400)
    (line,) = ax.plot(xs, result.predict(xs), color=color, lw=2.5, zorder=3, label=_fit_label(result, index))
    return line.get_color()


def _confidence_band(
    result: FitResult, xs: np.ndarray, ci_level: float = 0.68, prediction: bool = False
) -> tuple[np.ndarray, np.ndarray] | None:
    """Compute a confidence or prediction band for the fit line.

    Returns ``(lower, upper)`` arrays, or ``None`` if the covariance is
    unavailable.
    """
    if result.covariance is None or result.model is None:
        return None
    cov = result.covariance
    if cov.ndim != 2 or not np.all(np.isfinite(np.diag(cov))):
        return None

    names = result.param_names or tuple(result.params.keys())
    p0 = np.array([result.params[n] for n in names], dtype=float)
    n_params = len(p0)

    def model_at(x_val: float) -> float:
        return float(result.model(np.array([x_val]), *p0)[0])

    ys = np.asarray(result.predict(xs), dtype=float)
    sigma_fit = np.zeros_like(ys)

    eps = 1e-8
    for i, xv in enumerate(xs):
        grad = np.zeros(n_params, dtype=float)
        for j in range(n_params):
            step = max(abs(p0[j]) * eps, eps)
            up = p0.copy()
            up[j] += step
            down = p0.copy()
            down[j] -= step
            grad[j] = (
                float(result.model(np.array([xv]), *up)[0]) - float(result.model(np.array([xv]), *down)[0])
            ) / (2.0 * step)
        sigma_fit[i] = float(np.sqrt(max(grad @ cov @ grad, 0.0)))

    if prediction:
        if result.is_weighted:
            residual_var = float(result.reduced_chi2)
        elif result.y is not None and result.y_fit is not None:
            residual_var = float(np.var(result.y - result.y_fit))
        else:
            residual_var = 0.0
        sigma_fit = np.sqrt(sigma_fit**2 + residual_var)

    z = norm.ppf(0.5 + ci_level / 2.0)
    return ys - z * sigma_fit, ys + z * sigma_fit


def _plot_ci_band(ax, result: FitResult, xs: np.ndarray, color: str, ci_level: float, prediction: bool):
    band = _confidence_band(result, xs, ci_level=ci_level, prediction=prediction)
    if band is None:
        return
    lower, upper = band
    label = f"{ci_level * 100:.0f}% {'prediction' if prediction else 'confidence'}"
    ax.fill_between(xs, lower, upper, color=color, alpha=0.18, zorder=2, label=label)


def _plot_residual_axis(axr, result: FitResult, color: str | None = None):
    residuals = _residuals(result)
    if residuals is None:
        return
    if color is None:
        color = _OKABE_ITO[1]
    axr.axhline(0.0, color="0.35", lw=1.0, ls="--")
    axr.plot(result.x, residuals, marker="o", ls="none", ms=3.5, color=color)
    axr.set_ylabel("residuals")


def _style_axes(ax, axr=None, *, xlabel="x", ylabel="y", title=None):
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    # With sharex=True, only the bottom panel gets the x-label
    if axr is not None:
        axr.set_xlabel(xlabel)
    else:
        ax.set_xlabel(xlabel)


def _create_single_figure(*, show_residuals: bool, figsize):
    if show_residuals:
        fig, (ax, axr) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(figsize[0], figsize[1] * 1.35),
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06},
        )
        return fig, ax, axr
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax, None


def _single_axes_from_existing(ax, *, show_residuals: bool, figsize):
    fig = ax.figure
    if not show_residuals:
        return fig, ax, None
    try:
        spec = ax.get_subplotspec()
    except Exception:
        return _create_single_figure(show_residuals=show_residuals, figsize=figsize)
    if spec is None:
        return _create_single_figure(show_residuals=show_residuals, figsize=figsize)
    ax.remove()
    inner = spec.subgridspec(2, 1, height_ratios=[3, 1], hspace=0.06)
    ax_main = fig.add_subplot(inner[0])
    ax_res = fig.add_subplot(inner[1], sharex=ax_main)
    return fig, ax_main, ax_res


def _plot_single_fit(
    result: FitResult,
    *,
    data_series=None,
    ax=None,
    show_residuals: bool = True,
    show_ci: bool = False,
    ci_level: float = 0.68,
    prediction: bool = False,
    title: str | None = None,
    xlabel: str = "x",
    ylabel: str = "y",
    figsize=(7, 4),
    plotter: Plotter | None = None,
) -> Plotter:
    series = data_series
    if series is None:
        series = result.series
    if series is None:
        raise ValueError("plot_fit needs a DataSeries either via data_series or result.series")
    if isinstance(series, (list, tuple)):
        if len(series) != 1:
            raise TypeError("plot_fit expects a single DataSeries")
        series = series[0]
    if not isinstance(series, DataSeries):
        raise TypeError("plot_fit expects a DataSeries")

    if ax is None:
        fig, ax_main, ax_res = _create_single_figure(show_residuals=show_residuals, figsize=figsize)
    else:
        fig, ax_main, ax_res = _single_axes_from_existing(ax, show_residuals=show_residuals, figsize=figsize)

    plotter = plotter or Plotter()
    plotter.figure = fig
    plotter.axes = (ax_main, ax_res) if ax_res is not None else ax_main
    plotter.series = [series]

    color = _plot_series(ax_main, series, 0)
    fit_color = _plot_fit_line(ax_main, result, 0)
    if fit_color is None:
        fit_color = color
    if show_ci and result.x is not None:
        xs_band = np.linspace(float(np.min(result.x)), float(np.max(result.x)), 200)
        _plot_ci_band(ax_main, result, xs_band, fit_color, ci_level, prediction)
    if ax_res is not None:
        _plot_residual_axis(ax_res, result, fit_color)
    _style_axes(ax_main, ax_res, xlabel=xlabel, ylabel=ylabel, title=title)
    handles, labels = ax_main.get_legend_handles_labels()
    if handles:
        ax_main.legend(loc="best")
    if ax_res is None:
        fig.tight_layout()
    return plotter


def _multi_layout(count: int, layout: str):
    layout = (layout or "grid").lower()
    if layout != "grid":
        raise ValueError("layout must be 'grid'")
    return 2, max(count, 1)


def plot_multi_fit(
    results,
    series_list,
    layout: str = "grid",
    *,
    figsize=(5.0, 4.0),
    title: str | None = None,
    xlabel: str = "x",
    ylabel: str = "y",
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

    Returns
    -------
    Plotter
    """
    use_publication_style()
    results = _as_results(results)
    series_items = _as_series(series_list)
    if series_items is None:
        raise TypeError("series_list is required")
    if len(results) != len(series_items):
        raise ValueError("results and series_list must have the same length")

    nrows, ncols = _multi_layout(len(results), layout)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        sharex="col",
        figsize=(figsize[0] * ncols, figsize[1] * nrows),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08, "wspace": 0.25},
    )
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes[:, np.newaxis]

    for idx, (result, series) in enumerate(zip(results, series_items, strict=True)):
        ax = axes[0, idx]
        axr = axes[1, idx]
        color = _plot_series(ax, series, idx)
        fit_color = _plot_fit_line(ax, result, idx)
        if fit_color is None:
            fit_color = color
        _plot_residual_axis(axr, result, fit_color)
        subtitle = _series_label(series, idx)
        if title and idx == 0:
            ax.set_title(title)
        elif title is None:
            ax.set_title(subtitle)
        else:
            ax.set_title(f"{subtitle}")
        _style_axes(ax, axr, xlabel=xlabel, ylabel=ylabel)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="best")

    plotter = Plotter(series=series_items, figure=fig, axes=axes)
    return plotter


def _overlay_on_axes(ax, axr, result: FitResult, series: DataSeries | None, index: int):
    color = None
    if series is not None:
        color = _plot_series(ax, series, index)
    fit_color = _plot_fit_line(ax, result, index)
    if fit_color is None:
        fit_color = color
    if axr is not None:
        _plot_residual_axis(axr, result, fit_color)


def _add_result_axes(ax, result: FitResult, show_residuals: bool = False):
    series = result.series
    color = None
    if series is not None:
        color = _plot_series(ax, series, 0)
    fit_color = _plot_fit_line(ax, result, 0)
    if fit_color is None:
        fit_color = color
    ax.legend(loc="best")
    if show_residuals:
        ax2 = ax.twinx()
        _plot_residual_axis(ax2, result, fit_color)
        return ax2
    return ax


def plot_result(
    result=None,
    *,
    plotter: Plotter | None = None,
    show_residuals: bool = False,
    title: str | None = None,
    xlabel: str = "x",
    ylabel: str = "y",
    figsize=(7, 4),
    **kwargs,
) -> Plotter:
    """Plot fit results or raw data series on a single set of axes.

    Use this when you want to overlay several fits or plot raw series
    without a fit. Pass a sequence of :class:`~labfit.FitResult` objects
    to overlay them on the same axes.

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

    Returns
    -------
    Plotter
        Container with the figure, axes, and a ``.save(path)`` method.
    """
    use_publication_style()
    results = _as_results(result)
    plotter = plotter or Plotter()

    if plotter.figure is None:
        if show_residuals:
            fig, ax, axr = _create_single_figure(show_residuals=True, figsize=figsize)
        else:
            fig, ax, axr = _create_single_figure(show_residuals=False, figsize=figsize)
        plotter.figure = fig
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
            ax.legend(loc="best")
        _style_axes(ax, axr, xlabel=xlabel, ylabel=ylabel, title=title)
        if axr is None:
            fig.tight_layout()
        return plotter

    for idx, result_item in enumerate(results):
        series = result_item.series
        if series is None and idx < len(plotter.series):
            series = plotter.series[idx]
        _overlay_on_axes(ax, axr if show_residuals else None, result_item, series, idx)

    _style_axes(ax, axr if show_residuals else None, xlabel=xlabel, ylabel=ylabel, title=title)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best")
    if not show_residuals:
        fig.tight_layout()
    return plotter


def plot_fit(
    result=None,
    *,
    data_series=None,
    ax=None,
    show_residuals: bool = True,
    show_ci: bool = False,
    ci_level: float = 0.68,
    prediction: bool = False,
    **kwargs,
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
        Confidence level for the band (e.g. 0.68 for 1σ, 0.95 for 95%).
    prediction : bool, default ``False``
        If ``True``, draw a prediction band (includes data scatter)
        instead of a confidence band for the mean.
    title : str, optional
        Plot title.
    xlabel, ylabel : str
        Axis labels.

    Returns
    -------
    Plotter
    """
    use_publication_style()
    results = _as_results(result)
    if len(results) != 1:
        return plot_result(result=result, show_residuals=show_residuals, **kwargs)
    return _plot_single_fit(
        results[0],
        data_series=data_series,
        ax=ax,
        show_residuals=show_residuals,
        show_ci=show_ci,
        ci_level=ci_level,
        prediction=prediction,
        **kwargs,
    )


def plot_residuals(result=None, *, data_series=None, ax=None, **kwargs) -> Plotter:
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
