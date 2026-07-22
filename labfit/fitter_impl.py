from __future__ import annotations

import warnings
from collections import namedtuple
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2 as chi2_dist

from .io import load_csv as _load_csv
from .models import get_model, model_param_names
from .types import DataSeries, Dataset, FitResult
from .utils import effective_sigma


def _coerce_series_input(x, y=None, *, sigma=None, sigma_low=None, sigma_high=None, sigma_cov=None, label=""):
    if isinstance(x, DataSeries) and y is None:
        return x
    if isinstance(x, (str, Path)) and y is None:
        series = _load_csv(x)
        return DataSeries(
            x=series.x,
            y=series.y,
            sigma=sigma if sigma is not None else series.sigma,
            sigma_low=sigma_low if sigma_low is not None else series.sigma_low,
            sigma_high=sigma_high if sigma_high is not None else series.sigma_high,
            sigma_cov=sigma_cov,
            label=label,
        )
    if y is None:
        raise TypeError("fit requires either (x, y) arrays or a DataSeries / CSV path")
    return DataSeries(
        x=x, y=y, sigma=sigma, sigma_low=sigma_low, sigma_high=sigma_high, sigma_cov=sigma_cov, label=label
    )


_GuessStats = namedtuple(
    "_GuessStats", ["x", "y", "span", "yrange", "y_min", "y_max", "y_mean", "x_mean", "x_at_max"]
)


def _osc_freq(s: _GuessStats) -> float:
    """Estimate oscillation frequency from zero crossings."""
    freq = 1.0 / max(s.span, 1.0)
    if s.x.size > 1:
        dx = np.diff(s.x)
        if np.all(dx > 0):
            y_detrended = s.y - s.y_mean
            zero_crossings = np.where(np.signbit(y_detrended[:-1]) != np.signbit(y_detrended[1:]))[0]
            if zero_crossings.size >= 2:
                est_periods = np.diff(s.x[zero_crossings]) * 2.0
                period = float(np.median(est_periods)) if est_periods.size else s.span
                if period > 0:
                    freq = 1.0 / period
    return freq


def _guess_linear(s):
    if s.x.size >= 2:
        slope, intercept = np.polyfit(s.x, s.y, 1)
        return [float(slope), float(intercept)]
    return [1.0, s.y_mean]


def _guess_quadratic(s):
    coeffs = np.polyfit(s.x, s.y, 2) if s.x.size >= 3 else [0.0, 1.0, s.y_mean]
    return [float(v) for v in coeffs]


def _guess_cubic(s):
    coeffs = np.polyfit(s.x, s.y, 3) if s.x.size >= 4 else [0.0, 0.0, 1.0, s.y_mean]
    return [float(v) for v in coeffs]


def _guess_constant(s):
    return [s.y_mean]


def _guess_gaussian(s):
    sigma0 = s.span / 6.0 if s.span else 1.0
    amp = s.y_max - s.y_min if s.yrange else 1.0
    return [amp, s.x_at_max, max(sigma0, 1e-6)]


def _guess_lorentzian(s):
    amp = s.y_max - s.y_min if s.yrange else 1.0
    return [amp, s.x_at_max, max(s.span / 10.0, 1e-6)]


def _guess_exponential(s):
    amp = s.y[0] if s.y.size else 1.0
    decay = 1.0 / max(s.span, 1.0)
    positive = s.y > 0
    if positive.sum() >= 2 and s.x.size >= 2:
        slope, intercept = np.polyfit(s.x[positive], np.log(s.y[positive]), 1)
        amp = float(np.exp(intercept))
        decay = float(-slope)
    return [float(amp), float(decay)]


def _guess_power_law(s):
    amp = max(s.y_max, 1e-6)
    exponent = 1.0
    positive = (s.x > 0) & (s.y > 0)
    if positive.sum() >= 2:
        slope, intercept = np.polyfit(np.log(s.x[positive]), np.log(s.y[positive]), 1)
        amp = float(np.exp(intercept))
        exponent = float(slope)
    return [float(amp), float(exponent)]


def _guess_logistic(s):
    return [s.yrange or 1.0, s.x_mean, 1.0 / max(s.span, 1.0), s.y_min]


def _guess_damped_oscillator(s):
    freq = _osc_freq(s)
    amp = max(abs(s.y_min), abs(s.y_max), 1.0)
    return [float(amp), 0.1 / max(s.span, 1.0), float(freq), 0.0]


def _guess_damped_sine(s):
    freq = _osc_freq(s)
    amp = max(abs(s.y_min), abs(s.y_max), 1.0)
    return [float(amp), 0.1 / max(s.span, 1.0), float(freq), 0.0, float(s.y_mean)]


def _guess_sine(s):
    freq = _osc_freq(s)
    amp = max(abs(s.y_min), abs(s.y_max), 1.0)
    return [float(amp), float(freq), 0.0, float(s.y_mean)]


def _guess_cosine(s):
    return _guess_sine(s)


def _guess_beat(s):
    freq = _osc_freq(s)
    amp = max(abs(s.y_min), abs(s.y_max), 1.0)
    return [float(amp), float(freq), float(freq * 1.08), 0.0, float(s.y_mean)]


def _guess_voigt(s):
    amp = s.y_max - s.y_min if s.yrange else 1.0
    return [amp, s.x_at_max, max(s.span / 6.0, 1e-6), max(s.span / 10.0, 1e-6)]


def _guess_skew_normal(s):
    amp = s.y_max - s.y_min if s.yrange else 1.0
    return [amp, s.x_at_max, max(s.span / 6.0, 1e-6), 0.0]


def _guess_fwhm(s):
    amp = s.y_max - s.y_min if s.yrange else 1.0
    sigma0 = s.span / 6.0 if s.span else 1.0
    fwhm0 = sigma0 / 0.42466
    return [amp, s.x_at_max, max(fwhm0, 1e-6)]


def _guess_exgaussian(s):
    amp = s.y_max - s.y_min if s.yrange else 1.0
    return [amp, s.x_at_max, max(s.span / 8.0, 1e-6), max(s.span / 4.0, 1e-6)]


def _guess_stretched_exponential(s):
    amp = s.y[0] if s.y.size else 1.0
    return [float(amp), max(s.span, 1.0), 1.0, float(s.y_min)]


def _guess_step(s):
    return [s.yrange or 1.0, s.x_mean, max(s.span / 4.0, 1e-6), float(s.y_mean)]


def _guess_rational(s):
    return [1.0, s.x_mean]


def _guess_sinc(s):
    amp = s.y_max - s.y_min if s.yrange else 1.0
    return [amp, s.x_at_max, max(s.span / 6.0, 1e-6)]


def _guess_exponential_rise(s):
    amp = s.y_max - s.y_min if s.yrange else 1.0
    return [amp, max(s.span / 3.0, 1e-6), float(s.y_min)]


def _guess_double_exponential(s):
    amp1 = s.y[0] if s.y.size else 1.0
    return [float(amp1), max(s.span / 4.0, 1e-6), float(amp1) * 0.5, max(s.span, 1e-6)]


def _guess_moffat(s):
    amp = s.y_max - s.y_min if s.yrange else 1.0
    return [amp, s.x_at_max, max(s.span / 6.0, 1e-6), 2.0]


def _guess_gaussian_baseline(s):
    sigma0 = s.span / 6.0 if s.span else 1.0
    amp = s.y_max - s.y_min if s.yrange else 1.0
    if s.x.size >= 2:
        slope, intercept = np.polyfit(s.x, s.y, 1)
        return [amp, s.x_at_max, max(sigma0, 1e-6), float(slope), float(intercept)]
    return [amp, s.x_at_max, max(sigma0, 1e-6), 0.0, float(s.y_mean)]


def _guess_bimodal_gaussian(s):
    sigma0 = s.span / 6.0 if s.span else 1.0
    amp = (s.y_max - s.y_min) / 2.0 if s.yrange else 1.0
    return [amp, s.x_mean - s.span / 4.0, max(sigma0, 1e-6), amp, s.x_mean + s.span / 4.0, max(sigma0, 1e-6)]


def _guess_quartic(s):
    coeffs = np.polyfit(s.x, s.y, 4) if s.x.size >= 5 else [0.0, 0.0, 0.0, 1.0, s.y_mean]
    return [float(v) for v in coeffs]


def _guess_quintic(s):
    coeffs = np.polyfit(s.x, s.y, 5) if s.x.size >= 6 else [0.0, 0.0, 0.0, 0.0, 1.0, s.y_mean]
    return [float(v) for v in coeffs]


_GUESSERS = {
    "linear": _guess_linear,
    "quadratic": _guess_quadratic,
    "cubic": _guess_cubic,
    "constant": _guess_constant,
    "gaussian": _guess_gaussian,
    "lorentzian": _guess_lorentzian,
    "exponential": _guess_exponential,
    "power_law": _guess_power_law,
    "logistic": _guess_logistic,
    "damped_oscillator": _guess_damped_oscillator,
    "damped_sine": _guess_damped_sine,
    "sine": _guess_sine,
    "cosine": _guess_cosine,
    "beat": _guess_beat,
    "voigt": _guess_voigt,
    "skew_normal": _guess_skew_normal,
    "gaussian_fwhm": _guess_fwhm,
    "lorentzian_fwhm": _guess_fwhm,
    "exgaussian": _guess_exgaussian,
    "stretched_exponential": _guess_stretched_exponential,
    "tanh": _guess_step,
    "arctan": _guess_step,
    "rational": _guess_rational,
    "sinc": _guess_sinc,
    "exponential_rise": _guess_exponential_rise,
    "double_exponential": _guess_double_exponential,
    "moffat": _guess_moffat,
    "gaussian_baseline": _guess_gaussian_baseline,
    "bimodal_gaussian": _guess_bimodal_gaussian,
    "quartic": _guess_quartic,
    "quintic": _guess_quintic,
}


def _initial_guess(model, name, x, y, p0=None):
    param_names = model_param_names(model)
    if p0 is not None:
        if isinstance(p0, dict):
            return [float(p0.get(p, 1.0)) for p in param_names]
        arr = np.asarray(p0, dtype=float)
        if arr.ndim == 0:
            return [float(arr)]
        return arr.astype(float).tolist()

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    span = float(np.ptp(x)) if x.size else 1.0
    yrange = float(np.ptp(y)) if y.size else 1.0
    y_min = float(np.min(y)) if y.size else 0.0
    y_max = float(np.max(y)) if y.size else 1.0
    y_mean = float(np.mean(y)) if y.size else 0.0
    x_mean = float(np.mean(x)) if x.size else 0.0
    x_at_max = float(x[int(np.argmax(y))]) if y.size else 0.0

    stats = _GuessStats(x, y, span, yrange, y_min, y_max, y_mean, x_mean, x_at_max)
    guesser = _GUESSERS.get(name)
    if guesser is not None:
        return guesser(stats)
    return [1.0] * len(param_names)


def _coerce_bounds(bounds, param_names):
    if bounds is None:
        n = len(param_names)
        return (np.full(n, -np.inf, dtype=float), np.full(n, np.inf, dtype=float))
    if isinstance(bounds, dict):
        lower = []
        upper = []
        for name in param_names:
            lo, hi = bounds.get(name, (-np.inf, np.inf))
            lower.append(-np.inf if lo is None else lo)
            upper.append(np.inf if hi is None else hi)
        return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if isinstance(bounds, (tuple, list)) and len(bounds) == 2:
        lo, hi = bounds
        return np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    raise TypeError("bounds must be None, a (lower, upper) pair, or a dict keyed by parameter name")


def _normalize_sigma(
    series: DataSeries, sigma=None, weights=None, sigma_low=None, sigma_high=None, sigma_cov=None
):
    if sigma is None and series.effective_sigma is not None:
        sigma = series.effective_sigma
    if sigma_low is None and series.sigma_low is not None:
        sigma_low = series.sigma_low
    if sigma_high is None and series.sigma_high is not None:
        sigma_high = series.sigma_high
    if sigma_cov is None and series.sigma_cov is not None:
        sigma_cov = series.sigma_cov
    sigma = effective_sigma(sigma=sigma, sigma_low=sigma_low, sigma_high=sigma_high, sigma_cov=sigma_cov)
    if weights is not None:
        if sigma is not None:
            raise ValueError("Pass either sigma or weights, not both")
        weights = np.asarray(weights, dtype=float)
        sigma = np.where(weights > 0, 1.0 / np.sqrt(weights), np.inf)
    if sigma_cov is not None:
        sigma_cov = np.asarray(sigma_cov, dtype=float)
    return sigma, sigma_cov


def _model_wrapper(model):
    fn, model_name = get_model(model)
    if callable(model) and not isinstance(model, str):
        model_name = getattr(model, "__name__", "custom")
    param_names = model_param_names(fn)
    return fn, model_name, param_names


def _fit_single(
    series: DataSeries,
    *,
    model="linear",
    p0=None,
    bounds=None,
    sigma=None,
    weights=None,
    sigma_low=None,
    sigma_high=None,
    sigma_cov=None,
) -> FitResult:
    fn, model_name, param_names = _model_wrapper(model)
    x = np.asarray(series.x, dtype=float)
    y = np.asarray(series.y, dtype=float)
    sigma, sigma_cov = _normalize_sigma(
        series,
        sigma=sigma,
        weights=weights,
        sigma_low=sigma_low,
        sigma_high=sigma_high,
        sigma_cov=sigma_cov,
    )
    p0_vec = np.asarray(_initial_guess(model, model_name, x, y, p0=p0), dtype=float)
    if p0_vec.size != len(param_names):
        p0_vec = np.resize(p0_vec, len(param_names)).astype(float)
    lb, ub = _coerce_bounds(bounds, param_names)

    if sigma_cov is not None:
        sigma_cov = np.asarray(sigma_cov, dtype=float)
        chol = np.linalg.cholesky(sigma_cov)

        def residuals(params):
            r = y - fn(x, *params)
            return np.linalg.solve(chol, r)

    else:
        sigma_vec = None if sigma is None else np.asarray(sigma, dtype=float)

        def residuals(params):
            r = y - fn(x, *params)
            if sigma_vec is None:
                return r
            return r / sigma_vec

    opt = least_squares(residuals, p0_vec, bounds=(lb, ub), method="trf")
    popt = opt.x
    y_fit = np.asarray(fn(x, *popt), dtype=float)
    residual = y - y_fit
    dof = max(int(x.size - popt.size), 1)
    is_weighted = sigma is not None or sigma_cov is not None
    if sigma_cov is not None:
        chi2 = float(np.sum(np.square(np.linalg.solve(chol, residual))))
    elif sigma is not None:
        sigma_vec = np.asarray(sigma, dtype=float)
        chi2 = float(np.sum(np.square(residual / sigma_vec)))
    else:
        chi2 = float(np.sum(np.square(residual)))
    reduced_chi2 = chi2 / dof

    if opt.jac is not None and opt.jac.size:
        jtj = opt.jac.T @ opt.jac
        cov = np.linalg.pinv(jtj) * reduced_chi2
    else:
        cov = np.full((popt.size, popt.size), np.nan)

    diag_cov = np.diag(cov) if cov.ndim == 2 else np.asarray([])
    if np.any(~np.isfinite(diag_cov)):
        warnings.warn(
            "Covariance matrix is singular; parameter uncertainties are unreliable.",
            UserWarning,
            stacklevel=3,
        )

    params = {name: float(value) for name, value in zip(param_names, popt, strict=True)}
    uncertainties = {
        name: float(np.sqrt(max(float(value), 0.0)))
        for name, value in zip(param_names, diag_cov, strict=True)
    }
    p_value = float(chi2_dist.sf(chi2, dof)) if dof > 0 and np.isfinite(chi2) else float("nan")

    # ── actionable warnings for students ─────────────────────
    if not is_weighted:
        warnings.warn(
            "No uncertainties provided; 'reduced_chi2' is the unweighted SSR per "
            "degree of freedom, not a true reduced χ².",
            UserWarning,
            stacklevel=3,
        )
    if not opt.success:
        warnings.warn(
            f"Fit '{model_name}' did NOT converge. "
            f"Optimiser message: {opt.message}. "
            f"Try providing better initial guesses (p0), adding parameter bounds, "
            f"or checking whether '{model_name}' is the right model.",
            UserWarning,
            stacklevel=3,
        )
    elif is_weighted and reduced_chi2 > 10:
        warnings.warn(
            f"reduced chi2 = {reduced_chi2:.1f} is much larger than 1. "
            f"The model '{model_name}' may not describe the data well, "
            f"or the measurement uncertainties may be underestimated. "
            f"Inspect the residuals and check your model choice.",
            UserWarning,
            stacklevel=3,
        )

    return FitResult(
        reduced_chi2=float(reduced_chi2),
        params=params,
        covariance=cov,
        p_value=p_value,
        uncertainties=uncertainties,
        success=bool(opt.success),
        message=str(opt.message),
        model_name=model_name,
        param_names=tuple(param_names),
        x=x,
        y=y,
        sigma=None if sigma is None else np.asarray(sigma, dtype=float),
        y_fit=y_fit,
        series=series,
        model=fn,
        is_weighted=is_weighted,
    )


def fit(
    x,
    y=None,
    *,
    model="linear",
    p0=None,
    bounds=None,
    sigma=None,
    weights=None,
    sigma_low=None,
    sigma_high=None,
    sigma_cov=None,
    label="",
    **kwargs,
):
    """Fit a model to data and return parameter estimates with uncertainties.

    This is the main entry point. The first argument can be arrays,
    a :class:`~labfit.DataSeries`, or a path to a CSV file — the
    function adapts automatically.

    Parameters
    ----------
    x : array-like or DataSeries or Path or str
        x-values for the data. If a :class:`~labfit.DataSeries` or a CSV
        path is passed, ``y`` and the error columns are read from it.
    y : array-like, optional
        y-values (ignored if ``x`` is a DataSeries or CSV path).
    model : str or callable, default ``"linear"``
        Built-in model name (``"gaussian"``, ``"exponential"``, …) or a
        custom callable ``f(x, *params)``.
    p0 : dict or array-like, optional
        Initial parameter guesses. Dict keys must match the parameter
        names (e.g. ``{"amplitude": 1.0, "mean": 0.0}``).
    bounds : dict or tuple of arrays, optional
        Parameter bounds. Either a dict ``{"name": (lo, hi)}`` or a
        ``(lower, upper)`` tuple of arrays.
    sigma : array-like or AsymmetricError, optional
        1-σ uncertainties for each y-value.
    weights : array-like, optional
        Inverse-variance weights (alternative to ``sigma``).
    sigma_low, sigma_high : array-like, optional
        Asymmetric lower and upper uncertainties.
    sigma_cov : array-like, optional
        Full covariance matrix for correlated measurement errors.
    label : str, optional
        Label for the data series (used in plot legends).

    Returns
    -------
    FitResult
        Container with ``params``, ``uncertainties``, ``reduced_chi2``,
        ``p_value``, and convenience methods like ``predict(x)`` and
        ``residuals``.
    """
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected keyword arguments: {unexpected}")

    series = _coerce_series_input(
        x, y, sigma=sigma, sigma_low=sigma_low, sigma_high=sigma_high, sigma_cov=sigma_cov, label=label
    )
    return _fit_single(
        series,
        model=model,
        p0=p0,
        bounds=bounds,
        sigma=sigma,
        weights=weights,
        sigma_low=sigma_low,
        sigma_high=sigma_high,
        sigma_cov=sigma_cov,
    )


def fit_multi(dataset: Iterable[DataSeries] | Dataset, *, model="linear", p0=None, bounds=None, **kwargs):
    """Fit the same model to every series in a collection.

    Parameters
    ----------
    dataset : Dataset or iterable of DataSeries
        The data series to fit. Each series can carry its own label
        and error specification.
    model : str or callable, default ``"linear"``
        Model name or custom callable.
    p0 : dict or array-like, optional
        Shared initial parameter guesses for all series.
    bounds : dict or tuple of arrays, optional
        Shared parameter bounds for all series.

    Returns
    -------
    list of FitResult
        One result per input series.
    """
    series_list = list(dataset)
    return [_fit_single(series, model=model, p0=p0, bounds=bounds, **kwargs) for series in series_list]


__all__ = ["fit", "fit_multi"]
