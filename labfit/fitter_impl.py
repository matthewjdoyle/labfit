from __future__ import annotations

from pathlib import Path
from typing import Iterable

import warnings

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2 as chi2_dist

from .models import get_model, model_param_names
from .types import DataSeries, Dataset, FitResult
from .utils import effective_sigma, load_csv


def _coerce_series_input(x, y=None, *, sigma=None, sigma_low=None, sigma_high=None, sigma_cov=None, label=""):
    if isinstance(x, DataSeries) and y is None:
        return x
    if isinstance(x, (str, Path)) and y is None:
        x_arr, y_arr, sig, sig_low, sig_high = load_csv(x)
        return DataSeries(
            x=x_arr,
            y=y_arr,
            sigma=sigma if sigma is not None else sig,
            sigma_low=sigma_low if sigma_low is not None else sig_low,
            sigma_high=sigma_high if sigma_high is not None else sig_high,
            sigma_cov=sigma_cov,
            label=label,
        )
    if y is None:
        raise TypeError("fit requires either (x, y) arrays or a DataSeries / CSV path")
    return DataSeries(x=x, y=y, sigma=sigma, sigma_low=sigma_low, sigma_high=sigma_high, sigma_cov=sigma_cov, label=label)


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

    if name == "linear":
        if x.size >= 2:
            slope, intercept = np.polyfit(x, y, 1)
            return [float(slope), float(intercept)]
        return [1.0, y_mean]
    if name == "quadratic":
        coeffs = np.polyfit(x, y, 2) if x.size >= 3 else [0.0, 1.0, y_mean]
        return [float(v) for v in coeffs]
    if name == "cubic":
        coeffs = np.polyfit(x, y, 3) if x.size >= 4 else [0.0, 0.0, 1.0, y_mean]
        return [float(v) for v in coeffs]
    if name == "constant":
        return [y_mean]
    if name == "gaussian":
        sigma0 = span / 6.0 if span else 1.0
        amp = y_max - y_min if yrange else 1.0
        return [amp, x_at_max, max(sigma0, 1e-6)]
    if name == "lorentzian":
        amp = y_max - y_min if yrange else 1.0
        return [amp, x_at_max, max(span / 10.0, 1e-6)]
    if name == "exponential":
        amp = y[0] if y.size else 1.0
        decay = 1.0 / max(span, 1.0)
        positive = y > 0
        if positive.sum() >= 2 and x.size >= 2:
            slope, intercept = np.polyfit(x[positive], np.log(y[positive]), 1)
            amp = float(np.exp(intercept))
            decay = float(-slope)
        return [float(amp), float(decay)]
    if name == "power_law":
        amp = max(y_max, 1e-6)
        exponent = 1.0
        positive = (x > 0) & (y > 0)
        if positive.sum() >= 2:
            slope, intercept = np.polyfit(np.log(x[positive]), np.log(y[positive]), 1)
            amp = float(np.exp(intercept))
            exponent = float(slope)
        return [float(amp), float(exponent)]
    if name == "logistic":
        return [yrange or 1.0, x_mean, 1.0 / max(span, 1.0), y_min]
    if name in {"sine", "cosine", "damped_sine", "damped_oscillator", "beat"}:
        freq = 1.0 / max(span, 1.0)
        amp = max(abs(y_min), abs(y_max), 1.0)
        if x.size > 1:
            dx = np.diff(x)
            if np.all(dx > 0):
                y_detrended = y - y_mean
                zero_crossings = np.where(np.signbit(y_detrended[:-1]) != np.signbit(y_detrended[1:]))[0]
                if zero_crossings.size >= 2:
                    est_periods = np.diff(x[zero_crossings]) * 2.0
                    period = float(np.median(est_periods)) if est_periods.size else span
                    if period > 0:
                        freq = 1.0 / period
        if name == "damped_oscillator":
            return [float(amp), 0.1 / max(span, 1.0), float(freq), 0.0]
        if name == "damped_sine":
            return [float(amp), 0.1 / max(span, 1.0), float(freq), 0.0, float(y_mean)]
        if name == "sine":
            return [float(amp), float(freq), 0.0, float(y_mean)]
        if name == "cosine":
            return [float(amp), float(freq), 0.0, float(y_mean)]
        if name == "beat":
            return [float(amp), float(freq), float(freq * 1.08), 0.0, float(y_mean)]
    if name == "voigt":
        amp = y_max - y_min if yrange else 1.0
        return [amp, x_at_max, max(span / 6.0, 1e-6), max(span / 10.0, 1e-6)]
    if name == "skew_normal":
        amp = y_max - y_min if yrange else 1.0
        return [amp, x_at_max, max(span / 6.0, 1e-6), 0.0]
    if name in {"gaussian_fwhm", "lorentzian_fwhm"}:
        amp = y_max - y_min if yrange else 1.0
        sigma0 = span / 6.0 if span else 1.0
        fwhm0 = sigma0 / 0.42466
        return [amp, x_at_max, max(fwhm0, 1e-6)]
    if name == "exgaussian":
        amp = y_max - y_min if yrange else 1.0
        return [amp, x_at_max, max(span / 8.0, 1e-6), max(span / 4.0, 1e-6)]
    if name == "stretched_exponential":
        amp = y[0] if y.size else 1.0
        return [float(amp), max(span, 1.0), 1.0, float(y_min)]
    if name in {"tanh", "arctan"}:
        return [yrange or 1.0, x_mean, max(span / 4.0, 1e-6), float(y_mean)]
    if name == "rational":
        return [1.0, x_mean]
    if name == "quartic":
        coeffs = np.polyfit(x, y, 4) if x.size >= 5 else [0.0, 0.0, 0.0, 1.0, y_mean]
        return [float(v) for v in coeffs]
    if name == "quintic":
        coeffs = np.polyfit(x, y, 5) if x.size >= 6 else [0.0, 0.0, 0.0, 0.0, 1.0, y_mean]
        return [float(v) for v in coeffs]
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


def _normalize_sigma(series: DataSeries, sigma=None, weights=None, ysigma=None, sigma_low=None, sigma_high=None, sigma_cov=None):
    if ysigma is not None and sigma is None:
        sigma = ysigma
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
            raise ValueError("Pass either sigma/ysigma or weights, not both")
        weights = np.asarray(weights, dtype=float)
        sigma = np.where(weights > 0, 1.0 / np.sqrt(weights), np.inf)
    if sigma_cov is not None:
        sigma_cov = np.asarray(sigma_cov, dtype=float)
    return sigma, sigma_cov


def _model_wrapper(model, param_names):
    fn, model_name = get_model(model)
    if callable(model) and not isinstance(model, str):
        model_name = getattr(model, "__name__", "custom")
    return fn, model_name, param_names or model_param_names(fn)


def _fit_single(series: DataSeries, *, model="linear", p0=None, bounds=None, sigma=None, weights=None, ysigma=None, sigma_low=None, sigma_high=None, sigma_cov=None) -> FitResult:
    fn, model_name, param_names = _model_wrapper(model, model_param_names(model))
    x = np.asarray(series.x, dtype=float)
    y = np.asarray(series.y, dtype=float)
    sigma, sigma_cov = _normalize_sigma(series, sigma=sigma, weights=weights, ysigma=ysigma, sigma_low=sigma_low, sigma_high=sigma_high, sigma_cov=sigma_cov)
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

    params = {name: float(value) for name, value in zip(param_names, popt)}
    uncertainties = {
        name: float(np.sqrt(max(float(value), 0.0)))
        for name, value in zip(param_names, np.diag(cov))
    }
    p_value = float(chi2_dist.sf(chi2, dof)) if dof > 0 and np.isfinite(chi2) else float("nan")

    # ── actionable warnings for students ─────────────────────
    if not opt.success:
        warnings.warn(
            f"Fit '{model_name}' did NOT converge. "
            f"Optimiser message: {opt.message}. "
            f"Try providing better initial guesses (p0), adding parameter bounds, "
            f"or checking whether '{model_name}' is the right model.",
            UserWarning,
            stacklevel=3,
        )
    elif sigma is not None and reduced_chi2 > 10:
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
    )


def fit(x, y=None, *, model="linear", p0=None, bounds=None, sigma=None, weights=None, ysigma=None, sigma_low=None, sigma_high=None, sigma_cov=None, label="", **kwargs):
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
        # Preserve forwards compatibility for optional callers while keeping the core API explicit.
        if "ysigma" in kwargs and ysigma is None:
            ysigma = kwargs.pop("ysigma")
        if "sigma" in kwargs and sigma is None:
            sigma = kwargs.pop("sigma")
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected keyword arguments: {unexpected}")

    series = _coerce_series_input(x, y, sigma=sigma, sigma_low=sigma_low, sigma_high=sigma_high, sigma_cov=sigma_cov, label=label)
    return _fit_single(series, model=model, p0=p0, bounds=bounds, sigma=sigma, weights=weights, ysigma=ysigma, sigma_low=sigma_low, sigma_high=sigma_high, sigma_cov=sigma_cov)


def quick_fit(x, y=None, **kwargs):
    """Fit a model with automatic initial guesses.

    Convenience alias for :func:`fit`. All keyword arguments
    (``model``, ``sigma``, ``p0``, ``bounds``, …) are forwarded.

    Parameters
    ----------
    x : array-like or DataSeries or Path or str
        x-values, a DataSeries, or a CSV path.
    y : array-like, optional
        y-values.

    Returns
    -------
    FitResult
    """
    return fit(x, y, **kwargs)


def fit_to_model(x, y=None, **kwargs):
    """Fit a model, emphasising the model choice at the call site.

    Alias for :func:`fit` that reads naturally when you already know
    which model you want to use::

        result = fit_to_model("data.csv", model="exponential")

    Parameters
    ----------
    x : array-like or DataSeries or Path or str
        x-values, a DataSeries, or a CSV path.
    y : array-like, optional
        y-values.

    Returns
    -------
    FitResult
    """
    return fit(x, y, **kwargs)


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
    if isinstance(dataset, Dataset):
        series_list = list(dataset)
    else:
        series_list = list(dataset)
    return [
        _fit_single(series, model=model, p0=p0, bounds=bounds, **kwargs)
        for series in series_list
    ]


__all__ = ["fit", "quick_fit", "fit_to_model", "fit_multi"]
