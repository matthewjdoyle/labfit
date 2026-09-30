from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple, TypedDict

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import least_squares
from scipy.stats import chi2 as chi2_dist

from .io import load_csv as _load_csv
from .models import get_model, model_param_names
from .types import (
    AsymmetricError,
    Bounds,
    DataSeries,
    Dataset,
    FitInput,
    FitResult,
    InitialGuess,
    MaybeArray,
    ModelFunction,
    ModelSpec,
    _normalize_uncertainties,
)


def _coerce_series_input(
    x: FitInput,
    y: MaybeArray | None = None,
    *,
    sigma: MaybeArray | AsymmetricError | None = None,
    weights: MaybeArray | None = None,
    sigma_low: MaybeArray | None = None,
    sigma_high: MaybeArray | None = None,
    sigma_cov: np.ndarray | Sequence[Sequence[float]] | None = None,
    label: str = "",
) -> DataSeries:
    if isinstance(x, DataSeries) and y is None:
        return x
    if isinstance(x, str | Path) and y is None:
        explicit_errors = any(
            value is not None for value in (sigma, weights, sigma_low, sigma_high, sigma_cov)
        )
        series = _load_csv(x, error_mode="unweighted" if explicit_errors else "auto")
        return series.with_label(label)
    if y is None:
        raise TypeError("fit requires either (x, y) arrays or a DataSeries / CSV path")
    return DataSeries(
        x=np.asarray(x, dtype=float),
        y=y,
        sigma=sigma,
        sigma_low=sigma_low,
        sigma_high=sigma_high,
        sigma_cov=sigma_cov,
        label=label,
    )


class _GuessStats(NamedTuple):
    x: np.ndarray
    y: np.ndarray
    span: float
    yrange: float
    y_min: float
    y_max: float
    y_mean: float
    x_mean: float
    x_at_max: float


class _UncertaintySpec(TypedDict):
    sigma: MaybeArray | AsymmetricError | None
    weights: MaybeArray | None
    sigma_low: MaybeArray | None
    sigma_high: MaybeArray | None
    sigma_cov: np.ndarray | Sequence[Sequence[float]] | None


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


def _guess_linear(s: _GuessStats) -> list[float]:
    if s.x.size >= 2:
        slope, intercept = np.polyfit(s.x, s.y, 1)
        return [float(slope), float(intercept)]
    return [1.0, s.y_mean]


def _guess_quadratic(s: _GuessStats) -> list[float]:
    coeffs = np.polyfit(s.x, s.y, 2) if s.x.size >= 3 else [0.0, 1.0, s.y_mean]
    return [float(v) for v in coeffs]


def _guess_cubic(s: _GuessStats) -> list[float]:
    coeffs = np.polyfit(s.x, s.y, 3) if s.x.size >= 4 else [0.0, 0.0, 1.0, s.y_mean]
    return [float(v) for v in coeffs]


def _guess_constant(s: _GuessStats) -> list[float]:
    return [s.y_mean]


def _guess_gaussian(s: _GuessStats) -> list[float]:
    sigma0 = s.span / 6.0 if s.span else 1.0
    amp = s.y_max - s.y_min if s.yrange else 1.0
    return [amp, s.x_at_max, max(sigma0, 1e-6)]


def _guess_lorentzian(s: _GuessStats) -> list[float]:
    amp = s.y_max - s.y_min if s.yrange else 1.0
    return [amp, s.x_at_max, max(s.span / 10.0, 1e-6)]


def _guess_exponential(s: _GuessStats) -> list[float]:
    amp = s.y[0] if s.y.size else 1.0
    decay = 1.0 / max(s.span, 1.0)
    positive = s.y > 0
    if positive.sum() >= 2 and s.x.size >= 2:
        slope, intercept = np.polyfit(s.x[positive], np.log(s.y[positive]), 1)
        amp = float(np.exp(intercept))
        decay = float(-slope)
    return [float(amp), float(decay)]


def _guess_power_law(s: _GuessStats) -> list[float]:
    amp = max(s.y_max, 1e-6)
    exponent = 1.0
    positive = (s.x > 0) & (s.y > 0)
    if positive.sum() >= 2:
        slope, intercept = np.polyfit(np.log(s.x[positive]), np.log(s.y[positive]), 1)
        amp = float(np.exp(intercept))
        exponent = float(slope)
    return [float(amp), float(exponent)]


def _guess_logistic(s: _GuessStats) -> list[float]:
    return [s.yrange or 1.0, s.x_mean, 1.0 / max(s.span, 1.0), s.y_min]


def _guess_damped_oscillator(s: _GuessStats) -> list[float]:
    freq = _osc_freq(s)
    amp = max(abs(s.y_min), abs(s.y_max), 1.0)
    return [float(amp), 0.1 / max(s.span, 1.0), float(freq), 0.0]


def _guess_damped_sine(s: _GuessStats) -> list[float]:
    freq = _osc_freq(s)
    amp = max(abs(s.y_min), abs(s.y_max), 1.0)
    return [float(amp), 0.1 / max(s.span, 1.0), float(freq), 0.0, float(s.y_mean)]


def _guess_sine(s: _GuessStats) -> list[float]:
    freq = _osc_freq(s)
    amp = max(abs(s.y_min), abs(s.y_max), 1.0)
    return [float(amp), float(freq), 0.0, float(s.y_mean)]


def _guess_cosine(s: _GuessStats) -> list[float]:
    return _guess_sine(s)


def _guess_beat(s: _GuessStats) -> list[float]:
    freq = _osc_freq(s)
    amp = max(abs(s.y_min), abs(s.y_max), 1.0)
    return [float(amp), float(freq), float(freq * 1.08), 0.0, float(s.y_mean)]


def _guess_voigt(s: _GuessStats) -> list[float]:
    amp = s.y_max - s.y_min if s.yrange else 1.0
    return [amp, s.x_at_max, max(s.span / 6.0, 1e-6), max(s.span / 10.0, 1e-6)]


def _guess_skew_normal(s: _GuessStats) -> list[float]:
    amp = s.y_max - s.y_min if s.yrange else 1.0
    return [amp, s.x_at_max, max(s.span / 6.0, 1e-6), 0.0]


def _guess_fwhm(s: _GuessStats) -> list[float]:
    amp = s.y_max - s.y_min if s.yrange else 1.0
    sigma0 = s.span / 6.0 if s.span else 1.0
    fwhm0 = sigma0 / 0.42466
    return [amp, s.x_at_max, max(fwhm0, 1e-6)]


def _guess_exgaussian(s: _GuessStats) -> list[float]:
    amp = s.y_max - s.y_min if s.yrange else 1.0
    return [amp, s.x_at_max, max(s.span / 8.0, 1e-6), max(s.span / 4.0, 1e-6)]


def _guess_stretched_exponential(s: _GuessStats) -> list[float]:
    amp = s.y[0] if s.y.size else 1.0
    return [float(amp), max(s.span, 1.0), 1.0, float(s.y_min)]


def _guess_step(s: _GuessStats) -> list[float]:
    return [s.yrange or 1.0, s.x_mean, max(s.span / 4.0, 1e-6), float(s.y_mean)]


def _guess_rational(s: _GuessStats) -> list[float]:
    return [1.0, s.x_mean]


def _guess_sinc(s: _GuessStats) -> list[float]:
    amp = s.y_max - s.y_min if s.yrange else 1.0
    return [amp, s.x_at_max, max(s.span / 6.0, 1e-6)]


def _guess_exponential_rise(s: _GuessStats) -> list[float]:
    amp = s.y_max - s.y_min if s.yrange else 1.0
    return [amp, max(s.span / 3.0, 1e-6), float(s.y_min)]


def _guess_double_exponential(s: _GuessStats) -> list[float]:
    amp1 = s.y[0] if s.y.size else 1.0
    return [float(amp1), max(s.span / 4.0, 1e-6), float(amp1) * 0.5, max(s.span, 1e-6)]


def _guess_moffat(s: _GuessStats) -> list[float]:
    amp = s.y_max - s.y_min if s.yrange else 1.0
    return [amp, s.x_at_max, max(s.span / 6.0, 1e-6), 2.0]


def _guess_gaussian_baseline(s: _GuessStats) -> list[float]:
    sigma0 = s.span / 6.0 if s.span else 1.0
    amp = s.y_max - s.y_min if s.yrange else 1.0
    if s.x.size >= 2:
        slope, intercept = np.polyfit(s.x, s.y, 1)
        return [amp, s.x_at_max, max(sigma0, 1e-6), float(slope), float(intercept)]
    return [amp, s.x_at_max, max(sigma0, 1e-6), 0.0, float(s.y_mean)]


def _guess_bimodal_gaussian(s: _GuessStats) -> list[float]:
    sigma0 = s.span / 6.0 if s.span else 1.0
    amp = (s.y_max - s.y_min) / 2.0 if s.yrange else 1.0
    return [amp, s.x_mean - s.span / 4.0, max(sigma0, 1e-6), amp, s.x_mean + s.span / 4.0, max(sigma0, 1e-6)]


def _guess_quartic(s: _GuessStats) -> list[float]:
    coeffs = np.polyfit(s.x, s.y, 4) if s.x.size >= 5 else [0.0, 0.0, 0.0, 1.0, s.y_mean]
    return [float(v) for v in coeffs]


def _guess_quintic(s: _GuessStats) -> list[float]:
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


def _validate_parameter_keys(
    values: Mapping[str, object], param_names: tuple[str, ...], argument: str
) -> None:
    unknown = set(values) - set(param_names)
    if unknown:
        unexpected = ", ".join(sorted(map(str, unknown)))
        valid = ", ".join(param_names)
        raise ValueError(f"Unknown {argument} parameters: {unexpected}. Valid parameters: {valid}")


def _initial_guess(
    model: ModelSpec, name: str, x: np.ndarray, y: np.ndarray, p0: InitialGuess = None
) -> list[float]:
    param_names = model_param_names(model)
    if p0 is not None:
        if isinstance(p0, dict):
            _validate_parameter_keys(p0, param_names, "p0")
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


def _coerce_bounds(bounds: Bounds, param_names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    if bounds is None:
        n = len(param_names)
        return (np.full(n, -np.inf, dtype=float), np.full(n, np.inf, dtype=float))
    if isinstance(bounds, dict):
        _validate_parameter_keys(bounds, param_names, "bounds")
        lower = []
        upper = []
        for name in param_names:
            lo, hi = bounds.get(name, (-np.inf, np.inf))
            lower.append(-np.inf if lo is None else lo)
            upper.append(np.inf if hi is None else hi)
        return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if isinstance(bounds, tuple | list) and len(bounds) == 2:
        lower_values, upper_values = bounds
        return np.asarray(lower_values, dtype=float), np.asarray(upper_values, dtype=float)
    raise TypeError("bounds must be None, a (lower, upper) pair, or a dict keyed by parameter name")


def _uncertainty_spec(
    series: DataSeries,
    *,
    sigma: MaybeArray | AsymmetricError | None = None,
    weights: MaybeArray | None = None,
    sigma_low: MaybeArray | None = None,
    sigma_high: MaybeArray | None = None,
    sigma_cov: np.ndarray | Sequence[Sequence[float]] | None = None,
) -> _UncertaintySpec:
    supplied: _UncertaintySpec = dict(
        sigma=sigma, weights=weights, sigma_low=sigma_low, sigma_high=sigma_high, sigma_cov=sigma_cov
    )
    if any(value is not None for value in supplied.values()):
        return supplied
    return dict(
        sigma=series.sigma,
        weights=None,
        sigma_low=series.sigma_low,
        sigma_high=series.sigma_high,
        sigma_cov=series.sigma_cov,
    )


def _normalize_sigma(
    series: DataSeries,
    sigma: MaybeArray | AsymmetricError | None = None,
    weights: MaybeArray | None = None,
    sigma_low: MaybeArray | None = None,
    sigma_high: MaybeArray | None = None,
    sigma_cov: np.ndarray | Sequence[Sequence[float]] | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    specification = _uncertainty_spec(
        series, sigma=sigma, weights=weights, sigma_low=sigma_low, sigma_high=sigma_high, sigma_cov=sigma_cov
    )
    return _normalize_uncertainties(series.x.size, **specification)


def _model_wrapper(model: ModelSpec) -> tuple[ModelFunction, str, tuple[str, ...]]:
    fn, model_name = get_model(model)
    if callable(model) and not isinstance(model, str):
        model_name = getattr(model, "__name__", "custom")
    param_names = model_param_names(fn)
    return fn, model_name, param_names


def _fit_single(
    series: DataSeries,
    *,
    model: ModelSpec = "linear",
    p0: InitialGuess = None,
    bounds: Bounds = None,
    sigma: MaybeArray | AsymmetricError | None = None,
    weights: MaybeArray | None = None,
    sigma_low: MaybeArray | None = None,
    sigma_high: MaybeArray | None = None,
    sigma_cov: np.ndarray | Sequence[Sequence[float]] | None = None,
    absolute_sigma: bool = False,
) -> FitResult:
    fn, model_name, param_names = _model_wrapper(model)
    inference = (
        series.uncertainty_inference
        if all(value is None for value in (sigma, weights, sigma_low, sigma_high, sigma_cov))
        else None
    )
    data = DataSeries(series.x, series.y, label=series.label)
    x, y = data.x, data.y
    specification = _uncertainty_spec(
        series, sigma=sigma, weights=weights, sigma_low=sigma_low, sigma_high=sigma_high, sigma_cov=sigma_cov
    )
    sigma, sigma_cov = _normalize_sigma(series, **specification)
    if specification["weights"] is not None:
        assert sigma is not None
        sigma = np.broadcast_to(sigma, x.shape)
        included = np.isfinite(sigma)
        if not np.any(included):
            raise ValueError("weights must retain at least one data point")
        x, y, sigma = x[included], y[included], sigma[included]
        series = DataSeries(x, y, sigma=sigma, label=series.label)
    else:
        series = DataSeries(
            x,
            y,
            sigma=specification["sigma"],
            sigma_low=specification["sigma_low"],
            sigma_high=specification["sigma_high"],
            sigma_cov=sigma_cov,
            label=series.label,
            uncertainty_inference=inference,
        )
    p0_vec = np.asarray(_initial_guess(model, model_name, x, y, p0=p0), dtype=float)
    if p0_vec.ndim != 1 or p0_vec.size != len(param_names):
        raise ValueError(
            f"p0 must contain exactly {len(param_names)} values in a one-dimensional array "
            f"for parameters: {', '.join(param_names)}"
        )
    lb, ub = _coerce_bounds(bounds, param_names)

    if sigma_cov is not None:
        sigma_cov = np.asarray(sigma_cov, dtype=float)
        chol = np.linalg.cholesky(sigma_cov)

        def residuals(params: np.ndarray) -> np.ndarray:
            r = y - np.asarray(fn(x, *params), dtype=float)
            return solve_triangular(chol, r, lower=True)

    else:
        sigma_vec = None if sigma is None else np.asarray(sigma, dtype=float)

        def residuals(params: np.ndarray) -> np.ndarray:
            r = y - np.asarray(fn(x, *params), dtype=float)
            if sigma_vec is None:
                return r
            return r / sigma_vec

    opt = least_squares(residuals, p0_vec, bounds=(lb, ub), method="trf")
    popt = opt.x
    y_fit = np.asarray(fn(x, *popt), dtype=float)
    residual = y - y_fit
    dof = int(x.size - popt.size)
    is_weighted = sigma is not None or sigma_cov is not None
    if sigma_cov is not None:
        chi2 = float(np.sum(np.square(solve_triangular(chol, residual, lower=True))))
    elif sigma is not None:
        sigma_vec = np.asarray(sigma, dtype=float)
        chi2 = float(np.sum(np.square(residual / sigma_vec)))
    else:
        chi2 = float(np.sum(np.square(residual)))
    residual_variance = float(residual @ residual) / dof if dof > 0 else float("nan")
    objective_variance = chi2 / dof if dof > 0 else float("nan")
    reduced_chi2 = objective_variance if is_weighted else float("nan")

    # Normalize columns before testing conditioning so a change of parameter
    # units alone does not make an identifiable model appear ill-conditioned.
    jacobian_rank = 0
    jacobian_condition = float("inf")
    identifiable = False
    if opt.jac is not None and opt.jac.size and np.all(np.isfinite(opt.jac)):
        column_norms = np.linalg.norm(opt.jac, axis=0)
        if np.all(np.isfinite(column_norms)):
            scaled_jac = opt.jac / np.where(column_norms > 0, column_norms, 1.0)
            singular_values = np.linalg.svd(scaled_jac, compute_uv=False)
            tolerance = np.finfo(float).eps * max(scaled_jac.shape) * singular_values[0]
            jacobian_rank = int(np.count_nonzero(singular_values > tolerance))
            if singular_values.size == popt.size and singular_values[-1] > 0:
                jacobian_condition = float(singular_values[0] / singular_values[-1])
            # Numerical derivatives cannot resolve directions much smaller
            # than sqrt(machine epsilon) relative to the strongest direction.
            identifiable = jacobian_rank == popt.size and jacobian_condition < 1.0 / np.sqrt(
                np.finfo(float).eps
            )

    if identifiable:
        jtj = opt.jac.T @ opt.jac
        cov = np.linalg.pinv(jtj)
        if not np.all(np.isfinite(cov)):
            identifiable = False
        elif not absolute_sigma:
            # Residual scaling needs positive degrees of freedom, independently
            # of whether the Jacobian identifies the parameters.
            cov = cov * objective_variance if dof > 0 else np.full_like(cov, np.nan)
    if not identifiable:
        cov = np.full((popt.size, popt.size), np.nan)
        warnings.warn(
            f"Fit '{model_name}': optimizer success={bool(opt.success)}; parameters are not identifiable "
            f"(Jacobian rank {jacobian_rank}/{popt.size}, condition {jacobian_condition:.3g}); "
            "parameter covariance and uncertainties are unreliable.",
            UserWarning,
            stacklevel=3,
        )

    diag_cov = np.diag(cov)
    params = {name: float(value) for name, value in zip(param_names, popt, strict=True)}
    uncertainties = {
        name: float(np.sqrt(max(float(value), 0.0)))
        for name, value in zip(param_names, diag_cov, strict=True)
    }
    p_value = (
        float(chi2_dist.sf(chi2, dof))
        if is_weighted and dof > 0 and identifiable and opt.success and np.isfinite(chi2)
        else float("nan")
    )

    # ── actionable warnings for students ─────────────────────
    if not is_weighted:
        warnings.warn(
            "No uncertainties provided; chi-square statistics are unavailable. "
            "Use 'residual_variance' for the unweighted SSR per degree of freedom.",
            UserWarning,
            stacklevel=3,
        )
    if dof <= 0:
        warnings.warn(
            f"Fit '{model_name}' has {dof} degrees of freedom (N - k); "
            "residual variance, reduced chi-square and p-value are unavailable. "
            "Relative parameter covariance cannot be estimated; provide more data "
            "or use fewer parameters.",
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
    elif is_weighted and identifiable and reduced_chi2 > 10:
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
        x=np.asarray(x, dtype=float),
        y=y,
        sigma=None if sigma is None else np.asarray(sigma, dtype=float),
        y_fit=y_fit,
        series=series,
        model=fn,
        is_weighted=is_weighted,
        absolute_sigma=absolute_sigma,
        identifiable=identifiable,
        jacobian_rank=jacobian_rank,
        jacobian_condition=jacobian_condition,
        dof=dof,
        residual_variance=residual_variance,
    )


def fit(
    x: FitInput,
    y: MaybeArray | None = None,
    *,
    model: ModelSpec = "linear",
    p0: InitialGuess = None,
    bounds: Bounds = None,
    sigma: MaybeArray | AsymmetricError | None = None,
    weights: MaybeArray | None = None,
    sigma_low: MaybeArray | None = None,
    sigma_high: MaybeArray | None = None,
    sigma_cov: np.ndarray | Sequence[Sequence[float]] | None = None,
    absolute_sigma: bool = False,
    label: str = "",
    **kwargs: Any,
) -> FitResult:
    """Fit a model to data and return parameter estimates with uncertainties.

    This is the main entry point. The first argument can be arrays,
    a :class:`~labfit.DataSeries`, or a path to a CSV file — the
    function adapts automatically.

    Parameters
    ----------
    x : array-like or DataSeries or Path or str
        x-values for the data. If a :class:`~labfit.DataSeries` or a CSV
        path is passed, ``y`` and the error columns are read from it. CSVs
        without errors infer heuristic uncertainties with a warning and record
        the assumption on result.series. To load without errors or inference,
        pass ``load_csv(path, error_mode="unweighted")``. Explicit fit errors
        bypass file error columns and inference.
    y : array-like, optional
        y-values (ignored if ``x`` is a DataSeries or CSV path).
    model : str or callable, default ``"linear"``
        Built-in model name (``"gaussian"``, ``"exponential"``, …) or a
        custom callable ``f(x, *params)``.
    p0 : dict or array-like, optional
        Initial parameter guesses. Dict keys must match the parameter
        names (e.g. ``{"amplitude": 1.0, "mean": 0.0}``). Missing dict
        entries default to 1.0; unknown keys are rejected. Arrays must have
        exactly one value per parameter, in model signature order.
    bounds : dict or tuple of arrays, optional
        Parameter bounds. Either a dict ``{"name": (lo, hi)}`` or a
        ``(lower, upper)`` tuple of arrays. Omitted dict entries are
        unbounded; unknown keys are rejected.
    sigma : array-like or AsymmetricError, optional
        Finite, strictly positive 1-σ errors, as a scalar or 1D vector matching
        the data. Supply exactly one of sigma, weights, asymmetric pairs or
        sigma_cov. Explicit errors replace the entire stored DataSeries/CSV
        error specification; an asymmetric override requires both sides.
    weights : array-like, optional
        Finite nonnegative inverse-variance weights, as a scalar or vector
        matching the data. Zero weights exclude samples from fitting, stored
        result data, plots and all statistics; at least one weight must be positive.
    sigma_low, sigma_high : array-like, optional
        Asymmetric lower and upper uncertainties, supplied together. Both
        sides must be finite and strictly positive scalars or matching vectors.
    sigma_cov : array-like, optional
        Finite symmetric positive-definite covariance matrix with shape (N, N)
        for correlated measurement errors. Symmetry uses relative tolerance 1e-12;
        accepted rounding differences are symmetrized before factorization.
    absolute_sigma : bool, default False
        If True, use the supplied uncertainties as absolute measurement errors
        without rescaling parameter covariance by reduced chi-square. If False,
        treat them as relative weights and estimate the common noise scale from
        the residuals. Applies to sigma, weights, asymmetric errors (via their
        effective sigma), and sigma_cov. With no errors, True assumes unit
        standard deviation; False estimates it from residuals.
    label : str, optional
        Label for the data series (used in plot legends).

    Returns
    -------
    FitResult
        Container with ``params``, ``uncertainties``, ``reduced_chi2``,
        ``p_value``, ``dof`` (N minus the number of parameters), and
        ``residual_variance`` (unweighted SSR / dof, in squared y units).
        Reduced chi-square and p-value are NaN without supplied errors or
        positive dof; p-value also requires convergence and identifiability.
        Residual variance and relative parameter covariance are unavailable
        when dof <= 0. Convenience methods include ``predict(x)`` and ``residuals``.
    """
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected keyword arguments: {unexpected}")

    series = _coerce_series_input(
        x,
        y,
        sigma=sigma,
        weights=weights,
        sigma_low=sigma_low,
        sigma_high=sigma_high,
        sigma_cov=sigma_cov,
        label=label,
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
        absolute_sigma=absolute_sigma,
    )


def fit_multi(
    dataset: Iterable[DataSeries] | Dataset,
    *,
    model: ModelSpec = "linear",
    p0: InitialGuess = None,
    bounds: Bounds = None,
    absolute_sigma: bool = False,
    **kwargs: Any,
) -> list[FitResult]:
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
    absolute_sigma : bool, default False
        Covariance-scaling policy for every series; see :func:`fit`.

    Returns
    -------
    list of FitResult
        One result per input series.
    """
    series_list = list(dataset)
    return [
        _fit_single(series, model=model, p0=p0, bounds=bounds, absolute_sigma=absolute_sigma, **kwargs)
        for series in series_list
    ]


__all__ = ["fit", "fit_multi"]
