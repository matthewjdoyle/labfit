from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .io import load_csv as _load_csv_series
from .types import AsymmetricError


def propagate_errors(func: Callable, jacobian=None, covariance=None, **params):
    value_kwargs = {
        name: value
        for name, value in params.items()
        if not name.endswith("_error") and not name.endswith("_sigma")
    }
    y = func(**value_kwargs)

    if covariance is None:
        # Backwards-compatible convenience: if the caller passes a value and an associated
        # <name>_error keyword, use a diagonal covariance. Otherwise return zero uncertainty.
        errors = []
        for name in value_kwargs:
            err = params.get(f"{name}_error")
            if err is None:
                err = params.get(f"{name}_sigma")
            if err is None:
                err = 0.0
            errors.append(float(err))
        covariance = np.diag(np.square(errors))

    covariance = np.asarray(covariance, dtype=float)

    if jacobian is None:
        raise ValueError("jacobian is required when propagating parameter uncertainties")

    grad = jacobian(**value_kwargs)
    grad = np.atleast_1d(np.asarray(grad, dtype=float))
    variance = float(grad @ covariance @ grad.T)
    return y, float(np.sqrt(max(variance, 0.0)))


def as_array(value: Any, dtype=float):
    return np.asarray(value, dtype=dtype)


def effective_sigma(sigma=None, sigma_low=None, sigma_high=None, sigma_cov=None):
    if sigma_cov is not None:
        return None
    if sigma_low is not None and sigma_high is not None:
        return np.sqrt((np.asarray(sigma_low, dtype=float) ** 2 + np.asarray(sigma_high, dtype=float) ** 2) / 2.0)
    if isinstance(sigma, AsymmetricError):
        return sigma.effective
    if sigma is None:
        return None
    return np.asarray(sigma, dtype=float)


def load_csv(path, x_col="x", y_col="y", y_err_col=None, *, default_fraction=0.05, error_mode="auto", label=""):
    series = _load_csv_series(path, x_col=x_col, y_col=y_col, y_err_col=y_err_col, default_fraction=default_fraction, error_mode=error_mode, label=label)
    sigma = series.y_error
    return series.x, series.y, sigma, series.sigma_low, series.sigma_high


__all__ = ["propagate_errors", "as_array", "effective_sigma"]
