from __future__ import annotations

from .fitter_impl import fit as _fit


def fit_curve(model, x, y, y_err, *, p0=None, bounds=None, label="", **kwargs):
    """Fit a model to data with explicit 1-sigma y uncertainties.

    A convenience wrapper around :func:`fit` that keeps the error
    specification as a dedicated positional argument for clarity.

    Parameters
    ----------
    model : str or callable
        Built-in model name or custom callable.
    x : array-like
        x-values.
    y : array-like
        y-values.
    y_err : array-like
        1-sigma uncertainties for each y-value.
    p0 : dict or array-like, optional
        Initial parameter guesses.
    bounds : dict or tuple of arrays, optional
        Parameter bounds.
    label : str, optional
        Label for legends.

    Returns
    -------
    FitResult
    """

    return _fit(x, y, model=model, sigma=y_err, p0=p0, bounds=bounds, label=label, **kwargs)


fit = fit_curve


__all__ = ["fit_curve", "fit"]
