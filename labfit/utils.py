from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .types import AsymmetricError, MaybeArray, _normalize_uncertainties


def propagate_errors(
    func: Callable[..., float],
    jacobian: Callable[..., MaybeArray] | None = None,
    covariance: np.ndarray | Sequence[Sequence[float]] | None = None,
    **params: float,
) -> tuple[float, float]:
    """Propagate parameter uncertainties through a function.

    Given a function ``func`` and its parameters with associated
    uncertainties, compute the output value and its propagated
    uncertainty using the standard first-order (linear) formula:

    .. math::

        \\sigma_y = \\sqrt{\\mathbf{J} \\, \\Sigma \\, \\mathbf{J}^T}

    where :math:`\\mathbf{J}` is the Jacobian of *func* with respect to
    the parameters and :math:`\\Sigma` is their covariance matrix.

    Parameters
    ----------
    func : callable
        The function to evaluate. Called as ``func(**value_params)`` where
        ``value_params`` excludes any ``<name>_error`` / ``<name>_sigma``
        keywords.
    jacobian : callable, optional
        Function returning the partial derivatives of *func* with respect
        to each parameter, called as ``jacobian(**value_params)``.
        If omitted, a central-difference numerical Jacobian is used
        (requires ``covariance`` to be provided).
    covariance : array-like, optional
        Full covariance matrix of the parameters. If omitted, a diagonal
        covariance is constructed from ``<name>_error`` or
        ``<name>_sigma`` keywords passed as ``**params``.
    **params
        Parameter values and their uncertainties. For each parameter
        ``x``, pass ``x=...`` and optionally ``x_error=...`` or
        ``x_sigma=...``.

    Returns
    -------
    (value, uncertainty) : tuple of float
        The function output and its propagated 1-σ uncertainty.

    Raises
    ------
    ValueError
        If ``jacobian`` is None and ``covariance`` is None (cannot
        compute a numerical Jacobian without knowing the parameter
        count).
    """
    value_kwargs = {
        name: value
        for name, value in params.items()
        if not name.endswith("_error") and not name.endswith("_sigma")
    }
    y = func(**value_kwargs)

    if covariance is None:
        # Convenience: if the caller passes a value and an associated
        # <name>_error keyword, use a diagonal covariance.
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
        jacobian = _numerical_jacobian(func, value_kwargs)

    grad = jacobian(**value_kwargs)
    grad = np.atleast_1d(np.asarray(grad, dtype=float))
    variance = float(grad @ covariance @ grad.T)
    return y, float(np.sqrt(max(variance, 0.0)))


def _numerical_jacobian(
    func: Callable[..., float], value_kwargs: dict[str, float], eps: float = 1e-8
) -> Callable[..., np.ndarray]:
    """Build a central-difference Jacobian callable for *func*."""

    def jacobian(**kwargs: float) -> np.ndarray:
        names = list(value_kwargs.keys())
        base = np.array([float(value_kwargs[n]) for n in names], dtype=float)
        grad = np.zeros(len(names), dtype=float)
        for i in range(len(names)):
            step = max(abs(base[i]) * eps, eps)
            up = base.copy()
            up[i] += step
            down = base.copy()
            down[i] -= step
            f_up = func(**dict(zip(names, up, strict=True)))
            f_down = func(**dict(zip(names, down, strict=True)))
            grad[i] = (float(f_up) - float(f_down)) / (2.0 * step)
        return grad

    return jacobian


def effective_sigma(
    sigma: MaybeArray | AsymmetricError | None = None,
    sigma_low: MaybeArray | None = None,
    sigma_high: MaybeArray | None = None,
    sigma_cov: np.ndarray | Sequence[Sequence[float]] | None = None,
) -> np.ndarray | None:
    """Validate one uncertainty specification and return its effective sigma.

    Standard deviations must be finite, strictly positive scalars or 1D arrays.
    Asymmetric pairs use their root-mean-square standard deviation. Covariance
    must be symmetric positive-definite and returns None (correlations cannot
    be represented by independent sigmas). Competing specifications are rejected.
    DataSeries and fitting additionally validate the errors against data length.
    """
    effective, _ = _normalize_uncertainties(
        sigma=sigma, sigma_low=sigma_low, sigma_high=sigma_high, sigma_cov=sigma_cov
    )
    return effective


__all__ = ["propagate_errors", "effective_sigma"]
