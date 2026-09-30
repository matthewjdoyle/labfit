from __future__ import annotations

from collections.abc import Callable, ItemsView, Iterable, Iterator, KeysView, Sequence, ValuesView
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

MaybeArray = np.ndarray | Sequence[float] | float
ModelFunction = Callable[..., ArrayLike]
ModelSpec = str | ModelFunction | None
InitialGuess = dict[str, float] | MaybeArray | None
Bounds = (
    dict[str, tuple[float | None, float | None]] | tuple[MaybeArray, MaybeArray] | list[MaybeArray] | None
)


def _error_array(name: str, value: MaybeArray, n: int | None = None, *, weights: bool = False) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim > 1:
        raise ValueError(f"{name} must be a scalar or one-dimensional array")
    if n is not None and array.ndim == 1 and array.size != n:
        raise ValueError(f"{name} must have the same length as x and y")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if weights:
        if np.any(array < 0):
            raise ValueError("weights must be non-negative")
        if not np.any(array > 0):
            raise ValueError("weights must contain at least one positive value; zero weights exclude samples")
    elif np.any(array <= 0):
        raise ValueError(f"{name} must be strictly positive")
    return array


def _asymmetric_sigma(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    # Scaling before hypot avoids overflow when finite errors are very large.
    return np.hypot(lower / np.sqrt(2.0), upper / np.sqrt(2.0))


@dataclass(frozen=True)
class AsymmetricError:
    """Finite, strictly positive lower/upper errors of matching scalar or 1D shape."""

    lower: np.ndarray
    upper: np.ndarray

    if TYPE_CHECKING:

        def __init__(self, lower: MaybeArray, upper: MaybeArray) -> None: ...

    def __post_init__(self) -> None:
        lower = _error_array("sigma lower", self.lower)
        upper = _error_array("sigma upper", self.upper)
        if lower.shape != upper.shape:
            raise ValueError("sigma lower and upper arrays must have the same shape")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @property
    def effective(self) -> np.ndarray:
        sigma, _ = _normalize_uncertainties(sigma=self)
        assert sigma is not None
        return sigma


def _normalize_uncertainties(
    n: int | None = None,
    *,
    sigma: MaybeArray | AsymmetricError | None = None,
    weights: MaybeArray | None = None,
    sigma_low: MaybeArray | None = None,
    sigma_high: MaybeArray | None = None,
    sigma_cov: np.ndarray | Sequence[Sequence[float]] | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Validate one complete error specification, then return effective sigma/covariance."""
    if (sigma_low is None) != (sigma_high is None):
        raise ValueError("sigma_low and sigma_high must be supplied together")
    count = sum(value is not None for value in (sigma, weights, sigma_low, sigma_cov))
    if count > 1:
        raise ValueError("Pass exactly one of sigma, weights, sigma_low/sigma_high, or sigma_cov")
    if sigma_cov is not None:
        covariance = np.asarray(sigma_cov, dtype=float)
        if (
            covariance.ndim != 2
            or covariance.shape[0] != covariance.shape[1]
            or (n is not None and covariance.shape != (n, n))
        ):
            raise ValueError("sigma_cov must be a square covariance matrix matching x/y length")
        if not np.all(np.isfinite(covariance)):
            raise ValueError("sigma_cov must contain only finite values")
        if not np.allclose(covariance, covariance.T, rtol=1e-12, atol=0.0):
            raise ValueError("sigma_cov must be symmetric")
        covariance = covariance + 0.5 * (covariance.T - covariance)
        try:
            np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError("sigma_cov must be positive-definite") from exc
        return None, covariance
    if weights is not None:
        array = _error_array("weights", weights, n, weights=True)
        effective = np.full_like(array, np.inf)
        np.divide(1.0, np.sqrt(array), out=effective, where=array > 0)
        return effective, None
    if isinstance(sigma, AsymmetricError):
        lower = _error_array("sigma lower", sigma.lower, n)
        upper = _error_array("sigma upper", sigma.upper, n)
        if lower.shape != upper.shape:
            raise ValueError("sigma lower and upper arrays must have the same shape")
        return _asymmetric_sigma(lower, upper), None
    if sigma_low is not None:
        assert sigma_high is not None
        lower = _error_array("sigma_low", sigma_low, n)
        upper = _error_array("sigma_high", sigma_high, n)
        if lower.ndim == upper.ndim == 1 and lower.shape != upper.shape:
            raise ValueError("sigma_low and sigma_high arrays must have the same length")
        return _asymmetric_sigma(lower, upper), None
    return (None if sigma is None else _error_array("sigma", sigma, n)), None


@dataclass
class DataSeries:
    """Finite 1D data with one optional measurement-error specification.

    Errors must be finite and strictly positive scalars or vectors matching the
    data length. Supply sigma (or its y_err alias), both asymmetric sides, or a
    symmetric positive-definite covariance matrix, never competing specifications.
    File loaders record any assumed error formula in ``uncertainty_inference``;
    ``uncertainties_inferred`` indicates whether measurement errors were inferred.
    """

    x: np.ndarray
    y: np.ndarray
    sigma: MaybeArray | AsymmetricError | None = None
    y_err: MaybeArray | AsymmetricError | None = None
    label: str = ""
    sigma_low: np.ndarray | None = None
    sigma_high: np.ndarray | None = None
    sigma_cov: np.ndarray | None = None
    uncertainty_inference: str | None = None

    if TYPE_CHECKING:

        def __init__(
            self,
            x: MaybeArray,
            y: MaybeArray,
            sigma: MaybeArray | AsymmetricError | None = None,
            y_err: MaybeArray | AsymmetricError | None = None,
            label: str = "",
            sigma_low: MaybeArray | None = None,
            sigma_high: MaybeArray | None = None,
            sigma_cov: np.ndarray | Sequence[Sequence[float]] | None = None,
            uncertainty_inference: str | None = None,
        ) -> None: ...

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=float)
        self.y = np.asarray(self.y, dtype=float)
        if self.x.ndim != 1 or self.y.ndim != 1:
            raise ValueError("DataSeries x and y must be one-dimensional arrays")
        if self.x.size != self.y.size:
            raise ValueError("DataSeries x and y must have the same length")
        if not (np.all(np.isfinite(self.x)) and np.all(np.isfinite(self.y))):
            raise ValueError("DataSeries x and y must contain only finite values")

        n = self.x.size
        if self.y_err is not None:
            _normalize_uncertainties(n, sigma=self.y_err)
        if self.sigma is not None and self.y_err is not None:
            _normalize_uncertainties(n, sigma=self.sigma)
            if isinstance(self.sigma, AsymmetricError) and isinstance(self.y_err, AsymmetricError):
                same = np.allclose(self.sigma.lower, self.y_err.lower, rtol=1e-12, atol=0.0) and np.allclose(
                    self.sigma.upper, self.y_err.upper, rtol=1e-12, atol=0.0
                )
            elif not isinstance(self.sigma, AsymmetricError) and not isinstance(self.y_err, AsymmetricError):
                same = np.allclose(self.sigma, self.y_err, rtol=1e-12, atol=0.0)
            else:
                same = False
            if not same:
                raise ValueError("sigma and y_err must describe the same uncertainties")
        elif self.sigma is None:
            self.sigma = self.y_err

        _, self.sigma_cov = _normalize_uncertainties(
            n,
            sigma=self.sigma,
            sigma_low=self.sigma_low,
            sigma_high=self.sigma_high,
            sigma_cov=self.sigma_cov,
        )
        if self.sigma is not None and not isinstance(self.sigma, AsymmetricError):
            self.sigma = np.asarray(self.sigma, dtype=float)
        self.y_err = self.sigma
        if self.sigma_low is not None:
            self.sigma_low = np.asarray(self.sigma_low, dtype=float)
            self.sigma_high = np.asarray(self.sigma_high, dtype=float)

    @property
    def uncertainties_inferred(self) -> bool:
        """Whether this series carries a loader's heuristic measurement errors."""
        return self.uncertainty_inference is not None

    @property
    def effective_sigma(self) -> np.ndarray | None:
        sigma, _ = _normalize_uncertainties(
            self.x.size,
            sigma=self.sigma,
            sigma_low=self.sigma_low,
            sigma_high=self.sigma_high,
            sigma_cov=self.sigma_cov,
        )
        return sigma

    @property
    def y_error(self) -> np.ndarray | None:
        return self.effective_sigma

    def error(self) -> np.ndarray | None:
        return self.effective_sigma

    def with_label(self, label: str) -> DataSeries:
        return DataSeries(
            x=self.x,
            y=self.y,
            sigma=self.sigma,
            label=label,
            sigma_low=self.sigma_low,
            sigma_high=self.sigma_high,
            sigma_cov=self.sigma_cov,
            uncertainty_inference=self.uncertainty_inference,
        )


Series = DataSeries


@dataclass
class Dataset:
    series: list[DataSeries] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.series)

    def __iter__(self) -> Iterator[DataSeries]:
        return iter(self.series)

    def __getitem__(self, item: int) -> DataSeries:
        return self.series[item]

    def append(self, series: DataSeries) -> None:
        self.series.append(series)

    def extend(self, items: Iterable[DataSeries]) -> None:
        self.series.extend(items)


@dataclass
class FitResult:
    """Fitted parameters and diagnostics.

    ``dof`` is N - k, or None for manually constructed results without it.
    ``residual_variance`` is unweighted SSR / dof in squared y units, NaN
    for nonpositive dof. For heteroscedastic or correlated errors it is a
    measure of residual scatter, not a common measurement-noise variance.
    ``reduced_chi2`` and ``p_value`` are NaN for unweighted fits; the latter
    also requires positive dof, optimizer convergence and identifiable parameters.
    """

    reduced_chi2: float
    params: dict[str, float]
    covariance: np.ndarray | None = None
    p_value: float = float("nan")
    uncertainties: dict[str, float] = field(default_factory=dict)
    success: bool = True
    message: str = ""
    model_name: str = ""
    param_names: tuple[str, ...] = field(default_factory=tuple)
    x: np.ndarray | None = None
    y: np.ndarray | None = None
    sigma: np.ndarray | None = None
    y_fit: np.ndarray | None = None
    series: DataSeries | None = None
    model: ModelFunction | None = None
    is_weighted: bool = True
    absolute_sigma: bool = False
    identifiable: bool = True
    jacobian_rank: int | None = None
    jacobian_condition: float = float("nan")
    dof: int | None = None
    residual_variance: float = float("nan")

    def __post_init__(self) -> None:
        if self.covariance is not None:
            self.covariance = np.asarray(self.covariance, dtype=float)
        if self.x is not None:
            self.x = np.asarray(self.x, dtype=float)
        if self.y is not None:
            self.y = np.asarray(self.y, dtype=float)
        if self.sigma is not None:
            self.sigma = np.asarray(self.sigma, dtype=float)
        if self.y_fit is not None:
            self.y_fit = np.asarray(self.y_fit, dtype=float)
        if not isinstance(self.params, dict):
            self.params = dict(self.params)
        if not isinstance(self.uncertainties, dict):
            self.uncertainties = dict(self.uncertainties)
        if not self.uncertainties and self.covariance is not None:
            names = self.param_names or tuple(self.params.keys())
            diag = np.diag(self.covariance) if self.covariance.ndim == 2 else np.asarray([])
            self.uncertainties = {
                name: float(np.sqrt(max(float(value), 0.0))) for name, value in zip(names, diag, strict=True)
            }

    def __str__(self) -> str:
        """Human-readable summary of the fit result."""
        model = self.model_name or "custom"
        lines = [f"FitResult: {model}"]
        if self.series is not None and self.series.uncertainties_inferred:
            lines.append(f"  measurement errors inferred: {self.series.uncertainty_inference}")

        names = self.param_names or list(self.params.keys())
        if names:
            for name in names:
                val = self.params.get(name, float("nan"))
                unc = self.uncertainties.get(name, float("nan"))
                if np.isfinite(unc):
                    lines.append(f"  {name} = {val:.5g} +/- {unc:.5g}")
                else:
                    lines.append(f"  {name} = {val:.5g}  (uncertainty N/A)")

        if self.dof is not None:
            lines.append(f"  degrees of freedom = {self.dof}")
        positive_dof = self.dof is None or self.dof > 0
        if positive_dof and np.isfinite(self.residual_variance):
            lines.append(f"  residual variance = {self.residual_variance:.4g} (y units squared)")
        if self.dof is not None and self.dof <= 0:
            lines.append("  [!] Nonpositive degrees of freedom; residual-scale statistics unavailable")

        if self.is_weighted and positive_dof and np.isfinite(self.reduced_chi2):
            note = ""
            if self.success and self.identifiable:
                if self.reduced_chi2 > 10:
                    note = "  -- model may be wrong or errors underestimated"
                elif self.reduced_chi2 < 0.1:
                    note = "  -- errors may be overestimated"
            lines.append(f"  reduced chi2 = {self.reduced_chi2:.4g}{note}")

        if (
            self.is_weighted
            and positive_dof
            and self.success
            and self.identifiable
            and np.isfinite(self.p_value)
        ):
            if self.p_value < 0.001:
                lines.append(f"  p ~ {self.p_value:.1e}")
            else:
                lines.append(f"  p = {self.p_value:.3g}")

        if self.identifiable and any(not np.isfinite(value) for value in self.uncertainties.values()):
            lines.append("  [!] Parameter uncertainties are unavailable")

        if not self.identifiable:
            lines.append("  [!] Parameters are not identifiable; uncertainties are unreliable")

        if not self.success:
            lines.append("  [!] Fit did NOT converge")
            if self.message:
                lines.append(f"      Message: {self.message}")
                lines.append("      Try: better p0, add bounds, or check model choice")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"FitResult(model_name={self.model_name!r}, "
            f"reduced_chi2={self.reduced_chi2:.4g}, "
            f"success={self.success})"
        )

    @property
    def parameter_uncertainties(self) -> dict[str, float]:
        return self.uncertainties

    def __getitem__(self, item: str) -> float:
        return self.params[item]

    def __iter__(self) -> Iterator[float]:
        for name in self.param_names or tuple(self.params.keys()):
            yield self.params[name]

    def predict(self, x: MaybeArray) -> np.ndarray:
        if self.model is None:
            raise ValueError("No model is attached to this FitResult")
        x = np.asarray(x, dtype=float)
        values = [self.params[name] for name in (self.param_names or tuple(self.params.keys()))]
        return np.asarray(self.model(x, *values), dtype=float)

    @property
    def residuals(self) -> np.ndarray:
        if self.x is None or self.y is None:
            raise ValueError("FitResult does not contain raw data")
        return self.y - self.predict(self.x)

    def items(self) -> ItemsView[str, float]:
        return self.params.items()

    def keys(self) -> KeysView[str]:
        return self.params.keys()

    def values(self) -> ValuesView[float]:
        return self.params.values()


@dataclass
class Fitter:
    model: ModelSpec = "linear"
    p0: InitialGuess = None
    bounds: Bounds = None

    def fit(self, x: FitInput, y: MaybeArray | None = None, **kwargs: Any) -> FitResult:
        from .fitter_impl import fit as _fit

        return _fit(x, y, model=self.model, p0=self.p0, bounds=self.bounds, **kwargs)

    def fit_multi(self, dataset: Iterable[DataSeries] | Dataset, **kwargs: Any) -> list[FitResult]:
        from .fitter_impl import fit_multi as _fit_multi

        return _fit_multi(dataset, model=self.model, p0=self.p0, bounds=self.bounds, **kwargs)

    def __call__(self, x: FitInput, y: MaybeArray | None = None, **kwargs: Any) -> FitResult:
        return self.fit(x, y, **kwargs)


@dataclass
class Plotter:
    series: list[DataSeries] = field(default_factory=list)
    figure: Any = None
    axes: Any = None

    def add_series(self, *args: DataSeries | MaybeArray, **kwargs: Any) -> Plotter:
        if len(args) == 1 and isinstance(args[0], DataSeries):
            self.series.append(args[0])
            return self
        if len(args) >= 2:
            self.series.append(
                DataSeries(np.asarray(args[0], dtype=float), np.asarray(args[1], dtype=float), **kwargs)
            )
            return self
        raise TypeError("add_series expects a DataSeries or x, y arrays")

    def plot(self, result: ResultInput = None, **kwargs: Any) -> Plotter:
        from .plot import plot_result as _plot_result

        return _plot_result(result=result, plotter=self, **kwargs)

    def save(self, path: str | Path, **kwargs: Any) -> Path:
        if self.figure is None:
            raise ValueError("Nothing has been plotted yet")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.figure.savefig(str(path), **kwargs)
        return path

    def __call__(self, result: ResultInput = None, **kwargs: Any) -> Plotter:
        return self.plot(result=result, **kwargs)


FitInput = MaybeArray | DataSeries | str | Path
ResultInput = FitResult | list[FitResult] | tuple[FitResult, ...] | None
SeriesInput = DataSeries | list[DataSeries] | tuple[DataSeries, ...] | None

__all__ = [
    "AsymmetricError",
    "DataSeries",
    "Dataset",
    "Fitter",
    "FitResult",
    "Plotter",
    "Series",
]
