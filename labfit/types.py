from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

MaybeArray = np.ndarray | Sequence[float] | float


@dataclass(frozen=True)
class AsymmetricError:
    lower: np.ndarray
    upper: np.ndarray

    @property
    def effective(self) -> np.ndarray:
        lower = np.asarray(self.lower, dtype=float)
        upper = np.asarray(self.upper, dtype=float)
        return np.sqrt((lower**2 + upper**2) / 2.0)


@dataclass
class DataSeries:
    x: np.ndarray
    y: np.ndarray
    sigma: MaybeArray | AsymmetricError | None = None
    y_err: MaybeArray | AsymmetricError | None = None
    label: str = ""
    sigma_low: np.ndarray | None = None
    sigma_high: np.ndarray | None = None
    sigma_cov: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=float)
        self.y = np.asarray(self.y, dtype=float)

        # sigma is canonical; y_err is an alias that mirrors it
        if self.sigma is not None and self.y_err is not None:
            sigma_eff = (
                self.sigma.effective
                if isinstance(self.sigma, AsymmetricError)
                else np.asarray(self.sigma, dtype=float)
            )
            y_err_eff = (
                self.y_err.effective
                if isinstance(self.y_err, AsymmetricError)
                else np.asarray(self.y_err, dtype=float)
            )
            if not np.allclose(sigma_eff, y_err_eff):
                raise ValueError("sigma and y_err must describe the same uncertainties")
        elif self.sigma is None and self.y_err is not None:
            self.sigma = self.y_err
        self.y_err = self.sigma

        if self.sigma is not None and not isinstance(self.sigma, AsymmetricError):
            self.sigma = np.asarray(self.sigma, dtype=float)
        if self.sigma_low is not None:
            self.sigma_low = np.asarray(self.sigma_low, dtype=float)
        if self.sigma_high is not None:
            self.sigma_high = np.asarray(self.sigma_high, dtype=float)
        if self.sigma_cov is not None:
            self.sigma_cov = np.asarray(self.sigma_cov, dtype=float)

        if self.x.ndim != 1 or self.y.ndim != 1:
            raise ValueError("DataSeries x and y must be one-dimensional arrays")
        if self.x.size != self.y.size:
            raise ValueError("DataSeries x and y must have the same length")
        if not (np.all(np.isfinite(self.x)) and np.all(np.isfinite(self.y))):
            raise ValueError("DataSeries x and y must contain only finite values")

        n = self.x.size
        if self.sigma_cov is not None:
            if self.sigma_cov.ndim != 2 or self.sigma_cov.shape != (n, n):
                raise ValueError("sigma_cov must be a square covariance matrix matching x/y length")
            if not np.all(np.isfinite(self.sigma_cov)):
                raise ValueError("sigma_cov must contain only finite values")

        def _validate_error_array(name: str, value, *, allow_asymmetric: bool = False):
            if value is None:
                return
            if isinstance(value, AsymmetricError):
                lower = np.asarray(value.lower, dtype=float)
                upper = np.asarray(value.upper, dtype=float)
                if lower.shape != upper.shape:
                    raise ValueError(f"{name} lower and upper arrays must have the same shape")
                if lower.ndim > 0 and lower.size != n:
                    raise ValueError(f"{name} arrays must have the same length as x and y")
                if not (np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))):
                    raise ValueError(f"{name} arrays must contain only finite values")
                return
            arr = np.asarray(value, dtype=float)
            if arr.ndim > 0 and arr.size != n:
                raise ValueError(f"{name} must have the same length as x and y")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} must contain only finite values")
            if np.any(arr < 0):
                raise ValueError(f"{name} must be non-negative")

        _validate_error_array("sigma", self.sigma, allow_asymmetric=True)
        _validate_error_array("sigma_low", self.sigma_low)
        _validate_error_array("sigma_high", self.sigma_high)

    @property
    def effective_sigma(self) -> np.ndarray | None:
        if self.sigma_cov is not None:
            return None
        if self.sigma_low is not None and self.sigma_high is not None:
            return np.sqrt((self.sigma_low**2 + self.sigma_high**2) / 2.0)
        if isinstance(self.sigma, AsymmetricError):
            return self.sigma.effective
        if self.sigma is None:
            return None
        return np.asarray(self.sigma, dtype=float)

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
    model: Any = None
    is_weighted: bool = True

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

        names = self.param_names or list(self.params.keys())
        if names:
            for name in names:
                val = self.params.get(name, float("nan"))
                unc = self.uncertainties.get(name, float("nan"))
                if np.isfinite(unc):
                    lines.append(f"  {name} = {val:.5g} +/- {unc:.5g}")
                else:
                    lines.append(f"  {name} = {val:.5g}  (uncertainty N/A)")

        if np.isfinite(self.reduced_chi2):
            note = ""
            if self.reduced_chi2 > 10:
                note = "  -- model may be wrong or errors underestimated"
            elif self.reduced_chi2 < 0.1:
                note = "  -- errors may be overestimated"
            lines.append(f"  reduced chi2 = {self.reduced_chi2:.4g}{note}")

        if np.isfinite(self.p_value):
            if self.p_value < 0.001:
                lines.append(f"  p ~ {self.p_value:.1e}")
            else:
                lines.append(f"  p = {self.p_value:.3g}")

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

    def items(self):
        return self.params.items()

    def keys(self):
        return self.params.keys()

    def values(self):
        return self.params.values()


@dataclass
class Fitter:
    model: str | Any = "linear"
    p0: Any = None
    bounds: Any = None

    def fit(self, x, y=None, **kwargs) -> FitResult:
        from .fitter_impl import fit as _fit

        return _fit(x, y, model=self.model, p0=self.p0, bounds=self.bounds, **kwargs)

    def fit_multi(self, dataset, **kwargs):
        from .fitter_impl import fit_multi as _fit_multi

        return _fit_multi(dataset, model=self.model, p0=self.p0, bounds=self.bounds, **kwargs)

    def __call__(self, x, y=None, **kwargs) -> FitResult:
        return self.fit(x, y, **kwargs)


@dataclass
class Plotter:
    series: list[DataSeries] = field(default_factory=list)
    figure: Any = None
    axes: Any = None

    def add_series(self, *args, **kwargs) -> Plotter:
        if len(args) == 1 and isinstance(args[0], DataSeries):
            self.series.append(args[0])
            return self
        if len(args) >= 2:
            self.series.append(DataSeries(*args[:2], **kwargs))
            return self
        raise TypeError("add_series expects a DataSeries or x, y arrays")

    def plot(self, result=None, **kwargs) -> Plotter:
        from .plot import plot_result as _plot_result

        return _plot_result(result=result, plotter=self, **kwargs)

    def save(self, path, **kwargs):
        if self.figure is None:
            raise ValueError("Nothing has been plotted yet")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.figure.savefig(str(path), **kwargs)
        return path

    def __call__(self, result=None, **kwargs) -> Plotter:
        return self.plot(result=result, **kwargs)


__all__ = [
    "AsymmetricError",
    "DataSeries",
    "Dataset",
    "Fitter",
    "FitResult",
    "Plotter",
    "Series",
]
