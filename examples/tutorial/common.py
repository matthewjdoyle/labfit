"""Reporting helpers shared by the executable tutorial examples."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from labfit import FitResult, Plotter

matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
BLUE, ORANGE, GREEN = "#0072B2", "#D55E00", "#009E73"


@dataclass
class ExampleOutput:
    fits: dict[str, FitResult]
    derived: dict[str, tuple[float, float, str]] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)


def save_figure(figure: Figure | Plotter, name: str, output: Path) -> None:
    fig = figure.figure if isinstance(figure, Plotter) else figure
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"{name}.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def raw_labels(plot: Plotter, unit: str) -> None:
    for axis in np.asarray(plot.axes).flat:
        if axis.get_ylabel() == "residuals":
            axis.set_ylabel(f"Raw residual ({unit})")


def diagnostic_plot(x: np.ndarray, z: np.ndarray, xlabel: str, ylabel: str) -> Figure:
    fig, ax = plt.subplots(figsize=(7, 2.7), layout="constrained")
    ax.axhline(0, color="0.3", linewidth=1)
    ax.axhspan(-2, 2, color=BLUE, alpha=0.08)
    ax.plot(x, z, ".", color=ORANGE, markersize=5)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.grid(alpha=0.15)
    return fig


def correlation_plot(result: FitResult) -> Figure:
    assert result.covariance is not None
    errors = np.sqrt(np.diag(result.covariance))
    correlation = result.covariance / np.outer(errors, errors)
    names = result.param_names
    fig, ax = plt.subplots(figsize=(7, 5.7), layout="constrained")
    im = ax.imshow(correlation, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(names)):
            value = correlation[i, j]
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if abs(value) > 0.65 else "black",
            )
    fig.colorbar(im, ax=ax, shrink=0.8, label="Parameter correlation")
    return fig


def result_record(output: ExampleOutput) -> dict[str, object]:
    fits: dict[str, object] = {}
    for name, result in output.fits.items():
        fits[name] = {
            "model": result.model_name,
            "params": result.params,
            "uncertainties": result.uncertainties,
            "param_names": result.param_names,
            "covariance": result.covariance.tolist() if result.covariance is not None else None,
            "success": bool(result.success),
            "identifiable": bool(result.identifiable),
            "dof": result.dof,
            "reduced_chi2": result.reduced_chi2,
            "p_value": result.p_value,
            "residual_variance": result.residual_variance,
            "jacobian_rank": result.jacobian_rank,
            "jacobian_condition": result.jacobian_condition,
        }
    return {"fits": fits, "derived": output.derived, "metrics": output.metrics}


def print_output(output: ExampleOutput) -> None:
    for label, result in output.fits.items():
        print(f"\n{label}\n{result}")
    for label, (value, error, unit) in output.derived.items():
        print(f"{label}: {value:.6g} +/- {error:.3g} {unit}")


def write_json(path: Path, content: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8")
