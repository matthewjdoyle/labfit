"""Complex example: custom overlapping peaks with bounds and multiple starts."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from common import (
    BLUE,
    DATA,
    GREEN,
    ORANGE,
    ROOT,
    ExampleOutput,
    correlation_plot,
    print_output,
    raw_labels,
    save_figure,
)

from labfit import fit, plot_fit
from labfit.io import load_csv
from labfit.utils import propagate_errors


# tutorial:begin model
def double_peak(
    u: np.ndarray, A1: float, c1: float, w1: float, A2: float, c2: float, w2: float, m: float, b: float
) -> np.ndarray:
    first = A1 * np.exp(-0.5 * ((u - c1) / w1) ** 2)
    second = A2 * np.exp(-0.5 * ((u - c2) / w2) ** 2)
    return first + second + m * u + b


# tutorial:end model


def run(data: Path = DATA, figures: Path = ROOT / "figures") -> ExampleOutput:
    # tutorial:begin fit
    measured = load_csv(data / "double_peak.csv")
    u = measured.x - 550.0
    edge = (u < -1.2) | (u > 2.2)
    slope, baseline = np.polyfit(u[edge], measured.y[edge], 1)
    peak_height = float(np.max(measured.y - slope * u - baseline))
    candidates = []
    starts = [(-0.35, 0.65, 0.35), (0.0, 1.0, 0.55), (-0.2, 0.9, 0.75)]
    for left, right, width in starts:
        candidate = fit(
            u,
            measured.y,
            sigma=measured.sigma,
            model=double_peak,
            absolute_sigma=True,
            p0={
                "A1": 0.7 * peak_height,
                "c1": left,
                "w1": width,
                "A2": 0.5 * peak_height,
                "c2": right,
                "w2": width,
                "m": float(slope),
                "b": float(baseline),
            },
            bounds={
                "A1": (0, 10),
                "A2": (0, 10),
                "c1": (-0.6, 0.25),
                "c2": (0.4, 1.5),
                "w1": (0.15, 1.2),
                "w2": (0.15, 1.2),
            },
        )
        candidates.append(candidate)
    valid = [r for r in candidates if r.success and r.identifiable]
    if not valid:
        raise RuntimeError("No reliable overlapping-peak fit; inspect starts and bounds.")
    result = min(valid, key=lambda r: r.reduced_chi2)
    # tutorial:end fit
    # tutorial:begin derived
    names = ("c1", "c2")
    indices = [result.param_names.index(name) for name in names]
    centre_covariance = result.covariance[np.ix_(indices, indices)]
    separation, separation_error = propagate_errors(
        lambda c1, c2: c2 - c1,
        jacobian=lambda c1, c2: np.array([-1.0, 1.0]),
        covariance=centre_covariance,
        c1=result.params["c1"],
        c2=result.params["c2"],
    )
    # tutorial:end derived
    plot = plot_fit(
        result,
        xlabel="Wavelength offset from 550 nm (nm)",
        ylabel="Signal (mV)",
        title="Custom fit: two overlapping peaks plus background",
    )
    raw_labels(plot, "mV")
    save_figure(plot, "05_double_peak", figures)
    grid = np.linspace(u.min(), u.max(), 500)
    p = result.params
    background = p["m"] * grid + p["b"]
    fig, ax = plt.subplots(figsize=(7, 3.5), layout="constrained")
    ax.plot(grid, result.predict(grid), color=BLUE, label="Total fitted response")
    ax.plot(
        grid,
        p["A1"] * np.exp(-0.5 * ((grid - p["c1"]) / p["w1"]) ** 2),
        color=ORANGE,
        label="Peak 1 (background excluded)",
    )
    ax.plot(
        grid,
        p["A2"] * np.exp(-0.5 * ((grid - p["c2"]) / p["w2"]) ** 2),
        color=GREEN,
        label="Peak 2 (background excluded)",
    )
    ax.plot(grid, background, "--", color="0.4", label="Background")
    ax.set(xlabel="Wavelength offset from 550 nm (nm)", ylabel="Signal (mV)")
    ax.legend(fontsize=8)
    save_figure(fig, "05_components", figures)
    save_figure(correlation_plot(result), "05_correlations", figures)
    return ExampleOutput(
        {"double_peak": result},
        {"Peak separation": (separation, separation_error, "nm")},
        {f"start_{i}_chi2": r.reduced_chi2 for i, r in enumerate(candidates, start=1)},
    )


if __name__ == "__main__":
    print_output(run())
