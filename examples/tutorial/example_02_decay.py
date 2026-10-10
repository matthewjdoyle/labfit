"""Simple example: exponential decay and a derived half-life."""

from pathlib import Path

import numpy as np
from common import DATA, ROOT, ExampleOutput, print_output, raw_labels, save_figure

from labfit import fit, plot_fit
from labfit.io import load_csv
from labfit.utils import propagate_errors


def run(data: Path = DATA, figures: Path = ROOT / "figures") -> ExampleOutput:
    # tutorial:begin analysis
    series = load_csv(data / "decay.csv", label="Decaying signal")
    # Estimate a rate from two well-separated, still-positive readings.
    midpoint = len(series.y) // 2
    rate_guess = np.log(series.y[0] / series.y[midpoint]) / series.x[midpoint]
    result = fit(
        series,
        model="exponential",
        absolute_sigma=True,
        p0={"amplitude": float(series.y[0]), "decay": float(rate_guess)},
        bounds={"amplitude": (0, None), "decay": (0.001, 2)},
    )
    half_life, half_life_error = propagate_errors(
        lambda decay: np.log(2.0) / decay,
        jacobian=lambda decay: -np.log(2.0) / decay**2,
        decay=result.params["decay"],
        decay_error=result.uncertainties["decay"],
    )
    plot = plot_fit(
        result,
        show_ci=True,
        ci_level=0.95,
        xlabel="Time (s)",
        ylabel="Signal voltage (V)",
        title="Exponential signal decay",
    )
    raw_labels(plot, "V")
    save_figure(plot, "02_decay", figures)
    # tutorial:end analysis
    return ExampleOutput({"decay": result}, {"Half-life": (half_life, half_life_error, "s")})


if __name__ == "__main__":
    print_output(run())
