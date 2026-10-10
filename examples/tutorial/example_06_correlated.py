"""Complex example: generalized least squares for a thermal response."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from common import DATA, ROOT, ExampleOutput, diagnostic_plot, print_output, raw_labels, save_figure
from scipy.linalg import solve_triangular

from labfit import fit, plot_fit
from labfit.io import load_csv


def run(data: Path = DATA, figures: Path = ROOT / "figures") -> ExampleOutput:
    # tutorial:begin analysis
    series = load_csv(data / "thermal.csv", label="Thermal step response")
    covariance = np.loadtxt(data / "thermal_covariance.csv", delimiter=",")
    initial = {"amplitude": float(series.y[-1] - series.y[0]), "tau": 30.0, "offset": float(series.y[0])}
    bounds = {"amplitude": (0, 30), "tau": (1, 200), "offset": (10, 30)}
    correlated = fit(
        series.x,
        series.y,
        sigma_cov=covariance,
        model="exponential_rise",
        p0=initial,
        bounds=bounds,
        absolute_sigma=True,
    )
    diagonal = fit(
        series,
        model="exponential_rise",
        p0=initial,
        bounds=bounds,
        absolute_sigma=True,
    )
    cholesky = np.linalg.cholesky(covariance)
    whitened = solve_triangular(cholesky, correlated.residuals, lower=True)
    chi2_from_residuals = float(whitened @ whitened)
    # tutorial:end analysis
    plot = plot_fit(
        correlated,
        show_ci=True,
        ci_level=0.95,
        xlabel="Time (s)",
        ylabel="Temperature (deg C)",
        title="Thermal response: generalized least squares",
    )
    raw_labels(plot, "deg C")
    save_figure(plot, "06_correlated", figures)
    save_figure(diagnostic_plot(series.x, whitened, "Time (s)", "Whitened residual"), "06_whitened", figures)
    fig, ax = plt.subplots(figsize=(6, 4.3), layout="constrained")
    im = ax.imshow(covariance, cmap="viridis", origin="lower")
    ax.set(xlabel="Observation index j", ylabel="Observation index i")
    fig.colorbar(im, ax=ax, label="Measurement covariance (deg C squared)")
    save_figure(fig, "06_covariance", figures)
    return ExampleOutput(
        {"correlated": correlated, "diagonal": diagonal},
        metrics={
            "whitened_chi2": chi2_from_residuals,
            "reported_chi2": float(correlated.reduced_chi2 * correlated.dof),
            "tau_uncertainty_ratio": correlated.uncertainties["tau"] / diagonal.uncertainties["tau"],
        },
    )


if __name__ == "__main__":
    print_output(run())
