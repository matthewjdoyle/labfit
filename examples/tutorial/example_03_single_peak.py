"""Intermediate example: spectral peak and background misspecification."""

from pathlib import Path

import numpy as np
from common import DATA, ROOT, ExampleOutput, diagnostic_plot, print_output, raw_labels, save_figure

from labfit import fit, plot_fit
from labfit.io import load_csv
from labfit.utils import propagate_errors


def run(data: Path = DATA, figures: Path = ROOT / "figures") -> ExampleOutput:
    # tutorial:begin fit
    series = load_csv(data / "single_peak.csv", label="Peak plus background")
    edge = np.abs(series.x) > 2.5
    slope, intercept = np.polyfit(series.x[edge], series.y[edge], 1)
    corrected = series.y - slope * series.x - intercept
    centre = float(series.x[np.argmax(corrected)])
    result = fit(
        series,
        model="gaussian_baseline",
        absolute_sigma=True,
        p0={
            "amplitude": float(corrected.max()),
            "mean": centre,
            "sigma": 0.7,
            "m": float(slope),
            "b": float(intercept),
        },
        bounds={"amplitude": (0, None), "mean": (-1.5, 1.5), "sigma": (0.1, 2)},
    )
    wrong = fit(
        series,
        model="gaussian",
        absolute_sigma=True,
        p0={"amplitude": float(series.y.max()), "mean": centre, "sigma": 0.7},
        bounds={"amplitude": (0, None), "mean": (-1.5, 1.5), "sigma": (0.1, 4)},
    )
    # tutorial:end fit
    # tutorial:begin derived
    indices = [result.param_names.index(name) for name in ("amplitude", "sigma")]
    area_covariance = result.covariance[np.ix_(indices, indices)]
    area, area_error = propagate_errors(
        lambda amplitude, sigma: np.sqrt(2 * np.pi) * amplitude * sigma,
        jacobian=lambda amplitude, sigma: np.sqrt(2 * np.pi) * np.array([sigma, amplitude]),
        covariance=area_covariance,
        amplitude=result.params["amplitude"],
        sigma=result.params["sigma"],
    )
    factor = 2 * np.sqrt(2 * np.log(2))
    fwhm = factor * result.params["sigma"]
    fwhm_error = factor * result.uncertainties["sigma"]
    # tutorial:end derived
    for name, fitted in (("03_single_peak", result), ("03_missing_background", wrong)):
        plot = plot_fit(
            fitted, xlabel="Wavelength offset from 550 nm (nm)", ylabel="Signal (mV)", title=fitted.model_name
        )
        raw_labels(plot, "mV")
        save_figure(plot, name, figures)
    z = result.residuals / result.sigma
    save_figure(
        diagnostic_plot(series.x, z, "Wavelength offset (nm)", "Standardized residual"),
        "03_standardized",
        figures,
    )
    return ExampleOutput(
        {"baseline": result, "missing_background": wrong},
        {"FWHM": (fwhm, fwhm_error, "nm"), "Integrated peak area": (area, area_error, "mV nm")},
        {"background_free_chi2": wrong.reduced_chi2, "baseline_chi2": result.reduced_chi2},
    )


if __name__ == "__main__":
    print_output(run())
