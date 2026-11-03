#!/usr/bin/env python3
"""Generate committed gallery assets for the LabFit docs."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from labfit import Fitter, Series, fit, fit_curve, plot_fit
from labfit.plot import plot_result

OUT = ROOT / "_static" / "gallery"


def save_plot(plotter, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        path = OUT / f"{stem}.{ext}"
        kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = 220
        plotter.save(path, **kwargs)
    plt.close(plotter.figure)


def make_linear_fit() -> None:
    rng = np.random.default_rng(20240606)
    x = np.linspace(0.0, 8.0, 30)
    sigma = 0.10 + 0.02 * (1.0 + np.sin(x))
    y = 1.85 * x - 0.65 + rng.normal(0.0, sigma)
    result = fit_curve("linear", x, y, sigma)
    plot = plot_fit(
        result, show_residuals=True, title="Linear fit with residuals", xlabel="x", ylabel="signal"
    )
    save_plot(plot, "linear_fit")


def make_gaussian_fit() -> None:
    rng = np.random.default_rng(20240607)
    x = np.linspace(-5.0, 5.0, 260)
    y = 4.0 * np.exp(-0.5 * ((x - 0.75) / 1.10) ** 2) + 0.18
    y = y + rng.normal(0.0, 0.08, size=x.size)
    result = fit(x, y, model="gaussian")
    plot = plot_fit(result, show_residuals=True, title="Gaussian fit", xlabel="position", ylabel="intensity")
    save_plot(plot, "gaussian_fit")


def make_multi_fit() -> None:
    rng = np.random.default_rng(20240608)

    def damped_oscillator(t, amplitude, damping, f, phase):
        return amplitude * np.exp(-damping * t) * np.cos(2.0 * np.pi * f * t + phase)

    x1 = np.linspace(0.0, 6.0, 240)
    x2 = np.linspace(0.0, 6.0, 220)
    s1 = Series(
        x=x1,
        y=damped_oscillator(x1, 1.25, 0.22, 0.98, 0.10) + rng.normal(0, 0.03, size=x1.size),
        sigma=0.03,
        label="Trial 1",
    )
    s2 = Series(
        x=x2,
        y=damped_oscillator(x2, 1.05, 0.18, 1.02, -0.05) + rng.normal(0, 0.03, size=x2.size),
        sigma=0.03,
        label="Trial 2",
    )

    fitter = Fitter(
        model=damped_oscillator,
        p0={"amplitude": 1.0, "damping": 0.2, "f": 1.0, "phase": 0.0},
    )
    results = fitter.fit_multi([s1, s2])
    plot = plot_result(results, title="Damped oscillator comparison", xlabel="time / s", ylabel="signal")
    save_plot(plot, "damped_oscillators")


def make_bimodal_gaussian() -> None:
    rng = np.random.default_rng(20260607)
    x = np.linspace(-6.0, 10.0, 300)
    y_true = 3.5 * np.exp(-0.5 * ((x - 0.0) / 0.9) ** 2) + 2.0 * np.exp(-0.5 * ((x - 4.5) / 1.2) ** 2)
    y = y_true + rng.normal(0.0, 0.06, size=x.size)

    result = fit(
        x,
        y,
        model="bimodal_gaussian",
        p0={"amplitude1": 3.0, "mean1": 0.0, "sigma1": 1.0, "amplitude2": 1.5, "mean2": 4.0, "sigma2": 1.0},
    )
    plot = plot_fit(
        result, show_residuals=True, title="Bimodal Gaussian fit", xlabel="position", ylabel="intensity"
    )
    save_plot(plot, "bimodal_gaussian")


def make_exponential_fit() -> None:
    rng = np.random.default_rng(20260608)
    x = np.linspace(0.0, 10.0, 40)

    # True exponential decay model
    amplitude_true = 5.0
    decay_true = 0.4
    y_true = amplitude_true * np.exp(-decay_true * x)

    # Per-point uncertainties: counting-statistics style sigma proportional to sqrt(y)
    sigma = 0.05 + 0.08 * np.sqrt(y_true)
    y = y_true + rng.normal(0.0, sigma)

    result = fit_curve("exponential", x, y, sigma, p0={"amplitude": 5.0, "decay": 0.5})
    plot = plot_fit(
        result,
        show_residuals=True,
        title="Exponential decay with per-point errors",
        xlabel="time / s",
        ylabel="signal",
    )
    save_plot(plot, "exponential_fit")


def make_subset_fit() -> None:
    """Fit a Gaussian peak on a wavy background, using only the peak region."""
    rng = np.random.default_rng(20260618)
    x = np.linspace(0.0, 10.0, 300)
    y = (
        3.0 * np.exp(-0.5 * ((x - 5.0) / 0.8) ** 2)
        + 0.05 * np.sin(1.5 * x)
        + rng.normal(0, 0.03, size=x.size)
    )

    full = Series(x=x, y=y, sigma=0.03, label="full data")

    # Fit only the peak region (x = 3.5 ... 6.5)
    mask = (x > 3.5) & (x < 6.5)
    result = fit_curve("gaussian", x[mask], y[mask], 0.03)

    plot = plot_fit(
        result,
        data_series=full,
        title="Gaussian fit to peak region only",
        xlabel="x",
        ylabel="signal",
    )
    save_plot(plot, "subset_fit")


def make_model_comparison() -> None:
    """Fit the same data with a Gaussian (correct) and a Lorentzian (close but wrong).

    Both fit lines are overlaid on a single axis so the tail mismatch is
    immediately visible.  The reduced chi-squared values quantify what the
    eye sees: the Gaussian matches the data, the Lorentzian does not.
    """
    rng = np.random.default_rng(20260620)
    x = np.linspace(-4.0, 8.0, 200)
    y_true = 5.0 * np.exp(-0.5 * ((x - 2.0) / 1.0) ** 2)
    sigma = np.full_like(x, 0.08)
    y = y_true + rng.normal(0.0, sigma)

    result_gaussian = fit(x, y, model="gaussian", sigma=sigma, label="data")
    result_lorentzian = fit(x, y, model="lorentzian", sigma=sigma)

    # Both fits use the same observations and uncertainty.
    result_lorentzian.series = result_gaussian.series

    plot = plot_fit(
        [result_gaussian, result_lorentzian],
        show_residuals=False,
        title="Gaussian vs Lorentzian model comparison",
        xlabel="position",
        ylabel="intensity",
    )
    save_plot(plot, "model_comparison")

    print("=== Gaussian (correct model) ===")
    print(result_gaussian)
    print()
    print("=== Lorentzian (close but wrong) ===")
    print(result_lorentzian)


def make_confidence_bands() -> None:
    """Confidence and prediction bands for a linear calibration fit."""
    rng = np.random.default_rng(101)
    x = np.linspace(0, 50, 26)
    sigma = np.full(26, 0.045)
    y = 0.082 * x + 0.18 + rng.normal(0.0, sigma)

    result = fit(x, y, sigma=sigma, model="linear", absolute_sigma=True)

    # Mean-response confidence band
    plot_ci = plot_fit(
        result,
        show_ci=True,
        ci_level=0.95,
        xlabel="Reference temperature (°C)",
        ylabel="Sensor voltage (V)",
        title="Linear calibration: 95 % confidence band",
    )
    save_plot(plot_ci, "confidence_band")

    # Future-observation prediction band
    plot_pred = plot_fit(
        result,
        show_ci=True,
        prediction=True,
        ci_level=0.95,
        prediction_sigma=0.045,
        xlabel="Reference temperature (°C)",
        ylabel="Sensor voltage (V)",
        title="Linear calibration: 95 % prediction band",
    )
    save_plot(plot_pred, "prediction_band")


def make_error_propagation() -> None:
    """Exponential decay fit with derived half-life via error propagation."""
    from labfit.utils import propagate_errors

    rng = np.random.default_rng(202)
    x = np.linspace(0, 12, 49)
    sigma = np.full(49, 0.075)
    y = 4.8 * np.exp(-0.24 * x) + rng.normal(0.0, sigma)

    mid = len(y) // 2
    rate_guess = np.log(y[0] / y[mid]) / x[mid]

    result = fit(
        x,
        y,
        sigma=sigma,
        model="exponential",
        absolute_sigma=True,
        p0={"amplitude": float(y[0]), "decay": float(rate_guess)},
        bounds={"amplitude": (0, None), "decay": (0.001, 2)},
    )

    half_life, half_life_error = propagate_errors(
        lambda decay: np.log(2.0) / decay,
        jacobian=lambda decay: -np.log(2.0) / decay**2,
        decay=result.params["decay"],
        decay_error=result.uncertainties["decay"],
    )
    print(f"half-life = {half_life:.4f} ± {half_life_error:.4f} s")

    plot = plot_fit(
        result,
        show_ci=True,
        ci_level=0.95,
        xlabel="Time (s)",
        ylabel="Signal voltage (V)",
        title="Exponential signal decay",
    )
    save_plot(plot, "error_propagation")


def make_gaussian_baseline() -> None:
    """Spectral peak on a sloping background — correct vs missing-background model."""
    rng = np.random.default_rng(303)
    x = np.linspace(-3.5, 3.5, 141)
    y_true = 2.8 * np.exp(-0.5 * ((x - 0.35) / 0.62) ** 2) + 0.12 * x + 0.7
    sigma = 0.045 + 0.018 * np.sqrt(y_true)
    y = y_true + rng.normal(0.0, sigma)

    edge = np.abs(x) > 2.5
    slope, intercept = np.polyfit(x[edge], y[edge], 1)
    corrected = y - slope * x - intercept
    centre = float(x[np.argmax(corrected)])

    result = fit(
        x,
        y,
        sigma=sigma,
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
    plot = plot_fit(
        result,
        xlabel="Wavelength offset from 550 nm (nm)",
        ylabel="Signal (mV)",
        title="Gaussian + linear baseline",
    )
    save_plot(plot, "gaussian_baseline")

    wrong = fit(
        x,
        y,
        sigma=sigma,
        model="gaussian",
        absolute_sigma=True,
        p0={"amplitude": float(y.max()), "mean": centre, "sigma": 0.7},
        bounds={"amplitude": (0, None), "mean": (-1.5, 1.5), "sigma": (0.1, 4)},
    )
    print(f"With baseline   chi^2_r = {result.reduced_chi2:.3f}")
    print(f"Without baseline chi^2_r = {wrong.reduced_chi2:.3f}")


def make_custom_model() -> None:
    """Custom double-peak model fitted from multiple starting points."""
    rng = np.random.default_rng(505)
    u = np.linspace(-2.0, 3.0, 201)
    y_true = (
        2.5 * np.exp(-0.5 * ((u + 0.1) / 0.43) ** 2)
        + 1.8 * np.exp(-0.5 * ((u - 0.8) / 0.52) ** 2)
        + 0.06 * u
        + 0.35
    )
    sigma = np.full_like(u, 0.055)
    y = y_true + rng.normal(0.0, sigma)

    def double_peak(u, A1, c1, w1, A2, c2, w2, m, b):
        return A1 * np.exp(-0.5 * ((u - c1) / w1) ** 2) + A2 * np.exp(-0.5 * ((u - c2) / w2) ** 2) + m * u + b

    edge = (u < -1.2) | (u > 2.2)
    slope, baseline = np.polyfit(u[edge], y[edge], 1)
    peak_height = float(np.max(y - slope * u - baseline))

    starts = [(-0.35, 0.65, 0.35), (0.0, 1.0, 0.55), (-0.2, 0.9, 0.75)]
    candidates = []
    for left, right, width in starts:
        candidate = fit(
            u,
            y,
            sigma=sigma,
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
        raise RuntimeError("No reliable double-peak fit; inspect starts and bounds.")
    result = min(valid, key=lambda r: r.reduced_chi2)
    print(f"c1 = {result.params['c1']:.4f} +/- {result.uncertainties['c1']:.4f}")
    print(f"c2 = {result.params['c2']:.4f} +/- {result.uncertainties['c2']:.4f}")
    print(f"reduced chi^2 = {result.reduced_chi2:.4f}")

    # No figure for this example — the code snippet is the highlight.
    # We do not call save_plot; no image is referenced in gallery.rst.


def make_generalized_least_squares() -> None:
    """GLS fit of a thermal step response with AR(1)-like correlated errors."""
    rng = np.random.default_rng(606)
    x = np.linspace(0, 120, 81)
    y_true = 12 * (1 - np.exp(-x / 28)) + 21

    index = np.arange(x.size)
    rho, corr_sigma, indep_sigma = 0.78, 0.20, 0.06
    covariance = corr_sigma**2 * rho ** np.abs(index[:, None] - index[None, :]) + indep_sigma**2 * np.eye(
        x.size
    )
    y = y_true + np.linalg.cholesky(covariance) @ rng.normal(size=x.size)

    result = fit(
        x,
        y,
        sigma_cov=covariance,
        model="exponential_rise",
        p0={"amplitude": float(y[-1] - y[0]), "tau": 30.0, "offset": float(y[0])},
        bounds={"amplitude": (0, 30), "tau": (1, 200), "offset": (10, 30)},
        absolute_sigma=True,
    )
    plot = plot_fit(
        result,
        show_ci=True,
        ci_level=0.95,
        xlabel="Time (s)",
        ylabel="Temperature (°C)",
        title="Thermal response: generalized least squares",
    )
    save_plot(plot, "generalized_least_squares")
    print(result)


if __name__ == "__main__":
    make_linear_fit()
    make_gaussian_fit()
    make_multi_fit()
    make_bimodal_gaussian()
    make_exponential_fit()
    make_subset_fit()
    make_model_comparison()
    make_confidence_bands()
    make_error_propagation()
    make_gaussian_baseline()
    make_custom_model()
    make_generalized_least_squares()
    print(f"Wrote gallery assets to {OUT}")
