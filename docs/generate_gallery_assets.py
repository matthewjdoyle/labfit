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

from labfit import Fitter, Series, fit, fit_curve, plot_fit, plot_multi_fit

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
    plot = plot_multi_fit(
        results, [s1, s2], title="Damped oscillator comparison", xlabel="time / s", ylabel="signal"
    )
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


if __name__ == "__main__":
    make_linear_fit()
    make_gaussian_fit()
    make_multi_fit()
    make_bimodal_gaussian()
    make_exponential_fit()
    make_subset_fit()
    print(f"Wrote gallery assets to {OUT}")
