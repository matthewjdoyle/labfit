"""Recreate synthetic measurements; no fit uses the manifest's truth values."""

from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path

import numpy as np
from common import DATA, write_json


def generate(destination: Path = DATA) -> dict[str, object]:
    destination.mkdir(parents=True, exist_ok=True)
    datasets: dict[str, object] = {}

    def save(
        name: str,
        x: np.ndarray,
        mean: np.ndarray,
        sigma: np.ndarray,
        seed: int,
        equation: str,
        truth: dict[str, float],
        units: dict[str, str],
    ) -> None:
        rng = np.random.default_rng(seed)
        y = mean + rng.normal(size=x.size) * sigma
        np.savetxt(
            destination / f"{name}.csv",
            np.column_stack((x, y, sigma)),
            delimiter=",",
            header="x,y,sigma",
            comments="",
            fmt="%.12g",
        )
        datasets[name] = {
            "file": f"{name}.csv",
            "seed": seed,
            "samples": int(x.size),
            "equation": equation,
            "true_parameters": truth,
            "units": units,
            "noise": "Independent zero-mean Gaussian; sigma is calibrated 1-standard-deviation noise.",
            "x_uncertainty": "Negligible; x is treated as exact.",
        }

    x = np.linspace(0, 50, 26)
    save(
        "calibration",
        x,
        0.082 * x + 0.18,
        np.full_like(x, 0.045),
        101,
        "y = slope*x + intercept",
        {"slope": 0.082, "intercept": 0.18},
        {"x": "degree C", "y": "V", "slope": "V/degree C", "intercept": "V"},
    )

    x = np.linspace(0, 12, 49)
    save(
        "decay",
        x,
        4.8 * np.exp(-0.24 * x),
        np.full_like(x, 0.075),
        202,
        "y = amplitude*exp(-decay*x)",
        {"amplitude": 4.8, "decay": 0.24},
        {"x": "s", "y": "V", "amplitude": "V", "decay": "1/s"},
    )

    x = np.linspace(-3.5, 3.5, 141)
    mean = 2.8 * np.exp(-0.5 * ((x - 0.35) / 0.62) ** 2) + 0.12 * x + 0.7
    sigma = 0.045 + 0.018 * np.sqrt(mean)
    save(
        "single_peak",
        x,
        mean,
        sigma,
        303,
        "y = amplitude*exp(-0.5*((x-mean)/sigma)^2) + m*x + b",
        {"amplitude": 2.8, "mean": 0.35, "sigma": 0.62, "m": 0.12, "b": 0.7},
        {
            "x": "nm relative to 550 nm",
            "y": "mV",
            "amplitude": "mV",
            "mean": "nm",
            "sigma": "nm",
            "m": "mV/nm",
            "b": "mV",
        },
    )

    x = np.linspace(0, 8, 161)
    for index, (amp, damping, freq, phase, offset) in enumerate(
        ((3.1, 0.17, 0.88, 0.20, 0.12), (2.8, 0.19, 0.90, 0.35, 0.08), (3.3, 0.16, 0.87, 0.10, 0.15)), start=1
    ):
        mean = offset + amp * np.exp(-damping * x) * np.sin(2 * np.pi * freq * x + phase)
        save(
            f"oscillator_{index}",
            x,
            mean,
            np.full_like(x, 0.10),
            400 + index,
            "y = offset + amplitude*exp(-damping*x)*sin(2*pi*frequency*x + phase)",
            {"amplitude": amp, "damping": damping, "frequency": freq, "phase": phase, "offset": offset},
            {
                "x": "s",
                "y": "mm",
                "amplitude": "mm",
                "damping": "1/s",
                "frequency": "Hz",
                "phase": "rad",
                "offset": "mm",
            },
        )

    x = np.linspace(548, 553, 201)
    u = x - 550
    mean = (
        2.5 * np.exp(-0.5 * ((u + 0.1) / 0.43) ** 2)
        + 1.8 * np.exp(-0.5 * ((u - 0.8) / 0.52) ** 2)
        + 0.06 * u
        + 0.35
    )
    save(
        "double_peak",
        x,
        mean,
        np.full_like(x, 0.055),
        505,
        "u=x-550; y=A1*exp(-0.5*((u-c1)/w1)^2)+A2*exp(-0.5*((u-c2)/w2)^2)+m*u+b",
        {
            "A1": 2.5,
            "c1": -0.1,
            "w1": 0.43,
            "A2": 1.8,
            "c2": 0.8,
            "w2": 0.52,
            "m": 0.06,
            "b": 0.35,
            "reference_wavelength": 550.0,
        },
        {
            "x": "nm",
            "y": "mV",
            "A1": "mV",
            "c1": "nm",
            "w1": "nm",
            "A2": "mV",
            "c2": "nm",
            "w2": "nm",
            "m": "mV/nm",
            "b": "mV",
        },
    )

    x = np.linspace(0, 120, 81)
    mean = 12 * (1 - np.exp(-x / 28)) + 21
    index = np.arange(x.size)
    rho, correlated_sigma, independent_sigma = 0.78, 0.20, 0.06
    covariance = correlated_sigma**2 * rho ** np.abs(
        index[:, None] - index[None, :]
    ) + independent_sigma**2 * np.eye(x.size)
    rng = np.random.default_rng(606)
    y = mean + np.linalg.cholesky(covariance) @ rng.normal(size=x.size)
    np.savetxt(
        destination / "thermal.csv",
        np.column_stack((x, y, np.sqrt(np.diag(covariance)))),
        delimiter=",",
        header="x,y,sigma",
        comments="",
        fmt="%.12g",
    )
    np.savetxt(destination / "thermal_covariance.csv", covariance, delimiter=",", fmt="%.16g")
    datasets["thermal"] = {
        "file": "thermal.csv",
        "covariance_file": "thermal_covariance.csv",
        "seed": 606,
        "samples": int(x.size),
        "equation": "y=amplitude*(1-exp(-x/tau))+offset",
        "true_parameters": {"amplitude": 12.0, "tau": 28.0, "offset": 21.0},
        "units": {
            "x": "s",
            "y": "degree C",
            "amplitude": "degree C",
            "tau": "s",
            "offset": "degree C",
            "covariance": "degree C squared",
        },
        "noise": "Zero-mean Gaussian drawn using a Cholesky factor of the supplied covariance.",
        "covariance_equation": "C[i,j]=correlated_sigma^2*rho^abs(i-j)+independent_sigma^2*delta[i,j]",
        "noise_parameters": {
            "rho": rho,
            "correlated_sigma": correlated_sigma,
            "independent_sigma": independent_sigma,
        },
        "row_order": "Covariance rows/columns follow thermal.csv observation order; do not sort separately.",
        "x_uncertainty": "Negligible; x is treated as exact.",
    }
    manifest: dict[str, object] = {
        "description": "Synthetic laboratory measurements for education; no real instruments were measured.",
        "random_generator": "numpy.random.default_rng (PCG64)",
        "versions": {
            name: importlib.metadata.version(name) for name in ("labfit", "numpy", "scipy", "matplotlib")
        },
        "datasets": datasets,
    }
    write_json(destination / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DATA)
    args = parser.parse_args()
    generate(args.output)
    print(f"Generated eight measurement CSV files and covariance in {args.output}")
