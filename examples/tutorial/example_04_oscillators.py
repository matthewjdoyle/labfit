"""Intermediate example: independently fit three damped-oscillation runs."""

from pathlib import Path

import numpy as np
from common import DATA, ROOT, ExampleOutput, print_output, raw_labels, save_figure

from labfit import fit_multi, plot_multi_fit
from labfit.io import load_csv


def run(data: Path = DATA, figures: Path = ROOT / "figures") -> ExampleOutput:
    # tutorial:begin analysis
    series = [load_csv(data / f"oscillator_{i}.csv", label=f"Run {i}") for i in range(1, 4)]
    first = series[0]
    centred = first.y - np.mean(first.y)
    spectrum = np.abs(np.fft.rfft(centred))
    frequencies = np.fft.rfftfreq(len(first.x), d=float(first.x[1] - first.x[0]))
    frequency_guess = float(frequencies[1:][np.argmax(spectrum[1:])])
    results = fit_multi(
        series,
        model="damped_sine",
        absolute_sigma=True,
        p0={
            "amplitude": float(np.ptp(first.y) / 2),
            "damping": 0.1,
            "frequency": frequency_guess,
            "phase": 0.0,
            "offset": float(first.y.mean()),
        },
        bounds={
            "amplitude": (0, 10),
            "damping": (0.001, 1),
            "frequency": (0.5, 1.3),
            "phase": (-np.pi, np.pi),
            "offset": (-1, 1),
        },
    )
    plot = plot_multi_fit(results, series, figsize=(3.6, 3.7), xlabel="Time (s)", ylabel="Displacement (mm)")
    raw_labels(plot, "mm")
    save_figure(plot, "04_oscillators", figures)
    # tutorial:end analysis
    return ExampleOutput({f"run_{i}": fitted for i, fitted in enumerate(results, start=1)})


if __name__ == "__main__":
    print_output(run())
