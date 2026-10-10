"""Simple example: voltage calibration against reference temperature."""

from pathlib import Path

import numpy as np
from common import DATA, ROOT, ExampleOutput, print_output, raw_labels, save_figure

from labfit import fit, plot_fit
from labfit.io import load_csv


def run(data: Path = DATA, figures: Path = ROOT / "figures") -> ExampleOutput:
    # tutorial:begin analysis
    series = load_csv(data / "calibration.csv", label="Sensor calibration")
    result = fit(series, model="linear", absolute_sigma=True)
    temperature = 25.0
    prediction = float(result.predict([temperature])[0])
    gradient = np.array([temperature, 1.0])  # slope, intercept
    prediction_error = float(np.sqrt(gradient @ result.covariance @ gradient))

    confidence = plot_fit(
        result,
        show_ci=True,
        ci_level=0.95,
        xlabel="Reference temperature (deg C)",
        ylabel="Sensor voltage (V)",
        title="Linear calibration: mean-response confidence band",
    )
    raw_labels(confidence, "V")
    save_figure(confidence, "01_calibration", figures)
    future = plot_fit(
        result,
        show_ci=True,
        prediction=True,
        ci_level=0.95,
        prediction_sigma=0.045,
        xlabel="Reference temperature (deg C)",
        ylabel="Sensor voltage (V)",
        title="Linear calibration: future-observation prediction band",
    )
    raw_labels(future, "V")
    save_figure(future, "01_prediction", figures)
    # tutorial:end analysis
    return ExampleOutput(
        {"calibration": result}, {"Mean voltage at 25 deg C": (prediction, prediction_error, "V")}
    )


if __name__ == "__main__":
    print_output(run())
