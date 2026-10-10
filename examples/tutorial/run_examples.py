"""Run six demonstrations and record computed results for the tutorial."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
from pathlib import Path

import numpy as np
from common import DATA, ROOT, ExampleOutput, result_record, write_json

EXAMPLES = (
    "example_01_calibration",
    "example_02_decay",
    "example_03_single_peak",
    "example_04_oscillators",
    "example_05_double_peak",
    "example_06_correlated",
)


def run_all(data: Path = DATA, output: Path = ROOT) -> dict[str, ExampleOutput]:
    results = {}
    for name in EXAMPLES:
        module = importlib.import_module(name)
        result = module.run(data, output / "figures")
        for label, fitted in result.fits.items():
            if (
                not fitted.success
                or not fitted.identifiable
                or not all(np.isfinite(list(fitted.uncertainties.values())))
            ):
                raise RuntimeError(f"Unreliable fit: {name}/{label}: {fitted.message}")
        results[name] = result
        print(f"Completed {name}")
    write_json(
        output / "results.json",
        {
            "versions": {
                name: importlib.metadata.version(name) for name in ("labfit", "numpy", "scipy", "matplotlib")
            },
            "examples": {name: result_record(result) for name, result in results.items()},
        },
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    run_all(args.data, args.output)
