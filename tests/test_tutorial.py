"""Scientific integration checks for the six executable tutorial analyses."""

from __future__ import annotations

import importlib
import json
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from scipy.linalg import solve_triangular

TUTORIAL = Path(__file__).resolve().parents[1] / "examples" / "tutorial"


@pytest.fixture(scope="module")
def tutorial_modules() -> Iterator[dict[str, ModuleType]]:
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(TUTORIAL))
        yield {
            name: importlib.import_module(name)
            for name in ("generate_data", "run_examples", "build_tutorial")
        }


@pytest.fixture(scope="module")
def computed(
    tutorial_modules: dict[str, ModuleType], tmp_path_factory: pytest.TempPathFactory
) -> dict[str, Any]:
    destination = tmp_path_factory.mktemp("tutorial")
    manifest = tutorial_modules["generate_data"].generate(destination / "data")
    with pytest.warns(UserWarning, match="much larger than 1"):
        outputs = tutorial_modules["run_examples"].run_all(destination / "data", destination)
    return {"destination": destination, "manifest": manifest, "outputs": outputs}


def test_generated_measurements_are_reproducible(computed: dict[str, Any]) -> None:
    generated = computed["destination"] / "data"
    for source in (TUTORIAL / "data").glob("*.csv"):
        assert source.read_bytes() == (generated / source.name).read_bytes()


def test_thermal_covariance_matches_observation_order(computed: dict[str, Any]) -> None:
    data = computed["destination"] / "data"
    measured = np.loadtxt(data / "thermal.csv", delimiter=",", skiprows=1)
    covariance = np.loadtxt(data / "thermal_covariance.csv", delimiter=",")
    index = np.arange(len(measured))
    expected = 0.20**2 * 0.78 ** np.abs(index[:, None] - index[None, :]) + 0.06**2 * np.eye(len(index))
    np.testing.assert_allclose(covariance, expected, atol=1e-15)
    np.testing.assert_allclose(np.sqrt(np.diag(covariance)), measured[:, 2], rtol=1e-11)
    assert np.all(np.linalg.eigvalsh(covariance) > 0)


def test_fits_recover_truth_and_report_consistent_diagnostics(computed: dict[str, Any]) -> None:
    datasets = computed["manifest"]["datasets"]
    mapping = {
        "example_01_calibration": {"calibration": "calibration"},
        "example_02_decay": {"decay": "decay"},
        "example_03_single_peak": {"baseline": "single_peak"},
        "example_04_oscillators": {f"run_{i}": f"oscillator_{i}" for i in range(1, 4)},
        "example_05_double_peak": {"double_peak": "double_peak"},
        "example_06_correlated": {"correlated": "thermal"},
    }
    for example, fits in mapping.items():
        for label, dataset in fits.items():
            result = computed["outputs"][example].fits[label]
            assert result.success and result.identifiable
            assert result.absolute_sigma and result.dof > 0
            assert np.all(np.isfinite(result.predict(result.x)))
            truth = datasets[dataset]["true_parameters"]
            for name, fitted in result.params.items():
                error = result.uncertainties[name]
                assert np.isfinite(error) and error > 0
                floor = 1e-6 * max(1.0, abs(truth[name]))
                assert abs(fitted - truth[name]) <= 5 * error + floor
            residual = result.residuals
            if result.series.sigma_cov is not None:
                z = solve_triangular(np.linalg.cholesky(result.series.sigma_cov), residual, lower=True)
            else:
                z = residual / result.sigma
            assert result.reduced_chi2 * result.dof == pytest.approx(z @ z, rel=1e-11)
            assert result.residual_variance == pytest.approx(residual @ residual / result.dof)


def test_missing_background_is_detectably_inadequate(computed: dict[str, Any]) -> None:
    fits = computed["outputs"]["example_03_single_peak"].fits
    assert fits["missing_background"].success
    assert fits["missing_background"].reduced_chi2 > 10 * fits["baseline"].reduced_chi2
    assert fits["baseline"].reduced_chi2 < 2


def test_derived_peak_separation_retains_covariance(computed: dict[str, Any]) -> None:
    output = computed["outputs"]["example_05_double_peak"]
    result = output.fits["double_peak"]
    i, j = (result.param_names.index(name) for name in ("c1", "c2"))
    expected_variance = result.covariance[i, i] + result.covariance[j, j] - 2 * result.covariance[i, j]
    value, error, _ = output.derived["Peak separation"]
    assert value == pytest.approx(result.params["c2"] - result.params["c1"])
    assert error**2 == pytest.approx(expected_variance)
    assert expected_variance != pytest.approx(result.covariance[i, i] + result.covariance[j, j])


def test_gls_objective_matches_whitened_norm(computed: dict[str, Any]) -> None:
    metrics = computed["outputs"]["example_06_correlated"].metrics
    assert metrics["whitened_chi2"] == pytest.approx(metrics["reported_chi2"], rel=1e-12)


def test_document_source_resolves_code_results_and_assets(tutorial_modules: dict[str, ModuleType]) -> None:
    build = tutorial_modules["build_tutorial"]
    rendered = build.render_markdown()
    assert rendered == (TUTORIAL / "tutorial.md").read_text(encoding="utf-8")
    assert "{{" not in rendered
    records = json.loads((TUTORIAL / "results.json").read_text(encoding="utf-8"))
    value = records["examples"]["example_02_decay"]["derived"]["Half-life"][0]
    assert f"{value:.6g}" in rendered
    assert "absolute_sigma=True" in rendered
