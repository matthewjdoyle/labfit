"""Exercise file error policies, their provenance, and independent fit statistics."""

import textwrap
import warnings
from pathlib import Path

import numpy as np
import pytest

from labfit import DataSeries, Fitter, fit, fit_multi
from labfit.io import UncertaintyInferenceWarning, combine_series, load_csv, load_txt


@pytest.fixture(params=[load_csv, load_txt], ids=["csv", "txt"])
def table_loader(request, tmp_path):
    loader = request.param
    separator = "," if loader is load_csv else " "

    def load(y, *, columns=None, header=True, **kwargs):
        columns = columns or {}
        names = ["x", "y", *columns]
        rows = [separator.join(names)] if header else []
        for index, value in enumerate(y):
            rows.append(
                separator.join(map(str, [index, value, *(values[index] for values in columns.values())]))
            )
        path = tmp_path / ("data.csv" if loader is load_csv else "data.txt")
        path.write_text("\n".join(rows), encoding="utf-8")
        if not header:
            kwargs.setdefault("x_col", 0)
            kwargs.setdefault("y_col", 1)
        return loader(path, **kwargs), path

    return load


@pytest.mark.parametrize("mode", ["auto", "poisson", "POISSON"])
def test_zero_counts_have_positive_errors_and_an_explicit_assumption(table_loader, mode):
    with pytest.warns(UncertaintyInferenceWarning, match="one-count variance floor") as captured:
        series, path = table_loader([0, 1, 2], error_mode=mode)
    assert len(captured) == 1
    assert str(path) in str(captured[0].message)
    assert "error_mode='unweighted'" in str(captured[0].message)
    np.testing.assert_allclose(series.sigma, [1, 1, np.sqrt(2)])
    assert series.uncertainties_inferred
    assert series.uncertainty_inference == "poisson: sqrt(max(y, 1)) (one-count variance floor)"

    result = fit(series, absolute_sigma=True)
    np.testing.assert_allclose(list(result), [1, 0], atol=1e-8)
    design = np.array([[0, 1], [1, 1], [2, 1]])
    precision = np.diag([1, 1, 0.5])
    np.testing.assert_allclose(result.covariance, np.linalg.inv(design.T @ precision @ design), rtol=1e-6)
    assert result.dof == 1
    assert result.success and result.identifiable and result.is_weighted
    assert result.series.uncertainty_inference == series.uncertainty_inference
    assert "measurement errors inferred: poisson" in str(result)


def test_poisson_floor_keeps_all_zero_data(table_loader):
    with pytest.warns(UncertaintyInferenceWarning):
        series, _ = table_loader([0, 0, 0])
    np.testing.assert_array_equal(series.sigma, np.ones(3))
    np.testing.assert_array_equal(series.y, np.zeros(3))
    assert series.x.size == 3


@pytest.mark.parametrize("mode", ["auto", "fraction"])
def test_fractional_inference_records_the_selected_fraction(table_loader, mode):
    with pytest.warns(UncertaintyInferenceWarning, match=r"fraction: abs\(y\) \* 0.1"):
        series, _ = table_loader([-2.5, 5.5, 10.5], error_mode=mode, default_fraction=0.1)
    np.testing.assert_allclose(series.sigma, [0.25, 0.55, 1.05])
    assert series.uncertainty_inference == "fraction: abs(y) * 0.1"
    assert series.uncertainties_inferred


def test_near_integer_values_use_fractional_inference(table_loader):
    with pytest.warns(UncertaintyInferenceWarning, match="fraction"):
        series, _ = table_loader([1.000001, 2.000001, 3.000001])
    np.testing.assert_allclose(series.sigma, series.y * 0.05)


@pytest.mark.parametrize("y", [[-1, 2, 3], [0, 1.5, 2]])
def test_poisson_inference_rejects_data_that_are_not_raw_counts(table_loader, y):
    with pytest.raises(ValueError, match="nonnegative integer counts"):
        table_loader(y, error_mode="poisson")


@pytest.mark.parametrize("mode, y", [("fraction", [0, 1, 2]), ("auto", [0, 1.5, 2])])
def test_fractional_inference_does_not_invent_a_scale_for_zero_y(table_loader, mode, y):
    with (
        warnings.catch_warnings(record=True) as captured,
        pytest.raises(ValueError, match="zero y-values.*unweighted"),
    ):
        table_loader(y, error_mode=mode)
    assert not captured


@pytest.mark.parametrize("fraction", [0, -0.1, np.nan, np.inf, "invalid", [0.1, 0.2]])
def test_fractional_inference_validates_default_fraction(table_loader, fraction):
    with pytest.raises(ValueError, match="default_fraction must be finite and strictly positive"):
        table_loader([1.5, 2.5], error_mode="fraction", default_fraction=fraction)


@pytest.mark.parametrize("y, fraction", [([1.5, 2.5], np.finfo(float).max), ([1e-300, 2e-300], 1e-300)])
def test_fractional_inference_rejects_numerical_overflow_and_underflow(table_loader, y, fraction):
    with (
        np.errstate(over="raise", under="raise", invalid="raise"),
        pytest.raises(ValueError, match="at this scale"),
    ):
        table_loader(y, error_mode="fraction", default_fraction=fraction)


@pytest.mark.parametrize(
    "columns",
    [
        {},
        {"sigma": [0, -1, "invalid", np.nan]},
        {"sigma_low": ["invalid"] * 4},
        {"sigma_high": ["invalid"] * 4},
        {"sigma_low": ["invalid"] * 4, "sigma_high": ["invalid"] * 4},
        {"sigma": ["invalid"] * 4, "sigma_low": ["invalid"] * 4, "sigma_high": ["invalid"] * 4},
        {"y_err_frac": ["invalid"] * 4},
    ],
)
def test_unweighted_loading_ignores_errors_without_inference(table_loader, columns):
    with warnings.catch_warnings(record=True) as captured:
        series, _ = table_loader([0, 1.5, -2, 3], columns=columns, error_mode="unweighted", label="raw")
    assert not captured
    np.testing.assert_array_equal(series.y, [0, 1.5, -2, 3])
    assert series.label == "raw"
    assert series.sigma is series.y_err is series.sigma_low is series.sigma_high is series.sigma_cov is None
    assert series.effective_sigma is None
    assert series.uncertainty_inference is None
    assert not series.uncertainties_inferred
    with pytest.warns(UserWarning, match="No uncertainties provided"):
        result = fit(series)
    design = np.column_stack([series.x, np.ones(4)])
    reference = np.linalg.lstsq(design, series.y, rcond=None)[0]
    np.testing.assert_allclose(list(result), reference, atol=1e-7)
    assert not result.is_weighted
    assert np.isnan(result.reduced_chi2) and np.isnan(result.p_value)
    assert not result.series.uncertainties_inferred


def test_unweighted_mode_rejects_an_explicit_error_selector(table_loader):
    with pytest.raises(ValueError, match="y_err_col cannot be combined"):
        table_loader([0, 1, 2], error_mode="unweighted", y_err_col="sigma")


@pytest.mark.parametrize("y", [[np.nan, 1, 2], [np.inf, 1, 2]])
@pytest.mark.parametrize("mode", ["auto", "unweighted"])
def test_all_loading_modes_require_finite_data(table_loader, y, mode):
    with pytest.raises(ValueError, match="only finite values"):
        table_loader(y, error_mode=mode)


@pytest.mark.parametrize("mode", ["invalid", "", None, 5])
@pytest.mark.parametrize("explicit", [False, True])
def test_invalid_modes_are_rejected_even_with_error_columns(table_loader, mode, explicit):
    columns = {"sigma": [1, 1, 1]} if explicit else {}
    with pytest.raises(ValueError, match="error_mode must be one of"):
        table_loader([0, 1, 2], error_mode=mode, columns=columns)


@pytest.mark.parametrize("mode", ["auto", "poisson", "fraction"])
@pytest.mark.parametrize(
    "columns",
    [
        {"sigma": [0.2, 0.3, 0.4]},
        {"sigma_low": [0.2, 0.3, 0.4], "sigma_high": [0.4, 0.5, 0.6]},
        {"y_err_frac": [0.1, 0.2, 0.3]},
    ],
)
def test_supplied_error_columns_take_precedence_and_are_not_marked_inferred(table_loader, mode, columns):
    with warnings.catch_warnings(record=True) as captured:
        series, _ = table_loader([1.5, 2.5, -3.5], error_mode=mode, columns=columns, default_fraction=np.nan)
    assert not captured
    assert not series.uncertainties_inferred
    assert series.uncertainty_inference is None
    if "sigma" in columns:
        np.testing.assert_allclose(series.sigma, [0.2, 0.3, 0.4])
    elif "y_err_frac" in columns:
        np.testing.assert_allclose(series.sigma, [0.15, 0.5, 1.05])
    else:
        np.testing.assert_allclose(series.sigma_low, [0.2, 0.3, 0.4])
        np.testing.assert_allclose(series.sigma_high, [0.4, 0.5, 0.6])


def test_selected_custom_error_column_is_not_inferred(table_loader):
    with warnings.catch_warnings(record=True) as captured:
        series, _ = table_loader([0, 1, 2], columns={"custom": [0.1, 0.2, 0.3]}, y_err_col="custom")
    assert not captured
    np.testing.assert_allclose(series.sigma, [0.1, 0.2, 0.3])
    assert not series.uncertainties_inferred


@pytest.mark.parametrize("mode", ["auto", "unweighted"])
def test_headerless_tables_support_the_new_policies(table_loader, mode):
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        series, _ = table_loader([0, 1, 2], error_mode=mode, header=False)
    assert len(captured) == (1 if mode == "auto" else 0)
    assert series.uncertainties_inferred == (mode == "auto")


def test_inference_metadata_survives_relabeling_combination_and_fit_entrypoints(table_loader):
    with pytest.warns(UncertaintyInferenceWarning):
        series, _ = table_loader([0, 1, 3, 5])
    relabeled = series.with_label("counts")
    dataset = combine_series(relabeled, [series])
    assert all(s.uncertainty_inference == series.uncertainty_inference for s in dataset)
    with warnings.catch_warnings(record=True) as captured:
        results = [fit(relabeled), Fitter().fit(relabeled), *fit_multi(dataset), *Fitter().fit_multi(dataset)]
    assert not captured
    for result in results:
        assert result.series.uncertainties_inferred
        assert result.series.uncertainty_inference == series.uncertainty_inference
        assert "measurement errors inferred" in str(result)


@pytest.mark.parametrize(
    "override",
    [
        {"sigma": 1.0},
        {"weights": 1.0},
        {"sigma_low": 1.0, "sigma_high": 2.0},
        {"sigma_cov": np.eye(4)},
    ],
)
def test_explicit_fit_overrides_clear_inference_metadata(table_loader, override):
    with pytest.warns(UncertaintyInferenceWarning):
        series, _ = table_loader([0, 1, 3, 5])
    result = fit(series, **override)
    assert not result.series.uncertainties_inferred
    assert result.series.uncertainty_inference is None
    assert "measurement errors inferred" not in str(result)
    assert series.uncertainties_inferred


def test_direct_csv_fit_warns_and_preserves_inference(tmp_path):
    path = tmp_path / "counts.csv"
    path.write_text("x,y\n0,0\n1,1\n2,2\n")
    with pytest.warns(UncertaintyInferenceWarning) as captured:
        result = fit(path, label="counts")
    assert len(captured) == 1
    assert result.series.uncertainties_inferred
    assert result.series.label == "counts"
    assert np.all(np.isfinite(result.sigma)) and np.all(result.sigma > 0)


@pytest.mark.parametrize(
    "override",
    [
        {"sigma": 1.0},
        {"weights": 1.0},
        {"sigma_low": 1.0, "sigma_high": 2.0},
        {"sigma_cov": np.eye(4)},
    ],
)
@pytest.mark.parametrize("columns", ["", ",sigma", ",sigma_low"])
def test_csv_fit_with_explicit_errors_bypasses_unused_file_errors_and_inference(tmp_path, override, columns):
    path = tmp_path / "overridden.csv"
    rows = [f"x,y{columns}"]
    for index, y in enumerate([0, 1.5, 3.0, 5.5]):
        rows.append(f"{index},{y}" + (",invalid" if columns else ""))
    path.write_text("\n".join(rows))
    with warnings.catch_warnings(record=True) as captured:
        result = fit(path, **override)
    assert not captured
    assert not result.series.uncertainties_inferred
    reference = fit(np.arange(4), [0, 1.5, 3.0, 5.5], **override)
    np.testing.assert_allclose(list(result), list(reference), rtol=1e-7)
    np.testing.assert_allclose(result.covariance, reference.covariance, rtol=1e-7)


def test_manual_series_has_no_inference_provenance():
    assert not DataSeries([0, 1], [0, 1]).uncertainties_inferred
    assert not DataSeries([0, 1], [0, 1], sigma=1.0).uncertainties_inferred


def test_documented_file_loading_example(tmp_path, monkeypatch):
    document = (Path(__file__).resolve().parents[1] / "docs" / "utilities.rst").read_text(encoding="utf-8")
    section = document.split("Choosing file uncertainties", 1)[1]
    block = section.split(".. code-block:: python", 1)[1].lstrip("\n")
    lines = []
    for line in block.splitlines():
        if line and not line.startswith("   "):
            break
        lines.append(line)
    (tmp_path / "counts.csv").write_text("x,y\n0,0\n1,1\n2,2\n")
    monkeypatch.chdir(tmp_path)
    namespace = {}
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        exec(textwrap.dedent("\n".join(lines)), namespace)
    assert len(captured) == 2
    assert sum(isinstance(w.message, UncertaintyInferenceWarning) for w in captured) == 1
    assert any("No uncertainties provided" in str(w.message) for w in captured)
    assert namespace["raw_result"].is_weighted is False
    assert namespace["inferred_result"].series.uncertainties_inferred
