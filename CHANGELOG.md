# Changelog

All notable changes to LabFit are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - Unreleased

### Breaking
- Renamed the internal `labfit.fit` submodule to `labfit._fit` to resolve the
  naming collision with the public `labfit.fit` function. The two public entry
  points are now `fit(x, y, *, model=...)` (arrays/CSV first) and
  `fit_curve(model, x, y, y_err, ...)` (model first).
- Removed the `quick_fit` and `fit_to_model` aliases — both were thin wrappers
  around `fit`. Use `fit` directly.
- Removed `labfit.utils.load_csv` (returned a 5-tuple). Use `labfit.io.load_csv`
  (returns a `DataSeries`) everywhere.
- Removed `labfit.utils.as_array` (a trivial `np.asarray` wrapper with no added
  value).
- Dropped the `ysigma` keyword from `fit` and `_fit_single`; use `sigma`.
- `DataSeries.y_err` is now a read-only alias for `sigma`; `sigma` is the
  canonical field.
- `FitResult.__repr__` now produces a concise single-line representation
  distinct from the human-readable `__str__`.
- `plot.py` no longer forces the `Agg` matplotlib backend at import time. Callers
  (or tests) that need a non-interactive backend should set it themselves.

### Added
- `LICENSE` file (MIT).
- `CHANGELOG.md` and `CONTRIBUTING.md`.
- `FitResult.is_weighted` flag indicating whether the fit used y-uncertainties.
- Confidence/prediction bands on fit lines via `show_ci` / `ci_level` /
  `prediction` parameters on `plot_fit`.
- Numerical-Jacobian fallback in `propagate_errors` when `jacobian` is omitted
  but `covariance` is provided.
- Completed automatic initial guesses for `sinc`, `exponential_rise`,
  `double_exponential`, `moffat`, `gaussian_baseline`, `bimodal_gaussian`.

### Changed
- `reduced_chi2` is now accompanied by `is_weighted`; when no uncertainties are
  provided a `UserWarning` is emitted clarifying that the value is the
  unweighted SSR per degree of freedom, not a true reduced χ².
- `Dataset` uses the generated `@dataclass` `__init__` instead of a manual
  override.
- `use_publication_style` applies rcParams via a context manager instead of
  mutating global matplotlib state.
- Refactored `_initial_guess` into a per-model guess registry.
- Consolidated `mypy`, `ruff`, and `coverage` configuration into `pyproject.toml`.

### Fixed
- `DataSeries` no longer maintains duplicate `sigma`/`y_err` sync logic.
- `_model_wrapper` no longer resolves the model twice.
- Singular covariance matrices now emit a `UserWarning` instead of silently
  returning NaN uncertainties.
- Benchmark harness no longer hardcodes `/home/matt/projects/labfit`.
- Private helper functions removed from `plot.__all__`.

## [0.1.0] - 2026-06-08

Initial public release.
