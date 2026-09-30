# Changelog

All notable changes to LabFit are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - Unreleased

### Breaking
- Prediction bands for heterogeneous or correlated training errors now require
  `prediction_sigma` to define future observation noise. Invalid confidence
  levels raise ValueError instead of producing invalid band endpoints.
- Unweighted fits now return NaN for `reduced_chi2` and `p_value`. Use
  `residual_variance` for the unweighted SSR per degree of freedom in squared y units.
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
- `plot_fit(..., prediction_sigma=...)` accepts absolute future standard
  deviations in y units as a scalar, plotted-grid array, or function of x.
- `FitResult.dof` preserves N minus the number of parameters, including zero
  and negative values; `residual_variance` exposes raw residual scatter separately.
- `absolute_sigma` fitting option and result flag: use `True` for absolute
  measurement uncertainties, or the compatible default `False` to estimate
  a common noise scale from residuals. Applies to correlated errors and weights.
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
- Unweighted fits warn that chi-square statistics are unavailable, and summaries
  show residual variance without measurement-error advice.
- `Dataset` uses the generated `@dataclass` `__init__` instead of a manual
  override.
- `use_publication_style` applies rcParams via a context manager instead of
  mutating global matplotlib state.
- Refactored `_initial_guess` into a per-model guess registry.
- Consolidated `mypy`, `ruff`, and `coverage` configuration into `pyproject.toml`.

### Fixed
- Prediction bands now add observation variance in squared y units, using
  constant training errors and the covariance policy or unweighted SSR / dof.
  Explicit future errors are never scaled by reduced chi-square. Validate
  confidence levels and future errors before modifying plot axes; retain finite
  normal quantiles for confidence levels arbitrarily close to one.
- Stop clamping degrees of freedom to one. Nonpositive dof now yields unavailable
  residual variance, reduced chi-square, p-value and relative parameter covariance.
  Absolute covariance remains available for identifiable parameters. Suppress
  p-values for unidentifiable or unconverged fits.
- Detect rank-deficient and severely ill-conditioned fit Jacobians. Report
  convergence separately from identifiability, warn, and return unavailable
  parameter covariance and uncertainties when they cannot be trusted.
- Corrected the ExGaussian tail and normalization; amplitude now matches the
  documented integrated-area convention. Previously fitted values may change.
- Reject unknown initial-guess/bounds keys and incorrectly sized or shaped
  initial guesses instead of silently ignoring or resizing them.
- Preserve asymmetric error bars and show marginal errors for covariance inputs.
- Corrected chi-square interpretation and half-life propagation documentation.
- Read the documentation version from pyproject.toml.
- `DataSeries` no longer maintains duplicate `sigma`/`y_err` sync logic.
- `_model_wrapper` no longer resolves the model twice.
- Singular covariance matrices now emit a `UserWarning` instead of silently
  returning NaN uncertainties.
- Benchmark harness no longer hardcodes `/home/matt/projects/labfit`.
- Private helper functions removed from `plot.__all__`.

## [0.1.0] - 2026-06-08

Initial public release.
