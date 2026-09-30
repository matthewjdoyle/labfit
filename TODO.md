# LabFit improvement tracker

Based on the repository review of 2026-09-30. Keep the review IDs stable.
Check an item only when its acceptance criteria are implemented and verified;
record partial progress without closing the whole item.

Progress: **13 of 14 review items complete**; R14 partially addressed.

P1: incorrect or misleading scientific results. P2: reliability, usability,
performance, and maintenance.

## Scientific correctness

- [x] **R01 / P1: Correct the ExGaussian model.** Fix the reversed erfc argument
  and missing normalization; define amplitude as integrated area, matching the
  documented equation. Verify against an independent convolution and tail limits.
  Source: `labfit/models.py`, `docs/fitting-functions.rst`.
- [x] **R02 / P1: Distinguish absolute uncertainties from relative weights.**
  Add and document an explicit covariance-scaling policy (for example,
  `absolute_sigma`). Scaling all absolute sigmas by 10 must scale parameter
  uncertainties by 10. Test both policies and correlated covariance inputs.
  Source: `labfit/fitter_impl.py`.
- [x] **R03 / P1: Detect unidentifiable parameters.** Inspect Jacobian rank and
  conditioning rather than only finite covariance diagonals. A model `(a+b)*x`
  must not report both parameters as exactly known. Warn and mark uncertainties
  unreliable while distinguishing fit convergence from identifiability.
  Source: `labfit/fitter_impl.py`.
- [x] **R04 / P1: Correct prediction-band noise scaling.** Do not add
  dimensionless reduced chi-square to variance in y-units squared. Define future
  observation noise, including heteroscedastic data, and test invariance under
  unit changes. Validate confidence levels.
  Source: `labfit/plot.py`.
- [x] **R05 / P1: Report statistics only when valid.** Preserve actual degrees
  of freedom instead of clamping to one. Suppress chi-square p-values and
  measurement-error advice for unweighted fits; expose residual variance
  separately. Test zero/negative degrees of freedom and unweighted summaries.
  Source: `labfit/fitter_impl.py`, `labfit/types.py`.
- [x] **R06 / P1: Correct educational documentation.** Fix reversed chi-square
  explanations and the half-life formula and derivative; supply uncertainty in
  the propagation example. Replace unsupported asymmetric sigma tuple examples.
  Verify the half-life example numerically.
  Source: `docs/concepts.rst`.
- [x] **R07 / P1: Centralize uncertainty validation.** Validate normalized
  overrides as well as raw arrays: finite values, dimensions, positive sigmas,
  explicit zero-weight policy, nonnegative weights, complete asymmetric pairs,
  mutually exclusive specifications, symmetric positive-definite covariance.
  Negative weights must not silently discard all data and produce chi-square 0.
  Source: `labfit/types.py`, `labfit/fitter_impl.py`, `labfit/utils.py`.

## Input and plotting reliability

- [x] **R08 / P2: Make inferred file uncertainties explicit.** Add an unweighted
  loading mode and record/warn about inferred errors. Decide a defensible policy
  for zero counts; a CSV with y=[0,1,2] currently fails with zero sigma.
  Source: `labfit/io.py`.
- [x] **R09 / P2: Reject malformed guesses and parameter typos.** Reject unknown
  p0/bounds keys and incorrectly shaped or sized p0 arrays; never silently resize.
  Explain partial-dictionary defaults and list valid parameter names in errors.
  Source: `labfit/fitter_impl.py`.
- [x] **R10 / P2: Preserve supplied uncertainties in plots.** Render separate
  lower/upper error bars and marginal sqrt(diag(covariance)) bars. Explain that
  marginal bars do not visualize correlations. Test actual plotted endpoints.
  Source: `labfit/plot.py`, `docs/faq.rst`.

## Engineering quality

- [x] **R11 / P2: Add independent scientific regression tests.** Complement the
  90% coverage gate with analytical linear covariance, rank deficiency, unit
  scaling, independent model references, prediction bands and runnable examples.
  Avoid validating a formula only by fitting data generated with that formula.
  Source: `tests/`.
  Progress (2026-09-30): added 21 regression cases covering independent
  ExGaussian convolution/area/tails, parameter validation, plotted error-bar
  endpoints, and execution of the documented half-life example. Statistical
  additional covariance, rank-deficiency and prediction-band tests were planned.
  Progress (2026-09-30, R02): added 14 cases for analytical linear covariance,
  absolute/relative scaling with sigmas, weights and correlated covariance,
  unweighted policy, default compatibility, and public entry-point forwarding.
  Rank-deficiency and prediction-band tests remained open at that point.
  Progress (2026-09-30, R03): added five cases for redundant, nearly
  redundant, zero-sensitivity, and identifiable parameter fits. Prediction-band
  tests remain open.
  Progress (2026-09-30, R05): added 22 cases for analytical chi-square/p-values,
  raw residual variance and y-unit scaling, unweighted summaries, zero/negative
  degrees of freedom, absolute covariance without residual degrees of freedom,
  and p-value suppression after rank failure or nonconvergence. Prediction-band
  tests remain open.
  Progress (2026-09-30, R04): added 53 cases for analytical pointwise mean and
  prediction bands under absolute/relative covariance policies, homogeneous and
  heterogeneous future noise, correlated training errors, x/y unit scaling,
  invalid confidence levels/noise, actual rendered endpoints, and execution of
  the complete documented prediction-band example. Together with the preceding
  covariance, rank, model-reference and half-life-example regressions, all R11
  acceptance criteria are now covered and verified.
- [x] **R12 / P2: Optimize correlated fitting and confidence bands.** Use a
  triangular solve for Cholesky factors and vectorize model evaluations across
  the prediction grid. Extend benchmarks and compare numerical results/timings.
  Source: `labfit/fitter_impl.py`, `labfit/plot.py`, `benchmarks/`.
- [x] **R13 / P2: Enforce the documented typing policy.** Annotate public APIs,
  enable check_untyped_defs during migration, then enforce untyped-definition
  checks. Include a py.typed marker for downstream consumers.
  Source: `pyproject.toml`, `CONTRIBUTING.md`, `labfit/`.
- [ ] **R14 / P2: Verify distributions and gate releases.** Smoke-test the built
  wheel in a clean environment, verify the release commit before publishing,
  expand beyond Linux-only CI, and use reproducible development-tool versions.
  Derive documentation version from package metadata.
  Source: `.github/workflows/`, `pyproject.toml`, `docs/conf.py`.
  Progress (2026-09-30): documentation version now comes from pyproject.toml,
  with a conditional tomli dependency for Python 3.10. Release/CI work remains.

## Verification log

- Review baseline: Windows / Python 3.13.1; 40 tests passed; 90.95% coverage;
  formatting and mypy passed. Ruff 0.11.10 reported five UP038 findings.

- Quick-win batch (2026-09-30): completed R01, R06, R09 and R10. Added
  `tests/test_review_regressions.py`; updated CHANGELOG.md. Also fixed the five
  existing UP038 lint findings. No changes to covariance scaling or p-value
  semantics in this batch.
- Validation: 61 tests passed (28 existing warnings), 91.11% package coverage;
  Ruff 0.11.10 lint/format checks and mypy passed. HTML docs built successfully
  with warnings treated as errors using installed Sphinx 8.2.3 (the declared
  docs dependency range is Sphinx 7.x). Python 3.10 fallback was not exercised.
- Compatibility: ExGaussian results change because the old curve was incorrect;
  amplitude is now integrated area. Invalid p0/bounds keys and malformed p0
  arrays now raise ValueError. Partial dictionaries remain supported.

- R02 (2026-09-30): implemented `absolute_sigma` across fitting APIs and
  recorded the policy on FitResult; the default False preserves relative scaling.
  Independent analytical tests verify that 10x absolute sigmas produce 10x
  parameter uncertainties, including correlated covariance scaled by 100.
  Validation: 75 tests passed (28 existing warnings), 91.14% package coverage;
  Ruff lint/format checks and mypy passed. HTML docs built with warnings as
  errors using installed Sphinx 8.2.3 (declared docs range remains Sphinx 7.x).

- R03 (2026-09-30): diagnosed Jacobian rank and column-normalized condition
  after optimization. Unidentifiable fits warn, retain optimizer convergence
  status, and expose unavailable covariance and uncertainties as NaN.
  Validation: 80 tests passed (30 warnings), 91.46% package coverage;
  Ruff lint/format checks and mypy passed. HTML docs built successfully with
  warnings treated as errors using installed Sphinx 8.2.3.

- R05 (2026-09-30): preserved actual N - k degrees of freedom and exposed
  unweighted SSR / dof as `residual_variance` in squared y units. Unweighted
  chi-square statistics now return NaN and summaries omit measurement-error
  advice. Nonpositive dof makes residual-scale statistics and relative parameter
  covariance unavailable; identifiable absolute covariance remains supported.
  P-values also require convergence and identifiable parameters. Added 22
  independent statistics regression cases and updated the API docs/changelog.
  Validation: 102 tests passed (28 warnings), 91.99% package coverage;
  Ruff 0.11.10 lint/format checks and mypy passed. HTML docs built successfully
  with warnings treated as errors using installed Sphinx 8.2.3 (declared docs
  range remains Sphinx 7.x).

- R04 and R11 (2026-09-30): prediction bands now add future observation noise
  in squared y units. Added `prediction_sigma` for absolute future errors as a
  scalar, plotted-grid array, or callable. Constant training errors follow the
  covariance policy; heterogeneous/correlated errors require explicit future
  noise. Unweighted relative fits use residual variance; unweighted absolute
  fits retain their unit-variance assumption. Validate confidence levels and
  future sigmas before creating or modifying figures. Bands remain pointwise
  normal approximations for future errors independent of the fitted data.
  Added 53 regression cases; updated documentation and changelog. R11 is complete
  after verifying independent covariance/rank/model/unit/band references and
  runnable half-life and prediction examples across the regression suite.
  Validation: 155 tests passed (28 warnings), 92.81% package coverage;
  Ruff 0.11.10 lint/format checks and mypy passed. HTML docs built successfully
  with warnings treated as errors using installed Sphinx 8.2.3 (declared docs
  range remains Sphinx 7.x).

- R07 (2026-09-30): centralized error validation across DataSeries,
  AsymmetricError, effective_sigma and fitting, including normalized overrides.
  Reject nonfinite/malformed errors, nonpositive sigmas, incomplete asymmetric
  pairs, competing specifications, and nonsymmetric/non-positive-definite
  covariance. Negative weights now raise instead of silently discarding data;
  finite zero weights explicitly exclude samples from model evaluation, initial
  guesses, result data/default plots and all statistics. Require at least one
  retained sample and use its count for degrees of freedom. Explicit fitting
  errors replace the complete stored specification; result.series carries the
  errors actually fitted and preserves asymmetric sides. CSV/TXT asymmetric
  columns no longer trigger competing inferred symmetric errors. Alias agreement
  uses a relative tolerance independent of measurement units; asymmetric RMS
  avoids overflow. Added 282 regression cases, including independent weighted
  linear references, override/entry-point validation, plotted sample exclusion,
  covariance scaling and execution of the documented zero-weight example.
  Validation: 437 tests passed (29 warnings), 93.66% package coverage;
  Ruff 0.11.10 lint/format checks and mypy passed. HTML docs built successfully
  with warnings treated as errors using installed Sphinx 8.2.3 (declared docs
  range remains Sphinx 7.x). Zero-count inferred errors remain tracked by R08.

- R08 (2026-09-30): added error_mode="unweighted" to CSV/TXT loaders, reading
  only x/y and disabling both error columns and inference. The default auto mode
  remains, but every successful inference emits UncertaintyInferenceWarning
  naming the file and formula. DataSeries.uncertainties_inferred and
  uncertainty_inference retain the formula/settings through relabeling,
  combination and fitting; summaries identify inferred errors. Explicit fit
  overrides clear result provenance and bypass unused CSV error columns/inference.
  Counting inference now requires nonnegative integer values and uses
  sqrt(max(y, 1)): zero counts retain a finite sigma of 1, while positive counts
  keep sqrt(y). This is a documented one-count variance-floor approximation for
  exploration, not an exact Poisson interval. Fractional inference validates its
  scale, rejects zero-valued measurements with actionable guidance, and rejects
  overflow/underflow. Mode validation also runs when errors are supplied; explicit
  file errors still take precedence in all inference modes. Added 127 regression
  cases for both loaders, zero counts, warnings, provenance, explicit overrides,
  independent weighted/unweighted linear references and a runnable loading example.
  Validation: 564 tests passed (29 warnings), 94.21% package coverage;
  Ruff 0.11.10 lint/format checks and mypy passed. HTML docs built successfully
  with warnings treated as errors using installed Sphinx 8.2.3 (declared docs
  range remains Sphinx 7.x).

- R12 (2026-09-30): replaced general dense solves on lower Cholesky factors
  with scipy.linalg.solve_triangular for correlated residuals and chi-square.
  Confidence/prediction bands evaluate perturbations across the whole grid and
  contract parameter gradients/covariance in one array operation. Model calls
  now total 2*k + 1 rather than 2*k*N + 1. Added 22 regression cases for independent
  generalized linear statistics, analytical nonlinear Gaussian band derivatives,
  grid-independent call counts, edge grids and benchmark/report verification.
  Extended benchmarks with a deterministic reference/optimized runner covering
  whitening, full correlated linear/Gaussian fits and both band types. Alternating
  timing order, 2 warmups and 7 measured repetitions; seed 20260930, single-thread
  BLAS settings, sizes 32/128/512 and band grids 200/2000. All numerical comparisons
  passed at rtol=2e-6, atol=1e-9; maximum parameter difference 3.46e-10 and band
  endpoint difference 8.88e-16. At 512 observations, whitening improved 46.91x;
  complete linear/Gaussian fits improved 1.76x/2.39x. At 2000 grid points, linear
  confidence/prediction calculations improved 169.98x/136.07x, and Gaussian
  calculations improved 229.40x/192.34x. Band timings exclude rendering and I/O;
  full-fit timings include unchanged covariance validation/factorization.
  Results and method: benchmarks/R12_RESULTS.md and benchmarks/README.md.
  Raw timings, numerical deltas, configuration/environment and exact source hashes
  are saved under benchmarks/results/r12/20260930T131712874562Z/.
  Validation: 586 tests passed (29 warnings), 94.44% package coverage;
  Ruff 0.11.10 lint/format checks and mypy passed. HTML docs built successfully
  with warnings treated as errors using installed Sphinx 8.2.3 (declared docs
  range remains Sphinx 7.x).

- R13 (2026-09-30): annotated fitting, loading, plotting, model and uncertainty
  APIs plus container methods, private helpers and nested functions. Typed model
  decorators preserve array-like x inputs, named parameters and return types;
  dataclass constructor types accept array-like values while stored fields remain
  normalized arrays. Replaced untyped guess statistics with a typed NamedTuple
  and uncertainty dictionaries with a TypedDict. Enabled check_untyped_defs and
  disallow_untyped_defs without per-module enforcement exceptions. Added a
  py.typed marker and explicit setuptools package-data inclusion; updated
  CONTRIBUTING.md and CHANGELOG.md. Two consumer regression tests verify valid
  public calls and inferred types, reject invalid argument/return types and model
  parameter names, and enforce rejection of untyped definitions.
  Validation: 588 tests passed (29 existing warnings), 94.59% package coverage;
  Ruff 0.11.10 lint/format checks and mypy 1.17.0 passed, including a static
  check targeting Python 3.10. HTML docs built with
  warnings as errors using installed Sphinx 8.2.3 (declared docs range remains
  Sphinx 7.x). Built wheel and source distribution in a temporary staging copy;
  both include labfit/py.typed. Installed the wheel into a separate directory,
  confirmed its import path and a successful linear fit, and verified positive
  and negative external mypy consumers against that installed package. These
  checks reused the existing scientific dependencies; the fully clean-environment
  installation and release gates remain part of R14. Validation environment:
  Windows / Python 3.13.1; older supported Python runtimes were not executed.

## Next suggested batch

Address R14: smoke-test distributions in a clean environment, verify the release
commit before publishing, expand CI beyond Linux and make development-tool
versions reproducible. R01-R13 are complete; R14 remains partially addressed.
