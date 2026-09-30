# LabFit improvement tracker

Based on the repository review of 2026-09-30. Keep the review IDs stable.
Check an item only when its acceptance criteria are implemented and verified;
record partial progress without closing the whole item.

Progress: **6 of 14 review items complete**; R11 and R14 partially addressed.

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
- [ ] **R04 / P1: Correct prediction-band noise scaling.** Do not add
  dimensionless reduced chi-square to variance in y-units squared. Define future
  observation noise, including heteroscedastic data, and test invariance under
  unit changes. Validate confidence levels.
  Source: `labfit/plot.py`.
- [ ] **R05 / P1: Report statistics only when valid.** Preserve actual degrees
  of freedom instead of clamping to one. Suppress chi-square p-values and
  measurement-error advice for unweighted fits; expose residual variance
  separately. Test zero/negative degrees of freedom and unweighted summaries.
  Source: `labfit/fitter_impl.py`, `labfit/types.py`.
- [x] **R06 / P1: Correct educational documentation.** Fix reversed chi-square
  explanations and the half-life formula and derivative; supply uncertainty in
  the propagation example. Replace unsupported asymmetric sigma tuple examples.
  Verify the half-life example numerically.
  Source: `docs/concepts.rst`.
- [ ] **R07 / P1: Centralize uncertainty validation.** Validate normalized
  overrides as well as raw arrays: finite values, dimensions, positive sigmas,
  explicit zero-weight policy, nonnegative weights, complete asymmetric pairs,
  mutually exclusive specifications, symmetric positive-definite covariance.
  Negative weights must not silently discard all data and produce chi-square 0.
  Source: `labfit/types.py`, `labfit/fitter_impl.py`, `labfit/utils.py`.

## Input and plotting reliability

- [ ] **R08 / P2: Make inferred file uncertainties explicit.** Add an unweighted
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

- [ ] **R11 / P2: Add independent scientific regression tests.** Complement the
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
- [ ] **R12 / P2: Optimize correlated fitting and confidence bands.** Use a
  triangular solve for Cholesky factors and vectorize model evaluations across
  the prediction grid. Extend benchmarks and compare numerical results/timings.
  Source: `labfit/fitter_impl.py`, `labfit/plot.py`, `benchmarks/`.
- [ ] **R13 / P2: Enforce the documented typing policy.** Annotate public APIs,
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

## Next suggested batch

Address R05 with independent statistics tests from R11: preserve actual
degrees of freedom and report goodness-of-fit summaries only under their
valid assumptions. R02 and R03 are complete.
