.. _concepts:

Concepts: reduced χ² and error propagation
==========================================

What is reduced χ²?
-------------------

For a data set with :math:`N` points and a fit with :math:`k` free parameters, the **reduced** chi-square is

.. math::

   \chi^2_\nu = \frac{1}{N - k} \sum_{i=1}^{N} \left( \frac{y_i - M(x_i)}{\sigma_i} \right)^2

where :math:`M(x)` is the model prediction.

Interpretation:

- :math:`\chi^2_\nu \approx 1`  → consistent fit.
- :math:`\chi^2_\nu > 1`   → scatter larger than expected → underestimated errors, weak model, neglected structure, or outliers.
- :math:`\chi^2_\nu < 1`   → scatter smaller than expected → overestimated errors, overfitting, or data selection bias.

LabFit reports ``result.dof = N - k`` without clamping it. ``reduced_chi2``
is available when errors are supplied and ``dof > 0``. With no errors,
``reduced_chi2`` and ``p_value`` are ``NaN``: residuals in y units cannot be
compared to a chi-square distribution. Instead, ``result.residual_variance``
reports the unweighted sum of squared residuals divided by ``dof``, in squared
y units. For heterogeneous or correlated errors, this describes the raw
residual scatter rather than a common measurement-noise variance.

For ``dof <= 0``, residual variance, reduced chi-square and p-value are all
``NaN``, and LabFit warns. Relative parameter covariance cannot be estimated
without positive dof, so its uncertainties are also unavailable. Absolute
parameter covariance can still be computed when the Jacobian identifies the
parameters. Plotting the **standardized residuals** helps diagnose which
assumption failed.

A chi-square p-value also requires optimizer convergence and identifiable
parameters. Its interpretation assumes the supplied errors describe calibrated
Gaussian measurement noise and the model is appropriate; for nonlinear models
it is generally a local approximation. With purely relative weights, treat the
reported weighted chi-square as a diagnostic, and interpret its p-value only
if those weights also describe the absolute noise scale.

Absolute uncertainties and relative weights
-------------------------------------------

Choose the covariance policy explicitly when calling ``fit``, ``fit_curve``,
``fit_multi``, or the corresponding ``Fitter`` methods:

- ``absolute_sigma=True`` treats supplied errors as calibrated measurement
  uncertainties. Parameter covariance is not scaled by the residual scatter.
- ``absolute_sigma=False`` (the default, preserving earlier behavior) uses the
  errors only to set relative weights. The common noise scale is estimated from
  the residuals, multiplying parameter covariance by reduced chi-square.

For a locally identifiable model with a full-rank weighted residual Jacobian
:math:`J` and positive degrees of freedom :math:`N-k`, the policies give

.. math::

   C_{\mathrm{absolute}} = (J^T J)^{-1}, \qquad
   C_{\mathrm{relative}} = C_{\mathrm{absolute}}\,\frac{\chi^2}{N-k}.

The parameter standard errors are the square roots of the diagonal entries.
Multiplying every absolute sigma by 10 multiplies standard errors by 10;
under the relative policy, a common sigma rescaling leaves standard errors
unchanged. Neither policy changes the least-squares objective or best-fit
parameters for a fixed error specification. These are local linear covariance
estimates; bounds, nonlinear models, and unidentifiable parameters need care.

.. code-block:: python

   from labfit import fit

   result = fit(x, y, sigma=sigma, absolute_sigma=True)
   # For correlated measurement errors:
   correlated = fit(x, y, sigma_cov=C, absolute_sigma=True)

The same policy applies to inverse-variance ``weights`` and the effective sigma
used for asymmetric errors. With no error specification,
``absolute_sigma=True`` assumes independent unit standard deviations in y units;
the default estimates a common variance from the residuals instead.
``result.absolute_sigma`` records the policy used. Goodness-of-fit statistics
are calculated separately and are not changed by this option.

Valid uncertainty specifications
--------------------------------

LabFit validates measurement errors when constructing a ``DataSeries`` and
again when fitting, including explicit overrides. Supply one specification:

- ``sigma`` (or the ``DataSeries.y_err`` alias): a finite, strictly positive
  scalar or one-dimensional array matching the data length.
- ``sigma_low`` and ``sigma_high`` together, or an ``AsymmetricError`` supplied
  as ``sigma``: both sides must be finite and strictly positive. Scalars are
  supported; ``AsymmetricError`` requires matching lower/upper shapes.
  Fitting uses the root-mean-square error and plotting retains both sides.
- ``sigma_cov``: a finite symmetric positive-definite matrix of shape
  ``(N, N)``. LabFit checks symmetry with relative tolerance ``1e-12`` and
  symmetrizes accepted rounding differences. Singular, indefinite and
  asymmetric matrices raise ``ValueError`` before optimization.
- ``weights`` (fitting only): a finite nonnegative scalar or matching 1D array
  of inverse variances. A zero weight explicitly excludes that sample from
  model evaluation, initial guesses, stored result data, default plots and all
  statistics. At least one weight must be positive. Degrees of freedom use the
  number of retained samples, so excluded points cannot inflate confidence.

Competing specifications and incomplete asymmetric pairs raise ``ValueError``.
``sigma`` and ``y_err`` may both be supplied only if they describe the same
errors, including both asymmetric sides. Zero sigmas are rejected: use zero
weights to exclude observations instead of assigning them zero uncertainty.

Any explicit fitting error specification replaces the entire stored error
specification of a ``DataSeries`` or loaded CSV. Supplying just one asymmetric
side does not borrow the other side from the input. The input series is left
unchanged; ``result.series`` carries the errors actually used in the fit.

.. code-block:: python

   import numpy as np
   from labfit import DataSeries, fit

   series = DataSeries([0, 1, 2, 3], [1, 3, 100, 7], sigma=0.5)
   result = fit(series, weights=[4, 4, 0, 4], absolute_sigma=True)
   assert result.dof == 1  # three retained observations minus two parameters
   np.testing.assert_array_equal(result.x, [0, 1, 3])

Confidence and prediction bands
-------------------------------

``plot_fit(result, show_ci=True)`` draws a confidence band for the fitted mean.
Adding ``prediction=True`` draws a band for a future observation. Both use
pointwise normal approximations with the local parameter covariance, rather
than simultaneous confidence regions or exact small-sample intervals.
``ci_level`` must be finite and strictly between 0 and 1.

For a parameter gradient :math:`g(x)` and parameter covariance :math:`C`,
the mean variance is :math:`v_{\mathrm{mean}}(x) = g(x)^T C g(x)`. A prediction
band adds the future observation variance in squared y units:

.. math::

   v_{\mathrm{prediction}}(x) = v_{\mathrm{mean}}(x) + \sigma_{\mathrm{future}}(x)^2.

The band endpoints are the fitted mean plus or minus
:math:`z_{(1+\mathrm{ci\_level})/2}\sqrt{v(x)}`. This assumes future observation
errors are independent of the data used to estimate the parameters.

Pass ``prediction_sigma`` to specify absolute future 1-sigma errors in y units:

- A scalar applies the same error throughout the band.
- A function receives the plotted x grid and returns a scalar or an array with
  the same shape. This is useful for heteroscedastic noise, whose size varies
  with x.
- An array must match the plotted 200-point grid spanning the minimum and
  maximum fitted x values. Training-data error arrays usually have a different
  shape and cannot be reused directly.

Explicit future sigmas must be finite and nonnegative, and are never rescaled
by reduced chi-square. Zero specifies a noise-free future observation, so the
prediction band equals the confidence band. The option requires
``show_ci=True`` and ``prediction=True`` for a single fit.

When ``prediction_sigma`` is omitted, LabFit uses these defaults:

- Constant training sigmas are reused under ``absolute_sigma=True``. Under the
  relative policy, their variance is multiplied by ``reduced_chi2`` to estimate
  the common noise scale. Constant inverse-variance weights follow the same rule.
- An unweighted relative fit uses ``residual_variance`` (SSR / positive dof).
  An unweighted absolute fit assumes unit variance in the current y units,
  matching its parameter covariance policy.
- Heterogeneous or correlated training errors require an explicit future sigma.
  LabFit does not infer a noise function between data points or assume that the
  training covariance describes a new observation. For correlated training
  data, the supplied future sigma describes its marginal error; future errors
  must still be independent of the fitted data.

A missing residual noise scale requires an explicit ``prediction_sigma``.
If parameter covariance is unavailable, neither band can be drawn.
Changing y units requires scaling y and all absolute measurement and future
sigmas by the same factor; covariance scales by its square, and band widths
scale with y. An implicit unit-variance assumption describes a different noise
model when the y units change, so supply physical errors for unit invariance.

This complete example uses a future error function for heterogeneous data:

.. code-block:: python

   import numpy as np
   from labfit import fit, plot_fit

   x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
   y = np.array([-1.4, -0.5, 1.0, 2.0, 3.5, 4.3])
   sigma = 0.2 + 0.1 * np.abs(x)
   result = fit(x, y, sigma=sigma, absolute_sigma=True)
   plot = plot_fit(
       result, show_ci=True, prediction=True, ci_level=0.95,
       prediction_sigma=lambda grid: 0.2 + 0.1 * np.abs(grid),
   )
   plot.save("prediction-band.png")

Identifiability of fitted parameters
------------------------------------

An optimizer can converge even when the observations do not distinguish all
parameters. For example, in ``(a + b) * x``, only the sum is determined. LabFit
checks the weighted residual Jacobian after fitting. If it lacks full column
rank, has a zero column, or is too ill-conditioned for reliable numerical
uncertainties, ``result.identifiable`` is ``False``. It warns and sets the
parameter covariance and all parameter uncertainties to ``NaN``. The human
readable result also flags this condition. ``result.success`` still reports
whether the optimizer converged; it does not imply identifiable parameters.

``result.jacobian_rank`` and ``result.jacobian_condition`` expose the diagnostic.
The condition is computed after normalizing each Jacobian column, so a simple
change of parameter units does not trigger it. LabFit treats a condition
number at or above :math:`1/\sqrt{\epsilon_{\mathrm{machine}}}` as unreliable
for numerical covariance estimation. This local test cannot guarantee that a
nonlinear model is globally identifiable.

Correlated errors
-----------------

If your uncertainties share a common systematic, include a covariance matrix ``sigma_cov`` in :class:`Series`.

.. code-block:: python

   series = Series(x=x, y=y, sigma_cov=C)

LabFit uses Generalized Least Squares when ``sigma_cov`` is provided, so the fit accounts for correlated noise.

Asymmetric errors
-----------------

When ``sigma_low`` and ``sigma_high`` differ, pass them separately or use
:class:`labfit.types.AsymmetricError`.

.. code-block:: python

   fitter.fit(x, y, sigma_low=lower, sigma_high=upper)

Propagation to the fitted parameters is done via the covariance matrix returned by the optimizer.
The reported errors are the **square root of the diagonal** of the covariance matrix.

Errors on derived quantities
----------------------------

Use :func:`labfit.utils.propagate_errors` to transform parameter uncertainties into function uncertainty.

.. code-block:: python

   import numpy as np
   from labfit.utils import propagate_errors

   def half_life_from_decay(decay):
       return np.log(2.0) / decay

   half_life, half_life_error = propagate_errors(
       half_life_from_decay,
       decay=0.5,
       decay_error=0.02,
       jacobian=lambda decay: -np.log(2.0) / decay**2,
   )

For the built-in exponential ``A * exp(-decay * x)``, this example gives a
half-life of approximately 1.386 with uncertainty 0.055, in the units of x.
For fitted data, substitute ``result.params["decay"]`` and
``result.uncertainties["decay"]`` for the example values.

Initial guesses and bounds
--------------------------

Always provide an initial guess when possible. If the model is highly nonlinear,
use parameter bounds to prevent the optimizer from jumping to unphysical minima.

.. code-block:: python

   fitter = Fitter(
       model=gaussian,
       p0={"amplitude": 1.0, "mean": 0.0, "sigma": 1.0},
       bounds={
           "amplitude": (0.0, None),
           "mean": (-5.0, 5.0),
           "sigma": (1e-6, None),
       },
   )

Choosing weights
-----------------

Use ``sigma`` for standard deviations or ``weights=1.0 / sigma**2`` for
inverse-variance weights. Either representation accepts the same
``absolute_sigma`` policy; using ``weights`` does not select a policy automatically.

.. code-block:: python

   result = fitter.fit(x, y, weights=1.0 / sigma**2, absolute_sigma=True)

Weighted regression is mathematically equivalent to correlated errors with a diagonal covariance matrix.
