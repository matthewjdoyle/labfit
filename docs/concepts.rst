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

LabFit reports ``reduced_chi2`` on every :class:`FitResult`. Plotting the **standardized residuals**
helps diagnose which assumption failed.

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
