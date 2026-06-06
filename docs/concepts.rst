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
- :math:`\chi^2_\nu > 1`   → scatter larger than expected → overestimate errors, weak model, neglected structure, or outliers.
- :math:`\chi^2_\nu < 1`   → scatter smaller than expected → underestimated errors, overfitting, or data selection bias.

LabFit reports ``reduced_chi2`` on every :class:`FitResult`. Plotting the **standardized residuals**
helps diagnose which assumption failed.

Correlated errors
-----------------

If your uncertainties share a common systematic, include a correlation matrix ``sigma_cov`` in :class:`Series`.

.. code-block:: python

   series = Series(x=x, y=y, sigma=sigma, sigma_cov=C)

LabFit uses Generalized Least Squares when ``sigma_cov`` is provided, so the fit accounts for correlated noise.

Asymmetric errors
-----------------

When ``sigma_low`` and ``sigma_high`` differ, store them as a ``sigma`` tuple or use
:class:`labfit.types.AsymmetricError`.

.. code-block:: python

   labsig = (lower, upper)
   fitter.fit(x, y, sigma=labsig)

Propagation to the fitted parameters is done via the covariance matrix returned by the optimizer.
The reported errors are the **square root of the diagonal** of the covariance matrix.

Errors on derived quantities
----------------------------

Use :func:`labfit.utils.propagate_errors` to transform parameter uncertainties into function uncertainty.

.. code-block:: python

   from labfit.utils import propagate_errors

   def half_life_from_decay(decay):
       return decay / np.log(2.0)

   half_life, half_life_error = propagate_errors(
       half_life_from_decay,
       decay=result.params["decay"],
       jacobian=lambda decay: 1.0 / np.log(2.0),
   )

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

By default LabFit interprets ``sigma`` as 1-σ uncertainties. If you are supplying a
design matrix regression, set ``sigma=None`` and pass ``weights`` instead.

.. code-block:: python

   result = fitter.fit(x, y, weights=1.0 / sigma**2)

Weighted regression is mathematically equivalent to correlated errors with a diagonal covariance matrix.
