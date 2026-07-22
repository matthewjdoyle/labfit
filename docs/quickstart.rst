.. _quickstart:

Quickstart
==========

Installation
------------

Install from PyPI:

.. code-block:: shell

   pip install labfit

Or from source:

.. code-block:: shell

   git clone https://github.com/matthewjdoyle/labfit.git
   cd labfit
   pip install .

Dependencies
------------

- `numpy`
- `scipy`
- `matplotlib`

Input format
------------

Save your data to a CSV file with columns::

   x,y,sigma
   1.0,2.1,0.3
   2.0,3.8,0.4
   3.0,8.2,0.5
   4.0,16.0,1.0

``sigma`` is the **1-σ measurement uncertainty** in y.

Alternatively, when errors are asymmetric use ``sigma_low`` and ``sigma_high`` separately. LabFit converts them into an effective standard deviation for the fit and keeps the original bounds for plotting.

CSV → fit → plot (10 seconds)
--------------------------------

.. code-block:: python

   from labfit import fit, plot_fit

   result = fit("mydata.csv", model="exponential")
   plot_fit(result)
   print(f"reduced chi^2 = {result.reduced_chi2:.3f}")

One file, three lines, done.

Working with arrays directly
----------------------------

If your data is already in memory, use :func:`labfit.fit`:

.. code-block:: python

   import numpy as np
   from labfit import fit, plot_fit

   rng = np.random.default_rng(42)
   x = np.linspace(-5.0, 5.0, 200)
   y = 3.0 * np.exp(-0.5 * ((x - 0.5) / 1.2) ** 2) + rng.normal(0, 0.05, size=x.size)

   result = fit(x, y, model="gaussian")
   plot_fit(result, show_residuals=True)

:func:`labfit.fit` uses automatic initial-guess heuristics for all
built-in models so you don't have to think about it.

Choosing a built-in model
-------------------------

Pass any model name as the ``model`` argument:

.. code-block:: python

   result = fit(x, y, model="linear")
   result = fit(x, y, model="gaussian")
   result = fit(x, y, model="bimodal_gaussian")
   result = fit(x, y, model="damped_oscillator")

All available names are in :const:`labfit.models.MODEL_NAMES`:

.. code-block:: python

   from labfit.models import MODEL_NAMES
   print(MODEL_NAMES)   # ("constant", "linear", "quadratic", ...)

For the full list with equations and variable definitions see
:ref:`fitting-functions`.

Initial guesses (``p0``)
------------------------

For non-linear models a good initial guess helps the optimizer converge faster and avoid
local minima. Provide it as a dict keyed by parameter name:

.. code-block:: python

   result = fit(
       x, y, model="gaussian",
       p0={"amplitude": 3.0, "mean": 0.5, "sigma": 1.0},
   )

You can also use the class-based API:

.. code-block:: python

   from labfit import Fitter

   fitter = Fitter(
       model="gaussian",
       p0={"amplitude": 3.0, "mean": 0.0, "sigma": 1.0},
   )
   result = fitter(x, y)

Parameter bounds
----------------

Constrain parameters to a physically plausible range with ``bounds``.
Each bound is a ``(lower, upper)`` tuple; use ``None`` for unbounded:

.. code-block:: python

   result = fit(
       x, y, model="gaussian",
       p0={"amplitude": 3.0, "mean": 0.0, "sigma": 1.0},
       bounds={
           "amplitude": (0.0, None),    # amplitude must be positive
           "mean": (-5.0, 5.0),
           "sigma": (0.001, None),       # width must be positive
       },
   )

Without bounds the optimizer may wander into regions where the model is
ill-defined (e.g. negative sigma, zero amplitude in a denominator).

Reading the fit result
----------------------

Every fit returns a :class:`labfit.FitResult` with the fitted parameters,
their uncertainties, and a goodness-of-fit metric.

.. code-block:: python

   result = fit(x, y, model="gaussian")

   # Fitted values
   print(result.params["amplitude"])         # e.g. 2.987
   print(result.params["mean"])              # e.g. 0.498
   print(result.params["sigma"])             # e.g. 1.203

   # Uncertainties
   print(result.uncertainties["amplitude"])  # e.g. 0.012

   # Goodness-of-fit
   print(result.reduced_chi2)               # reduced chi-squared
   print(result.p_value)                     # p-value of the fit

   # Convergence check
   if not result.success:
       print(f"WARNING: Fit did NOT converge!")
       print(f"  Reason: {result.message}")

   # Full parameter listing
   for name in result.param_names:
       val = result.params[name]
       unc = result.uncertainties.get(name, float("nan"))
       print(f"  {name} = {val:.4f} ± {unc:.4f}")

Interpreting reduced χ²
------------------------

The **reduced chi-squared** (``reduced_chi2``) tells you whether the model
describes the data within the stated uncertainties:

- **≈ 1.0** → the model and uncertainties are consistent.
- **> 1.0** → the scatter is larger than the uncertainties suggest.
  Possible causes: underestimated errors, missing model terms, or outliers.
- **< 1.0** → the scatter is smaller than expected.
  Possible causes: overestimated errors, overfitting, or data selection bias.

For a deeper discussion see :ref:`concepts`.

Plotting multiple series
------------------------

.. code-block:: python

   from labfit import Series, Plotter

   series_a = Series(x=x1, y=y1, sigma=sigma1, label="Sensor A")
   series_b = Series(x=x2, y=y2, sigma=sigma2, label="Sensor B")

   plot = Plotter()
   plot.add_series(series_a)
   plot.add_series(series_b)
   plot.plot(title="Two sensor fit")
   plot.save("output.png")

Fitting a subset of your data
-----------------------------

Often your data has a region of interest — a peak, a linear portion, or a
decay tail — while the rest is background or irrelevant. Fit only the
region you care about, then overlay the fit on the full dataset.

.. code-block:: python

   import numpy as np
   from labfit import fit, DataSeries, plot_fit

   rng = np.random.default_rng(20260618)

   # Simulated data: a Gaussian peak on a slight wavy background
   x = np.linspace(0.0, 10.0, 300)
   y = 3.0 * np.exp(-0.5 * ((x - 5.0) / 0.8) ** 2) \
       + 0.05 * np.sin(1.5 * x) \
       + rng.normal(0, 0.03, size=x.size)

   full = DataSeries(x=x, y=y, sigma=0.03, label="full data")

   # Fit only the peak region (x = 3.5 … 6.5)
   mask = (x > 3.5) & (x < 6.5)
   result = fit(x[mask], y[mask], model="gaussian", sigma=0.03)

   # Plot the fit on the full dataset — the fit was done on the
   # subset, but you still see all the data
   plot_fit(result, data_series=full,
            title="Gaussian fit to peak region only",
            xlabel="x", ylabel="y")

   print(result)

The same technique works with any boolean mask — select a linear region,
exclude outliers, or isolate a single oscillation from a multi-cycle
measurement. The fit itself only sees the masked points, while the plot
shows everything.

Next steps
----------

- See :ref:`concepts` for the meaning of reduced χ² and how asymmetric errors propagate.
- Browse :ref:`gallery` for plotting examples.
- Jump to the :ref:`api` for the complete reference.