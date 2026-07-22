.. _faq:

FAQ
===

What units should errors be in?
-------------------------------

Use 1-σ measurement uncertainties. LabFit weights the fit by the inverse variance.

Should I pass ``sigma_low`` and ``sigma_high``?
-----------------------------------------------

Yes, when the measurement errors are asymmetric. LabFit converts them to an effective standard deviation for fitting and preserves the original bounds for plotting.

Why is reduced χ² not near 1?
------------------------------

Check the error scale and whether the model really describes the data. If residuals are structured, the model is likely wrong; if the size is wrong, the error estimates probably need rescaling.

Why did my fit not converge?
----------------------------

There are a few common culprits:

**Bad initial guesses.** Non-linear fits (like Gaussian or exponential) need a
decent starting point. The automatic heuristics usually handle this, but if
you're fitting a weird dataset try passing ``p0`` with sensible guesses.
See :ref:`quickstart` for how.

**Unconstrained parameters.** Without bounds the optimizer can wander into
forbidden territory — negative sigma, zero amplitude in a denominator, etc.
Add ``bounds`` to keep parameters in a physically plausible range.

**Wrong model for the data.** Not every bell-shaped curve is a Gaussian. If
the data has two peaks you probably need ``bimodal_gaussian``; if it decays
to a baseline you may need to add a constant offset. Plot your data and
think about what shape it really has.

**Very noisy data.** If the signal-to-noise ratio is really low the
optimizer may struggle to find a meaningful minimum. More data (or less
noise) helps. You can also try fixing one or two parameters to sensible
values to reduce the number of free parameters.

If none of these help, check the residuals — structured residuals (a
wiggle, a trend) almost always mean the model is missing something.

What's the difference between ``fit()`` and ``fit_curve()``?
-------------------------------------------------------------

Both do least-squares fitting — they just differ in how you call them.

- :func:`labfit.fit` is the **main** function. It takes ``x``, ``y``, and
  optionally ``sigma`` as keyword arguments::

      fit(x=x, y=y, model="gaussian", p0={...})

- :func:`labfit.fit_curve` is a **slimmed-down variant** that takes ``model``
  first and ``y_err`` as a **positional argument** (after ``x`` and ``y``)::

      fit_curve("gaussian", x, y, y_err)

  It is designed for quick interactive use when your errors are already in a
  separate array and you want to save a few keystrokes.

In short: use ``fit()`` in scripts you plan to keep, and ``fit_curve()`` at
the REPL for one-liners.

How do I use a custom model with Fitter?
----------------------------------------

Define any function that takes ``x`` as the first argument and your
parameters as keyword arguments, then pass it to
:class:`labfit.Fitter` as ``model``:

.. code-block:: python

   import numpy as np
   from labfit import Fitter

   def my_model(x, amplitude, center, width):
       return amplitude * np.exp(-0.5 * ((x - center) / width) ** 2)

   x = np.linspace(-5, 5, 100)
   y = 3.0 * np.exp(-0.5 * ((x - 0.5) / 1.2) ** 2)

   fitter = Fitter(
       model=my_model,
       p0={"amplitude": 2.5, "center": 0.0, "width": 1.0},
   )
   result = fitter(x, y)
   print(result.params)

The same works with :func:`labfit.fit`:

.. code-block:: python

   from labfit import fit

   result = fit(
       x=x, y=y,
       model=my_model,
       p0={"amplitude": 2.5, "center": 0.0, "width": 1.0},
   )

You can also add ``bounds`` the same way as with built-in models. The
function body can use any NumPy operations — just make sure it is
vectorised (works on arrays).

How do I interpret the p-value?
-------------------------------

The ``p_value`` in a fit result answers this question:

   *If the model is correct and the uncertainties are right, how likely
   would I be to see a χ² this large (or larger) just by chance?*

- A **p-value near 1** means the fit looks suspiciously good — the χ² is
  much smaller than expected. This can happen when errors are overestimated
  or the model has too many free parameters (overfitting).
- A **p-value near 0** means the χ² is too large to be explained by random
  scatter alone. Something is probably off — the model may be wrong, the
  uncertainties may be underestimated, or there may be outliers.
- A **p-value around 0.05–0.50** is typical for a reasonable fit.

Think of it as a sanity check. If the p-value is tiny (say < 0.001) you
should definitely investigate. But even a "good" p-value doesn't prove the
model is correct — it just means the data doesn't give you a reason to
doubt it.