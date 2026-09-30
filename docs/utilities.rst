.. _utilities:

Utilities and I/O
=================

These helpers are part of the public surface for loading data, combining
series, and propagating uncertainties.

Data loading helpers
--------------------

CSV and TXT error columns obey the same validation as ``DataSeries``. Files
with both asymmetric columns retain those bounds and do not infer a competing
symmetric sigma. Files containing both symmetric and asymmetric errors, or
only one asymmetric side, raise ``ValueError``. These rules apply when using
error columns; the explicit unweighted mode reads only x/y.

Choosing file uncertainties
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``error_mode="unweighted"`` when no measurement-error model is available.
It loads x/y without inferring errors and ignores any error columns in the file.
It cannot be combined with an explicit ``y_err_col`` selector. Unweighted fits
report residual variance; reduced chi-square and its p-value are unavailable.

The compatible default, ``error_mode="auto"``, uses supplied error columns when
present. Without errors, it chooses a heuristic based on the y-values:

- Nonnegative integer counts use ``sqrt(max(y, 1))``. The one-count variance
  floor gives a zero count sigma of 1, keeps that observation in the fit, and
  prevents an infinite inverse-variance weight. Positive integer counts retain
  the previous ``sqrt(y)`` values.
- Other values use ``abs(y) * default_fraction``, with a default fraction of
  0.05. The fraction must be finite and strictly positive when used. Zero y-values
  have no positive fractional-error scale; supply explicit errors or load
  unweighted instead. Overflowing or underflowing inferred sigmas are rejected.

Select ``error_mode="poisson"`` or ``"fraction"`` to choose the fallback rule
explicitly. Poisson inference requires nonnegative integer counts; negative or
fractional data are rejected rather than clipped. The auto mode uses exact
integer values when deciding whether data are counts. Other modes still use
explicit file errors when available; supplied fractional-error columns also
count as explicit errors.

Every successful inference emits :class:`labfit.io.UncertaintyInferenceWarning`
with the file and formula. ``series.uncertainties_inferred`` flags inferred
errors and ``series.uncertainty_inference`` records the formula and its numerical
settings. Relabeling, combining series and fitting preserve this record, and
fit summaries identify inferred errors. Explicit fit overrides clear the
record on the result; the original loaded series retains it. Passing explicit
errors to ``fit(path, ...)`` bypasses file error columns and inference entirely.

These heuristics are conveniences for exploration. The count floor is a stated
approximation, not an exact Poisson confidence interval or calibrated error
estimate. Low-count data may require a count-likelihood analysis; a chi-square
p-value from inferred errors depends on the assumed noise model. For measured
uncertainties, supply explicit errors rather than treating inference as evidence
that the error model is correct.

.. code-block:: python

   import numpy as np
   from labfit import fit
   from labfit.io import load_csv

   # counts.csv contains x,y rows (0,0), (1,1), (2,2).
   raw = load_csv("counts.csv", error_mode="unweighted")
   raw_result = fit(raw)  # warns that chi-square statistics are unavailable
   assert not raw_result.is_weighted
   assert np.isnan(raw_result.reduced_chi2)

   inferred = load_csv("counts.csv")  # warns and records the assumed errors
   np.testing.assert_allclose(inferred.sigma, [1, 1, np.sqrt(2)])
   assert inferred.uncertainties_inferred
   assert "one-count variance floor" in inferred.uncertainty_inference
   inferred_result = fit(inferred)
   assert inferred_result.series.uncertainty_inference == inferred.uncertainty_inference

.. autoclass:: labfit.io.UncertaintyInferenceWarning

.. autofunction:: labfit.io.load_csv

.. autofunction:: labfit.io.load_txt

.. autofunction:: labfit.io.combine_series

Uncertainty helpers
-------------------

.. autofunction:: labfit.utils.propagate_errors

.. autofunction:: labfit.utils.effective_sigma
