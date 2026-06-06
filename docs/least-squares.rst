.. _least-squares:

How least-squares fitting works
===============================

.. raw:: html

   <style>
   .ls-fig { margin: 1.5em 0; }
   .ls-fig img { width: 100%; max-width: 650px; display: block; margin: 0 auto; }
   .ls-fig p { text-align: center; font-size: 0.9em; color: #7f8c8d; margin-top: 0.3em; }
   </style>

This page explains what happens when you call ``fit()``.  The math is kept
to what you need to read a fit result intelligently.  No background in
optimisation is assumed.

If you just want to run a fit, start with the :ref:`quickstart`.

1. What is a residual?
----------------------

Every data point is a measurement :math:`(x_i, y_i)` with uncertainty
:math:`\sigma_i`.  We choose a model :math:`M(x; \boldsymbol{\theta})` with
adjustable parameters :math:`\boldsymbol{\theta}`.  For a straight line the
parameters are :math:`\boldsymbol{\theta} = [m, c]` and
:math:`M(x) = m x + c`.

The **residual** is the difference between the measured value and the model
prediction:

.. math::

   r_i = y_i - M(x_i; \boldsymbol{\theta})

A good fit makes all the residuals small.

.. raw:: html

   <div class="ls-fig">
   <img src="_static/residuals.svg" alt="Data points, best-fit line, and vertical residual lines">
   <p>Each dashed vertical line is a residual: the distance from the data point to the fitted curve.</p>
   </div>

2. Why square the residuals?
----------------------------

Adding residuals directly lets positive and negative values cancel.  We could
take absolute values (:math:`\sum |r_i|`), but squaring has two advantages:

* Large deviations are penalised more than small ones.  A point twice as far
  from the curve contributes *four* times as much to the cost.
* The squared sum is differentiable, which lets us use calculus to find the
  best parameters efficiently.

Define the **objective function** (also called the cost):

.. math::

   \chi^2(\boldsymbol{\theta}) = \sum_{i=1}^{N}
   \frac{\bigl[y_i - M(x_i; \boldsymbol{\theta})\bigr]^2}{\sigma_i^2}

Dividing by :math:`\sigma_i^2` means points with large uncertainties matter
less than points with small ones.

3. The :math:`\chi^2` landscape
-------------------------------

For a two-parameter model like a straight line, every pair :math:`(m, c)`
produces a :math:`\chi^2` value.  Plotting :math:`\chi^2` against the
parameters gives a surface.

* For **linear** models (line, polynomial) the surface is a perfect parabola
  with one global minimum.
* For **nonlinear** models (Gaussian, exponential) the surface can be bumpy
  with local minima and ridges.

Finding the best-fit parameters means sliding downhill on this surface
until you reach the bottom.

.. raw:: html

   <div class="ls-fig">
   <img src="_static/chi2-surface.svg" alt="Contour plot of chi squared for slope and intercept parameters">
   <p>Each contour connects parameter pairs with the same \(\chi^2\).  The star marks the minimum: the best-fit parameters.</p>
   </div>

4. How the optimiser walks downhill
-----------------------------------

The surface shows the destination.  The **optimiser** is the algorithm that
gets you there.  LabFit uses SciPy's ``least_squares`` with the Trust Region
Reflective (TRF) method.  The idea is the same for any gradient-based
method:

1. **Start somewhere** with an initial guess :math:`\boldsymbol{\theta}_0`
   (supplied via ``p0`` or from automatic heuristics).
2. **Look at the local slope** by computing the gradient of :math:`\chi^2`
   with respect to each parameter.
3. **Take a step downhill** in the direction that reduces :math:`\chi^2`
   most.
4. **Repeat** until the gradient is nearly zero or improvement stalls.

The TRF method goes further.  It builds a local quadratic model of the
surface (a "trust region") to pick better step sizes.  It also respects
parameter **bounds**: if you set ``amplitude`` to be positive, the optimiser
never tries negative values.

5. Goodness-of-fit: reduced :math:`\chi^2`
------------------------------------------

Once the optimiser finds the minimum :math:`\chi^2_{\min}`, we compute the
**reduced chi-square**:

.. math::

   \chi^2_\nu = \frac{\chi^2_{\min}}{N - k}

where :math:`N` is the number of data points and :math:`k` is the number of
free parameters.  The denominator is the **degrees of
freedom**.

The reduced :math:`\chi^2` tells you whether your model and your
uncertainties are consistent:

.. raw:: html

   <div class="ls-fig">
   <img src="_static/goodness-of-fit.svg" alt="Three panels showing overestimated errors, good fit, and underestimated errors">
   <p>Same data-generating process but different reported uncertainties produce different \(\chi^2_\nu\) values.</p>
   </div>

* :math:`\chi^2_\nu \approx 1` means the scatter matches the error bars.
* :math:`\chi^2_\nu \gg 1` means the data scatter more than the error
  bars suggest.  The model may be too simple, uncertainties may be
  underestimated, or there may be outliers.
* :math:`\chi^2_\nu \ll 1` means the scatter is smaller than expected.
  The error bars may be overestimated.

LabFit also returns a **p-value** for a more formal test.  A very small
p-value (below 0.01) suggests the model is a poor description of the data.

6. Parameter uncertainties
--------------------------

The best-fit parameters are only part of the result.  We also need to know
how **confident** we are in each value.  That information comes from the
shape of the :math:`\chi^2` surface at the minimum.

If the :math:`\chi^2` valley is **steep and narrow** in a particular
direction, that parameter is tightly constrained.  Small changes increase
the cost a lot, so the uncertainty is small.

If the valley is **shallow and flat**, the parameter can vary widely without
changing :math:`\chi^2` much.  Its uncertainty is large.

.. raw:: html

   <div class="ls-fig">
   <img src="_static/curvature-uncertainty.svg" alt="Chi squared contours showing the 1-sigma and 2-sigma uncertainty ellipses">
   <p>The \(\chi^2_{\min}+1\) contour encloses the 1-\(\sigma\) confidence region.  A steep valley means a small uncertainty; a flat valley means a large uncertainty.</p>
   </div>

The covariance matrix :math:`C` captures this curvature:

.. math::

   C = \chi^2_\nu \; (J^T J)^{-1}

where :math:`J` is the Jacobian (the gradient of the residuals with respect
to the parameters) at the best fit.  The parameter uncertainties in
``result.uncertainties`` are the square roots of the diagonal entries:

.. math::

   \sigma_{\theta_j} = \sqrt{C_{jj}}


For more detail on reduced :math:`\chi^2` and error propagation, see
:ref:`concepts`.  For the full API reference, see :ref:`api`.
