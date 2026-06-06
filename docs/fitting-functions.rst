.. _fitting-functions:

Fitting functions
=================

LabFit exposes a small set of entry points for fitting and plotting, plus a
collection of built-in model functions for common textbook curves.

The public fitting helpers are intentionally small:

- :func:`labfit.fit` — the main least-squares entry point.
- :func:`labfit.fit_curve` — a convenience wrapper that takes ``y_err`` explicitly.
- :func:`labfit.fit_to_model` — alias for :func:`labfit.fit` for readability.
- :func:`labfit.quick_fit` — convenience alias for quick experiments.
- :func:`labfit.fit_multi` — fit a whole sequence of :class:`labfit.Series` objects.
- :func:`labfit.plot_fit` — plot one fit and, by default, its residuals.
- :func:`labfit.plot_multi_fit` — compare multiple fits in a grid.
- :func:`labfit.plot_residuals` — explicit residual-focused wrapper.

Entry points
------------

.. autofunction:: labfit.fit

.. autofunction:: labfit.fit_curve

.. autofunction:: labfit.fit_to_model

.. autofunction:: labfit.quick_fit

.. autofunction:: labfit.fit_multi

.. autofunction:: labfit.plot_fit

.. autofunction:: labfit.plot_multi_fit

.. autofunction:: labfit.plot_residuals

Built-in model functions
------------------------

The built-in models live in :mod:`labfit.models`. They can be passed either by
name or as a callable to :func:`labfit.fit` and :class:`labfit.Fitter`.

The sections below show each function signature, then the equation, then the
variable definitions used in the formula.

constant
~~~~~~~~

.. autofunction:: labfit.models.constant

Equation
^^^^^^^^

.. math::

   y = k

Variables
^^^^^^^^^

- ``x``: input x-values.
- ``k``: constant output level.

linear
~~~~~~

.. autofunction:: labfit.models.linear

Equation
^^^^^^^^

.. math::

   y = m x + c

Variables
^^^^^^^^^

- ``x``: input x-values.
- ``m``: linear gradient.
- ``c``: vertical offset.

quadratic
~~~~~~~~~

.. autofunction:: labfit.models.quadratic

Equation
^^^^^^^^

.. math::

   y = a x^2 + b x + c

Variables
^^^^^^^^^

- ``x``: input x-values.
- ``a``: quadratic coefficient.
- ``b``: linear coefficient.
- ``c``: constant term.

cubic
~~~~~

.. autofunction:: labfit.models.cubic

Equation
^^^^^^^^

.. math::

   y = a x^3 + b x^2 + c x + d

Variables
^^^^^^^^^

- ``x``: input x-values.
- ``a``: cubic coefficient.
- ``b``: quadratic coefficient.
- ``c``: linear coefficient.
- ``d``: constant term.

gaussian
~~~~~~~~

.. autofunction:: labfit.models.gaussian

Equation
^^^^^^^^

.. math::

   y = A\,
       e^{-\frac{1}{2}\bigl((x-\mu)/\sigma\bigr)^2}

Variables
^^^^^^^^^

- ``x``: input x-values.
- ``A``: peak height scale.
- ``\mu``: center position.
- ``\sigma``: Gaussian width parameter.

lorentzian
~~~~~~~~~~

.. autofunction:: labfit.models.lorentzian

Equation
^^^^^^^^

.. math::

   y = A\,
       \frac{\gamma^{2}}{(x - x_0)^2 + \gamma^{2}}

Variables
^^^^^^^^^

- ``x``: input x-values.
- ``A``: peak height scale.
- ``x_0``: line center.
- ``\gamma``: half-width parameter.

exponential
~~~~~~~~~~~

.. autofunction:: labfit.models.exponential

Equation
^^^^^^^^

.. math::

   y = A\,e^{-\lambda x}

Variables
^^^^^^^^^

- ``x``: input x-values.
- ``A``: starting scale at ``x = 0``.
- ``\lambda``: exponential decay constant.

power_law
~~~~~~~~~

.. autofunction:: labfit.models.power_law

Equation
^^^^^^^^

.. math::

   y = A\,x^{n}

Variables
^^^^^^^^^

- ``x``: input x-values.
- ``A``: scale factor.
- ``n``: power-law exponent.

logistic
~~~~~~~~

.. autofunction:: labfit.models.logistic

Equation
^^^^^^^^

.. math::

   y = b + \frac{A}{1 + e^{-k(x - x_0)}}

Variables
^^^^^^^^^

- ``x``: input x-values.
- ``A``: curve height above the baseline.
- ``x_0``: midpoint / inflection location.
- ``k``: steepness parameter.
- ``b``: low-end offset, default ``0.0``.

sine
~~~~

.. autofunction:: labfit.models.sine

Equation
^^^^^^^^

.. math::

   y = o + A \sin(2\pi f x + \phi)

Variables
^^^^^^^^^

- ``x``: input x-values.
- ``A``: oscillation amplitude.
- ``f``: cycles per unit of ``x``.
- ``\phi``: phase shift in radians.
- ``o``: vertical offset, default ``0.0``.

cosine
~~~~~~

.. autofunction:: labfit.models.cosine

Equation
^^^^^^^^

.. math::

   y = o + A \cos(2\pi f x + \phi)

Variables
^^^^^^^^^

- ``x``: input x-values.
- ``A``: oscillation amplitude.
- ``f``: cycles per unit of ``x``.
- ``\phi``: phase shift in radians.
- ``o``: vertical offset, default ``0.0``.

damped_oscillator
~~~~~~~~~~~~~~~~~

.. autofunction:: labfit.models.damped_oscillator

Equation
^^^^^^^^

.. math::

   y = A\,e^{-\gamma x}\cos(2\pi f x + \phi)

Variables
^^^^^^^^^

- ``x``: input x-values.
- ``A``: initial oscillation amplitude.
- ``\gamma``: exponential damping constant.
- ``f``: cycles per unit of ``x``.
- ``\phi``: phase shift in radians.

damped_sine
~~~~~~~~~~~

.. autofunction:: labfit.models.damped_sine

Equation
^^^^^^^^

.. math::

   y = o + A\,e^{-\gamma x}\sin(2\pi f x + \phi)

Variables
^^^^^^^^^

- ``x``: input x-values.
- ``A``: initial oscillation amplitude.
- ``\gamma``: exponential damping constant.
- ``f``: cycles per unit of ``x``.
- ``\phi``: phase shift in radians.
- ``o``: vertical offset, default ``0.0``.

sinc
~~~~

.. autofunction:: labfit.models.sinc

Equation
^^^^^^^^

.. math::

   y = A\,
       \frac{\sin(u)}{u},
       \quad
       u = \frac{\pi (x - x_0)}{w}

Variables
^^^^^^^^^

- ``x``: input x-values.
- ``A``: peak amplitude.
- ``x_0``: center position.
- ``w``: characteristic width.

exponential_rise
~~~~~~~~~~~~~~~~

.. autofunction:: labfit.models.exponential_rise

Equation
^^^^^^^^

.. math::

   y = A\bigl(1 - e^{-x/\tau}\bigr) + o

Variables
^^^^^^^^^

- ``x``: input x-values.
- ``A``: asymptotic amplitude.
- ``\tau``: time constant.
- ``o``: vertical offset, default ``0.0``.

double_exponential
~~~~~~~~~~~~~~~~~~

.. autofunction:: labfit.models.double_exponential

Equation
^^^^^^^^

.. math::

   y = A_1 e^{-x/\tau_1} + A_2 e^{-x/\tau_2}

Variables
^^^^^^^^^

- ``x``: input x-values.
- ``A_1``: amplitude of the fast component.
- ``\tau_1``: decay constant of the fast component.
- ``A_2``: amplitude of the slow component.
- ``\tau_2``: decay constant of the slow component.

moffat
~~~~~~

.. autofunction:: labfit.models.moffat

Equation
^^^^^^^^

.. math::

   y = A\,
       \Bigl[1 + \Bigl(\frac{x - x_0}{\alpha}\Bigr)^2\Bigr]^{-\beta}

Variables
^^^^^^^^^

- ``x``: input x-values.
- ``A``: peak amplitude.
- ``x_0``: center position.
- ``\alpha``: width parameter.
- ``\beta``: power-law exponent controlling the tail weight.

gaussian_baseline
~~~~~~~~~~~~~~~~~

.. autofunction:: labfit.models.gaussian_baseline

Equation
^^^^^^^^

.. math::

   y = A\,
       e^{-\frac{1}{2}\bigl((x-\mu)/\sigma\bigr)^2}
       + m x + b

Variables
^^^^^^^^^

- ``x``: input x-values.
- ``A``: peak height scale.
- ``\mu``: center position.
- ``\sigma``: Gaussian width parameter.
- ``m``: linear baseline slope.
- ``b``: linear baseline intercept.

bimodal_gaussian
~~~~~~~~~~~~~~~~

.. autofunction:: labfit.models.bimodal_gaussian

Equation
^^^^^^^^

.. math::

   y = A_1\,
       e^{-\frac{1}{2}\bigl((x-\mu_1)/\sigma_1\bigr)^2}
       +\;
       A_2\,
       e^{-\frac{1}{2}\bigl((x-\mu_2)/\sigma_2\bigr)^2}

Variables
^^^^^^^^^

- ``x``: input x-values.
- ``A_1``: amplitude of the first peak.
- ``\mu_1``: center of the first peak.
- ``\sigma_1``: width of the first peak.
- ``A_2``: amplitude of the second peak.
- ``\mu_2``: center of the second peak.
- ``\sigma_2``: width of the second peak.

voigt
~~~~~

.. autofunction:: labfit.models.voigt

Equation
^^^^^^^^

.. math::

   y = A\,
       V(x - x_0, \sigma, \gamma)

where :math:`V` is the Voigt profile (convolution of a Gaussian and Lorentzian).

Variables
^^^^^^^^^

- ``x``: input x-values.
- ``A``: peak amplitude scale.
- ``x_0``: center position.
- ``\sigma``: Gaussian width parameter.
- ``\gamma``: Lorentzian half-width parameter.

skew_normal
~~~~~~~~~~~

.. autofunction:: labfit.models.skew_normal

Equation
^^^^^^^^

.. math::

   y = A\,
       f(x; \alpha, \mu, \sigma)

where :math:`f` is the skew-normal probability density function.

Variables
^^^^^^^^^

- ``x``: input x-values.
- ``A``: amplitude scale factor.
- ``\mu``: location (center) parameter.
- ``\sigma``: scale (width) parameter.
- ``\alpha``: shape parameter (:math:`\alpha=0` recovers a symmetric Gaussian).

gaussian_fwhm
~~~~~~~~~~~~~

.. autofunction:: labfit.models.gaussian_fwhm

Equation
^^^^^^^^

.. math::

   y = A\,
       e^{-\frac{1}{2}\bigl((x-\mu)/\sigma\bigr)^2},
   \qquad
   \sigma = \frac{\text{FWHM}}{2\sqrt{2\ln 2}}

Variables
^^^^^^^^^

- ``x``: input x-values.
- ``A``: peak amplitude.
- ``\mu``: center position.
- ``\text{FWHM}``: full width at half maximum.

lorentzian_fwhm
~~~~~~~~~~~~~~~

.. autofunction:: labfit.models.lorentzian_fwhm

Equation
^^^^^^^^

.. math::

   y = A\,
       \frac{(\text{FWHM}/2)^2}{(x - x_0)^2 + (\text{FWHM}/2)^2}

Variables
^^^^^^^^^

- ``x``: input x-values.
- ``A``: peak amplitude.
- ``x_0``: center position.
- ``\text{FWHM}``: full width at half maximum.

exgaussian
~~~~~~~~~~

.. autofunction:: labfit.models.exgaussian

Equation
^^^^^^^^

.. math::

   y = A\,
       \frac{1}{2\tau}
       \exp\!\Bigl(\frac{\sigma^2}{2\tau^2} - \frac{x - \mu}{\tau}\Bigr)
       \operatorname{erfc}\!\Bigl(
           -\frac{x - \mu}{\sqrt{2}\,\sigma}
           + \frac{\sigma}{\sqrt{2}\,\tau}
       \Bigr)

Variables
^^^^^^^^^

- ``x``: input x-values.
- ``A``: amplitude scale.
- ``\mu``: Gaussian center.
- ``\sigma``: Gaussian width.
- ``\tau``: exponential decay constant (tailing).

stretched_exponential
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: labfit.models.stretched_exponential

Equation
^^^^^^^^

.. math::

   y = o + A\,
       \exp\!\Bigl[-\Bigl(\frac{x}{\tau}\Bigr)^\beta\Bigr]

Variables
^^^^^^^^^

- ``x``: input x-values.
- ``A``: amplitude at ``x = 0``.
- ``\tau``: decay constant.
- ``\beta``: stretching exponent (:math:`\beta = 1` recovers a simple exponential).
- ``o``: vertical offset, default ``0.0``.

tanh
~~~~

.. autofunction:: labfit.models.tanh

Equation
^^^^^^^^

.. math::

   y = o + A\,
       \tanh\!\Bigl(\frac{x - x_0}{w}\Bigr)

Variables
^^^^^^^^^

- ``x``: input x-values.
- ``A``: step height (half the total amplitude).
- ``x_0``: inflection point.
- ``w``: transition width.
- ``o``: vertical offset, default ``0.0``.

arctan
~~~~~~

.. autofunction:: labfit.models.arctan

Equation
^^^^^^^^

.. math::

   y = o + A\,
       \arctan\!\Bigl(\frac{x - x_0}{w}\Bigr)

Variables
^^^^^^^^^

- ``x``: input x-values.
- ``A``: step amplitude scale.
- ``x_0``: inflection point.
- ``w``: transition width.
- ``o``: vertical offset, default ``0.0``.

beat
~~~~

.. autofunction:: labfit.models.beat

Equation
^^^^^^^^

.. math::

   y = o + \frac{A}{2}\,
       \bigl[\cos(2\pi f_1 x + \phi) + \cos(2\pi f_2 x + \phi)\bigr]

Variables
^^^^^^^^^

- ``x``: input x-values.
- ``A``: overall amplitude.
- ``f_1``: cycles per unit ``x`` (first frequency).
- ``f_2``: cycles per unit ``x`` (second frequency).
- ``\phi``: phase shift in radians.
- ``o``: vertical offset, default ``0.0``.

rational
~~~~~~~~

.. autofunction:: labfit.models.rational

Equation
^^^^^^^^

.. math::

   y = \frac{A}{x - x_0}

Variables
^^^^^^^^^

- ``x``: input x-values.
- ``A``: amplitude scale.
- ``x_0``: pole location.

quartic
~~~~~~~

.. autofunction:: labfit.models.quartic

Equation
^^^^^^^^

.. math::

   y = a x^4 + b x^3 + c x^2 + d x + e

Variables
^^^^^^^^^

- ``x``: input x-values.
- ``a``: quartic coefficient.
- ``b``: cubic coefficient.
- ``c``: quadratic coefficient.
- ``d``: linear coefficient.
- ``e``: constant term.

quintic
~~~~~~~

.. autofunction:: labfit.models.quintic

Equation
^^^^^^^^

.. math::

   y = a x^5 + b x^4 + c x^3 + d x^2 + e x + f

Variables
^^^^^^^^^

- ``x``: input x-values.
- ``a``: quintic coefficient.
- ``b``: quartic coefficient.
- ``c``: cubic coefficient.
- ``d``: quadratic coefficient.
- ``e``: linear coefficient.
- ``f``: constant term.
