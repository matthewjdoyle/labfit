from __future__ import annotations

import functools
import inspect
from collections import OrderedDict
from collections.abc import Callable
from typing import Concatenate, ParamSpec, Protocol

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import voigt_profile
from scipy.stats import exponnorm, skewnorm

from .types import ModelFunction, ModelSpec

_P = ParamSpec("_P")


class _ArrayModel(Protocol[_P]):
    def __call__(self, x: ArrayLike, *args: _P.args, **kwargs: _P.kwargs) -> np.ndarray: ...


def _as_x(func: Callable[Concatenate[np.ndarray, _P], np.ndarray]) -> _ArrayModel[_P]:
    """Convert x to a float array while preserving the complete model signature."""

    @functools.wraps(func)
    def wrapper(x: ArrayLike, *args: _P.args, **kwargs: _P.kwargs) -> np.ndarray:
        return func(np.asarray(x, dtype=float), *args, **kwargs)

    return wrapper


@_as_x
def constant(x: np.ndarray, level: float) -> np.ndarray:
    """Horizontal line at a fixed y-value."""
    return np.full_like(x, level, dtype=float)


@_as_x
def linear(x: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    """Straight line with constant gradient."""
    return slope * x + intercept


@_as_x
def quadratic(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Second-degree polynomial (parabola)."""
    return a * x**2 + b * x + c


@_as_x
def cubic(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """Third-degree polynomial."""
    return ((a * x + b) * x + c) * x + d


@_as_x
def gaussian(x: np.ndarray, amplitude: float, mean: float, sigma: float) -> np.ndarray:
    """Symmetric bell-shaped peak (normal distribution)."""
    sigma_array = np.asarray(sigma, dtype=float)
    return amplitude * np.exp(-0.5 * ((x - mean) / sigma_array) ** 2)


@_as_x
def lorentzian(x: np.ndarray, amplitude: float, center: float, gamma: float) -> np.ndarray:
    """Peak with a narrower core and heavier tails than a Gaussian.

    Also known as the Cauchy or Breit-Wigner distribution. Common in
    spectroscopy for natural line shapes and in particle physics for
    resonance profiles.
    """
    gamma_array = np.asarray(gamma, dtype=float)
    return amplitude * (gamma_array**2 / ((x - center) ** 2 + gamma_array**2))


@_as_x
def exponential(x: np.ndarray, amplitude: float, decay: float) -> np.ndarray:
    """Exponential decay starting from ``amplitude`` at ``x = 0``."""
    return amplitude * np.exp(-decay * x)


@_as_x
def power_law(x: np.ndarray, amplitude: float, exponent: float) -> np.ndarray:
    """Power-law scaling with a variable exponent."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return amplitude * np.power(x, exponent)


@_as_x
def logistic(x: np.ndarray, amplitude: float, x0: float, k: float, baseline: float = 0.0) -> np.ndarray:
    """Sigmoidal curve with a tunable steepness.

    Transitions smoothly from ``baseline`` to ``baseline + amplitude``
    centred at ``x0``. Used for population growth, dose-response curves,
    and phase transitions with a continuous order parameter.
    """
    return baseline + amplitude / (1.0 + np.exp(-k * (x - x0)))


@_as_x
def sine(x: np.ndarray, amplitude: float, frequency: float, phase: float, offset: float = 0.0) -> np.ndarray:
    """Sinusoidal oscillation with tunable frequency and phase."""
    return offset + amplitude * np.sin(2.0 * np.pi * frequency * x + phase)


@_as_x
def cosine(
    x: np.ndarray, amplitude: float, frequency: float, phase: float, offset: float = 0.0
) -> np.ndarray:
    """Cosinusoidal oscillation with tunable frequency and phase."""
    return offset + amplitude * np.cos(2.0 * np.pi * frequency * x + phase)


@_as_x
def damped_oscillator(
    x: np.ndarray, amplitude: float, damping: float, frequency: float, phase: float
) -> np.ndarray:
    """Oscillation that decays exponentially (cosine form).

    Represents a harmonic oscillator with friction, such as a swinging
    pendulum subject to air resistance or an RLC circuit with resistance.
    """
    return amplitude * np.exp(-damping * x) * np.cos(2.0 * np.pi * frequency * x + phase)


@_as_x
def damped_sine(
    x: np.ndarray, amplitude: float, damping: float, frequency: float, phase: float, offset: float = 0.0
) -> np.ndarray:
    """Oscillation that decays exponentially (sine form).

    Identical in form to the damped oscillator but uses sine instead of
    cosine. Suitable when the signal starts at the equilibrium point.
    """
    return offset + amplitude * np.exp(-damping * x) * np.sin(2.0 * np.pi * frequency * x + phase)


@_as_x
def sinc(x: np.ndarray, amplitude: float, center: float, width: float) -> np.ndarray:
    """Central peak with oscillating sidelobes.

    Defined as sin(u)/u where u depends on the distance from ``center``
    and the characteristic ``width``. Arises in diffraction patterns,
    signal processing (ideal low-pass filter response), and Fourier
    optics.
    """
    u = np.pi * (x - center) / width
    with np.errstate(invalid="ignore"):
        return amplitude * np.sinc(u / np.pi)
    # np.sinc(x) = sin(pi*x) / (pi*x), so np.sinc(u/pi) = sin(u)/u


@_as_x
def exponential_rise(x: np.ndarray, amplitude: float, tau: float, offset: float = 0.0) -> np.ndarray:
    """Saturation curve approaching an asymptotic value.

    Starts at ``offset`` and rises exponentially toward
    ``offset + amplitude`` with time constant ``tau``. Describes
    charging capacitors, thermal equilibration, and approach to steady
    state in first-order systems.
    """
    return amplitude * (1.0 - np.exp(-x / tau)) + offset


@_as_x
def double_exponential(
    x: np.ndarray, amplitude1: float, tau1: float, amplitude2: float, tau2: float
) -> np.ndarray:
    """Sum of two exponential decays with distinct time constants.

    Useful when a process has both a fast and a slow relaxation
    channel, such as biexponential fluorescence decay, two-component
    nuclear magnetic resonance relaxation, or mixed kinetics.
    """
    return amplitude1 * np.exp(-x / tau1) + amplitude2 * np.exp(-x / tau2)


@_as_x
def moffat(x: np.ndarray, amplitude: float, x0: float, alpha: float, beta: float) -> np.ndarray:
    """Peak with power-law tails controlled by an exponent.

    Similar to a Lorentzian near the centre but with a variable
    power-law fall-off (``beta`` controls the tail weight). Widely used
    in astronomy for point-spread functions and in X-ray diffraction
    profile analysis.
    """
    return amplitude * (1.0 + ((x - x0) / alpha) ** 2) ** (-beta)


@_as_x
def gaussian_baseline(
    x: np.ndarray, amplitude: float, mean: float, sigma: float, m: float, b: float
) -> np.ndarray:
    """Gaussian peak superimposed on a linear background."""
    sigma_array = np.asarray(sigma, dtype=float)
    g = amplitude * np.exp(-0.5 * ((x - mean) / sigma_array) ** 2)
    return g + m * x + b


@_as_x
def bimodal_gaussian(
    x: np.ndarray,
    amplitude1: float,
    mean1: float,
    sigma1: float,
    amplitude2: float,
    mean2: float,
    sigma2: float,
) -> np.ndarray:
    """Sum of two independent Gaussian peaks.

    Models data with two resolved components — for example, emission
    lines from closely spaced energy levels, overlapping diffraction
    peaks, or multi-species velocity distributions.
    """
    sigma1_array = np.asarray(sigma1, dtype=float)
    sigma2_array = np.asarray(sigma2, dtype=float)
    g1 = amplitude1 * np.exp(-0.5 * ((x - mean1) / sigma1_array) ** 2)
    g2 = amplitude2 * np.exp(-0.5 * ((x - mean2) / sigma2_array) ** 2)
    return g1 + g2


# ── New models ──────────────────────────────────────────────────────

FWHM2SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
"""Conversion factor:  sigma = FWHM * FWHM2SIGMA."""


@_as_x
def voigt(x: np.ndarray, amplitude: float, center: float, sigma: float, gamma: float) -> np.ndarray:
    """Symmetric peak shape with a Gaussian core and Lorentzian wings.

    The Voigt profile is the convolution of a Gaussian and a Lorentzian
    of equal widths. It arises naturally in spectroscopy when Doppler
    (Gaussian) and natural/pressure (Lorentzian) broadening act together.
    """
    return amplitude * voigt_profile(x - center, sigma, gamma)


@_as_x
def skew_normal(x: np.ndarray, amplitude: float, location: float, scale: float, alpha: float) -> np.ndarray:
    """Asymmetric bell-shaped curve with a shape parameter.

    Generalises the normal distribution by adding a skewness parameter
    ``alpha``. When ``alpha = 0`` the curve is a symmetric Gaussian; positive
    values tilt the peak leftwards and negative values tilt it rightwards.
    Useful for modelling peaks that are not symmetric about their centre.
    """
    return amplitude * skewnorm.pdf(x, alpha, loc=location, scale=scale)


@_as_x
def gaussian_fwhm(x: np.ndarray, amplitude: float, center: float, fwhm: float) -> np.ndarray:
    """Gaussian peak parameterised by its full width at half maximum.

    Identical to the standard Gaussian model but accepts the peak width
    as the directly measurable FWHM instead of the standard deviation.
    The conversion between FWHM and sigma is handled internally.
    """
    sigma = np.asarray(fwhm, dtype=float) * FWHM2SIGMA
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)


@_as_x
def lorentzian_fwhm(x: np.ndarray, amplitude: float, center: float, fwhm: float) -> np.ndarray:
    """Lorentzian peak parameterised by its full width at half maximum.

    Identical to the standard Lorentzian model but accepts the peak
    width as the directly measurable FWHM instead of the half-width at
    half maximum. The conversion is handled internally.
    """
    hwhm = np.asarray(fwhm, dtype=float) / 2.0
    return amplitude * (hwhm**2 / ((x - center) ** 2 + hwhm**2))


@_as_x
def exgaussian(x: np.ndarray, amplitude: float, mu: float, sigma: float, tau: float) -> np.ndarray:
    """Asymmetric peak with a Gaussian rise and exponential tail.

    The exponentially modified Gaussian (ExGaussian) is the convolution
    of a normal distribution with an exponential decay. It produces a
    peak that rises symmetrically but decays with a long tail, matching
    the characteristic shape of chromatographic signals, reaction-time
    data, and detector pulses. ``amplitude`` is the integrated area;
    ``sigma`` and ``tau`` must be positive.
    """
    sigma_array = np.asarray(sigma, dtype=float)
    tau_array = np.asarray(tau, dtype=float)
    return amplitude * exponnorm.pdf(x, tau_array / sigma_array, loc=mu, scale=sigma_array)


@_as_x
def stretched_exponential(
    x: np.ndarray, amplitude: float, tau: float, beta: float, offset: float = 0.0
) -> np.ndarray:
    """Decay function with a variable stretching exponent.

    A generalisation of the exponential decay where the time constant
    is raised to a power ``beta``. When ``beta = 1`` it reduces to a
    simple exponential; values of ``beta`` between zero and one produce
    a slower, stretched decay. Commonly used to model relaxation in
    disordered systems such as polymers, glasses, and biological tissues.
    """
    tau_array = np.asarray(tau, dtype=float)
    beta_array = np.asarray(beta, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        return offset + amplitude * np.exp(-((x / tau_array) ** beta_array))


@_as_x
def tanh(x: np.ndarray, amplitude: float, center: float, width: float, offset: float = 0.0) -> np.ndarray:
    """Smooth sigmoidal step with a hyperbolic-tangent shape.

    Produces a monotonic transition from one level to another, centred
    at ``center`` with transition width ``width``. Appears in models of
    magnetic phase transitions, adsorption isotherms, and switching
    phenomena where the transition has a well-defined midpoint.
    """
    return offset + amplitude * np.tanh((x - center) / width)


@_as_x
def arctan(x: np.ndarray, amplitude: float, center: float, width: float, offset: float = 0.0) -> np.ndarray:
    """Smooth step with broad polynomial tails.

    Similar to a hyperbolic-tangent step but approaches the asymptotic
    levels as inverse-power tails rather than exponentially, giving a
    broader transition zone. Useful for resistivity jumps, specific-heat
    anomalies, and other phase-transition signatures with wide tails.
    """
    return offset + amplitude * np.arctan((x - center) / width)


@_as_x
def beat(
    x: np.ndarray, amplitude: float, frequency1: float, frequency2: float, phase: float, offset: float = 0.0
) -> np.ndarray:
    """Superposition of two cosine waves of nearby frequencies.

    Models the acoustic or electronic beat phenomenon where two
    oscillations of similar frequency interfere, producing an amplitude
    modulation (envelope) at the difference frequency. The output is the
    average of the two cosines, scaled by the amplitude.
    """
    return offset + amplitude * 0.5 * (
        np.cos(2.0 * np.pi * frequency1 * x + phase) + np.cos(2.0 * np.pi * frequency2 * x + phase)
    )


@_as_x
def rational(x: np.ndarray, amplitude: float, x0: float) -> np.ndarray:
    """First-order rational function with a single pole.

    Produces a simple resonance lineshape with a singularity at ``x0``.
    The magnitude diverges as the independent variable approaches the
    pole, making this suitable for modelling resonant behaviour in
    driven systems and susceptibility measurements away from the
    singular point.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return amplitude / (x - x0)


@_as_x
def quartic(x: np.ndarray, a: float, b: float, c: float, d: float, e: float) -> np.ndarray:
    """Fourth-degree polynomial."""
    return ((a * x + b) * x + c) * x * x + d * x + e


@_as_x
def quintic(x: np.ndarray, a: float, b: float, c: float, d: float, e: float, f: float) -> np.ndarray:
    """Fifth-degree polynomial."""
    return (((a * x + b) * x + c) * x + d) * x * x + e * x + f


MODEL_REGISTRY: OrderedDict[str, ModelFunction] = OrderedDict(
    [
        ("constant", constant),
        ("linear", linear),
        ("quadratic", quadratic),
        ("cubic", cubic),
        ("gaussian", gaussian),
        ("lorentzian", lorentzian),
        ("exponential", exponential),
        ("power_law", power_law),
        ("logistic", logistic),
        ("sine", sine),
        ("cosine", cosine),
        ("damped_oscillator", damped_oscillator),
        ("damped_sine", damped_sine),
        ("sinc", sinc),
        ("exponential_rise", exponential_rise),
        ("double_exponential", double_exponential),
        ("moffat", moffat),
        ("gaussian_baseline", gaussian_baseline),
        ("bimodal_gaussian", bimodal_gaussian),
        ("voigt", voigt),
        ("skew_normal", skew_normal),
        ("gaussian_fwhm", gaussian_fwhm),
        ("lorentzian_fwhm", lorentzian_fwhm),
        ("exgaussian", exgaussian),
        ("stretched_exponential", stretched_exponential),
        ("tanh", tanh),
        ("arctan", arctan),
        ("beat", beat),
        ("rational", rational),
        ("quartic", quartic),
        ("quintic", quintic),
    ]
)

MODEL_NAMES = tuple(MODEL_REGISTRY.keys())


def get_model(model: ModelSpec) -> tuple[ModelFunction, str]:
    if callable(model):
        return model, getattr(model, "__name__", "custom")
    if model is None:
        return linear, "linear"
    if isinstance(model, str):
        key = model.lower()
        if key not in MODEL_REGISTRY:
            raise KeyError(f"Unknown model {model!r}; available: {', '.join(MODEL_NAMES)}")
        return MODEL_REGISTRY[key], key
    raise TypeError("model must be a callable or model name string")


def model_param_names(model: ModelSpec) -> tuple[str, ...]:
    fn, _ = get_model(model)
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())
    if not params:
        return ()
    return tuple(p.name for p in params[1:])


__all__ = [
    "MODEL_NAMES",
    "MODEL_REGISTRY",
    "constant",
    "linear",
    "quadratic",
    "cubic",
    "gaussian",
    "lorentzian",
    "exponential",
    "power_law",
    "logistic",
    "sine",
    "cosine",
    "damped_oscillator",
    "damped_sine",
    "sinc",
    "exponential_rise",
    "double_exponential",
    "moffat",
    "gaussian_baseline",
    "bimodal_gaussian",
    "voigt",
    "skew_normal",
    "gaussian_fwhm",
    "lorentzian_fwhm",
    "exgaussian",
    "stretched_exponential",
    "tanh",
    "arctan",
    "beat",
    "rational",
    "quartic",
    "quintic",
    "get_model",
    "model_param_names",
    "FWHM2SIGMA",
]
