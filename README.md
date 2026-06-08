<p align="center">
  <img src="https://raw.githubusercontent.com/matthewjdoyle/labfit/master/docs/_static/banner.svg" alt="LabFit - least-squares curve fitting for Python" width="100%">
</p>

<p align="center">
  <a href="https://matthewd0yle.com/labfit/"><strong>Documentation</strong></a> ·
  <a href="https://github.com/matthewjdoyle/labfit"><strong>GitHub</strong></a> ·
  <a href="https://matthewd0yle.com/labfit/gallery.html"><strong>Gallery</strong></a>
</p>

**LabFit** is a small, student-friendly Python library for least-squares curve fitting,
error propagation, and publication-quality plotting - built on NumPy, SciPy, and Matplotlib.

```python
pip install labfit
```

## Quick start

```python
from labfit import fit, plot_fit

result = fit("data.csv", model="exponential")
plot_fit(result)
print(f"χ²/ν = {result.reduced_chi2:.3f}")
```

Three lines: CSV → fit → plot → goodness-of-fit.

## Features

- **Built-in models** - linear, quadratic, Gaussian, Lorentzian, exponential, power law,
  logistic, sinc, bimodal Gaussian, damped oscillator, and more - all by name.
- **Reduced χ²** on every fit result, so you can immediately assess goodness-of-fit.
- **Asymmetric & correlated errors** - handle `sigma_low`/`sigma_high` and full
  covariance matrices without rewriting propagation code.
- **Multi-series fitting** - fit and compare many data sets on shared or independent axes.
- **Plots** - residual panels, multi-fit grids, proper axis labels.
- **100 % Python** - no compiled extensions, installs everywhere.

## Longer example

```python
import numpy as np
from labfit import quick_fit, plot_fit

rng = np.random.default_rng(42)
x = np.linspace(-5, 5, 200)
y = 3.0 * np.exp(-0.5 * ((x - 0.5) / 1.2) ** 2) + rng.normal(0, 0.05, size=x.size)

result = quick_fit(x, y, model="gaussian")
print(f"amplitude = {result.params['amplitude']:.3f}")
print(f"mean      = {result.params['mean']:.3f}")
print(f"sigma     = {result.params['sigma']:.3f}")
print(f"χ²/ν      = {result.reduced_chi2:.4f}")

plot = plot_fit(result, show_residuals=True)
plot.save("gaussian_fit.png")
```

## Dependencies

- Python ≥ 3.10
- NumPy, SciPy, Matplotlib
