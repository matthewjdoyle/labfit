"""Generate single-peak fit figures for slide 9."""
from labfit.io import load_csv
from labfit import fit, plot_fit
import numpy as np

data = load_csv('examples/data/mercury_int10s.csv',
                x_col='wavelength_nm', y_col='intensity', y_err_col='y_error')

# Single Gaussian at 297 nm
m1 = (data.x > 296) & (data.x < 299)
x1 = data.x[m1]; y1 = data.y[m1]; e1 = data.y_err[m1]
r1 = fit(x1, y1, model='gaussian_baseline', sigma=e1,
         p0={'amplitude': 150, 'mean': 297.4, 'sigma': 0.5, 'm': 0, 'b': 780})
print('297 nm fit chi2 =', r1.reduced_chi2)
p1 = plot_fit(r1, show_residuals=True)
p1.save('examples/data/mercury_297nm_single_fit.png')
print('  -> saved')

# Single Gaussian at 302.8 nm
m2 = (data.x > 301) & (data.x < 305)
x2 = data.x[m2]; y2 = data.y[m2]; e2 = data.y_err[m2]
r2 = fit(x2, y2, model='gaussian_baseline', sigma=e2,
         p0={'amplitude': 400, 'mean': 302.8, 'sigma': 0.6, 'm': 0, 'b': 800})
print('302.8 nm fit chi2 =', r2.reduced_chi2)
p2 = plot_fit(r2, show_residuals=True)
p2.save('examples/data/mercury_302nm_single_fit.png')
print('  -> saved')
