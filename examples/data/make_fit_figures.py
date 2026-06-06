"""Generate fit figures for the revised Beamer walkthrough using 300 nm region."""
from labfit.io import load_csv
from labfit import fit, plot_fit
import numpy as np
import matplotlib.pyplot as plt

data = load_csv('examples/data/mercury_int10s.csv',
                x_col='wavelength_nm', y_col='intensity', y_err_col='y_error')

# ── 1) Full spectrum with highlight box ──
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(data.x, data.y, 'k-', lw=0.5, label='Hg spectrum (10 ms)')
# Highlight the 300 nm region
ax.axvspan(294, 306, color='#D55E00', alpha=0.12)
ax.text(300, 4000, 'our region', ha='center', fontsize=9, color='#D55E00', style='italic')
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Intensity (counts)')
ax.set_title('Mercury Emission Spectrum')
fig.tight_layout()
fig.savefig('examples/data/mercury_full_spectrum.png', dpi=150)
print('1/5  Full spectrum saved')
plt.close(fig)

# ── 2) Bimodal Gaussian fit: 294-306 nm (two small peaks) ──
mask = (data.x > 294) & (data.x < 306)
x_sub = data.x[mask]
y_sub = data.y[mask]
e_sub = data.y_err[mask]

print(f'\nSubset 294-306 nm: {len(x_sub)} points')
print(f'y range: {y_sub.min():.1f} - {y_sub.max():.1f}')

# Fit bimodal Gaussian
# Initial guesses: peak1 ~297.4, peak2 ~302.8, both ~small amplitude
r_bi = fit(x_sub, y_sub, model='bimodal_gaussian', sigma=e_sub,
           p0={'amplitude1': 200, 'mean1': 297.4, 'sigma1': 1.0,
               'amplitude2': 400, 'mean2': 302.8, 'sigma2': 1.2})
print('\n=== Bimodal Gaussian (two small peaks ~300 nm) ===')
print(r_bi)

p_bi = plot_fit(r_bi, show_residuals=True)
p_bi.save('examples/data/mercury_300nm_bimodal_fit.png')
print('2/5  300 nm bimodal fit saved')

# ── 3) Single Gaussian on the 302.8 nm peak ──
mask3 = (data.x > 301.0) & (data.x < 304.8)
x3 = data.x[mask3]
y3 = data.y[mask3]
e3 = data.y_err[mask3]

r_single = fit(x3, y3, model='gaussian_baseline', sigma=e3,
               p0={'amplitude': 400, 'mean': 302.8, 'sigma': 0.6, 'm': 0, 'b': 800})
print('\n=== Single Gaussian at 302.8 nm ===')
print(r_single)

p_s = plot_fit(r_single, show_residuals=True)
p_s.save('examples/data/mercury_302nm_single_fit.png')
print('3/5  302 nm single fit saved')

# ── 4) Show the region data as a simple overview plot ──
fig2, ax2 = plt.subplots(figsize=(7, 3.5))
ax2.plot(x_sub, y_sub, 'o-', color='#0072B2', ms=3, lw=0.8)
ax2.set_xlabel('Wavelength (nm)')
ax2.set_ylabel('Intensity (counts)')
ax2.set_title('Mercury spectrum: 294-306 nm region')
ax2.annotate('Peak ~297.4 nm', xy=(297.4, 925), xytext=(295, 980),
             arrowprops=dict(arrowstyle='->', color='#D55E00'), fontsize=9, color='#D55E00')
ax2.annotate('Peak ~302.8 nm', xy=(302.8, 1198), xytext=(304, 1260),
             arrowprops=dict(arrowstyle='->', color='#D55E00'), fontsize=9, color='#D55E00')
fig2.tight_layout()
fig2.savefig('examples/data/mercury_300nm_region.png', dpi=150)
print('4/5  300 nm region overview saved')
plt.close(fig2)

print('\n5/5  All figures generated successfully.')
