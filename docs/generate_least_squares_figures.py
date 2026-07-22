#!/usr/bin/env python3
"""Generate pedagogical SVG figures for least-squares docs.

Matches LabFit's publication style (see labfit/plot.py use_publication_style).
"""

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "_static"

# ── Okabe-Ito palette (matches labfit/plot.py) ─────────────────
OKABE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00", "#000000"]


def _labfit_style():
    """Apply rcParams consistent with LabFit's use_publication_style()."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 9.5,
            "mathtext.fontset": "dejavusans",
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.linewidth": 0.8,
            "axes.labelsize": 10.5,
            "axes.titlesize": 11,
            "axes.grid": False,
            "axes.facecolor": "white",
            "xtick.direction": "in",
            "xtick.major.size": 4,
            "xtick.major.width": 0.7,
            "xtick.minor.size": 2,
            "xtick.minor.width": 0.5,
            "xtick.labelsize": 9,
            "xtick.top": True,
            "ytick.direction": "in",
            "ytick.major.size": 4,
            "ytick.major.width": 0.7,
            "ytick.minor.size": 2,
            "ytick.minor.width": 0.5,
            "ytick.labelsize": 9,
            "ytick.right": True,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "legend.handlelength": 1.2,
            "lines.linewidth": 1.4,
            "lines.markersize": 5.5,
            "figure.facecolor": "white",
        }
    )


# ── 1. residuals.svg ───────────────────────────────────────────
def fig_residuals():
    _labfit_style()
    rng = np.random.default_rng(42)
    x = np.linspace(0.5, 9.5, 12)
    true_slope, true_int = 0.8, 0.5
    y = true_slope * x + true_int + rng.normal(0, 0.5, size=x.size)

    A = np.vstack([x, np.ones_like(x)]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    x_fine = np.linspace(0, 10, 200)
    y_fit = m * x_fine + c

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    # residuals — grey dashed lines
    for xi, yi in zip(x, y, strict=True):
        y_on_line = m * xi + c
        ax.plot([xi, xi], [yi, y_on_line], color="#7f8c8d", linewidth=1.0, linestyle="--", zorder=1)
        ax.scatter([xi], [y_on_line], color="#7f8c8d", s=8, zorder=2, marker=".")

    # fit line — Okabe-Ito[1] (orange, matches LabFit model line)
    ax.plot(x_fine, y_fit, color=OKABE[1], linewidth=1.4, zorder=3, label="Best-fit model $M(x)$")

    # data — Okabe-Ito[0] (blue, matches LabFit data markers)
    ax.scatter(x, y, color=OKABE[0], s=5.5**2, zorder=4, label="Data $(x_i, y_i)$")

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 9)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")

    # r_i annotation on the right of the residual line
    mid_idx = 5
    xi, yi = x[mid_idx], y[mid_idx]
    y_on_line = m * xi + c
    mid_y = (yi + y_on_line) / 2
    ax.annotate(
        "$r_i$",
        xy=(xi, mid_y),
        xytext=(xi + 1.3, mid_y + 0.1),
        fontsize=10,
        color="#7f8c8d",
        ha="center",
        arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=1.0),
    )

    ax.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "residuals.svg", format="svg")
    plt.close(fig)
    print("  ✓ residuals.svg")


# ── 2. chi2-surface.svg ────────────────────────────────────────
def fig_chi2_surface():
    _labfit_style()
    rng = np.random.default_rng(42)
    x = np.linspace(0.5, 9.5, 10)
    true_m, true_c = 0.8, 0.5
    y = true_m * x + true_c + rng.normal(0, 0.4, size=x.size)

    slopes = np.linspace(-0.2, 1.8, 80)
    ints = np.linspace(-1.5, 2.5, 80)
    SS = np.zeros((len(slopes), len(ints)))
    for i, m_val in enumerate(slopes):
        for j, c_val in enumerate(ints):
            y_pred = m_val * x + c_val
            SS[i, j] = np.sum((y - y_pred) ** 2)

    A = np.vstack([x, np.ones_like(x)]).T
    m_best, c_best = np.linalg.lstsq(A, y, rcond=None)[0]

    fig, ax = plt.subplots(figsize=(6.5, 5))

    contours = ax.contour(ints, slopes, SS, levels=20, cmap="viridis", linewidths=0.8)
    ax.clabel(contours, inline=True, fontsize=7, fmt="%.0f")

    ax.plot(c_best, m_best, "*", markersize=12, color=OKABE[1], zorder=5, label="Best fit $(m^*, c^*)$")

    ax.set_xlabel("Intercept $c$")
    ax.set_ylabel("Slope $m$")
    ax.set_title("$\\chi^2$ landscape")

    annot_x = c_best + 0.5 if c_best < 1.0 else c_best - 0.7
    annot_y = m_best + 0.25
    ax.annotate(
        "Minimum $\\chi^2$",
        xy=(c_best, m_best),
        xytext=(annot_x, annot_y),
        fontsize=9.5,
        color=OKABE[1],
        ha="center",
        arrowprops=dict(arrowstyle="->", color=OKABE[1], lw=1.0),
    )

    ax.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "chi2-surface.svg", format="svg")
    plt.close(fig)
    print("  ✓ chi2-surface.svg")


# ── 3. optimizer-path.svg ──────────────────────────────────────
def fig_optimizer_path():
    """Trajectory from a poor initial guess to the minimum.

    Uses properly-scaled gradient descent (small steps, monotonic
    χ² decrease) then samples evenly for visual clarity.
    """
    _labfit_style()
    rng = np.random.default_rng(42)
    x = np.linspace(0.5, 9.5, 10)
    true_m, true_c = 0.8, 0.5
    y = true_m * x + true_c + rng.normal(0, 0.4, size=x.size)

    # Contour grid
    slopes = np.linspace(-0.5, 2.0, 80)
    ints = np.linspace(-1.8, 2.8, 80)
    SS = np.zeros((len(slopes), len(ints)))
    for i, m_val in enumerate(slopes):
        for j, c_val in enumerate(ints):
            y_pred = m_val * x + c_val
            SS[i, j] = np.sum((y - y_pred) ** 2)

    A = np.vstack([x, np.ones_like(x)]).T
    m_best, c_best = np.linalg.lstsq(A, y, rcond=None)[0]

    # ── Stable gradient-descent path ─────────────────────────
    m0, c0 = 1.8, -1.4
    m_curr, c_curr = m0, c0
    alpha = 0.0008  # small, safe step
    raw = [(m0, c0)]
    for _ in range(1200):
        y_pred = m_curr * x + c_curr
        resid = y - y_pred  # data minus model
        dm = 2 * np.sum(resid * x) / x.size  # gradient of χ²/N
        dc = 2 * np.sum(resid) / x.size
        m_curr += alpha * dm  # move downhill
        c_curr += alpha * dc
        raw.append((m_curr, c_curr))
        # stop when close enough
        if abs(m_curr - m_best) < 0.005 and abs(c_curr - c_best) < 0.005:
            break

    # ── Plot with smooth path (no intermediate dots) ─────────
    fig, ax = plt.subplots(figsize=(6.5, 5))

    # Soft heatmap
    ax.contourf(ints, slopes, SS, levels=30, cmap="YlOrRd", alpha=0.12, zorder=0)
    # Reference contours
    cs_lines = ax.contour(ints, slopes, SS, levels=10, cmap="Greys", linewidths=0.5, zorder=1)
    ax.clabel(cs_lines, inline=True, fontsize=6.5, fmt="%.0f")

    raw_ms = [p[0] for p in raw]
    raw_cs = [p[1] for p in raw]

    # Smooth path line (no dot markers — clean visual)
    ax.plot(raw_cs, raw_ms, "-", color=OKABE[3], linewidth=2.0, zorder=4)

    # Directional arrows at evenly-spaced positions along the path
    n_arrows = 4
    for k in range(1, n_arrows + 1):
        i = min(k * len(raw) // (n_arrows + 1), len(raw) - 2)
        dc = raw_cs[i + 1] - raw_cs[i]
        dm = raw_ms[i + 1] - raw_ms[i]
        ax.arrow(
            raw_cs[i],
            raw_ms[i],
            dc * 0.7,
            dm * 0.7,
            head_width=0.06,
            head_length=0.08,
            fc=OKABE[3],
            ec=OKABE[3],
            linewidth=0,
            zorder=5,
        )

    # Start and end markers
    ax.scatter([c0], [m0], color=OKABE[2], s=70, zorder=6, marker="s", edgecolors="white", linewidth=1.2)
    ax.plot(c_best, m_best, "*", markersize=16, color=OKABE[1], zorder=7)

    # Labels
    ax.annotate(
        "Start  $\\boldsymbol{\\theta}_0$",
        xy=(c0, m0),
        xytext=(c0 - 0.55, m0 - 0.45),
        fontsize=9,
        color=OKABE[2],
        ha="right",
        va="top",
        arrowprops=dict(arrowstyle="->", color=OKABE[2], lw=0.8),
    )
    ax.annotate(
        "Best fit  $\\boldsymbol{\\theta}^*$",
        xy=(c_best, m_best),
        xytext=(c_best + 0.25, m_best + 0.45),
        fontsize=9,
        color=OKABE[1],
        ha="center",
        va="bottom",
        arrowprops=dict(arrowstyle="->", color=OKABE[1], lw=0.8),
    )

    ax.set_xlabel("Intercept $c$")
    ax.set_ylabel("Slope $m$")
    ax.set_title("Walking downhill on the $\\chi^2$ surface")
    fig.tight_layout()
    fig.savefig(OUT / "optimizer-path.svg", format="svg")
    plt.close(fig)
    print("  ✓ optimizer-path.svg")


# ── 4. goodness-of-fit.svg ─────────────────────────────────────
def fig_goodness():
    _labfit_style()
    rng = np.random.default_rng(2001)
    x = np.linspace(0.5, 9.5, 15)

    def true_fn(xv):
        return 0.7 * xv + 0.5

    # (title, true_noise, reported_err)
    #   χ² << 1 → reported err is much larger than the true scatter
    #   χ² ≈ 1  → reported err matches the true scatter
    #   χ² >> 1 → reported err is much smaller than the true scatter
    scenarios = [
        ("Overestimated errors\n$\\chi^2_\\nu \\ll 1$", 0.08, 1.0),
        ("Good fit\n$\\chi^2_\\nu \\approx 1$", 0.25, 0.25),
        ("Underestimated errors\n$\\chi^2_\\nu \\gg 1$", 0.55, 0.07),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True)

    for ax, (title, true_noise, reported_err) in zip(axes, scenarios, strict=True):
        y = true_fn(x) + rng.normal(0, true_noise, size=x.size)

        A = np.vstack([x, np.ones_like(x)]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        x_fine = np.linspace(0, 10, 200)
        y_fit = m * x_fine + c

        residual = y - (m * x + c)
        dof = x.size - 2
        chi2 = np.sum((residual / reported_err) ** 2)
        red_chi2 = chi2 / dof

        ax.errorbar(
            x,
            y,
            yerr=reported_err,
            fmt="o",
            color=OKABE[0],
            markersize=4.0,
            capsize=3.5,
            capthick=1.2,
            elinewidth=1.2,
            ecolor="#7f8c8d",
            zorder=2,
        )
        ax.plot(x_fine, y_fit, color=OKABE[1], linewidth=1.4, zorder=3)

        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("$x$")

        ax.text(
            0.95,
            0.92,
            f"$\\chi^2_\\nu = {red_chi2:.2f}$",
            transform=ax.transAxes,
            fontsize=9,
            va="top",
            ha="right",
            bbox=dict(facecolor="white", edgecolor="none", pad=2),
        )

    axes[0].set_ylabel("$y$")
    fig.tight_layout()
    fig.savefig(OUT / "goodness-of-fit.svg", format="svg")
    plt.close(fig)
    print("  ✓ goodness-of-fit.svg")


# ── 5. curvature-uncertainty.svg ───────────────────────────────
def fig_curvature():
    _labfit_style()
    rng = np.random.default_rng(42)
    x = np.linspace(0.5, 9.5, 10)
    true_m, true_c = 0.8, 0.5
    y = true_m * x + true_c + rng.normal(0, 0.4, size=x.size)

    slopes = np.linspace(0.2, 1.4, 80)
    ints = np.linspace(-0.5, 1.8, 80)
    SS = np.zeros((len(slopes), len(ints)))
    for i, m_val in enumerate(slopes):
        for j, c_val in enumerate(ints):
            y_pred = m_val * x + c_val
            SS[i, j] = np.sum((y - y_pred) ** 2)

    A = np.vstack([x, np.ones_like(x)]).T
    m_best, c_best = np.linalg.lstsq(A, y, rcond=None)[0]

    fig, ax = plt.subplots(figsize=(6, 5))

    chi2_min = np.min(SS)

    # thin background contours for depth (no fill)
    ax.contour(ints, slopes, SS, levels=15, cmap="Greys", linewidths=0.3, alpha=0.5)

    # three highlighted contour levels: min, +1 (1σ), +4 (2σ)
    lvls = [chi2_min, chi2_min + 1, chi2_min + 4]
    cs = ax.contour(
        ints, slopes, SS, levels=lvls, colors=[OKABE[2], OKABE[1], OKABE[3]], linewidths=[1.0, 1.6, 1.0]
    )
    ax.clabel(
        cs,
        fmt={lvls[0]: "$\\chi^2_{\\min}$", lvls[1]: "$\\chi^2_{\\min}+1$", lvls[2]: "$\\chi^2_{\\min}+4$"},
        fontsize=7.5,
    )

    ax.plot(c_best, m_best, "*", markersize=12, color=OKABE[1], zorder=5, label="Best fit")

    ax.annotate(
        "1-$\\sigma$ contour",
        xy=(c_best + 0.3, m_best + 0.25),
        fontsize=9,
        color=OKABE[1],
        ha="left",
    )

    ax.set_xlabel("Intercept $c$")
    ax.set_ylabel("Slope $m$")
    ax.set_title("Curvature determines uncertainty")
    ax.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "curvature-uncertainty.svg", format="svg")
    plt.close(fig)
    print("  ✓ curvature-uncertainty.svg")


# ── main ────────────────────────────────────────────────────────
if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    print("Generating least-squares pedagogical figures...")
    fig_residuals()
    fig_chi2_surface()
    fig_optimizer_path()
    fig_goodness()
    fig_curvature()
    print("Done -- all figures saved to", OUT)
