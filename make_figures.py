# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib"]
# ///
"""Build publication figures from results/abm.npz (made by discourse_sims.py).

Style follows figures4papers (Chen Liu): PALETTE, top/right spines off,
frameless legends, thick axes, PNG+PDF export.
Run:  uv run make_figures.py
"""
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
DATA = HERE / "results" / "abm.npz"
FIGS = HERE / "figures"

# --- figures4papers conventions ---------------------------------------------
PALETTE = {
    "blue_main": "#0F4D92", "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE", "green_2": "#AADCA9", "green_3": "#8BCF8B",
    "red_1": "#F6CFCB", "red_2": "#E9A6A1", "red_strong": "#B64342",
    "neutral": "#CFCECE", "highlight": "#FFD700",
    "teal": "#42949E", "violet": "#9A4D8E",
}


@dataclass(frozen=True)
class FigureStyle:
    font_size: int = 16
    axes_linewidth: float = 2.0
    font_family: tuple = ("DejaVu Sans", "Helvetica", "Arial", "sans-serif")


def apply_publication_style(style=FigureStyle()):
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": list(style.font_family),
        "font.size": style.font_size, "axes.titlesize": style.font_size,
        "axes.labelsize": style.font_size, "legend.fontsize": style.font_size - 3,
        "xtick.labelsize": style.font_size - 2, "ytick.labelsize": style.font_size - 2,
        "axes.linewidth": style.axes_linewidth,
        "xtick.major.width": style.axes_linewidth, "ytick.major.width": style.axes_linewidth,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "svg.fonttype": "none", "pdf.fonttype": 42,
    })


def finalize_figure(fig, name, formats=("png", "pdf"), dpi=300, pad=0.05):
    FIGS.mkdir(parents=True, exist_ok=True)
    for f in formats:
        fig.savefig(FIGS / f"{name}.{f}", dpi=dpi, bbox_inches="tight", pad_inches=pad)
    plt.close(fig)


def hist_line(ax, x, bins, **kw):
    d, e = np.histogram(x, bins=bins, density=True)
    ax.plot((e[:-1] + e[1:]) / 2, d, **kw)


# --- figures -----------------------------------------------------------------
C = PALETTE


def movavg(y, w=9):
    k = np.ones(w) / w; yp = np.pad(y, (w // 2, w // 2), mode="edge")
    return np.convolve(yp, k, mode="valid")


def boxed(ax):
    for sp in ax.spines.values(): sp.set_visible(True)


def smooth_hist(ax, x, w, bins, sigma=1.5, **kw):
    d, e = np.histogram(x, bins=bins, weights=w, density=True)
    k = np.exp(-0.5 * (np.arange(-4, 5) / sigma) ** 2); k /= k.sum()
    ax.plot((e[:-1] + e[1:]) / 2, np.convolve(d, k, mode="same"), **kw)


def fig_offset(R):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]; bins = np.linspace(-4, 4, 121)
    ax.hist(R["off_x"], bins=bins, density=True, color=C["neutral"], label="Population")
    smooth_hist(ax, R["off_x"], R["off_w"], bins, color=C["red_strong"], lw=2.5, label="Feed")
    ax.set(xlabel="Opinion $x$", ylabel="Density"); ax.legend(loc="upper left", fontsize=12)
    ax = axes[1]; bins = np.linspace(-3, 3, 61)
    ax.hist(R["off_a"], bins=bins, density=True, color=C["neutral"], label="Population")
    smooth_hist(ax, R["off_a"], R["off_w"], bins, color=C["red_strong"], lw=2.5, label="Feed (who is heard)")
    ax.axvline(0, color="black", lw=1, ls="--")
    ax.set(xlabel="Agreeability $a$  (< 0: disagreeable)", ylabel="Density"); ax.legend(loc="upper left", fontsize=12)
    fig.tight_layout(); finalize_figure(fig, "1_offset")


def fig_gap(R):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.axhline(1, color=C["neutral"], lw=2, ls="--")
    ax.errorbar(R["gap_lam"], R["gap_ratio"], yerr=1.96 * R["gap_ratio_se"], color=C["red_strong"],
                lw=2.5, marker="o", ms=7, mfc="white", mew=2, capsize=3)
    ax.set(xlabel="Online share of attention $\\lambda$", ylabel="Perceived / true\nout-group extremity")
    fig.tight_layout(); finalize_figure(fig, "2_perception_gap")


def fig_who_wins(R):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    f, da = R["win_f"], R["win_da"]; ext = [f[0] - 0.05, f[-1] + 0.05, da[0] - 0.3, da[-1] + 0.3]
    for ax, key, title in [(axes[0], "win_pop", "People: share of R"), (axes[1], "win_feed", "Feed: share of R posts")]:
        im = ax.imshow(R[key], origin="lower", cmap="RdBu_r", vmin=0, vmax=1, aspect="auto", extent=ext)
        for i in range(len(da)):
            for j in range(len(f)):
                v = R[key][i, j]; ax.text(f[j], da[i], f"{v:.2f}", ha="center", va="center", fontsize=11,
                                          color="white" if abs(v - 0.5) > 0.3 else "black")
        ax.set(title=title, xlabel="Initial share of R"); boxed(ax)
    axes[0].set_ylabel("$a_R - a_L$\n(< 0: R less deterred by disagreement)")
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02); cb.set_label("Share of R")
    finalize_figure(fig, "3_who_wins")


def fig_two_party(R):
    fig, axes = plt.subplots(1, 3, figsize=(19, 4.8), gridspec_kw=dict(width_ratios=[1, 1, 1.05]))
    for ax, tag, title in [(axes[0], "shared", "Shared feed ($h$=0)"), (axes[1], "local", "Homophilous feed ($h$=0.9)")]:
        h = R[f"tp_hs_{tag}"]
        ax.imshow(h.T, origin="lower", aspect="auto", cmap="Blues", vmin=0, vmax=0.06,
                  extent=[0, len(h), -4, 4], interpolation="nearest")
        ax.set(title=title, xlabel="Time"); boxed(ax)
    axes[0].set_ylabel("Opinion $x$")
    ax = axes[2]; ax.axhline(2, color=C["neutral"], lw=2, ls="--")
    ax.plot(R["tp_h"], R["tp_nf"], color=C["blue_main"], lw=2.5, marker="o", mfc="white", mew=2)
    ax.set(xlabel="Homophily $h$ (0 = one shared feed)", ylabel="Effective no. of factions", ylim=(0.8, 2.6))
    fig.tight_layout(); finalize_figure(fig, "4_two_party")


if __name__ == "__main__":
    apply_publication_style()
    R = dict(np.load(DATA))
    for f in (fig_offset, fig_gap, fig_who_wins, fig_two_party):
        f(R)
    print(f"Figures in {FIGS}")
