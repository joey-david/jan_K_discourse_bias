# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib"]
# ///
"""Build publication figures from results/sims.npz (made by discourse_sims.py).

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
DATA = HERE / "results" / "sims.npz"
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
def fig_offset(R):
    bins = np.linspace(-5, 5, 101)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    pop = R["offset_population"]
    ax.hist(pop, bins=bins, density=True, color=PALETTE["neutral"], label="Population")
    for k, c in [(0, PALETTE["green_3"]), (1, PALETTE["blue_secondary"]), (2, PALETTE["red_strong"])]:
        hist_line(ax, R[f"offset_feed_k{k}"], bins, color=c, lw=2.5, label=f"Feed, $k$={k}")
    ax.set(xlabel="Opinion $x$", ylabel="Density", xlim=(-5, 5))
    ax.legend(loc="upper right")
    finalize_figure(fig, "1_offset")


def fig_diabolisation(R):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.axhline(1, color=PALETTE["neutral"], lw=2, ls="--")
    ax.plot(R["diab_k"], R["diab_exaggeration"], "-", color=PALETTE["red_strong"], lw=2.5)
    ax.set(xlabel="Selection strength $k$", ylabel="Seen / true out-group mean")
    finalize_figure(fig, "2_diabolisation")


def fig_bubble(R):
    bins = np.linspace(-4, 4, 61)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    for ax, (k, a) in zip(axes.flat, [(0, 0.0), (2, 0.0), (0, 0.05), (2, 0.05)]):
        t = f"k{k}_a{a:.2f}"
        ax.hist(R[f"bubble_{t}_x0"], bins=bins, density=True, color=PALETTE["neutral"],
                label="Initial")
        ax.hist(R[f"bubble_{t}_x"], bins=bins, density=True, histtype="step", lw=2.5,
                color=PALETTE["blue_main"] if k == 0 else PALETTE["red_strong"], label="Final")
        ax.set_title(f"$k$={k}, anchor={a}")
        ax.text(0.97, 0.95, f"BC={float(R[f'bubble_{t}_bc']):.2f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=13)
    for ax in axes[1]: ax.set_xlabel("Opinion $x$")
    for ax in axes[:, 0]: ax.set_ylabel("Density")
    axes[0, 0].legend(loc="upper left")
    fig.tight_layout()
    finalize_figure(fig, "3_bubble")


def fig_minority(R):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(R["minority_p"] * 100, R["minority_adopt"] * 100, "-", color=PALETTE["blue_main"], lw=2.5)
    for v, c, ls, lab in [(R["minority_offline"], PALETTE["teal"], ":", "Offline committed share"),
                          (R["minority_online"], PALETTE["red_strong"], "--", "Online post share ($k$=2)")]:
        ax.axvline(float(v) * 100, color=c, ls=ls, lw=2.5, label=lab)
    ax.set(xlabel="Committed minority (%)", ylabel="Final adoption (%)", xlim=(0, 35))
    ax.legend(loc="center right", bbox_to_anchor=(0.9, 0.45))
    finalize_figure(fig, "4_minority")


def fig_control(R):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    m = R["control_means"]
    cols = [PALETTE["neutral"], PALETTE["red_2"], PALETTE["red_strong"]]
    bars = ax.bar(range(len(m)), m, color=cols, edgecolor="black", lw=1.5, width=0.6)
    for b, v in zip(bars, m):
        ax.text(b.get_x() + b.get_width() / 2, max(v, 0) + 0.04, f"{v:+.2f}", ha="center", fontsize=13)
    ax.set_xticks(range(len(m)), R["control_labels"], rotation=15)
    ax.axhline(0, color="black", lw=1)
    ax.set_ylabel("Feed mean (pop. mean = 0)")
    finalize_figure(fig, "5_control")


def fig_who_wins(R):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    f, dk = R["win_f"], R["win_dk"]
    for ax, key, title in [(axes[0], "win_pop", "Population: share of R"),
                           (axes[1], "win_feed", "Feed: share of R posts")]:
        im = ax.imshow(R[key], origin="lower", cmap="RdBu_r", vmin=0, vmax=1, aspect="auto",
                       extent=[f[0], f[-1], dk[0], dk[-1]])
        cs = ax.contour(f, dk, R[key], levels=[0.5], colors="black", linewidths=2)
        ax.set(title=title, xlabel="Initial share of R in population")
        for sp in ax.spines.values(): sp.set_visible(True)
    axes[0].set_ylabel("Loudness gap $k_R - k_L$")
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02); cb.set_label("Share of R")
    finalize_figure(fig, "6_who_wins")


def fig_cross_bias(R):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    kR = R["cross_kR"]
    for rho, c in [(0.0, PALETTE["neutral"]), (0.01, PALETTE["blue_secondary"]), (0.02, PALETTE["blue_main"])]:
        axes[0].plot(kR, R[f"cross_mL_rho{rho}"], "-", color=c, lw=2.5, label=f"backfire $\\rho$={rho}")
        axes[1].plot(kR, R[f"cross_seenR_rho{rho}"], "-", color=c, lw=2.5)
    axes[0].set(xlabel="R loudness $k_R$ ($k_L$=1)", ylabel="Mean opinion of L camp")
    axes[1].set(xlabel="R loudness $k_R$ ($k_L$=1)", ylabel="R as seen in feed (mean)")
    axes[0].legend(loc="lower left")
    fig.tight_layout()
    finalize_figure(fig, "7_cross_bias")


def fig_events(R):
    t0 = int(R["event_t0"]); names = [("none", "No event", PALETTE["neutral"]),
                                      ("sym", "Symmetric event", PALETTE["violet"]),
                                      ("right", "One-sided event (R)", PALETTE["red_strong"])]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for key, lab, c in names:
        ts = R[f"event_{key}_ts"]; t = np.arange(len(ts)) - t0
        axes[0].plot(t, ts[:, 3], color=c, lw=2.5, label=lab)
        axes[0].plot(t, ts[:, 1], color=c, lw=2, ls=":")
        axes[1].plot(t, ts[:, 2], color=c, lw=2.5)
        axes[2].plot(t, ts[:, 5], color=c, lw=2.5)
    for ax in axes: ax.axvline(0, color="black", lw=1, ls="--"); ax.set_xlabel("Time since event")
    axes[0].set_ylabel("Spread (sd)  — feed, ··· pop.")
    axes[1].set_ylabel("Feed mean opinion"); axes[2].set_ylabel("Share of feed from R")
    axes[0].legend(loc="upper left", fontsize=12)
    fig.tight_layout()
    finalize_figure(fig, "8_events")

    bins = np.linspace(-5, 5, 81)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    for ax, (ph, title) in zip(axes, [("pre", "Before"), ("peak", "Peak (+30)"), ("post", "End")]):
        ax.hist(R[f"event_none_{ph}_pop"], bins=bins, density=True, color=PALETTE["neutral"], label="Population (no event)")
        for key, lab, c in names[1:]:
            hist_line(ax, R[f"event_{key}_{ph}_feed"], bins, color=c, lw=2.5, label=f"Feed: {lab.lower()}")
        ax.set(title=title, xlabel="Opinion $x$")
    axes[0].set_ylabel("Density"); axes[0].legend(loc="upper left", fontsize=11)
    fig.tight_layout()
    finalize_figure(fig, "9_event_snapshots")


def fig_two_party(R):
    pos = R["tp_pos"]
    fig, axes = plt.subplots(1, 4, figsize=(21, 4.8), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.25]))
    titles = {"consensus": "No selection: centre wins", "fragmented": "Selection, short reach",
              "twoparty": "Selection, shared feed"}
    for ax, name in zip(axes[:3], titles):
        h = R[f"tp_hist_{name}"]; k, r, n = R[f"tp_meta_{name}"]
        ax.imshow(h.T, origin="lower", aspect="auto", cmap="Blues", vmin=0, vmax=0.25,
                  extent=[0, len(h) - 1, pos[0] - 0.05, pos[-1] + 0.05], interpolation="nearest")
        ax.set(title=titles[name], xlabel="Time")
        ax.text(0.97, 0.04, f"$k$={k:.0f}, reach={r:.1f}\n$N_{{eff}}$={n:.1f}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=12, color="black")
        for sp in ax.spines.values(): sp.set_visible(True)
    axes[0].set_ylabel("Position")
    ax = axes[3]; k, r = R["tp_k"], R["tp_reach"]
    im = ax.imshow(R["tp_neff"], origin="lower", aspect="auto", cmap="magma_r", vmin=1, vmax=5,
                   extent=[k[0], k[-1], r[0], r[-1]])
    ax.set(xlabel="Selection strength $k$", ylabel="Reach of feed", title="Effective no. of factions")
    for sp in ax.spines.values(): sp.set_visible(True)
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    fig.tight_layout()
    finalize_figure(fig, "10_two_party")


if __name__ == "__main__":
    apply_publication_style()
    R = dict(np.load(DATA))
    for f in (fig_offset, fig_diabolisation, fig_bubble, fig_minority, fig_control,
              fig_who_wins, fig_cross_bias, fig_events, fig_two_party):
        f(R)
    print(f"Figures in {FIGS}")
