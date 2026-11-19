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
MODEL_LABELS = ["M0\nfixed", "M1\n+spiral", "M2\n+agree.", "M3\n+likes", "M4\n+backlash"]
MODEL_COLORS = [C["neutral"], C["green_3"], C["teal"], C["blue_secondary"], C["blue_main"]]


def movavg(y, w=9):
    k = np.ones(w) / w; yp = np.pad(y, (w // 2, w // 2), mode="edge")
    return np.convolve(yp, k, mode="valid")


def boxed(ax):
    for sp in ax.spines.values(): sp.set_visible(True)


def smooth_hist(ax, x, w, bins, sigma=1.5, **kw):
    """Weighted histogram smoothed with a small Gaussian kernel (in bins)."""
    d, e = np.histogram(x, bins=bins, weights=w, density=True)
    k = np.exp(-0.5 * (np.arange(-4, 5) / sigma) ** 2); k /= k.sum()
    ax.plot((e[:-1] + e[1:]) / 2, np.convolve(d, k, mode="same"), **kw)


def fig_offset(R):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    bins = np.linspace(-4, 4, 121); ax = axes[0]
    ax.hist(R["off_full_x"], bins=bins, density=True, color=C["neutral"], label="Population (full model)")
    smooth_hist(ax, R["off_m0_x"], R["off_m0_w"], bins, color=C["green_3"], lw=2.5, label="Feed, M0 (fixed)")
    smooth_hist(ax, R["off_full_x"], R["off_full_w"], bins, color=C["red_strong"], lw=2.5, label="Feed, full model")
    ax.set(xlabel="Opinion $x$", ylabel="Density"); ax.legend(loc="upper left", fontsize=12)
    ax = axes[1]; bins = np.linspace(-3, 3, 61)
    ax.hist(R["off_full_a"], bins=bins, density=True, color=C["neutral"], label="Population")
    smooth_hist(ax, R["off_full_a"], R["off_full_w"], bins, color=C["red_strong"], lw=2.5, label="Feed (who is heard)")
    ax.axvline(0, color="black", lw=1, ls="--")
    ax.set(xlabel="Agreeability $a$  (< 0: disagreeable)", ylabel="Density")
    ax.legend(loc="upper left", fontsize=12)
    fig.tight_layout(); finalize_figure(fig, "1_offset")


def fig_gap(R):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.axhline(1, color=C["neutral"], lw=2, ls="--")
    ax.errorbar(R["gap_lam"], R["gap_ratio"], yerr=1.96 * R["gap_ratio_se"], color=C["red_strong"],
                lw=2.5, marker="o", ms=7, mfc="white", mew=2, capsize=3)
    ax.set(xlabel="Online share of attention $\\lambda$ (decile mean)",
           ylabel="Perceived / true\nout-group extremity")
    fig.tight_layout(); finalize_figure(fig, "2_perception_gap")


def fig_nested(R):
    keys = list(R["nest_keys"]); show = [("feed_sd", "Feed spread (sd)"), ("dis_feed", "Disagreeable share of feed"),
                                         ("top1", "Top 1% share of posts"), ("n_fac", "Effective no. of factions")]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    for ax, (k, lab) in zip(axes, show):
        j = keys.index(k); m = [R[f"nest_{i}_mean"][j] for i in range(5)]; e = [R[f"nest_{i}_sd"][j] for i in range(5)]
        ax.bar(range(5), m, yerr=e, color=MODEL_COLORS, edgecolor="black", lw=1.2, capsize=3, width=0.7)
        ax.set_xticks(range(5), MODEL_LABELS, fontsize=11); ax.set_title(lab)
        ref = {"dis_feed": "dis_pop", "feed_sd": "pop_sd"}
        if k in ref:
            y = [R[f"nest_{i}_mean"][keys.index(ref[k])] for i in range(5)]
            if k == "dis_feed": y[0] = y[1] = np.nan
            ax.scatter(range(5), y, marker="_", s=900, lw=3, color="black", zorder=5, label="Population")
            ax.legend(loc="upper left", fontsize=11)
        if k == "dis_feed":
            for i in (0, 1): ax.text(i, 0.01, "no trait", ha="center", va="bottom", fontsize=10, rotation=90)
    fig.tight_layout(); finalize_figure(fig, "3_nested_models")


def fig_minority(R):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    f = R["min_f"] * 100
    ax.plot(f, R["min_feedR"] * 100, color=C["red_strong"], lw=2.5, marker="o", mfc="white", mew=2, label="R share of feed")
    ax.plot(f, R["min_popR"] * 100, color=C["blue_main"], lw=2.5, marker="s", mfc="white", mew=2, label="R share of people")
    ax.axhline(50, color=C["neutral"], lw=2, ls="--")
    ax.set(xlabel="Committed R zealots (% of population)", ylabel="Share (%)", ylim=(40, 90))
    ax.legend(loc="upper left")
    finalize_figure(fig, "4_minority")


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
    axes[0].set_ylabel("Agreeability gap $a_R - a_L$\n(< 0: R more disagreeable)")
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02); cb.set_label("Share of R")
    finalize_figure(fig, "5_who_wins")


def fig_cross(R):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labs = ["No backlash", "Backlash in L only", "Backlash in both"]
    cols = [C["neutral"], C["blue_secondary"], C["blue_main"]]
    for c in range(3):
        ax.plot(-R["cross_da"], R[f"cross_mL_{c}"], color=cols[c], lw=2.5, marker="o", mfc="white", mew=2, label=labs[c])
    ax.set(xlabel="How much less agreeable R is ($a_L - a_R$)", ylabel="Mean opinion of L camp")
    ax.legend(loc="lower left", fontsize=12)
    finalize_figure(fig, "6_cross_bias")


def fig_ratchet(R):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i in (2, 3, 4):
        y = movavg(R[f"nest_{i}_ts"][1:, 4] * 100)
        ax.plot(np.arange(1, len(y) + 1), y, color=MODEL_COLORS[i], lw=2.5, label=MODEL_LABELS[i].replace("\n", " "))
    ax.axhline(R["nest_2_ts"][0, 5] * 100, color="black", lw=1.5, ls="--")
    ax.text(len(R["nest_2_ts"]) - 1, R["nest_2_ts"][0, 5] * 100 + 1, "share of population", ha="right", fontsize=12)
    ax.set(xlabel="Time", ylabel="Disagreeable share of feed (%)", ylim=(0, None))
    ax.legend(loc="lower right", fontsize=12)
    finalize_figure(fig, "7_hostility_ratchet")


def fig_events(R):
    t0 = int(R["ev_t0"]); sc = [("none", "No event", C["neutral"]), ("sym", "Symmetric event", C["violet"]),
                               ("right", "One-sided event (R)", C["red_strong"])]
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    cols = [(1, "Feed spread (sd)"), (3, "R share of feed"), (4, "Disagreeable share of feed")]
    lo = -60
    for ax, (j, yl) in zip(axes, cols):
        if j != 4:
            ts = R["ev_m0_sym_ts"]; t = np.arange(len(ts)) - t0; keep = t >= lo
            ax.plot(t[keep], movavg(ts[:, j], 5)[keep], color="black", lw=1.8, ls=":", label="M0, symmetric event")
        for key, lab, c in sc[::-1]:
            ts = R[f"ev_full_{key}_ts"]; t = np.arange(len(ts)) - t0; keep = t >= lo
            ax.plot(t[keep], movavg(ts[:, j], 5)[keep], color=c, lw=2.5, label=lab)
        ax.axvline(0, color="black", lw=1, ls="--"); ax.set(ylabel=yl, xlabel="Time since event")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h[::-1], l[::-1], loc="upper center", ncol=4, fontsize=13, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout(); finalize_figure(fig, "8_events")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, m, title in [(axes[0], "m0", "M0 fixed selection"), (axes[1], "full", "Full model")]:
        h = R[f"ev_{m}_sym_hs"]
        im = ax.imshow(h.T, origin="lower", aspect="auto", cmap="Blues", vmin=0, vmax=0.06,
                       extent=[-t0, len(h) - t0, -4, 4], interpolation="nearest")
        ax.axvline(0, color=C["red_strong"], lw=2, ls="--")
        ax.set(title=f"{title}: symmetric event", xlabel="Time since event"); boxed(ax)
    axes[0].set_ylabel("Opinion $x$")
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02); cb.set_label("Population share")
    finalize_figure(fig, "9_event_density")


def fig_two_party(R):
    fig, axes = plt.subplots(1, 3, figsize=(19, 4.8), gridspec_kw=dict(width_ratios=[1, 1, 1.05]))
    for ax, tag, title in [(axes[0], "shared", "Shared feed ($h$=0)"), (axes[1], "local", "Homophilous feed ($h$=0.9)")]:
        h = R[f"tp_hs_{tag}"]
        ax.imshow(h.T, origin="lower", aspect="auto", cmap="Blues", vmin=0, vmax=0.06,
                  extent=[0, len(h), -4, 4], interpolation="nearest")
        ax.set(title=f"Full model, {title}", xlabel="Time"); boxed(ax)
    axes[0].set_ylabel("Opinion $x$")
    ax = axes[2]; ax.axhline(2, color=C["neutral"], lw=2, ls="--")
    for tag, lab, c in [("m0", "M0 fixed", C["green_3"]), ("m2", "M2 +agreeability", C["teal"]), ("full", "M4 full", C["blue_main"])]:
        ax.plot(R["tp_h"], R[f"tp_nf_{tag}"], color=c, lw=2.5, marker="o", mfc="white", mew=2, label=lab)
    ax.set(xlabel="Homophily $h$ (0 = one shared feed)", ylabel="Effective no. of factions", ylim=(0.8, 2.6))
    ax.legend(loc="lower left", fontsize=12)
    fig.tight_layout(); finalize_figure(fig, "10_two_party")


if __name__ == "__main__":
    apply_publication_style()
    R = dict(np.load(DATA))
    for f in (fig_offset, fig_gap, fig_nested, fig_minority, fig_who_wins, fig_cross,
              fig_ratchet, fig_events, fig_two_party):
        f(R)
    print(f"Figures in {FIGS}")
