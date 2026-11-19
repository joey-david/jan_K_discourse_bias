# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""Experiments on the agent-based discourse model (abm.py).

Run:  uv run discourse_sims.py              all experiments (< 1 min, parallel)
      uv run discourse_sims.py events two_party   only these, merged into results/abm.npz
Then: uv run make_figures.py
Old fixed-selection version: legacy/.
"""
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from abm import FULL, MODELS, n_factions, params, simulate

OUT = Path(__file__).parent / "results"; OUT.mkdir(exist_ok=True)
FILE = OUT / "abm.npz"
RES = {}
POOL = None
SMALL = dict(n=2000, steps=150)          # for sweeps


def run(P, seed=0, record=False):
    return simulate(P, seed, record)


def pmap(jobs):
    """jobs: list of (P, seed, record) -> list of results, in parallel."""
    return list(POOL.map(run, *zip(*jobs)))


def summary(o):
    x, r = o["x"], o["r"]; fw = r / r.sum(); fm = (fw * x).sum()
    return dict(pop_sd=x.std(), feed_sd=np.sqrt((fw * (x - fm) ** 2).sum()),
                n_fac=n_factions(x), dis_feed=fw[o["a"] < 0].sum(), dis_pop=np.mean(o["a"] < 0),
                top1=np.sort(fw)[-len(x) // 100:].sum(),
                gap=np.mean(o["perceived"] / o["true_out"]))


# 1-3, 9: nested models (who is seen, perception gap, ablation, hostility ratchet)
def exp_nested():
    names = list(MODELS)
    outs = pmap([(params(**MODELS[m]), sd, True) for m in names for sd in range(3)])
    print("\n[nested] mean over 3 seeds")
    keys = ["pop_sd", "feed_sd", "n_fac", "dis_feed", "dis_pop", "top1", "gap"]
    print(f"   {'model':24s} " + " ".join(f"{k:>8s}" for k in keys))
    for i, m in enumerate(names):
        S = [summary(o) for o in outs[3 * i:3 * i + 3]]
        vals = np.array([[s_[k] for k in keys] for s_ in S])
        RES[f"nest_{i}_mean"], RES[f"nest_{i}_sd"] = vals.mean(0), vals.std(0)
        RES[f"nest_{i}_ts"] = outs[3 * i]["ts"]
        print(f"   {m:24s} " + " ".join(f"{v:8.2f}" for v in vals.mean(0)))
    RES["nest_names"], RES["nest_keys"] = np.array(names), np.array(keys)
    # offset + trait exaggeration: M0 vs full, one seed
    for tag, oo in (("m0", outs[:3]), ("full", outs[-3:])):     # pool 3 seeds; r = posting volume
        for v in ("x", "a"):
            RES[f"off_{tag}_{v}"] = np.concatenate([o[v] for o in oo])
        RES[f"off_{tag}_w"] = np.concatenate([o["r"] / o["r"].sum() for o in oo])
    # perception gap vs online share lambda (full model, 3 seeds pooled)
    lam = np.concatenate([o["lam"] for o in outs[-3:]])
    ratio = np.concatenate([o["perceived"] / o["true_out"] for o in outs[-3:]])
    edges = np.quantile(lam, np.linspace(0, 1, 11)); idx = np.clip(np.digitize(lam, edges) - 1, 0, 9)
    RES["gap_lam"] = np.array([lam[idx == b].mean() for b in range(10)])
    RES["gap_ratio"] = np.array([ratio[idx == b].mean() for b in range(10)])
    RES["gap_ratio_se"] = np.array([ratio[idx == b].std() / np.sqrt((idx == b).sum()) for b in range(10)])
    print(f"   perception gap, lowest vs highest online-share decile: "
          f"{RES['gap_ratio'][0]:.2f}x vs {RES['gap_ratio'][-1]:.2f}x")


# 4: committed minority
def exp_minority():
    fs = np.linspace(0, 0.10, 11)
    outs = pmap([(params(**FULL, **SMALL, committed=f), sd, False) for f in fs for sd in range(2)])
    popR, feedR = [], []
    for i in range(len(fs)):
        o2 = outs[2 * i:2 * i + 2]
        popR.append(np.mean([np.mean(o["x"][~o["com"]] > 0) for o in o2]))
        feedR.append(np.mean([(o["r"] * (o["x"] > 0)).sum() / o["r"].sum() for o in o2]))
    RES["min_f"], RES["min_popR"], RES["min_feedR"] = fs, np.array(popR), np.array(feedR)
    print("\n[minority] committed R zealots (taken from L camp) -> share of non-zealots on R | R feed share")
    for f, a, b in list(zip(fs, popR, feedR))[::2]:
        print(f"   {f:4.0%}: {a:.2f} | {b:.2f}")


# 5: who wins: camp size x agreeability gap
def exp_who_wins():
    fs = np.linspace(0.3, 0.7, 5); das = np.linspace(-1.2, 1.2, 5)
    outs = pmap([(params(**FULL, **SMALL, f_right=f, da_R=d), 0, False) for d in das for f in fs])
    pop = np.array([np.mean(o["x"] > 0) for o in outs]).reshape(5, 5)
    feed = np.array([(o["r"] * (o["x"] > 0)).sum() / o["r"].sum() for o in outs]).reshape(5, 5)
    RES["win_f"], RES["win_da"], RES["win_pop"], RES["win_feed"] = fs, das, pop, feed
    print("\n[who wins] rows = agreeability gap a_R - a_L, cols = initial R share; pop|feed share R")
    for i, d in enumerate(das):
        print(f"   {d:+5.1f} | " + " ".join(f"{pop[i, j]:.2f}|{feed[i, j]:.2f}" for j in range(5)))


# 6: cross-group bias (Bail asymmetry)
def exp_cross():
    das = np.linspace(0, -1.6, 9)                 # R increasingly disagreeable
    cfg = [(0.0, 0.0), (0.02, 0.0), (0.02, 0.02)]  # (rho_L, rho_R)
    outs = pmap([(params(**{**FULL, "rho_L": rl, "rho_R": rr}, **SMALL, da_R=d), sd, False)
                 for rl, rr in cfg for d in das for sd in range(2)])
    RES["cross_da"] = das
    print("\n[cross] mean opinion of L camp as R becomes disagreeable")
    for c, (rl, rr) in enumerate(cfg):
        mL = [np.mean([o["x"][o["x0"] < 0].mean() for o in outs[(c * 9 + i) * 2:(c * 9 + i) * 2 + 2]])
              for i in range(9)]
        RES[f"cross_mL_{c}"] = np.array(mL)
        print(f"   rho_L={rl}, rho_R={rr}: " + ", ".join(f"{v:+.2f}" for v in mL[::2]))
    RES["cross_cfg"] = np.array(cfg)


# 7-8: charged events, fixed selection vs full model (hysteresis)
def exp_events():
    T, t0 = 300, 150
    scen = {"none": None, "sym": (t0, 1.5, 25, "both"), "right": (t0, 1.5, 25, "R")}
    seeds = range(3)
    jobs = [(params(**MODELS[m], n=3000, steps=T, shock=sh), sd, True)
            for m in ("M0 fixed selection", "M4 +backlash (full)") for sh in scen.values() for sd in seeds]
    outs = pmap(jobs); RES["ev_t0"] = t0
    outs = [dict(ts=np.mean([o["ts"] for o in outs[i:i + 3]], 0), hs=outs[i]["hs"])
            for i in range(0, len(outs), 3)]          # seed-averaged time series
    print(f"\n[events] t0={t0}: pop sd before -> +20 -> end | disagreeable feed share")
    for i, (m, name) in enumerate([(m, s) for m in ("m0", "full") for s in scen]):
        ts = outs[i]["ts"]; RES[f"ev_{m}_{name}_ts"] = ts; RES[f"ev_{m}_{name}_hs"] = outs[i]["hs"]
        print(f"   {m:4s} {name:5s}: " + " -> ".join(f"{ts[t, 0]:.2f}" for t in (t0 - 1, t0 + 20, T - 1))
              + "  |  " + " -> ".join(f"{ts[t, 4]:.2f}" for t in (t0 - 1, t0 + 20, T - 1)))


# 10: two-party
def exp_two_party():
    hs = np.linspace(0, 0.9, 7)
    cfgs = {"m0": MODELS["M0 fixed selection"], "m2": MODELS["M2 +agreeability"], "full": FULL}
    outs = pmap([(params(**c, **SMALL, h=h), sd, False) for c in cfgs.values() for h in hs for sd in range(2)])
    RES["tp_h"] = hs
    print("\n[two-party] effective no. of factions vs homophily h (0 = one shared feed)")
    for c, tag in enumerate(cfgs):
        nf = [np.mean([n_factions(o["x"]) for o in outs[(c * 7 + i) * 2:(c * 7 + i) * 2 + 2]]) for i in range(7)]
        RES[f"tp_nf_{tag}"] = np.array(nf)
        print(f"   {tag:4s}: " + " ".join(f"{v:.2f}" for v in nf))
    for tag, o in zip(("shared", "local"), pmap([(params(**FULL, h=h), 0, True) for h in (0.0, 0.9)])):
        RES[f"tp_hs_{tag}"] = o["hs"]


ALL = dict(nested=exp_nested, minority=exp_minority, who_wins=exp_who_wins,
           cross=exp_cross, events=exp_events, two_party=exp_two_party)

if __name__ == "__main__":
    chosen = sys.argv[1:] or list(ALL)
    if sys.argv[1:] and FILE.exists():
        RES.update(np.load(FILE))
    t = time.time()
    with ProcessPoolExecutor() as POOL:
        for name in chosen:
            ALL[name]()
    np.savez_compressed(FILE, **RES)
    print(f"\nResults in {FILE}  ({time.time() - t:.0f}s)")
