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

from abm import MODEL as FULL, n_factions, params, simulate

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


# offset + perception gap
def exp_base():
    outs = pmap([(params(**FULL), sd, False) for sd in range(3)])
    RES["off_x"] = np.concatenate([o["x"] for o in outs])
    RES["off_a"] = np.concatenate([o["a"] for o in outs])
    RES["off_w"] = np.concatenate([o["r"] / o["r"].sum() for o in outs])
    lam = np.concatenate([o["lam"] for o in outs])
    ratio = np.concatenate([o["perceived"] / o["true_out"] for o in outs])
    edges = np.quantile(lam, np.linspace(0, 1, 11)); idx = np.clip(np.digitize(lam, edges) - 1, 0, 9)
    RES["gap_lam"] = np.array([lam[idx == b].mean() for b in range(10)])
    RES["gap_ratio"] = np.array([ratio[idx == b].mean() for b in range(10)])
    RES["gap_ratio_se"] = np.array([ratio[idx == b].std() / np.sqrt((idx == b).sum()) for b in range(10)])
    S = [summary(o) for o in outs]
    print("\n[base] " + ", ".join(f"{k}={np.mean([s_[k] for s_ in S]):.2f}" for k in S[0]))
    print(f"   perception gap, lowest vs highest online-share decile: "
          f"{RES['gap_ratio'][0]:.2f}x vs {RES['gap_ratio'][-1]:.2f}x")


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


# decoupling: change the feed a lot, measure how much people move
def exp_decouple():
    ks = np.linspace(0, 3, 7)
    outs = pmap([(params(**{**FULL, "k": k}, **SMALL), sd, False) for k in ks for sd in range(2)])
    fsd, psd = [], []
    for i in range(len(ks)):
        S = [summary(o) for o in outs[2 * i:2 * i + 2]]
        fsd.append(np.mean([s_["feed_sd"] for s_ in S])); psd.append(np.mean([s_["pop_sd"] for s_ in S]))
    RES["dec_k"], RES["dec_feed_sd"], RES["dec_pop_sd"] = ks, np.array(fsd), np.array(psd)
    print("\n[decouple A] k: feed sd | pop sd")
    for k, a, b in zip(ks, fsd, psd): print(f"   k={k:.1f}: {a:.2f} | {b:.2f}")
    das = np.linspace(-1.2, 1.2, 9); alphas = (0.005, 0.02, 0.08)
    outs = pmap([(params(**FULL, **SMALL, da_R=d, alpha=al), sd, False)
                 for al in alphas for d in das for sd in range(2)])
    RES["dec_da"], RES["dec_alpha"] = das, np.array(alphas)
    print("\n[decouple B] R share of feed | people, by agreeability gap (f_R = 0.5)")
    for c, al in enumerate(alphas):
        fe, po = [], []
        for i in range(len(das)):
            oo = outs[(c * 9 + i) * 2:(c * 9 + i) * 2 + 2]
            fe.append(np.mean([(o["r"] * (o["x"] > 0)).sum() / o["r"].sum() for o in oo]))
            po.append(np.mean([np.mean(o["x"] > 0) for o in oo]))
        RES[f"dec_feed_{c}"], RES[f"dec_pop_{c}"] = np.array(fe), np.array(po)
        print(f"   alpha={al}: feed {fe[0]:.2f}->{fe[-1]:.2f}  people {po[0]:.2f}->{po[-1]:.2f}")


# 10: two-party
def exp_two_party():
    hs = np.linspace(0, 0.9, 7)
    outs = pmap([(params(**FULL, **SMALL, h=h), sd, False) for h in hs for sd in range(2)])
    RES["tp_h"] = hs
    nf = [np.mean([n_factions(o["x"]) for o in outs[2 * i:2 * i + 2]]) for i in range(7)]
    RES["tp_nf"] = np.array(nf)
    print("\n[two-party] effective no. of factions vs homophily h: " + " ".join(f"{v:.2f}" for v in nf))
    for tag, o in zip(("shared", "local"), pmap([(params(**FULL, h=h), 0, True) for h in (0.0, 0.9)])):
        RES[f"tp_hs_{tag}"] = o["hs"]


ALL = dict(base=exp_base, who_wins=exp_who_wins, two_party=exp_two_party, decouple=exp_decouple)

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
