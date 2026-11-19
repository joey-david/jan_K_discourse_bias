"""Agent-based model of selection in online discourse (engine only).

Per agent i: opinion x_i (camp = sign x), offline opinion x0_i, agreeability a_i,
activity act_i (lognormal), online share of attention lam_i.

Posting rate (one rule, nested terms):
    log r_i = k|x_i|  -  a_i * |x_i - norm_i|        (+ floor)
      k|x|          : intensity selection (model M0)
      -a_i * dist   : a_i > 0 quiet when far from perceived norm (spiral of silence),
                      a_i < 0 louder when far (disagreeable); mean(a) > 0 = spiral
Feed: each agent sees `sample` posts, a fraction h from opinion-neighbours (homophily),
the rest from the global feed; both volume-weighted by r.
Perceived norm_i = lam_i * mean(seen posts) + (1 - lam_i) * mean(offline contacts).
Update: dx = mu*(nearby seen - x) + alpha*(x0 - x)
"""
import numpy as np

BASE = dict(n=4000, steps=200, sample=20, k=1.5, a_mu=0.0, a_sd=0.0, mu=0.1, eps=0.6, alpha=0.02, h=0.5, lam_mu=0.6,
            floor=0.02, f_right=0.5, da_R=0.0, dk_R=0.0, committed=0.0, xmax=4.0,
            shock=None)   # shock = (t0, amplitude, tau, "both"|"R"): charged event boosting k

MODEL = dict(a_mu=0.5, a_sd=0.8)   # agreeability a_i ~ N(a_mu, a_sd); a_i < 0: disagreeable


def params(**kw):
    P = dict(BASE); P.update(kw); return P


def camps(rng, n, f_right):
    z = rng.normal(0, 1, n)
    return z - np.quantile(z, 1 - f_right)


def simulate(P, seed=0, record=False):
    """Run one simulation; record=True stores time series + opinion histograms."""
    rng = np.random.default_rng(seed)
    n, S = P["n"], P["sample"]
    x0 = camps(rng, n, P["f_right"]); x = x0.copy()
    R0 = x0 > 0
    a = rng.normal(P["a_mu"], P["a_sd"], n) + P["da_R"] * R0
    act = rng.lognormal(0, 1, n)
    lam = rng.beta(4 * P["lam_mu"], 4 * (1 - P["lam_mu"]), n)
    kc = P["k"] + P["dk_R"] * R0
    com = np.zeros(n, bool)
    if P["committed"] > 0:                       # committed R zealots: fixed, loud, disagreeable
        com[rng.choice(np.flatnonzero(~R0), int(P["committed"] * n), replace=False)] = True
        x[com] = x0[com] = 2.5; a[com] = -1.0; act[com] = np.maximum(act[com], 1.0)
    norm = np.full(n, x.mean())
    nl = int(round(P["h"] * S)); ng = S - nl; w = max(1, int(0.05 * n))
    ts, hs = [], []
    sh = P["shock"]
    for t in range(P["steps"]):
        ke = kc
        if sh and t >= sh[0]:
            ke = kc + sh[1] * np.exp(-(t - sh[0]) / sh[2]) * (R0 if sh[3] == "R" else 1)
        logr = ke * np.abs(x) - a * np.abs(x - norm)
        r = act * np.exp(np.clip(logr, -20, 20)) + P["floor"]
        p = r / r.sum()
        seen = rng.choice(n, (n, ng), p=p) if ng else np.empty((n, 0), int)
        if nl:                                    # local: rank-neighbours, volume-weighted (pick 1 of 4)
            order = np.argsort(x); rank = np.empty(n, int); rank[order] = np.arange(n)
            cand = order[np.clip(rank[:, None, None] + rng.integers(-w, w + 1, (n, nl, 4)), 0, n - 1)]
            g = np.log(r[cand]) + rng.gumbel(size=cand.shape)
            seen = np.concatenate([seen, np.take_along_axis(cand, g.argmax(2)[..., None], 2)[..., 0]], 1)
        xs = x[seen]
        offline = x[rng.integers(0, n, (n, 5))].mean(1)
        norm = lam * xs.mean(1) + (1 - lam) * offline
        same = np.sign(xs) == np.sign(x)[:, None]
        # opinion update
        near = np.abs(xs - x[:, None]) < P["eps"]
        att = np.where(near.any(1), (xs * near).sum(1) / np.maximum(near.sum(1), 1) - x, 0)
        x = np.clip(x + P["mu"] * att + P["alpha"] * (x0 - x), -P["xmax"], P["xmax"])
        x[com] = x0[com]
        if record:
            fw = r / r.sum(); dis = a < 0
            fm = (fw * x).sum()
            ts.append((x.std(), np.sqrt((fw * (x - fm) ** 2).sum()), fm, fw[x > 0].sum(),
                       fw[dis].sum(), dis.mean(), np.mean(x > 0)))
            hs.append(np.histogram(x, bins=80, range=(-4, 4))[0] / n)
    # final perception of the out-group (|mean opinion|), online + offline mix
    out_seen = np.where(~same, np.abs(xs), np.nan)
    on = np.nanmean(np.where((~same).any(1)[:, None], out_seen, np.abs(x).mean()), 1)
    true_out = np.where(x > 0, np.abs(x[x < 0]).mean(), np.abs(x[x > 0]).mean())
    perceived = lam * on + (1 - lam) * true_out
    return dict(x=x, x0=x0, a=a, act=act, lam=lam, r=r, com=com,
                perceived=perceived, true_out=true_out, ts=np.array(ts), hs=np.array(hs))


def n_factions(x, lo=-4, hi=4, bins=80, tol=0.002):
    """Effective number of factions: contiguous occupied histogram runs, inverse Simpson."""
    hgram, _ = np.histogram(x, bins=bins, range=(lo, hi)); q = hgram / hgram.sum()
    masses, cur = [], 0.0
    for v in q:
        if v > tol: cur += v
        elif cur: masses.append(cur); cur = 0.0
    if cur: masses.append(cur)
    m = np.array(masses); m /= m.sum()
    return 1 / np.sum(m ** 2)
