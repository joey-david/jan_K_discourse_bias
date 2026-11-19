# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""Participation-selection bias in public discourse: four toy models.

Run:  uv run discourse_sims.py   (~10 s, writes results/sims.npz)
Then: uv run make_figures.py     (reads results/, writes figures/)

Core assumption (the "tautology" made explicit):
  each person has a latent opinion x ~ N(0,1) on an issue, and a propensity
  to post that grows with intensity |x|:   rate(x) = exp(k * |x|).
  k = 0 -> participation independent of opinion (no bias).
Activity is also heavy-tailed across people (lognormal), as on real platforms.
"""
import numpy as np
from pathlib import Path

rng = np.random.default_rng(0)
OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)
RES = {}  # everything the figure script needs
N = 1_000_000


def post_weights(x, k, activity_sigma=1.0):
    """Expected posts per person: intensity selection x heavy-tailed activity."""
    act = rng.lognormal(0, activity_sigma, size=x.shape)
    return np.exp(k * np.abs(x)) * act


# ---------------------------------------------------------------------------
# 1. Offset between real life and online: what the feed looks like vs population
# ---------------------------------------------------------------------------
def sim_offset():
    x = rng.normal(0, 1, N)
    RES["offset_population"] = x
    print("\n[1] Offset: population vs posts-weighted ('the feed')")
    print(f"{'k':>4} {'sd(feed)':>9} {'P(|x|>2) feed':>14} {'top1% share':>12}")
    for k in [0, 1, 2]:
        w = post_weights(x, k)
        feed = rng.choice(x, size=N, p=w / w.sum())
        top1 = np.sort(w)[-N // 100:].sum() / w.sum()
        print(f"{k:>4} {feed.std():9.2f} {np.mean(np.abs(feed) > 2):14.1%} {top1:12.1%}")
        RES[f"offset_feed_k{k}"] = feed
    print(f"     population P(|x|>2) = {np.mean(np.abs(x) > 2):.1%}")


# ---------------------------------------------------------------------------
# 2. Diabolisation: perceived out-group vs actual out-group
# ---------------------------------------------------------------------------
def sim_diabolisation():
    x = rng.normal(0, 1, N)
    print("\n[2] Diabolisation: how extreme the 'other side' looks")
    print(f"{'k':>4} {'true mean R':>11} {'seen mean R':>12} {'exaggeration':>13}")
    ks = np.linspace(0, 3, 121); ex = []
    for k in ks:
        w = post_weights(x, k)
        R = x > 0
        true_m = x[R].mean()
        seen_m = np.average(x[R], weights=w[R])
        ex.append(seen_m / true_m)
        if np.isclose(k, round(k)):
            print(f"{k:>4.0f} {true_m:11.2f} {seen_m:12.2f} {seen_m/true_m:12.2f}x")
    RES["diab_k"], RES["diab_exaggeration"] = ks, np.array(ex)


# ---------------------------------------------------------------------------
# 3. Feedback bubble: opinion dynamics with/without selection & real-world anchor
#    Each step, agents see a sample of posts (weighted by current intensity),
#    move toward posts within confidence bound eps (homophily), and are pulled
#    back toward their offline anchor with strength a (real-world feedback).
# ---------------------------------------------------------------------------
def run_bubble(k, anchor, n=20_000, steps=300, eps=0.8, mu=0.1, sample=20):
    x0 = rng.normal(0, 1, n); x = x0.copy()
    for _ in range(steps):
        w = np.exp(k * np.abs(x)); p = w / w.sum()
        seen = x[rng.choice(n, size=(n, sample), p=p)]
        near = np.abs(seen - x[:, None]) < eps
        target = np.where(near.any(1), (seen * near).sum(1) / np.maximum(near.sum(1), 1), x)
        x = x + mu * (target - x) + anchor * (x0 - x)
        x = np.clip(x, -5, 5)
    return x0, x


def bimodality(x):  # Sarle's coefficient; > 0.555 suggests bimodal
    m = x - x.mean(); s = x.std()
    g = (m**3).mean() / s**3; kurt = (m**4).mean() / s**4 - 3
    n = len(x)
    return (g**2 + 1) / (kurt + 3 * (n - 1)**2 / ((n - 2) * (n - 3)))


def sim_bubble():
    print("\n[3] Feedback bubble (bounded-confidence + intensity-weighted feed)")
    print(f"{'k':>4} {'anchor':>7} {'sd':>6} {'bimodality':>11}")
    for k, a in [(0, .0), (2, .0), (0, .05), (2, .05)]:
        x0, x = run_bubble(k, a)
        b = bimodality(x)
        print(f"{k:>4} {a:7.2f} {x.std():6.2f} {b:11.2f}")
        tag = f"k{k}_a{a:.2f}"
        RES[f"bubble_{tag}_x0"], RES[f"bubble_{tag}_x"], RES[f"bubble_{tag}_bc"] = x0, x, b


# ---------------------------------------------------------------------------
# 4. Minority rule: committed minority in a binary naming game (Xie et al. 2011)
#    Online, the committed fraction is inflated by selection -> crosses the
#    ~10% tipping point online while offline it does not.
# ---------------------------------------------------------------------------
def naming_game(p_committed, n=1000, steps=60_000):
    # states: 0 = A, 1 = B, 2 = {A,B}; committed hold A forever; others start B
    s = np.ones(n, dtype=int); c = np.zeros(n, bool)
    c[: int(p_committed * n)] = True; s[c] = 0
    for _ in range(steps):
        i, j = rng.integers(n, size=2)
        word = s[i] if s[i] != 2 else rng.integers(2)
        if s[j] == 2 or s[j] == word:   # listener knows word -> both agree
            if not c[j]: s[j] = word
            if not c[i]: s[i] = word
        else:                           # listener learns it
            if not c[j]: s[j] = 2
    return np.mean(s == 0)


def sim_minority():
    print("\n[4] Minority rule: committed minority vs final adoption of its view")
    ps = np.linspace(0, 0.2, 101)
    adopt = [np.mean([naming_game(p) for _ in range(3)]) for p in ps]
    for p, a in zip(ps, adopt):
        print(f"   committed {p:5.0%} -> adoption {a:5.0%}")
    # Selection inflation: committed = top |x| tail, k=2
    x = rng.normal(0, 1, N); w = post_weights(x, 2)
    tail = x > 1.75
    print(f"   offline share of strongly committed (x>1.75): {tail.mean():.1%}")
    print(f"   online share of posts from them (k=2):        {w[tail].sum()/w.sum():.1%}")
    RES["minority_p"], RES["minority_adopt"] = ps, np.array(adopt)
    RES["minority_offline"], RES["minority_online"] = tail.mean(), w[tail].sum() / w.sum()


# ---------------------------------------------------------------------------
# 5. Control: what selection does NOT do
#    Symmetric selection on |x| leaves the mean unchanged -> it can't explain
#    a directional shift in the feed unless participation is asymmetric
#    (one side more motivated). Shows mean of feed under symmetric vs asymmetric k.
# ---------------------------------------------------------------------------
def sim_control():
    x = rng.normal(0, 1, N)
    RES["control_labels"] = np.array(["k_L=2, k_R=2", "k_L=1, k_R=2", "k_L=0, k_R=2"])
    means = []
    print("\n[5] Control: mean of feed (population mean = 0)")
    for kL, kR in [(2, 2), (1, 2), (0, 2)]:
        w = np.exp(np.where(x < 0, kL, kR) * np.abs(x))
        means.append(np.average(x, weights=w))
        print(f"   k_left={kL}, k_right={kR}: feed mean = {means[-1]:+.2f}")
    RES["control_means"] = np.array(means)


# ---------------------------------------------------------------------------
# Two-camp dynamics shared by experiments 6-7.
#   camp = sign(x); each camp posts at rate exp(k_camp * |x|)
#   attraction : move toward seen posts within eps (any camp -> moderates can be won over)
#   backfire   : pushed away from the out-group in proportion to how much, and how
#                extreme, out-group content you see (rho = reactivity)
#   anchor     : pull back toward offline opinion x0 (real-world feedback)
# ---------------------------------------------------------------------------
def two_camp(x0, kfun, steps, eps=0.8, mu=0.1, rho=0.02, anchor=0.02, sample=20, record=False):
    n = len(x0); x = x0.copy(); rec = []
    for t in range(steps):
        kL, kR = kfun(t)
        w = np.exp(np.where(x < 0, kL, kR) * np.abs(x)); p = w / w.sum()
        if record:
            rec.append((x.mean(), x.std(), np.average(x, weights=w),
                        np.sqrt(np.average((x - np.average(x, weights=w))**2, weights=w)),
                        np.mean(x > 0), w[x > 0].sum() / w.sum()))
            if t in record: RES[f"{record[t]}_pop"] = x.copy(); RES[f"{record[t]}_feed"] = rng.choice(x, n, p=p)
        seen = x[rng.choice(n, size=(n, sample), p=p)]
        near = np.abs(seen - x[:, None]) < eps
        att = np.where(near.any(1), (seen * near).sum(1) / np.maximum(near.sum(1), 1) - x, 0)
        out = np.sign(seen) != np.sign(x)[:, None]
        back = np.sign(x) * (np.abs(seen) * out).sum(1) / sample
        x = np.clip(x + mu * att + rho * back + anchor * (x0 - x), -5, 5)
    return x, np.array(rec)


def camps(n, f_right):
    z = rng.normal(0, 1, n)
    return z - np.quantile(z, 1 - f_right)   # exactly f_right of people start with x > 0


# ---------------------------------------------------------------------------
# 6. Who wins: group size vs loudness; and how one camp's loudness moves the other
# ---------------------------------------------------------------------------
def sim_who_wins():
    print("\n[6a] Who wins: final share of R (population | feed); k_L + k_R = 3")
    fs = np.linspace(0.3, 0.7, 9); dks = np.linspace(-1.5, 1.5, 9)
    pop = np.zeros((len(dks), len(fs))); feed = np.zeros_like(pop)
    for i, dk in enumerate(dks):
        for j, f in enumerate(fs):
            kL, kR = 1.5 - dk / 2, 1.5 + dk / 2
            x, _ = two_camp(camps(4000, f), lambda t: (kL, kR), 150)
            w = np.exp(np.where(x < 0, kL, kR) * np.abs(x))
            pop[i, j], feed[i, j] = np.mean(x > 0), w[x > 0].sum() / w.sum()
    RES["win_f"], RES["win_dk"], RES["win_pop"], RES["win_feed"] = fs, dks, pop, feed
    print(f"   {'k_R-k_L':>8} | " + " ".join(f"f={f:.2f}" for f in fs[::2]))
    for i in range(0, len(dks), 2):
        print(f"   {dks[i]:+8.2f} | " + " ".join(f"{pop[i,j]:.2f}|{feed[i,j]:.2f}" for j in range(0, len(fs), 2)))

    print("\n[6b] Cross-group bias: L camp's mean opinion as R gets louder (k_L = 1)")
    kRs = np.linspace(0, 3, 13); RES["cross_kR"] = kRs
    for rho in (0.0, 0.01, 0.02):
        mL, seenR = [], []
        for kR in kRs:
            x0 = camps(10_000, 0.5)
            x, _ = two_camp(x0, lambda t: (1.0, kR), 150, rho=rho)
            w = np.exp(np.where(x < 0, 1.0, kR) * np.abs(x))
            mL.append(x[x0 < 0].mean())                      # people who started on L
            seenR.append(np.average(x[x > 0], weights=w[x > 0]))  # R as L sees it
        RES[f"cross_mL_rho{rho}"], RES[f"cross_seenR_rho{rho}"] = np.array(mL), np.array(seenR)
        print(f"   rho={rho:.2f}: mean(L) at k_R=0,1,2,3 -> " +
              ", ".join(f"{mL[i]:+.2f}" for i in (0, 4, 8, 12)))


# ---------------------------------------------------------------------------
# 7. Politically charged event at t0 (after burn-in): salience spike k(t) = k0 + A exp(-(t-t0)/tau)
#    Scenarios: none, symmetric (both camps fired up), one-sided (only R fired up).
# ---------------------------------------------------------------------------
def sim_events(T=600, t0=300, k0=0.5, A=2.0, tau=40):
    print(f"\n[7] Charged event at t={t0} (after burn-in): before -> peak (+30) -> end")
    spike = lambda t: A * np.exp(-(t - t0) / tau) if t >= t0 else 0.0
    scen = {"none": lambda t: (k0, k0),
            "sym": lambda t: (k0 + spike(t), k0 + spike(t)),
            "right": lambda t: (k0, k0 + spike(t))}
    x0 = camps(10_000, 0.5)
    RES["event_t0"], RES["event_k"] = t0, np.array([k0 + spike(t) for t in range(T)])
    print(f"   {'scenario':>8} {'pop sd':>20} {'feed sd':>20} {'feed mean':>22}")
    for name, kf in scen.items():
        _, rec = two_camp(x0, kf, T, record={t0 - 1: f"event_{name}_pre",
                                              t0 + 30: f"event_{name}_peak",
                                              T - 1: f"event_{name}_post"})
        RES[f"event_{name}_ts"] = rec
        f = lambda c: " -> ".join(f"{rec[t, c]:+.2f}" for t in (t0 - 1, t0 + 30, T - 1))
        print(f"   {name:>8} {f(1):>20} {f(3):>20} {f(2):>22}")

# ---------------------------------------------------------------------------
# 8. Why issues become two-party: spiral of silence on a line of positions.
#    Factions sit at positions p_i with population share s_i. Visibility
#    v_i = s_i exp(k |p_i|). Each step, people drift toward a neighbouring
#    position that is more visible than their own ("my view seems rare, the
#    one next to it is everywhere"). k=0: majority (centre) absorbs everyone.
#    k>0: the centre goes silent and mass ends in two poles.
# ---------------------------------------------------------------------------
def spiral(k, pos, steps=1500, eta=0.05, reach=0.5):
    """People at p_i drift toward any position within `reach` that is more visible."""
    s = np.exp(-pos**2 / 2); s /= s.sum(); hist = [s.copy()]
    D = np.abs(pos[:, None] - pos[None, :]); W = (D <= reach) & (D > 0)
    W = W / W.sum(1, keepdims=True)
    for _ in range(steps):
        v = s * np.exp(k * np.abs(pos))
        rel = np.maximum(v[None, :] / (v[:, None] + v[None, :] + 1e-12) - 0.5, 0)
        m = eta * s[:, None] * W * rel                 # m[i, j]: flow i -> j
        s = s - m.sum(1) + m.sum(0); hist.append(s.copy())
    return np.array(hist)


def n_factions(s, tol=1e-3):
    """Effective number of factions: merge contiguous occupied positions into clusters."""
    occ = s > tol; masses, cur = [], 0.0
    for o, x in zip(occ, s):
        if o: cur += x
        elif cur: masses.append(cur); cur = 0.0
    if cur: masses.append(cur)
    q = np.array(masses) / sum(masses)
    return 1 / np.sum(q**2)


def sim_two_party():
    pos = np.linspace(-2, 2, 41); RES["tp_pos"] = pos
    ks = np.linspace(0, 3, 16); reaches = np.linspace(0.2, 2.0, 13)
    grid = np.array([[n_factions(spiral(k, pos, reach=r)[-1]) for k in ks] for r in reaches])
    RES["tp_k"], RES["tp_reach"], RES["tp_neff"] = ks, reaches, grid
    for name, k, r in [("consensus", 0.0, 2.0), ("fragmented", 2.0, 0.5), ("twoparty", 2.0, 2.0)]:
        h = spiral(k, pos, reach=r); RES[f"tp_hist_{name}"] = h
        RES[f"tp_meta_{name}"] = np.array([k, r, n_factions(h[-1])])
    print("\n[8] Two-party: effective no. of factions; rows = reach, cols = k")
    print("   reach | " + " ".join(f"k={k:.1f}" for k in ks[::3]))
    for i in range(0, len(reaches), 3):
        print(f"   {reaches[i]:5.2f} | " + " ".join(f"{grid[i, j]:5.2f}" for j in range(0, len(ks), 3)))


if __name__ == "__main__":
    # `uv run discourse_sims.py two_party events` runs only those, merging into sims.npz
    import sys
    ALL = dict(offset=sim_offset, diabolisation=sim_diabolisation, bubble=sim_bubble,
               minority=sim_minority, control=sim_control, who_wins=sim_who_wins,
               events=sim_events, two_party=sim_two_party)
    chosen = sys.argv[1:] or list(ALL)
    if sys.argv[1:] and (OUT / "sims.npz").exists():
        RES.update(np.load(OUT / "sims.npz"))
    for name in chosen: ALL[name]()
    np.savez_compressed(OUT / "sims.npz", **RES)
    print(f"\nResults in {OUT / 'sims.npz'}")
