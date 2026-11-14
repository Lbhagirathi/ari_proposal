"""Hyena whoop ABM: 4 hyenas moving in 2D, three social contexts.
Context sets the care matrix (who attends to whom) and the amplification shape g(s).
Cascades emerge from iterated local decisions; no global coordination.

Last block: virtual PAM grid that sees only localised call positions (no identity)
-> how well does it recover relay probability and branching ratio R?
"""
import numpy as np, matplotlib.pyplot as plt
rng = np.random.default_rng(5)

AUD  = 1.0    # audible range (r)
L    = 3.0    # arena side length (× r)
N    = 4      # group size
TOP  = 0      # index of highest-ranked individual
T    = 1000   # time steps per replicate
REPS = 100    # replicates per context
U0   = 0.2    # baseline whoop rate (whoops/s)
SIG  = 0.08   # random-walk s.d. per step (× r)
STEP = 0.3    # fraction of distance moved toward replied-to animal
P_EP = 0.3    # probability of a call episode per time step

COL = {'defence': '#c0392b', 'cohesion': '#1f7a8c', 'hunting': '#e67e22'}

# CARE[ctx][i, j] = how much listener i attends to caller j (scales reply probability)
idx = np.arange(N)
CARE = {
    # Defence: clan-wide mutual support; top animal most responsive
    'defence':  np.tile(np.where(idx == TOP, 1.0, 0.8)[:, None], (1, N)),
    # Cohesion: strong bond with top animal; little attention among subordinates
    'cohesion': np.tile(np.where(idx == TOP, 1.0, 0.3)[None, :], (N, 1)),
    # Hunting: moderate uniform attention; coordinate active pursuers
    'hunting':  np.full((N, N), 0.5),
}

def decide(d, u, ctx, care):
    """Receiver at distance d hears bout of rate u; returns 'amp' | 'ack' | None."""
    s = d / AUD
    if rng.random() > care * (0.1 + 0.8 * u) * (1 - 0.5 * s):
        return None
    if ctx == 'defence':
        shape = 0.2 + 0.8 * s          # rising: distant animals most need a relay
    elif ctx == 'cohesion':
        shape = 0.25 + 0.6 * s**4      # flat with uptick near range edge
    else:  # hunting
        shape = 0.5                     # flat intermediate: relay regardless of distance
    return 'amp' if rng.random() < u * shape else 'ack'

def cascade(pos, ctx, caller, u=1.0):
    """Run one cascade from `caller`; returns (calls, events, relay_occurred)."""
    calls = [(caller, u, None)]
    q, done, ev, relay = [0], {caller}, [], False
    while q:
        w, uc, _ = calls[q.pop(0)]
        for j in range(N):
            d = np.linalg.norm(pos[j] - pos[w])
            if j in done or d > AUD: continue
            k = decide(d, uc, ctx, CARE[ctx][j, w])
            ev.append((j, w, k is not None))
            if k is None: continue
            done.add(j)
            calls.append((j, uc if k == 'amp' else U0, w))
            q.append(len(calls) - 1)
            relay |= np.linalg.norm(pos[j] - pos[caller]) > AUD
    return calls, ev, relay

def run(ctx):
    """Simulate T steps; return (log, support_matrix, mean_distance, trajectory, episodes)."""
    pos   = rng.uniform(0, L, (N, 2))
    B     = np.zeros((N, N))   # running support matrix (exponential moving average)
    log, ratio, traj, eps = [], [], [], []
    for t in range(T):
        # random walk with reflecting walls
        pos = pos + rng.normal(0, SIG, pos.shape)
        pos = np.where(pos < 0, -pos, np.where(pos > L, 2*L - pos, pos))
        if rng.random() < P_EP:
            c = rng.integers(N)
            calls, ev, relay = cascade(pos, ctx, c)
            log.append((c, len(calls) > 1, relay))
            eps.append(pos[[w for w, _, _ in calls]].copy())
            for i, j, responded in ev:
                B[i, j] += 0.1 * (responded - B[i, j])
            # repliers move toward the animal they answered
            for who, _, src in calls[1:]:
                pos[who] += STEP * (pos[src] - pos[who])
        D = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
        ratio.append(D.sum() / (N * (N - 1)))
        traj.append(pos.copy())
    return np.array(log), B, np.array(ratio), np.array(traj), eps

# run all contexts (takes a few minutes)
R = {c: [run(c) for _ in range(REPS)] for c in COL}

# ── Fig 1: example movement trajectories (last 200 steps) ───────────────────────────────────
fig, ax = plt.subplots(1, 3, figsize=(13, 4.4))
for a, ctx in zip(ax, COL):
    tr = R[ctx][0][3][-200:]
    for i in range(N):
        top = i == TOP
        a.plot(*tr[:, i].T,
               color='goldenrod' if top else '.55',
               lw=1.4 if top else .7,
               alpha=.9 if top else .6)
        a.scatter(*tr[-1, i],
                  s=120 if top else 40,
                  color='goldenrod' if top else '.35',
                  edgecolor='k', zorder=5)
    a.set_title(ctx.capitalize(), color=COL[ctx], weight='bold')
    a.set_xlim(0, L); a.set_ylim(0, L)
    a.set_aspect('equal'); a.set_xticks([]); a.set_yticks([])
plt.tight_layout(); plt.savefig('fig1_rank_movement.png', dpi=200); plt.close()

# ── Fig 2: P(answered) and P(relay) by caller rank ──────────────────────────────────────────
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4)); w = .25
for k, ctx in enumerate(COL):
    lg = np.vstack([r[0] for r in R[ctx]])
    for a, col in zip(ax, (1, 2)):
        a.bar(np.arange(N) + (k - 1) * w,
              [lg[lg[:, 0] == c, col].mean() for c in range(N)],
              w, color=COL[ctx], label=ctx)
for a, t in zip(ax, ('P(call is answered)', 'P(relay beyond caller range)')):
    a.set_xticks(range(N))
    a.set_xticklabels(['top'] + [f'rank {i+1}' for i in range(1, N)])
    a.set_xlabel('caller'); a.set_ylabel(t)
ax[0].set_ylim(0, 1.05)
ax[0].legend(frameon=False, ncol=3, loc='upper right')
plt.tight_layout(); plt.savefig('fig2_answered_by_rank.png', dpi=200); plt.close()

# ── Fig 3: support heatmaps + group cohesion over time ──────────────────────────────────────
fig, ax = plt.subplots(1, 4, figsize=(15, 3.5),
                       gridspec_kw={'width_ratios': [1, 1, 1, 1.3]})
for a, ctx in zip(ax[:3], COL):
    M = np.mean([r[1] for r in R[ctx]], 0)
    np.fill_diagonal(M, np.nan)
    im = a.imshow(M, vmin=0, vmax=.7, cmap='Oranges')
    a.set_title(f'Support: {ctx}', color=COL[ctx], weight='bold', fontsize=9)
    a.set_xticks(range(N)); a.set_yticks(range(N))
    a.set_xlabel('caller j'); a.set_ylabel('listener i')
    a.set_xticklabels(['top', 'r2', 'r3', 'r4'])
    a.set_yticklabels(['top', 'r2', 'r3', 'r4'])
plt.colorbar(im, ax=ax[2], fraction=.046)
for ctx in COL:
    m = np.convolve(np.mean([r[2] for r in R[ctx]], 0),
                    np.ones(25) / 25, 'valid')
    ax[3].plot(m, color=COL[ctx], lw=2, label=ctx)
ax[3].set_xlabel('time step')
ax[3].set_ylabel('mean pairwise distance (× audible range)')
ax[3].legend(frameon=False)
plt.tight_layout(); plt.savefig('fig3_support_and_tightness.png', dpi=200); plt.close()

# ── PAM view: grid sees only localised positions and temporal order (no identity) ────────────
NODES0 = 6      # default nodes per side
LOC0   = 0.15   # default localisation error (× audible range)

def pam(P, n, loc):
    """Localise cascade calls: detected if >=3 nodes within r; positions get TDOA error."""
    g = np.linspace(0, L, n)
    nodes = np.array([(x, y) for x in g for y in g])
    detected = (np.linalg.norm(P[:, None] - nodes[None], axis=-1) < AUD).sum(1) >= 3
    Q = P[detected]
    return Q + rng.normal(0, loc, Q.shape)

def relay_hat(Q):
    """Infer relay A->B->C: C beyond A's range, B within range of both."""
    if len(Q) < 3: return False
    d = np.linalg.norm(Q[:, None] - Q[None], axis=-1)
    return any(
        d[0, m] > AUD and any(d[0, b] <= AUD and d[b, m] <= AUD
                               for b in range(1, m))
        for m in range(2, len(Q)))

def estimate(eps, n, loc):
    """Return (P(relay), branching ratio R) as seen by a PAM grid."""
    Qs    = [pam(P, n, loc) for P in eps]
    calls = sum(map(len, Qs))
    return (np.mean([relay_hat(Q) for Q in Qs]),
            sum(max(len(Q) - 1, 0) for Q in Qs) / max(calls, 1))

# ── Fig 4: PAM estimates vs ABM ground truth ────────────────────────────────────────────────
SW = {
    'localisation error (× audible range)': [0, .1, .2, .3, .5],
    'nodes per side':                        [4, 5, 6, 8, 10],
}
fig, ax = plt.subplots(2, 2, figsize=(9, 6.4))
for ctx in COL:
    eps    = [P for r in R[ctx] for P in r[4]]
    n_calls = sum(map(len, eps))
    truth  = (np.vstack([r[0] for r in R[ctx]])[:, 2].mean(),
              (n_calls - len(eps)) / n_calls)
    for row, (name, xs) in enumerate(SW.items()):
        est = np.array([
            estimate(eps, NODES0, x) if row == 0 else estimate(eps, x, LOC0)
            for x in xs])
        for col in (0, 1):
            ax[row, col].plot(xs, est[:, col], 'o-', color=COL[ctx], label=ctx)
            ax[row, col].axhline(truth[col], color=COL[ctx], ls='--', lw=1)
for row, name in enumerate(SW):
    for col, ylabel in enumerate(('P(relay episode)',
                                  'branching ratio R (replies per call)')):
        ax[row, col].set_xlabel(name)
        ax[row, col].set_ylabel(ylabel)
        ax[row, col].set_ylim(bottom=0)
ax[0, 0].plot([], [], 'k--', lw=1, label='truth (ABM)')
ax[0, 0].legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.savefig('fig4_pam_view.png', dpi=200); plt.close()
