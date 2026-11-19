"""Hyena whoop ABM: 4 hyenas moving in 2D, three social contexts.
Context sets the care matrix (who attends to whom) and the amplification shape g(s).
Cascades emerge from iterated local decisions; no global coordination.

Fig 1: P(answered) and P(relay) by caller rank and context.
       Shows how assumed rank-dependent care matrices affect participation.
       These are consequences of the assumed CARE matrices, not independently
       discovered patterns.

Fig 2: Virtual sensor-grid cascade detection and relay-path recovery.
       Detection depends on true spatial coverage; localisation error affects
       whether the estimated positions preserve the full relay path.

Network centrality (P4, P5) is not calculated in this four-individual model.
A larger simulation and episode-level network analyses would be needed.
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

rng = np.random.default_rng(5)
rng_sensor = np.random.default_rng(5005)

AUD  = 1.0
L    = 3.0
N    = 4
TOP  = 0
T    = 1000
REPS = 100
U0   = 0.2
SIG  = 0.08
STEP = 0.3
P_EP = 0.3

COL = {'defence': '#c0392b', 'cohesion': '#1f7a8c', 'hunting': '#e67e22'}

idx = np.arange(N)
CARE = {
    # Defence: clan-wide mutual support; top animal most responsive
    'defence':  np.tile(np.where(idx == TOP, 1.0, 0.8)[:, None], (1, N)),
    # Cohesion: strong attention to top animal; little care among subordinates
    'cohesion': np.tile(np.where(idx == TOP, 1.0, 0.3)[None, :], (N, 1)),
    # Hunting: moderate uniform attention
    'hunting':  np.full((N, N), 0.5),
}

def decide(d, u, ctx, care):
    s = d / AUD
    if rng.random() > care * (0.1 + 0.8 * u) * (1 - 0.5 * s):
        return None
    if ctx == 'defence':
        shape = 0.2 + 0.8 * s
    elif ctx == 'cohesion':
        shape = 0.25 + 0.6 * s**4
    else:
        shape = 0.5
    return 'amp' if rng.random() < u * shape else 'ack'

def cascade(pos, ctx, caller, u=1.0):
    """Returns (calls, ev, relay).
    calls: list of (who, u, src) tuples.
    ev: list of (listener, caller, responded) triples.
    relay: True if any responder is beyond AUD of original caller."""
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
    pos = rng.uniform(0, L, (N, 2))
    log, eps = [], []
    for t in range(T):
        pos = pos + rng.normal(0, SIG, pos.shape)
        pos = np.where(pos < 0, -pos, np.where(pos > L, 2*L - pos, pos))
        if rng.random() < P_EP:
            c = rng.integers(N)
            calls, ev, relay = cascade(pos, ctx, c)
            log.append((c, len(calls) > 1, relay))
            # Preserve the episode's positions and actual caller-to-responder links.
            eps.append({'positions': pos.copy(), 'calls': calls.copy()})
            for who, _, src in calls[1:]:
                pos[who] += STEP * (pos[src] - pos[who])
    return np.array(log), eps

R = {c: [run(c) for _ in range(REPS)] for c in COL}

# ── Fig 1: P(answered) and P(relay) by caller rank ───────────────────────────
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
w = 0.25
for k, ctx in enumerate(COL):
    lg = np.vstack([r[0] for r in R[ctx]])
    for a, col in zip(ax, (1, 2)):
        a.bar(np.arange(N) + (k - 1) * w,
              [lg[lg[:, 0] == c, col].mean() for c in range(N)],
              w, color=COL[ctx], label=ctx)
for a, t in zip(ax, ('P(call is answered)', 'P(relay beyond caller range)')):
    a.set_xticks(range(N))
    a.set_xticklabels(['top'] + [f'rank {i+1}' for i in range(1, N)])
    a.set_xlabel('caller rank'); a.set_ylabel(t)
ax[0].set_ylim(0, 1.05)
ax[0].legend(frameon=False, ncol=3, loc='upper right')
ax[0].set_title('Rank effects on cascade participation\n'
                '(consequence of assumed CARE matrices)',
                fontweight='bold', fontsize=9)
plt.tight_layout()
plt.savefig('fig1_rank_responses.png', dpi=200)
plt.close()
print('Fig 1 done')

# ── PAM view: virtual sensor grid ────────────────────────────────────────────
NODES0, LOC0 = 6, 0.15

def pam_detect(P, n, loc):
    """Return a detection mask and noisy estimated positions for all callers.

    Detection depends on true spatial coverage; localisation error affects
    only the subsequent inference of whether the detected calls form a relay.
    """
    g = np.linspace(0, L, n)
    nodes = np.array([(x, y) for x in g for y in g])
    covering = (np.linalg.norm(P[:, None] - nodes[None], axis=-1) < AUD).sum(1)
    detected = covering >= 3
    estimated = P + rng_sensor.normal(0, loc, P.shape)
    return detected, estimated

def cascade_detect_rate(episodes, n, loc):
    """Proportion of all cascade calls detected by the virtual grid."""
    n_detected, n_calls = 0, 0
    for episode in episodes:
        calls = episode['calls']
        P = episode['positions'][[who for who, _, _ in calls]]
        detected, _ = pam_detect(P, n, loc)
        n_detected += detected.sum()
        n_calls += len(calls)
    return n_detected / n_calls if n_calls else 0.0

def relay_detect_rate(episodes, n, loc):
    """Proportion of true-relay episodes in which a complete relay path is recovered.

    The original caller, all intermediate callers, and a relayer beyond the
    original caller's audible range must be detected. Estimated positions must
    also be consistent with each successive audible-range link.
    """
    relay_episode_results = []
    for episode in episodes:
        calls = episode['calls']
        P_all = episode['positions']
        P = P_all[[who for who, _, _ in calls]]
        detected, estimated = pam_detect(P, n, loc)

        orig_who = calls[0][0]
        orig_pos = P_all[orig_who]
        who_to_idx = {who: i for i, (who, _, _) in enumerate(calls)}

        relay_indices = [
            i for i, (who, _, src) in enumerate(calls)
            if src is not None
            and np.linalg.norm(P_all[who] - orig_pos) > AUD
        ]
        if not relay_indices:
            continue

        episode_recovered = False
        for i in relay_indices:
            path = [i]
            current = calls[i][2]
            while current is not None:
                parent_idx = who_to_idx[current]
                path.append(parent_idx)
                current = calls[parent_idx][2]
            path.reverse()

            all_detected = all(detected[j] for j in path)
            links_consistent = all(
                np.linalg.norm(estimated[b] - estimated[a]) <= AUD
                for a, b in zip(path[:-1], path[1:])
            )
            beyond_origin = (
                np.linalg.norm(estimated[path[-1]] - estimated[path[0]]) > AUD
            )
            if all_detected and links_consistent and beyond_origin:
                episode_recovered = True
                break

        relay_episode_results.append(episode_recovered)

    return np.mean(relay_episode_results) if relay_episode_results else np.nan

# ── Fig 2: virtual-grid cascade detection and relay recovery ──────────────────────────────
NODES_SW = [4, 5, 6, 8, 10]
LOC_SW   = [0, .1, .2, .3, .5]

fig, axes = plt.subplots(2, 2, figsize=(10, 7))

for ctx in COL:
    eps = [P for r in R[ctx] for P in r[1]]

    # Relay detection is conditional on an episode containing a true relay.
    # A value of 1 indicates complete recovery of all such episodes.

    # Row 0: vs localisation error (nodes fixed at NODES0)
    det_loc = [cascade_detect_rate(eps, NODES0, x) for x in LOC_SW]
    rel_loc = [relay_detect_rate(eps, NODES0, x) for x in LOC_SW]

    # Row 1: vs grid density (loc error fixed at LOC0)
    det_nod = [cascade_detect_rate(eps, n, LOC0) for n in NODES_SW]
    rel_nod = [relay_detect_rate(eps, n, LOC0) for n in NODES_SW]

    axes[0, 0].plot(LOC_SW, det_loc, 'o-', color=COL[ctx], lw=2, ms=5, label=ctx)
    axes[0, 1].plot(LOC_SW, rel_loc, 'o-', color=COL[ctx], lw=2, ms=5)
    axes[1, 0].plot(NODES_SW, det_nod, 'o-', color=COL[ctx], lw=2, ms=5)
    axes[1, 1].plot(NODES_SW, rel_nod, 'o-', color=COL[ctx], lw=2, ms=5)
    # Ideal recovery reference for episodes known to contain a true relay.
    axes[0, 1].axhline(1.0, color=COL[ctx], ls='--', lw=1)
    axes[1, 1].axhline(1.0, color=COL[ctx], ls='--', lw=1)

for ax, (xlab, ylab, title) in zip(
        axes.flat,
        [('Localisation error (× AUD)', 'Cascade detection rate',
          f'(a) Cascade detection vs loc. error\n(nodes = {NODES0}/side)'),
         ('Localisation error (× AUD)', 'Relay-episode recovery rate',
          f'(b) Relay recovery vs loc. error\n(dashed = ideal recovery)'),
         ('Nodes per side', 'Cascade detection rate',
          f'(c) Cascade detection vs grid density\n(loc. error = {LOC0} r)'),
         ('Nodes per side', 'Relay-episode recovery rate',
          f'(d) Relay recovery vs grid density')]):
    ax.set_xlabel(xlab, fontsize=9); ax.set_ylabel(ylab, fontsize=9)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.set_ylim(0, 1.05); ax.grid(axis='y', lw=0.4, color='#ddd')

axes[0, 0].legend(frameon=False, fontsize=8)
axes[0, 1].plot([], [], 'k--', lw=1, label='ideal relay recovery')
fig.suptitle('Virtual PAM grid: cascade and relay detection rates',
             fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig('fig2_pam_detection.png', dpi=200)
plt.close()
print('Fig 2 done')
