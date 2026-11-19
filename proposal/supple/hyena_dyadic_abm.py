"""Minimal ABM: dyadic whoop replies -> cascades.
4 hyenas on a line, three contexts: defence, cohesion, hunting.

Fig 1: one example cascade per context showing relay chain emergence.
        Uses context-specific amplification shapes g(s) at full asymmetry.

Fig 2: Virtual sensor coverage analysis.
        How spatial coverage (nodes along transect) and localisation error
        affect detection of simulated cascade events.
        The sensitivity sweep uses a FLAT g(s)=0.5 for both defence and
        cohesion so that alpha=0 makes the two contexts genuinely identical.
        alpha controls care asymmetry only; amplification shape is held constant.
        This isolates the contribution of care asymmetry to detectability.

Note on the PAM simulation: nodes detect calls within audible range and add
Gaussian localisation error. This is a simplified spatial-coverage model,
not a full TDOA pipeline.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

rng = np.random.default_rng(3)
rng_sensor = np.random.default_rng(3003)

AUD, U0, NW = 1.0, 0.2, 4
POS = np.array([0, .7, 1.4, 2.1])
COL = {'defence': '#c0392b', 'cohesion': '#1f7a8c', 'hunting': '#e67e22'}

# ── Care asymmetry parameter ──────────────────────────────────────────────────
# alpha=0: defence and cohesion have identical care (0.5) -> contexts indistinguishable
# alpha=1: defence=0.8, cohesion=0.2 (maximum spread)
def care_level(ctx, alpha):
    base = {'defence': 0.8, 'hunting': 0.5, 'cohesion': 0.2}
    return 0.5 + alpha * (base[ctx] - 0.5)

# ── Two decide() variants ─────────────────────────────────────────────────────
def decide_biological(d, u, ctx, alpha):
    """Full biological model: context-specific g(s) and care asymmetry.
    Used for Fig 1 (example cascades)."""
    s = d / AUD
    care = care_level(ctx, alpha)
    p_reply = care * (0.1 + 0.8 * u) * (1 - 0.5 * s)
    if ctx == 'defence':
        shape = 0.2 + 0.8 * s
    elif ctx == 'cohesion':
        shape = 0.25 + 0.6 * s**4
    else:
        shape = 0.5
    if rng.random() > p_reply: return 'none'
    return 'amp' if rng.random() < u * shape else 'ack'

def decide_flat(d, u, ctx, alpha):
    """Sensitivity analysis: flat g(s)=0.5 for all contexts.
    alpha=0 makes defence and cohesion genuinely identical.
    Used for Fig 2 (sensitivity sweep)."""
    s = d / AUD
    care = care_level(ctx, alpha)
    p_reply = care * (0.1 + 0.8 * u) * (1 - 0.5 * s)
    if rng.random() > p_reply: return 'none'
    return 'amp' if rng.random() < 0.5 * u else 'ack'

def cascade(ctx, u, alpha=1.0, pos=POS, decide_fn=decide_biological):
    calls = [dict(t=0., who=0, u=u, kind='orig', src=None)]
    done = {0}; q = [0]
    while q:
        c = calls[q.pop(0)]
        for j in range(len(pos)):
            d = abs(pos[j] - pos[c['who']])
            if j in done or d > AUD: continue
            k = decide_fn(d, c['u'], ctx, alpha)
            if k == 'none': continue
            done.add(j)
            calls.append(dict(
                t    = c['t'] + NW / c['u'] + 1 + rng.exponential(2),
                who  = j,
                u    = c['u'] if k == 'amp' else U0,
                kind = k,
                src  = c['who']))
            q.append(len(calls) - 1)
    return calls

# ── Virtual sensor: simplified spatial coverage (not full TDOA) ───────────────
def pam_detect(calls, pos, n_nodes, loc_err, L=3.0):
    """Returns (detected_mask, estimated_positions).
    A call is detected if >= 2 nodes lie within AUD of its true position.
    Estimated position has zero-mean Gaussian error (std=loc_err).
    Grid is 1D (nodes along transect)."""
    true_pos = np.array([pos[c['who']] for c in calls])
    nodes = np.linspace(0, L, n_nodes)
    # For each call, count how many nodes are within AUD
    covering_per_call = np.array([
        np.sum(np.abs(p - nodes) < AUD) for p in true_pos])
    detected = covering_per_call >= 2
    est_pos = true_pos + rng_sensor.normal(0, loc_err, true_pos.shape)
    return detected, est_pos

def cascade_detection_rate(calls, pos, n_nodes, loc_err):
    """Proportion of all calls in the cascade detected by the grid."""
    detected, _ = pam_detect(calls, pos, n_nodes, loc_err)
    return detected.mean()

def relay_detection_rate(calls, pos, n_nodes, loc_err):
    """Proportion of true relay events whose full call path is recoverable.

    The original caller, all intermediate callers, and the relayer must be
    detected. Their estimated positions must also be consistent with each
    successive audible-range link and with the relayer lying beyond the
    original caller's audible range.
    """
    detected, est_pos = pam_detect(calls, pos, n_nodes, loc_err)
    who_to_idx = {c['who']: i for i, c in enumerate(calls)}
    orig_who = calls[0]['who']
    orig_pos = pos[orig_who]

    relay_events = [
        i for i, c in enumerate(calls)
        if c['src'] is not None
        and abs(pos[c['who']] - orig_pos) > AUD
    ]
    if not relay_events:
        return np.nan

    recoverable = []
    for i in relay_events:
        path = [i]
        current = calls[i]['src']
        while current is not None:
            parent_idx = who_to_idx[current]
            path.append(parent_idx)
            current = calls[parent_idx]['src']
        path.reverse()

        all_detected = all(detected[j] for j in path)
        links_consistent = all(
            abs(est_pos[b] - est_pos[a]) <= AUD
            for a, b in zip(path[:-1], path[1:])
        )
        beyond_origin = abs(est_pos[path[-1]] - est_pos[path[0]]) > AUD
        recoverable.append(all_detected and links_consistent and beyond_origin)

    return np.mean(recoverable)

# ── Fig 1: example cascade per context (biological model, full asymmetry) ─────
fig, ax = plt.subplots(1, 3, figsize=(14, 3.4), sharey=True)
kc = {'orig': 'k', 'amp': '#e67e22', 'ack': '#7f8c8d'}
for a, ctx in zip(ax, COL):
    for seed in range(500):
        rng = np.random.default_rng(seed)
        cs = cascade(ctx, 1.0, alpha=1.0, decide_fn=decide_biological)
        if len(cs) >= 3 and cs[-1]['who'] >= 2: break
    for c in cs:
        n = 1 if c['kind'] == 'ack' else NW
        ts = c['t'] + np.arange(n) / c['u']
        a.vlines(ts, c['who'] - .25, c['who'] + .25,
                 color=kc[c['kind']], lw=2.5)
        if c['src'] is not None:
            src_idx = [x['who'] for x in cs].index(c['src'])
            a.annotate('', (c['t'], c['who']),
                       (cs[src_idx]['t'], c['src']),
                       arrowprops=dict(arrowstyle='->', color='.6', lw=1))
    a.set_yticks(range(4))
    a.set_yticklabels([f'{n}  (x={p})' for n, p in zip('ABCD', POS)])
    a.set_title(ctx.capitalize(), color=COL[ctx], weight='bold')
    a.set_xlabel('time (s)'); a.invert_yaxis()
for k, l in [('orig','original bout'),('amp','amplified relay'),('ack','acknowledge')]:
    ax[0].plot([], [], color=kc[k], lw=3, label=l)
ax[0].legend(frameon=False, fontsize=8, loc='lower right')
ax[0].set_title('Example cascade per context\n(biological amplification shapes)',
                fontweight='bold', fontsize=9)
plt.tight_layout()
plt.savefig('fig1_example_cascade.png', dpi=200)
plt.close()
print('Fig 1 done')

# ── Fig 2: virtual sensor coverage analysis (flat g(s), alpha sweep) ──────────
# For each alpha x n_nodes x loc_err:
# measure cascade detection rate and relay detection rate for defence vs cohesion.
# Standardised paired context contrast:
# d = mean(defence - cohesion) / SD(defence - cohesion) across replicates.
# d=1 is a descriptive one-SD contrast, not a significance threshold.
# At alpha=0 the expected context contrast is zero.

ALPHAS    = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
NODES     = [4, 5, 6, 8, 10]       # nodes along 1D transect
LOC_ERRS  = [0.0, 0.1, 0.2, 0.3]
N_CASCADE = 300
N_REP     = 20
U_CALL    = 1.0

print('Running sensor coverage sweep (flat g(s), care asymmetry only)...')
D_detect = np.zeros((len(ALPHAS), len(NODES), len(LOC_ERRS)))
D_relay  = np.zeros((len(ALPHAS), len(NODES), len(LOC_ERRS)))

for ai, alpha in enumerate(ALPHAS):
    for ni, n_nodes in enumerate(NODES):
        for li, loc_err in enumerate(LOC_ERRS):
            det_def_reps, det_coh_reps = [], []
            rel_def_reps, rel_coh_reps = [], []
            for rep in range(N_REP):
                rng = np.random.default_rng(rep*1000 + ai*100 + ni*10 + li)
                dd, dc, rd, rc = [], [], [], []
                for _ in range(N_CASCADE):
                    cs_d = cascade('defence', U_CALL, alpha, decide_fn=decide_flat)
                    dd.append(cascade_detection_rate(cs_d, POS, n_nodes, loc_err))
                    r = relay_detection_rate(cs_d, POS, n_nodes, loc_err)
                    if not np.isnan(r): rd.append(r)

                    cs_c = cascade('cohesion', U_CALL, alpha, decide_fn=decide_flat)
                    dc.append(cascade_detection_rate(cs_c, POS, n_nodes, loc_err))
                    r = relay_detection_rate(cs_c, POS, n_nodes, loc_err)
                    if not np.isnan(r): rc.append(r)

                det_def_reps.append(np.mean(dd))
                det_coh_reps.append(np.mean(dc))
                rel_def_reps.append(np.mean(rd) if rd else 0.0)
                rel_coh_reps.append(np.mean(rc) if rc else 0.0)

            for diff_arr, D_arr in [
                    (np.array(det_def_reps) - np.array(det_coh_reps), D_detect),
                    (np.array(rel_def_reps) - np.array(rel_coh_reps), D_relay)]:
                sd = np.std(diff_arr)
                D_arr[ai, ni, li] = np.mean(diff_arr) / sd if sd > 0 else 0.0

    print(f'  alpha={alpha:.1f} done')

# Plot: 2x2 — cascade detection d and relay detection d, each with line plot and heatmap
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
li_fix = LOC_ERRS.index(0.1)
THRESHOLD = 1.0
cmap_nodes = plt.cm.Blues(np.linspace(0.4, 0.95, len(NODES)))

for row, (D_arr, label) in enumerate([(D_detect, 'cascade detection rate'),
                                       (D_relay,  'relay detection rate')]):
    # Line plot: d vs alpha
    ax = axes[row, 0]
    for ni, n_nodes in enumerate(NODES):
        ax.plot(ALPHAS, D_arr[:, ni, li_fix], 'o-',
                color=cmap_nodes[ni], lw=2, ms=5,
                label=f'{n_nodes} nodes along transect')
    ax.axhline(THRESHOLD, color='k', lw=1.2, linestyle='--',
               label='one-SD contrast  d = 1')
    ax.fill_between(ALPHAS, 0, THRESHOLD, alpha=0.06, color='grey')
    ax.set_xlabel('Care asymmetry  α  (0 = contexts identical)', fontsize=9)
    ax.set_ylabel(f'Standardised context contrast  d\n({label})', fontsize=8.5)
    ax.set_title(f'({"ab"[row]})  d vs α  |  {label}\n(localisation error = 0.1 r)',
                 fontsize=9, fontweight='bold')
    if row == 0:
        ax.legend(fontsize=7.5, frameon=False, loc='upper left')
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(bottom=0)
    ax.grid(axis='y', lw=0.4, color='#ddd')

    # Heatmap: min detectable alpha
    ax = axes[row, 1]
    min_alpha = np.full((len(NODES), len(LOC_ERRS)), np.nan)
    for ni in range(len(NODES)):
        for li in range(len(LOC_ERRS)):
            above = np.where(D_arr[:, ni, li] >= THRESHOLD)[0]
            if len(above) > 0:
                min_alpha[ni, li] = ALPHAS[above[0]]
    levels = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    norm = BoundaryNorm(levels, plt.cm.RdYlGn_r.N)
    im = ax.imshow(min_alpha, aspect='auto', origin='lower',
                   cmap=plt.cm.RdYlGn_r, norm=norm,
                   extent=[-0.5, len(LOC_ERRS)-0.5, -0.5, len(NODES)-0.5])
    for ni in range(len(NODES)):
        for li in range(len(LOC_ERRS)):
            val = min_alpha[ni, li]
            txt = f'{val:.1f}' if not np.isnan(val) else '—'
            ax.text(li, ni, txt, ha='center', va='center', fontsize=8.5,
                    fontweight='bold',
                    color='white' if (not np.isnan(val) and val > 0.3) else 'k')
    ax.set_xticks(range(len(LOC_ERRS)))
    ax.set_xticklabels([str(e) for e in LOC_ERRS])
    ax.set_yticks(range(len(NODES)))
    ax.set_yticklabels([str(n) for n in NODES])
    ax.set_xlabel('Localisation error  (× audible range)', fontsize=9)
    ax.set_ylabel('Nodes along transect', fontsize=9)
    ax.set_title(f'({"cd"[row]})  Min detectable α  |  {label}',
                 fontsize=9, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(
        'minimum α for d ≥ 1', fontsize=8)

fig.suptitle('Virtual sensor coverage analysis\n'
             '(flat amplification shape; care asymmetry isolated)',
             fontsize=10, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('fig2_sensor_coverage.png', dpi=200, bbox_inches='tight')
plt.close()
print('Fig 2 done')
