"""Minimal ABM: dyadic whoop replies -> cascades.
4 hyenas on a line, three contexts: defence, cohesion, hunting.

Context sets the amplification shape g(s), which determines whether a receiver
relays the bout (amplification) or only acknowledges (single whoop at baseline).
The cascade emerges from iterated local relay decisions; no global coordination.
"""
import numpy as np, matplotlib.pyplot as plt
rng = np.random.default_rng(3)

AUD, U0, NW = 1.0, 0.2, 4            # audible range (r), baseline whoop rate (whoops/s), whoops per bout
POS = np.array([0, .7, 1.4, 2.1])    # A..D: neighbours within r; C and D reachable only via relay
COL = {'defence': '#c0392b', 'cohesion': '#1f7a8c', 'hunting': '#e67e22'}

def decide(d, u, ctx):
    """Receiver heard a bout of rate u from distance d.
    Returns 'none' | 'ack' | 'amp'.
    p_reply: urgent bouts more likely answered; distant callers less so.
    g(s):    probability that a reply is an amplification (relay) rather than ack.
    """
    s = d / AUD
    p_reply = (0.1 + 0.8 * u) * (1 - 0.5 * s)
    if ctx == 'defence':
        shape = 0.2 + 0.8 * s          # rising: distant animals most need a relay during conflict
    elif ctx == 'cohesion':
        shape = 0.25 + 0.6 * s**4      # flat with uptick at range edge: re-establish contact
    else:  # hunting
        shape = 0.5                     # flat intermediate: coordinate active pursuers regardless of distance
    if rng.random() > p_reply: return 'none'
    return 'amp' if rng.random() < u * shape else 'ack'

def cascade(ctx, u, pos=POS):
    """Run one cascade: returns list of call events (dicts with t, who, u, kind, src)."""
    calls = [dict(t=0., who=0, u=u, kind='orig', src=None)]
    done = {0}; q = [0]
    while q:
        c = calls[q.pop(0)]
        for j in range(len(pos)):
            d = abs(pos[j] - pos[c['who']])
            if j in done or d > AUD: continue
            k = decide(d, c['u'], ctx)
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

# ── Fig 1: one example cascade per context (first seed producing a relay) ────────────────────
fig, ax = plt.subplots(1, 3, figsize=(16, 3.6), sharey=True)
kc = {'orig': 'k', 'amp': '#e67e22', 'ack': '#7f8c8d'}
for a, ctx in zip(ax, COL):
    for seed in range(500):
        rng = np.random.default_rng(seed); cs = cascade(ctx, 1.0)
        if len(cs) >= 3 and cs[-1]['who'] >= 2: break
    for c in cs:
        n = 1 if c['kind'] == 'ack' else NW
        ts = c['t'] + np.arange(n) / c['u']
        a.vlines(ts, c['who'] - .25, c['who'] + .25, color=kc[c['kind']], lw=2.5)
        if c['src'] is not None:
            a.annotate('', (c['t'], c['who']),
                       (cs[[x['who'] for x in cs].index(c['src'])]['t'], c['src']),
                       arrowprops=dict(arrowstyle='->', color='.6', lw=1))
    a.set_yticks(range(4))
    a.set_yticklabels([f'{n}  (x={p})' for n, p in zip('ABCD', POS)])
    a.set_title(ctx.capitalize(), color=COL[ctx], weight='bold')
    a.set_xlabel('time (s)'); a.invert_yaxis()
for k, l in [('orig', 'original bout'), ('amp', 'amplified relay'), ('ack', 'acknowledge')]:
    ax[0].plot([], [], color=kc[k], lw=3, label=l)
ax[0].legend(frameon=False, fontsize=8, loc='lower right')
plt.tight_layout(); plt.savefig('fig1_example_cascade.png', dpi=200); plt.close()

# ── Fig 2: P(reply | heard) vs distance, by context and urgency ─────────────────────────────
ds, N = np.linspace(0, 1, 15), 4000
fig, ax = plt.subplots(1, 3, figsize=(13, 3.4), sharey=True)
for a, ctx in zip(ax, COL):
    for u, ls in [(1.0, '-'), (U0, '--')]:
        r = np.array([[decide(d, u, ctx) for _ in range(N)] for d in ds])
        a.plot(ds, (r == 'ack').mean(1), ls, color='#7f8c8d', lw=2)
        a.plot(ds, (r == 'amp').mean(1), ls, color='#e67e22', lw=2)
    a.set_title(ctx.capitalize(), color=COL[ctx], weight='bold')
    a.set_xlabel('distance to caller (× audible range)')
ax[0].set_ylabel('P(reply | heard)')
for c, l in [('#7f8c8d', 'acknowledge'), ('#e67e22', 'amplify')]:
    ax[0].plot([], [], color=c, lw=2, label=l)
ax[0].plot([], [], 'k-',  label='urgent bout (1 whoop/s)')
ax[0].plot([], [], 'k--', label='baseline (0.2 whoop/s)')
ax[0].legend(frameon=False, fontsize=7)
plt.tight_layout(); plt.savefig('fig2_reply_vs_distance.png', dpi=200); plt.close()

# ── Fig 3: urgency vector — cascade properties vs bout rate ─────────────────────────────────
us, N = np.linspace(U0, 1, 9), 3000
fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
for ctx in COL:
    reach, relay, rate = [], [], []
    for u in us:
        runs = [cascade(ctx, u) for _ in range(N)]
        reach.append(np.mean([max(abs(POS[c['who']] - POS[0]) for c in r) for r in runs]) / AUD)
        relay.append(np.mean([any(c['who'] >= 2 for c in r) for r in runs]))
        rep = [c['u'] / U0 for r in runs for c in r[1:]]
        rate.append(np.mean(rep) if rep else np.nan)
    for a, y in zip(ax, (reach, relay, rate)):
        a.plot(us, y, 'o-', color=COL[ctx], label=ctx)
for a, t in zip(ax, ('Reach (audible ranges)',
                     'P(relay beyond first range)',
                     'Reply rate / baseline')):
    a.set_xlabel('caller bout rate (whoops/s)'); a.set_ylabel(t)
ax[0].legend(frameon=False)
plt.tight_layout(); plt.savefig('fig3_urgency_vector.png', dpi=200); plt.close()
