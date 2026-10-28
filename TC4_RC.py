"""theory/98: set-membership (cone-intersection) rule vs age-discounted hold and other baselines, risk-coverage on between-node frames (Amendment 20).
Reuses locked helpers (tc_cert.calib, tc2_cert.calib_M/stat_series/sprt_llr, tc3_rc.aurc/op_point). Causal rules only.
Methods: cone_all (margin = max(lo - tau, tau - hi) of the interval I(t) = intersection over ALL past nodes of [z_i -+ (e + L|t - t_i|)]; state = centre of I), cone_w5 (last 5 nodes), cert_k1 (last node cone, as in Amendment 19), hold_c (|z_last - tau| / (age + c*), c* tuned on dev_calib over {.5,1,2,3,4,6,8}),
hold, ema, median5, sprt5 (margin |LLR|, window 5). L = M_1 (q = 1.0, V1) and e (q = 1.0) from dev_calib. usage: python -I tc4_rc.py -> experiments/out/tc/result_tc4.json"""
import json, hashlib, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tc_cert as t1  # noqa: E402
import tc2_cert as t2  # noqa: E402
import tc3_rc as t3  # noqa: E402

OUT = HERE / 'out' / 'tc'
DS = ('mutex', 'sirius', 'droid')
MS = (2, 4, 8)
CGRID = (0.5, 1, 2, 3, 4, 6, 8)
NAMES = ('cone_all', 'cone_w5', 'cert_k1', 'hold_c', 'hold', 'ema', 'median5', 'sprt5')


def episode(s, m, fps, tau, e, L, c, mu, sd):
    n = len(s); nodes = np.arange(0, n, m); j = np.searchsorted(nodes, np.arange(n), side='right') - 1
    t = np.arange(n) / fps; tn = nodes / fps; v = s[nodes]
    D = np.abs(t[:, None] - tn[None, :]); past = (nodes[None, :] <= np.arange(n)[:, None])
    up = np.where(past, v[None, :] + e + L * D, np.inf); lo_ = np.where(past, v[None, :] - e - L * D, -np.inf)
    hi_all, lo_all = up.min(1), lo_.max(1)
    idxw = np.arange(len(nodes))[None, :]; w5 = past & (idxw > (j[:, None] - 5))
    hi_w, lo_w = np.where(w5, up, np.inf).min(1), np.where(w5, lo_, -np.inf).max(1)
    age = (np.arange(n) - nodes[j]).astype(float)
    last = v[j]
    out = {}
    for name, lo, hi in (('cone_all', lo_all, hi_all), ('cone_w5', lo_w, hi_w)):
        out[name] = (np.where(hi >= lo, np.maximum(lo - tau, tau - hi), 0.0), (lo + hi) / 2 > tau, np.mean(hi < lo))
    out['cert_k1'] = (np.abs(last - tau) / (e + L * age / fps), last > tau, 0.0)
    out['hold_c'] = (np.abs(last - tau) / (age + c), last > tau, 0.0)
    out['hold'] = (np.abs(last - tau), last > tau, 0.0)
    ema = t2.stat_series(s, m, 'ema'); out['ema'] = (np.abs(ema - tau), ema > tau, 0.0)
    med = t2.stat_series(s, m, 'median5'); out['median5'] = (np.abs(med - tau), med > tau, 0.0)
    llr = t2.sprt_llr(s, m, mu, sd); out['sprt5'] = (np.abs(llr), llr > 0, 0.0)
    return out


def collect(eps_list, z, ep, label, fps, tau, m, e, L, c, mu, sd):
    acc = {k: ([], [], [], []) for k in NAMES}
    for e_i in eps_list:
        s = z[ep == e_i]; l = label[ep == e_i]; n = len(s)
        btw = (np.arange(n) % m != 0) & (l >= 0)
        if not btw.any():
            continue
        r = (s > tau).astype(int)
        for k, (marg, state, empty) in episode(s, m, fps, tau, e, L, c, mu, sd).items():
            dec = state.astype(int)
            acc[k][0].append(marg[btw]); acc[k][1].append((dec != r)[btw]); acc[k][2].append((dec != l)[btw]); acc[k][3].append(np.full(int(btw.sum()), empty))
    return {k: tuple(np.concatenate(v) for v in vs) for k, vs in acc.items()}


def run():
    res = {}
    for ds in DS:
        d = np.load(OUT / f'z_{ds}.npz'); z, ep, label, split, fps, tau = d['z'], d['ep'], d['label'], d['split'], float(d['fps']), float(d['tS'])
        cal, ev = split == 'calib', split == 'eval'
        e_q = t1.calib(z[cal], ep[cal], fps, 2, 1.0)[0]; L = t2.calib_M(z[cal], ep[cal], fps, 1, 1.0, 'V1')
        zc, lc = z[cal], label[cal]; mu = [float(zc[lc == c].mean()) for c in (0, 1)]; sd = [float(zc[lc == c].std()) + 1e-6 for c in (0, 1)]
        res[ds] = dict(fps=fps, e=e_q, L=L, cells=[])
        for m in MS:
            best = min(CGRID, key=lambda c: t3.aurc(*[collect(list(np.unique(ep[cal])), z, ep, label, fps, tau, m, e_q, L, c, mu, sd)['hold_c'][i] for i in (0, 1)]))
            D = collect(list(np.unique(ep[cal])), z, ep, label, fps, tau, m, e_q, L, best, mu, sd)
            E = collect(list(np.unique(ep[ev])), z, ep, label, fps, tau, m, e_q, L, best, mu, sd)
            cell = dict(m=m, c_star=best, n_eval_frames=int(len(E['hold'][0])), empty_frac_cone_all=float(E['cone_all'][3].mean()), methods={})
            for k in NAMES:
                cell['methods'][k] = dict(aurc_reader=t3.aurc(E[k][0], E[k][1]), aurc_label=t3.aurc(E[k][0], E[k][2]),
                                          op={str(eps): t3.op_point(D[k][0], D[k][1], E[k][0], E[k][1], eps) for eps in (0.0, 0.002, 0.01)})
            res[ds]['cells'].append(cell)
            print(ds, 'm', m, 'c*', best, {k: round(v['aurc_reader'], 5) for k, v in cell['methods'].items()}, flush=True)
    p = OUT / 'result_tc4.json'; p.write_text(json.dumps(res, indent=1))
    print('wrote', p, hashlib.sha256(p.read_bytes()).hexdigest())


if __name__ == '__main__':
    run()
