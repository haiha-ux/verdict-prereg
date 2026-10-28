"""theory/97: risk-coverage test of the temporal certificate on BETWEEN-NODE frames (Amendment 19). Imports locked helpers of tc_cert / tc2_cert (calib, calib_M, lag_basis, stat_series).
Every method gives each between-node labelled frame a margin (abstain below a threshold theta); we sweep theta. Methods (causal, online): cert_k1 / cert_k2 (margin = |p - tau| / w with the q = 1.0, V1 radius from dev_calib), hold (|last node - tau|), ema (|ema - tau|, alpha .5).
Metrics: AURC (mean over coverage levels of the cumulative risk of the most confident frames) against the reader's own thresholded output 1[z_t > tau] and against the sensor label; dev-calibrated operating points: theta* = largest coverage with dev_calib risk <= eps (eps in {0, .002, .01}), then eval coverage and eval risk (reader target).
usage: python -I tc3_rc.py -> experiments/out/tc/result_tc3.json"""
import json, hashlib, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tc_cert as t1  # noqa: E402
import tc2_cert as t2  # noqa: E402

OUT = HERE / 'out' / 'tc'
DS = ('mutex', 'sirius', 'droid')
MS = (2, 4, 8)
EPS = (0.0, 0.002, 0.01)


def margins(s, m, k, e, M, fps, tau):
    """causal margins for every native frame: returns (ratio |p - tau| / w, p) arrays; k = 1 is the last node with a Lipschitz radius."""
    n = len(s); nodes = np.arange(0, n, m); ratio = np.zeros(n)
    for t in range(n):
        j = int(np.searchsorted(nodes, t, side='right')) - 1
        lo = max(j - k + 1, 0); idx = nodes[lo:j + 1]; kk = len(idx); tn = idx / fps; tt = t / fps
        L = t1.lag_basis(tn, tt)
        p = float(L @ s[idx]); w = M / float(math.factorial(kk)) * float(np.prod(np.abs(tt - tn))) + e * float(np.abs(L).sum())
        ratio[t] = abs(p - tau) / w
    return ratio


def collect(eps_list, z, ep, label, fps, tau, m, e, M):
    """per method: arrays of margin, err_reader, err_label over between-node labelled frames of the given episodes."""
    out = {k: ([], [], []) for k in ('cert_k1', 'cert_k2', 'hold', 'ema')}
    for e_i in eps_list:
        s = z[ep == e_i]; l = label[ep == e_i]; n = len(s)
        btw = (np.arange(n) % m != 0) & (l >= 0)
        if not btw.any():
            continue
        r = (s > tau).astype(int)
        marg = {'cert_k1': margins(s, m, 1, e, M[1], fps, tau), 'cert_k2': margins(s, m, 2, e, M[2], fps, tau),
                'hold': np.abs(t2.stat_series(s, m, 'hold') - tau), 'ema': np.abs(t2.stat_series(s, m, 'ema') - tau)}
        # predicted state of each rule = side of its centre; for cert_k* the centre p (sign of p - tau); recompute cheaply from the same nodes
        nodes = np.arange(0, n, m)
        cen = {'hold': t2.stat_series(s, m, 'hold'), 'ema': t2.stat_series(s, m, 'ema')}
        for k in (1, 2):
            pc = np.zeros(n)
            for t in range(n):
                j = int(np.searchsorted(nodes, t, side='right')) - 1
                idx = nodes[max(j - k + 1, 0):j + 1]
                pc[t] = float(t1.lag_basis(idx / fps, t / fps) @ s[idx])
            cen[f'cert_k{k}'] = pc
        for name in out:
            dec = (cen[name] > tau).astype(int)
            out[name][0].append(marg[name][btw]); out[name][1].append((dec != r)[btw]); out[name][2].append((dec != l)[btw])
    return {k: tuple(np.concatenate(v) for v in vs) for k, vs in out.items()}


def aurc(marg, err):
    o = np.argsort(-marg, kind='stable'); c = np.cumsum(err[o]) / np.arange(1, len(err) + 1)
    return float(c.mean())


def op_point(marg_dev, err_dev, marg_ev, err_ev, eps):
    """theta*: smallest margin threshold such that dev risk of frames with margin >= theta is <= eps; apply to eval."""
    o = np.argsort(-marg_dev, kind='stable'); c = np.cumsum(err_dev[o]) / np.arange(1, len(o) + 1)
    ok = np.flatnonzero(c <= eps + 1e-12)
    if len(ok) == 0:
        return dict(theta=None, cov_dev=0.0, cov_eval=0.0, risk_eval=None)
    theta = float(marg_dev[o][ok[-1]]); sel = marg_ev >= theta
    return dict(theta=theta, cov_dev=float((ok[-1] + 1) / len(o)), cov_eval=float(sel.mean()), risk_eval=float(err_ev[sel].mean()) if sel.any() else None)


def run():
    res = {}
    for ds in DS:
        d = np.load(OUT / f'z_{ds}.npz'); z, ep, label, split, fps, tau = d['z'], d['ep'], d['label'], d['split'], float(d['fps']), float(d['tS'])
        cal, ev = split == 'calib', split == 'eval'
        e_q = t1.calib(z[cal], ep[cal], fps, 2, 1.0)[0]
        M = {k: t2.calib_M(z[cal], ep[cal], fps, k, 1.0, 'V1') for k in (1, 2)}
        res[ds] = dict(fps=fps, e=e_q, M=M, cells=[])
        for m in MS:
            D = collect(list(np.unique(ep[cal])), z, ep, label, fps, tau, m, e_q, M)
            E = collect(list(np.unique(ep[ev])), z, ep, label, fps, tau, m, e_q, M)
            cell = dict(m=m, n_eval_frames=int(len(E['hold'][0])), n_dev_frames=int(len(D['hold'][0])), methods={})
            for name in E:
                cell['methods'][name] = dict(aurc_reader=aurc(E[name][0], E[name][1]), aurc_label=aurc(E[name][0], E[name][2]),
                                             op={str(eps): op_point(D[name][0], D[name][1], E[name][0], E[name][1], eps) for eps in EPS})
            res[ds]['cells'].append(cell)
            print(ds, 'm', m, {n: round(v['aurc_reader'], 5) for n, v in cell['methods'].items()}, flush=True)
    p = OUT / 'result_tc3.json'
    p.write_text(json.dumps(res, indent=1))
    print('wrote', p, hashlib.sha256(p.read_bytes()).hexdigest())


if __name__ == '__main__':
    run()
