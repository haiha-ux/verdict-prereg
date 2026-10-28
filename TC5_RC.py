"""theory/99: maximum-ratio cone-intersection margin (second stage after tc4). Reuses locked helpers tc_cert.calib, tc2_cert.calib_M/stat_series, tc3_rc.aurc/op_point.
cone_max: margin = max over ALL past nodes of |z_i - tau| / (e + L |t - t_i|), state = side of the maximising node; cone_max_w5: last 5 nodes; cert_k1 (last node ratio), hold_c (tuned age discount on dev_calib), hold. usage: python -I tc5_rc.py -> experiments/out/tc/result_tc5.json"""
import json, hashlib, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tc_cert as t1  # noqa: E402
import tc2_cert as t2  # noqa: E402
import tc3_rc as t3  # noqa: E402

OUT = HERE / 'out' / 'tc'
DS = ('mutex', 'sirius', 'droid'); MS = (2, 4, 8); CGRID = (0.5, 1, 2, 3, 4, 6, 8)
NAMES = ('cone_max', 'cone_max_w5', 'cert_k1', 'hold_c', 'hold')


def episode(s, m, fps, tau, e, L, c):
    n = len(s); nodes = np.arange(0, n, m); j = np.searchsorted(nodes, np.arange(n), side='right') - 1
    t = np.arange(n) / fps; tn = nodes / fps; v = s[nodes]
    D = np.abs(t[:, None] - tn[None, :]); past = nodes[None, :] <= np.arange(n)[:, None]
    ratio = np.abs(v - tau)[None, :] / (e + L * D)
    r_all = np.where(past, ratio, -np.inf); ia = r_all.argmax(1)
    w5 = past & (np.arange(len(nodes))[None, :] > (j[:, None] - 5)); r_w = np.where(w5, ratio, -np.inf); iw = r_w.argmax(1)
    age = (np.arange(n) - nodes[j]).astype(float); last = v[j]
    return {'cone_max': (r_all.max(1), v[ia] > tau), 'cone_max_w5': (r_w.max(1), v[iw] > tau),
            'cert_k1': (np.abs(last - tau) / (e + L * age / fps), last > tau), 'hold_c': (np.abs(last - tau) / (age + c), last > tau), 'hold': (np.abs(last - tau), last > tau)}


def collect(eps_list, z, ep, label, fps, tau, m, e, L, c):
    acc = {k: ([], [], []) for k in NAMES}
    for e_i in eps_list:
        s = z[ep == e_i]; l = label[ep == e_i]; n = len(s)
        btw = (np.arange(n) % m != 0) & (l >= 0)
        if not btw.any():
            continue
        r = (s > tau).astype(int)
        for k, (marg, state) in episode(s, m, fps, tau, e, L, c).items():
            dec = state.astype(int); acc[k][0].append(marg[btw]); acc[k][1].append((dec != r)[btw]); acc[k][2].append((dec != l)[btw])
    return {k: tuple(np.concatenate(v) for v in vs) for k, vs in acc.items()}


def run():
    res = {}
    for ds in DS:
        d = np.load(OUT / f'z_{ds}.npz'); z, ep, label, split, fps, tau = d['z'], d['ep'], d['label'], d['split'], float(d['fps']), float(d['tS'])
        cal, ev = split == 'calib', split == 'eval'
        e_q = t1.calib(z[cal], ep[cal], fps, 2, 1.0)[0]; L = t2.calib_M(z[cal], ep[cal], fps, 1, 1.0, 'V1')
        res[ds] = dict(fps=fps, e=e_q, L=L, cells=[])
        for m in MS:
            best = min(CGRID, key=lambda c: t3.aurc(*[collect(list(np.unique(ep[cal])), z, ep, label, fps, tau, m, e_q, L, c)['hold_c'][i] for i in (0, 1)]))
            D = collect(list(np.unique(ep[cal])), z, ep, label, fps, tau, m, e_q, L, best); E = collect(list(np.unique(ep[ev])), z, ep, label, fps, tau, m, e_q, L, best)
            cell = dict(m=m, c_star=best, n_eval_frames=int(len(E['hold'][0])), methods={})
            for k in NAMES:
                cell['methods'][k] = dict(aurc_reader=t3.aurc(E[k][0], E[k][1]), aurc_label=t3.aurc(E[k][0], E[k][2]),
                                          op={str(eps): t3.op_point(D[k][0], D[k][1], E[k][0], E[k][1], eps) for eps in (0.0, 0.002, 0.01)})
            res[ds]['cells'].append(cell)
            print(ds, 'm', m, 'c*', best, {k: round(v['aurc_reader'], 5) for k, v in cell['methods'].items()}, flush=True)
    p = OUT / 'result_tc5.json'; p.write_text(json.dumps(res, indent=1))
    print('wrote', p, hashlib.sha256(p.read_bytes()).hexdigest())


if __name__ == '__main__':
    run()
