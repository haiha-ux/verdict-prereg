"""theory/96: second, tighter test of the temporal certificate (Amendment 18). Imports the locked tc_cert helpers (calib for e, lag_basis).
Changes vs tc_cert.py: (i) M estimator V0 (finite differences of SavGol(9,3), as before) and V1 (SavGol derivative filter, window 21, poly k+2, q-quantile of |s^(k)|); (ii) causal (online: last k nodes up to t) and retrospective (k nodes around t) variants, k in {1,2,3};
(iii) baselines matched to the certificate's abstained fraction ON LABELLED FRAMES: linear interpolation band (retrospective), causal hold, causal EMA, causal median-of-5, causal SPRT (window 5, Gaussian class models of the logit from dev_calib); (iv) two errors: vs the sensor label (open = 1, label >= 0) and vs the reader's own thresholded output 1[z_t > tau] on the same frames.
usage: python -I tc2_cert.py -> experiments/out/tc/result_tc2.json"""
import json, hashlib, math, sys
from pathlib import Path
import numpy as np
from scipy.signal import savgol_filter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tc_cert as t1  # noqa: E402  (locked helpers)

OUT = HERE / 'out' / 'tc'
DS = ('mutex', 'sirius', 'droid')
KS, QS, MS, VARS, MODES = (1, 2, 3), (0.99, 1.0), (2, 4, 8), ('V0', 'V1'), ('retro', 'causal')
B = 1000
SEED = 20261028
WIN = 5


def calib_M(z, ep, fps, k, q, var):
    if var == 'V0':
        return t1.calib(z, ep, fps, k, q)[1]
    dk = []
    for e in np.unique(ep):
        s = z[ep == e]
        if len(s) < 25:
            continue
        dk.append(np.abs(savgol_filter(s, 21, k + 2, deriv=k, delta=1.0 / fps)))
    return float(np.quantile(np.concatenate(dk), q))


def decide(s, m, k, e, M, fps, tau, causal):
    n = len(s); nodes = np.arange(0, n, m); dec = np.full(n, -1); fact = float(math.factorial(k))
    for t in range(n):
        j = int(np.searchsorted(nodes, t, side='right')) - 1
        if causal:
            lo = max(j - k + 1, 0); idx = nodes[lo:j + 1]
        else:
            lo = min(max(j - (k - 1) // 2, 0), len(nodes) - k); idx = nodes[lo:lo + k]
        kk = len(idx); tn = idx / fps; tt = t / fps
        L = t1.lag_basis(tn, tt)
        p = float(L @ s[idx]); w = M / float(math.factorial(kk)) * float(np.prod(np.abs(tt - tn))) + e * float(np.abs(L).sum())
        dec[t] = 1 if p - w > tau else (0 if p + w < tau else -1)
    return dec


def stat_series(s, m, kind):
    """baseline statistic for every native frame."""
    n = len(s); nodes = np.arange(0, n, m); v = s[nodes]
    if kind == 'linear':
        return np.interp(np.arange(n), nodes, v)
    j = np.searchsorted(nodes, np.arange(n), side='right') - 1
    if kind == 'hold':
        return v[j]
    if kind == 'ema':
        ema = np.zeros(len(v)); ema[0] = v[0]
        for i in range(1, len(v)):
            ema[i] = 0.5 * v[i] + 0.5 * ema[i - 1]
        return ema[j]
    if kind == 'median5':
        med = np.array([np.median(v[max(0, i - WIN + 1):i + 1]) for i in range(len(v))])
        return med[j]
    raise ValueError(kind)


def sprt_llr(s, m, mu, sd):
    n = len(s); nodes = np.arange(0, n, m); v = s[nodes]
    ll = (-(v - mu[1]) ** 2 / (2 * sd[1] ** 2) - np.log(sd[1])) - (-(v - mu[0]) ** 2 / (2 * sd[0] ** 2) - np.log(sd[0]))
    cs = np.array([ll[max(0, i - WIN + 1):i + 1].sum() for i in range(len(v))])
    j = np.searchsorted(nodes, np.arange(n), side='right') - 1
    return cs[j]


def metrics(decs, labs, zs, tau, rng):
    per = []
    for d, l, z in zip(decs, labs, zs):
        m = l >= 0; r = (z > tau).astype(int); dd = d >= 0
        per.append((int(((d != l) & dd & m).sum()), int(((d != r) & dd & m).sum()), int((dd & m).sum()), int(m.sum())))
    p = np.array(per, float)
    out = dict(err_label=float(p[:, 0].sum() / max(p[:, 2].sum(), 1)), err_reader=float(p[:, 1].sum() / max(p[:, 2].sum(), 1)), abstain=float(1 - p[:, 2].sum() / p[:, 3].sum()), n_err_reader=int(p[:, 1].sum()), n_decided=int(p[:, 2].sum()))
    bs = []
    for _ in range(B):
        q = p[rng.integers(0, len(p), len(p))]
        bs.append(q[:, 1].sum() / max(q[:, 2].sum(), 1))
    out['err_reader_ci'] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    return out


def match(fn, target, lo, hi):
    """bisection on a monotone parameter so that the abstained fraction on labelled frames equals target; fn(param) -> abstained fraction."""
    for _ in range(30):
        mid = (lo + hi) / 2
        if fn(mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def run():
    res = {}
    for ds in DS:
        d = np.load(OUT / f'z_{ds}.npz')
        z, ep, label, split, fps, tau = d['z'], d['ep'], d['label'], d['split'], float(d['fps']), float(d['tS'])
        cal = split == 'calib'; ev = list(np.unique(ep[split == 'eval']))
        S = {e: z[ep == e] for e in ev}; Lb = {e: label[ep == e] for e in ev}; Z = [S[e] for e in ev]; LL = [Lb[e] for e in ev]
        zc, lc = z[cal], label[cal]
        mu = [float(zc[lc == c].mean()) for c in (0, 1)]; sd = [float(zc[lc == c].std()) + 1e-6 for c in (0, 1)]
        res[ds] = dict(fps=fps, tau=tau, n_eval_eps=len(ev), mu=mu, sd=sd, cells=[])
        frac = lambda decs: 1 - sum(int(((dd >= 0) & (l >= 0)).sum()) for dd, l in zip(decs, LL)) / sum(int((l >= 0).sum()) for l in LL)
        for k in KS:
            for q in QS:
                e_q = t1.calib(z[cal], ep[cal], fps, 2, q)[0]
                for var in VARS:
                    M = calib_M(z[cal], ep[cal], fps, k, q, var)
                    for mode in MODES:
                        for m in MS:
                            decs = [decide(s, m, k, e_q, M, fps, tau, mode == 'causal') for s in Z]
                            rng = np.random.default_rng([SEED, 2, k, int(q * 100), m, VARS.index(var), MODES.index(mode)])
                            cert = metrics(decs, LL, Z, tau, rng); ab = cert['abstain']
                            bases = {}
                            kinds = ('linear',) if mode == 'retro' else ('hold', 'ema', 'median5')
                            for kind in kinds:
                                st = [stat_series(s, m, kind) for s in Z]
                                f = lambda b, st=st: frac([np.where(a - b > tau, 1, np.where(a + b < tau, 0, -1)) for a in st])
                                b = match(f, ab, 0.0, 30.0)
                                bases[kind] = metrics([np.where(a - b > tau, 1, np.where(a + b < tau, 0, -1)) for a in st], LL, Z, tau, np.random.default_rng([SEED, 3, k, m]))
                            if mode == 'causal':
                                st = [sprt_llr(s, m, mu, sd) for s in Z]
                                f = lambda A, st=st: frac([np.where(a > A, 1, np.where(a < -A, 0, -1)) for a in st])
                                A = match(f, ab, 0.0, 200.0)
                                bases['sprt5'] = metrics([np.where(a > A, 1, np.where(a < -A, 0, -1)) for a in st], LL, Z, tau, np.random.default_rng([SEED, 4, k, m]))
                            res[ds]['cells'].append(dict(k=k, q=q, var=var, mode=mode, m=m, e=e_q, M=M, cert=cert, base=bases))
                            print(ds, k, q, var, mode, m, 'cert err_reader', round(cert['err_reader'], 4), 'abst', round(ab, 3), '| ' + ' '.join(f'{b} {v["err_reader"]:.4f}/{v["abstain"]:.3f}' for b, v in bases.items()), flush=True)
    p = OUT / 'result_tc2.json'
    p.write_text(json.dumps(res, indent=1))
    print('wrote', p, hashlib.sha256(p.read_bytes()).hexdigest())


if __name__ == '__main__':
    run()
