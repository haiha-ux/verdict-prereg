"""theory/95: temporal smoothness certificate on dense reader logits. numpy only.
For each dataset: calibrate (e_q, M_k,q) on dev_calib episodes; on eval episodes observe every m-th frame (nodes), enclose the latent score between nodes by Lagrange interpolation
of k nodes + Lagrange remainder M/k! prod|t-tj| + e * Lebesgue function; decide 1 / 0 / abstain against tS; score on ALL native frames with label >= 0 (open = 1).
Baseline at equal abstention: linear interpolation of the nodes with a band b chosen (per config) to match the abstention fraction. Episode bootstrap CI of the error among non-abstained decisions.
usage: python -I tc_cert.py -> experiments/out/tc/result_tc.json"""
import json, hashlib, math
from pathlib import Path
import numpy as np
from scipy.signal import savgol_filter

HERE = Path(__file__).resolve().parent
OUT = HERE / 'out' / 'tc'
DS = ('mutex', 'sirius', 'droid')
KS, QS, MS = (2, 3, 4), (0.9, 0.99, 1.0), (1, 2, 4, 8, 16)
B = 1000
SEED = 20261028


def calib(z, ep, fps, k, q):
    e_res, dk = [], []
    for e in np.unique(ep):
        s = z[ep == e]
        if len(s) < 15:
            continue
        sh = savgol_filter(s, 9, 3)
        e_res.append(np.abs(s - sh))
        d = sh.copy()
        for _ in range(k):
            d = np.diff(d)
        dk.append(np.abs(d) * fps ** k)
    return float(np.quantile(np.concatenate(e_res), q)), float(np.quantile(np.concatenate(dk), q))


def lag_basis(tn, t):
    """Lagrange basis values l_j(t) for nodes tn."""
    L = np.ones(len(tn))
    for j in range(len(tn)):
        for i in range(len(tn)):
            if i != j:
                L[j] *= (t - tn[i]) / (tn[j] - tn[i])
    return L


def episode_decisions(s, m, k, e, M, fps, tau):
    """returns (decision array in {1, 0, -1 abstain}, centre value p) for every native frame of one episode."""
    n = len(s); nodes = np.arange(0, n, m)
    if len(nodes) < k:
        nodes = np.arange(0, n, max(1, n // k))
    dec = np.full(n, -1); pc = np.zeros(n)
    fact = float(math.factorial(k))
    for t in range(n):
        j = int(np.searchsorted(nodes, t, side='right')) - 1
        lo = min(max(j - (k - 1) // 2, 0), len(nodes) - k)
        idx = nodes[lo:lo + k]; tn = idx / fps; tt = t / fps
        L = lag_basis(tn, tt)
        p = float(L @ s[idx])
        w = M / fact * float(np.prod(np.abs(tt - tn))) + e * float(np.abs(L).sum())
        pc[t] = p
        if p - w > tau:
            dec[t] = 1
        elif p + w < tau:
            dec[t] = 0
    return dec, pc


def baseline(s, m, fps, tau, b):
    n = len(s); nodes = np.arange(0, n, m)
    lin = np.interp(np.arange(n), nodes, s[nodes])
    return np.where(lin - b > tau, 1, np.where(lin + b < tau, 0, -1))


def stats(decs, labs, eps_ids, rng):
    """error among non-abstained and abstain fraction over labelled frames, episode bootstrap CI of the error."""
    per = []
    for d, l in zip(decs, labs):
        m = l >= 0
        per.append((int(((d != l) & (d >= 0) & m).sum()), int(((d >= 0) & m).sum()), int(m.sum())))
    per = np.array(per, float)
    err = per[:, 0].sum() / max(per[:, 1].sum(), 1); ab = 1 - per[:, 1].sum() / per[:, 2].sum()
    bs = []
    for _ in range(B):
        i = rng.integers(0, len(per), len(per)); p = per[i]
        bs.append(p[:, 0].sum() / max(p[:, 1].sum(), 1))
    return float(err), float(ab), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], int(per[:, 0].sum()), int(per[:, 1].sum())


def run():
    res = {}
    for ds in DS:
        d = np.load(OUT / f'z_{ds}.npz')
        z, ep, label, split, fps, tau = d['z'], d['ep'], d['label'], d['split'], float(d['fps']), float(d['tS'])
        cal = split == 'calib'
        ev_eps = [e for e in np.unique(ep[split == 'eval'])]
        res[ds] = dict(fps=fps, tau=tau, n_eval_eps=len(ev_eps), cells=[])
        for k in KS:
            for q in QS:
                e_q, M_q = calib(z[cal], ep[cal], fps, k, q)
                for m in MS:
                    decs, labs, bdec = [], [], []
                    for ei in ev_eps:
                        s = z[ep == ei]; l = label[ep == ei]
                        dd, _ = episode_decisions(s, m, k, e_q, M_q, fps, tau)
                        decs.append(dd); labs.append(l)
                    rng = np.random.default_rng([SEED, k, int(q * 100), m])
                    err, ab, ci, nerr, nn = stats(decs, labs, ev_eps, rng)
                    # baseline with the same abstention
                    lo, hi = 0.0, 20.0
                    for _ in range(25):
                        b = (lo + hi) / 2
                        fr = np.mean(np.concatenate([baseline(z[ep == ei], m, fps, tau, b) == -1 for ei in ev_eps]))
                        lo, hi = (b, hi) if fr < ab else (lo, b)
                    bd = [baseline(z[ep == ei], m, fps, tau, (lo + hi) / 2) for ei in ev_eps]
                    berr, bab, bci, bnerr, bnn = stats(bd, labs, ev_eps, np.random.default_rng([SEED, 1, k, m]))
                    res[ds]['cells'].append(dict(k=k, q=q, m=m, e=e_q, M=M_q, err=err, abstain=ab, ci=ci, n_err=nerr, n_decided=nn, base_err=berr, base_abstain=bab, base_ci=bci))
                    print(ds, k, q, m, 'err', round(err, 4), 'abst', round(ab, 3), '| base err', round(berr, 4), 'abst', round(bab, 3), flush=True)
    p = OUT / 'result_tc.json'
    p.write_text(json.dumps(res, indent=1))
    print('wrote', p, hashlib.sha256(p.read_bytes()).hexdigest())


if __name__ == '__main__':
    run()
