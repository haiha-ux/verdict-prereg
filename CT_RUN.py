"""theory/90 stages `feat <set>` and `run`: DINOv2-small CLS+mean features of the fetched frames, site cross-fitted linear probe,
threshold rules R0/Q/Otsu/oracle, claims C1-C4. Written and locked before any feature was computed.
usage: python -I ct_run.py feat <set> | run
"""
import json, sys, hashlib
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import t2_run as t2  # noqa: E402  (fit_probe, logits, la_threshold, cp_upper unchanged)
import theory82_check as th82  # noqa: E402  (wsr_upper: cluster betting bound, unchanged)
import theory84_check as th84  # noqa: E402  (Src, regret_matrix: identified-set certificate, unchanged)

ROOT = HERE.parent
DATA, OUT = ROOT / 'data', HERE / 'out' / 'camtrap'
SETS = ('cct', 'wel', 'ss3')
SEED = 20261028
K = 5
B = 2000
BAND = 0.30
KS = (5, 10, 20)  # audit sizes in SITES (clarification of the plan's 'm'; 25 frames per site)
DRAWS = 2000
ALPHA = 0.05


def load(name):
    m = json.load(open(DATA / f'ct_{name}_manifest.json'))
    ok = np.load(DATA / f'ct_{name}_ok.npy')
    rows = [r for r, o in zip(m['rows'], ok) if o]
    keep = np.flatnonzero(ok)
    return rows, keep


def stage_feat(name):
    import h113_run as h13
    from PIL import Image
    rows, keep = load(name)
    arr = np.load(DATA / f'ct_{name}_frames.npy', mmap_mode='r')
    F = []
    for a in range(0, len(keep), 1000):
        F.append(h13.embed([Image.fromarray(np.asarray(arr[i])) for i in keep[a:a + 1000]]).astype(np.float32))
    np.savez(OUT / f'feat_{name}.npz', F=np.concatenate(F), keep=keep)
    print(name, 'features', sum(len(f) for f in F))


# ---------------------------------------------------------------- rules (label-free except oracle)
def otsu(z):
    h, e = np.histogram(z, 64)
    p = h / h.sum()
    c = (e[:-1] + e[1:]) / 2
    w0, w1 = np.cumsum(p), 1 - np.cumsum(p)
    m0 = np.cumsum(p * c) / np.maximum(w0, 1e-12)
    m1 = (np.sum(p * c) - np.cumsum(p * c)) / np.maximum(w1, 1e-12)
    return float(c[np.argmax(w0 * w1 * (m0 - m1) ** 2)])


def thresholds(z, y, tS, rho):
    """R0 = source threshold; Q = quantile matching to the source positive rate rho; Otsu; oracle (uses labels). Positive = z > t."""
    tQ = float(np.quantile(z, 1 - rho))
    return dict(R0=tS, Q=tQ, Otsu=otsu(z), oracle=t2.la_threshold(z, y))


def errs(z, y, th):
    return {k: ((z > t) != y).astype(np.int8) for k, t in th.items()}


def fit_fold(F, y, site, tr_sites, rng):
    """probe from `tr_sites` (75% train / 25% calibration, by site); returns probe dict."""
    s = np.array(sorted(tr_sites))
    rng.shuffle(s)
    cal = set(s[: max(1, len(s) // 4)])
    tr = np.array([x not in cal for x in site])
    ca = np.array([x in cal for x in site])
    m = np.isin(site, s)
    P = t2.fit_probe(F[m & tr], y[m & tr], F[m & ca], y[m & ca])
    P['ycal'] = y[m & ca]
    return P


def cert(P, z, tS_rules, y_cal_z=None):
    """identified-set certificate rho(r) = max_p Reg_p(r) over p in [pi_dev -+ BAND], source class atoms from the calibration scores."""
    zc, yc = P['zcal'], P['ycal']
    edges = np.unique(np.quantile(zc, np.linspace(0, 1, 41)))
    edges[0], edges[-1] = -np.inf, np.inf
    m1 = np.histogram(zc[yc == 1], edges)[0].astype(float) + 0.5
    m0 = np.histogram(zc[yc == 0], edges)[0].astype(float) + 0.5
    src = th84.Src(m1 / m1.sum(), m0 / m0.sum())
    pd_ = float(yc.mean())
    ps = np.linspace(max(0.02, pd_ - BAND), min(0.98, pd_ + BAND), 13)
    out = {}
    for k, t in tS_rules.items():
        r = float((z <= t).mean())
        Reg, _, _ = th84.regret_matrix(src, ps, np.array([r]))
        out[k] = float(Reg.max())
    return out


def run_set(name, rows, F):
    y = np.array([r['label_animal'] for r in rows])
    site = np.array([r['site'] for r in rows])
    role = np.array([r['role'] for r in rows])
    share = np.array([r['site_empty_share'] for r in rows])
    fold = np.array([int(r.split('_')[0][1:]) for r in role])
    cases, frames = [], []
    for f in range(K):
        rng = np.random.default_rng([SEED, SETS.index(name), f])
        dev_sites = set(site[fold != f])
        P = fit_fold(F, y, site, dev_sites, rng)
        pools = {'P0': (fold == f)}
        for tag, cond in (('PH', share >= .60), ('PL', share <= .20)):
            pools[tag] = (fold == f) & cond
        for pool, m in pools.items():
            ev = m & np.array([x.endswith('eval') for x in role])
            if ev.sum() < 150 or min(y[ev].sum(), (1 - y[ev]).sum()) < 10:
                continue
            z = t2.logits(P, F[ev]); yy = y[ev]
            th = thresholds(z, yy, P['tS'], P['rho'])
            e = errs(z, yy, th)
            cz = cert(P, z, {k: th[k] for k in ('R0', 'Q')})
            prior_shift = float(yy.mean() - P['ycal'].mean())
            cases.append(dict(set=name, fold=f, pool=pool, n=int(ev.sum()), n_sites=int(len(set(site[ev]))), prior_shift=prior_shift,
                              err={k: float(v.mean()) for k, v in e.items()}, auc=t2.auc(z, yy), cert=cz,
                              regret={k: float(e[k].mean() - e['oracle'].mean()) for k in ('R0', 'Q', 'Otsu')}))
            frames.append(dict(set=name, fold=f, pool=pool, site=site[ev], e={k: v for k, v in e.items()}))
        # audit data for C4 (R0 of this fold): audit-pool frames + per-site clusters
        au = (fold == f) & np.array([x.endswith('audit') for x in role])
        za = t2.logits(P, F[au]); ya = y[au]
        frames.append(dict(set=name, fold=f, pool='AUD', site=site[au], e=dict(R0=((za > P['tS']) != ya).astype(np.int8))))
    return cases, frames


def site_boot(frames, set_, pool, stat, rng, B=B):
    """bootstrap over sites inside (set, pool): frames of a resampled site are all included; stat(dict k -> concatenated 0/1 arrays)"""
    fs = [f for f in frames if f['set'] == set_ and f['pool'] == pool]
    if not fs:
        return None
    sites = sorted({(f['fold'], s) for f in fs for s in set(f['site'])})
    idx = {}
    for f in fs:
        for s in set(f['site']):
            idx[(f['fold'], s)] = {k: v[f['site'] == s] for k, v in f['e'].items()}
    vals = []
    for _ in range(B):
        pick = rng.integers(0, len(sites), len(sites))
        d = {k: np.concatenate([idx[sites[i]][k] for i in pick]) for k in idx[sites[0]]}
        vals.append(stat(d))
    v = np.array(vals)
    pt = stat({k: np.concatenate([idx[s][k] for s in sites]) for k in idx[sites[0]]})
    return dict(point=float(pt), lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5)), n_sites=len(sites))


def audit_validity(frames, set_, rng):
    """C4: R0 of each fold; clusters = sites; truth = pooled R0 error on the evaluated (P0) frames of all folds;
    audit sample = k audit sites; betting bound (cluster) vs frame-level Clopper-Pearson on the same frames."""
    ev = [f for f in frames if f['set'] == set_ and f['pool'] == 'P0']
    au = [f for f in frames if f['set'] == set_ and f['pool'] == 'AUD']
    truth = float(np.concatenate([f['e']['R0'] for f in ev]).mean())
    cl = [(int((f['e']['R0'][f['site'] == s]).sum()), int((f['site'] == s).sum())) for f in au for s in set(f['site'])]
    cl = np.array(cl, float)
    res = {}
    for k in KS:
        if len(cl) < k:
            continue
        ub, uc = [], []
        for _ in range(DRAWS):
            c = cl[rng.choice(len(cl), k, replace=False)]
            ub.append(float(th82.wsr_upper(c[:, 0][None], c[:, 1][None], ALPHA)[0]))
            uc.append(float(t2.cp_upper(c[:, 0].sum(), c[:, 1].sum(), ALPHA)))
        ub, uc = np.array(ub), np.array(uc)
        res[k] = dict(truth=truth, betting_cover=float((truth <= ub).mean()), cp_cover=float((truth <= uc).mean()), betting_U=float(ub.mean()), cp_U=float(uc.mean()))
    return res


def stage_run():
    res = dict(sets={}, claims={})
    allcases, allframes, Fs = [], [], {}
    for name in SETS:
        rows, keep = load(name)
        z = np.load(OUT / f'feat_{name}.npz')
        assert (z['keep'] == keep).all()
        Fs[name] = (rows, z['F'])
        c, f = run_set(name, rows, z['F'])
        allcases += c; allframes += f
    rng = np.random.default_rng([SEED, 9])
    # C1: descriptive, P0 per set
    c1 = {}
    for name in SETS:
        cs = [c for c in allcases if c['set'] == name and c['pool'] == 'P0']
        c1[name] = dict(n_folds=len(cs), **{f'regret_{k}': float(np.mean([c['regret'][k] for c in cs])) for k in ('R0', 'Q', 'Otsu')},
                        auc=float(np.mean([c['auc'] for c in cs])), q_beats_r0=int(sum(c['regret']['Q'] < c['regret']['R0'] for c in cs)))
    # C2: prior-shifted pools (|shift| >= .2), mean of err_Q - err_R0, site bootstrap pooled over sets
    shifted = [(c['set'], c['pool'], c['fold']) for c in allcases if c['pool'] in ('PH', 'PL') and abs(c['prior_shift']) >= .20]
    d = {}
    for (s, p, f) in shifted:
        fr = next(x for x in allframes if x['set'] == s and x['pool'] == p and x['fold'] == f)
        d.setdefault(s, []).append(fr)
    sh = [x for v in d.values() for x in v]
    if sh:
        sites = [(i, s) for i, x in enumerate(sh) for s in set(x['site'])]
        diff = lambda picks: float(np.mean(np.concatenate([(sh[i]['e']['Q'] - sh[i]['e']['R0'])[sh[i]['site'] == s] for i, s in picks])))
        pt = diff(sites)
        vals = [diff([sites[j] for j in rng.integers(0, len(sites), len(sites))]) for _ in range(B)]
        c2 = dict(n_pools=len(sh), n_sites=len(sites), mean_Q_minus_R0=pt, lo=float(np.percentile(vals, 2.5)), hi=float(np.percentile(vals, 97.5)),
                  passed=bool(pt >= .02 and np.percentile(vals, 2.5) > 0))
    else:
        c2 = dict(n_pools=0, note='no pool with |prior shift| >= .20')
    # C3: certificate coverage of realised regret
    cov = [(c['regret'][k] <= c['cert'][k] + 1e-12, c['cert'][k]) for c in allcases for k in ('R0', 'Q')]
    c3 = dict(n_cases=len(cov), coverage=float(np.mean([a for a, _ in cov])) if cov else None, mean_width=float(np.mean([w for _, w in cov])) if cov else None,
              informative=bool(cov and np.mean([w for _, w in cov]) <= .30))
    # C4
    c4 = {name: audit_validity(allframes, name, np.random.default_rng([SEED, 10, i])) for i, name in enumerate(SETS)}
    res['cases'], res['claims'] = allcases, dict(C1=c1, C2=c2, C3=c3, C4=c4)
    p = OUT / 'result_ct.json'
    p.write_text(json.dumps(res, indent=1, default=float))
    print('wrote', p, hashlib.sha256(p.read_bytes()).hexdigest())
    print(json.dumps(res['claims'], indent=1, default=float)[:3000])


if __name__ == '__main__':
    if sys.argv[1] == 'feat':
        stage_feat(sys.argv[2])
    elif sys.argv[1] == 'run':
        stage_run()
