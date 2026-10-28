"""Amendment 16: PX cross-set pools. Probe trained on ALL sites of set A (75/25 train/calibration by site, 10 calibration seeds), evaluated on ALL frames of set B (A != B).
Rules R0/Q/Otsu/oracle, constant-rule error, AUC, identified-set certificate (band +-.30 around the calibration prior of A) for R0 and Q. Reuses ct_run functions unchanged.
usage: python -I ct_px.py -> experiments/out/camtrap/result_px.json"""
import json, sys, hashlib, itertools
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ct_run as cr  # noqa: E402
import t2_run as t2  # noqa: E402

SEEDS = 10


def main():
    D = {}
    for name in cr.SETS:
        rows, keep = cr.load(name)
        z = np.load(cr.OUT / f'feat_{name}.npz'); assert (z['keep'] == keep).all()
        D[name] = (np.array([r['label_animal'] for r in rows]), np.array([r['site'] for r in rows]), z['F'])
    out = []
    for a, b in itertools.permutations(cr.SETS, 2):
        ya, sa, Fa = D[a]; yb, sb, Fb = D[b]
        reps = []
        for s in range(SEEDS):
            rng = np.random.default_rng([cr.SEED, 99, cr.SETS.index(a), cr.SETS.index(b), s])
            P = cr.fit_fold(Fa, ya, sa, set(sa), rng)
            z = t2.logits(P, Fb)
            th = cr.thresholds(z, yb, P['tS'], P['rho'])
            e = cr.errs(z, yb, th)
            cz = cr.cert(P, z, {k: th[k] for k in ('R0', 'Q')})
            reg = {k: float(e[k].mean() - e['oracle'].mean()) for k in ('R0', 'Q', 'Otsu')}
            reps.append(dict(seed=s, auc=t2.auc(z, yb), prior_dev=float(P['ycal'].mean()), prior_target=float(yb.mean()), err={k: float(v.mean()) for k, v in e.items()},
                             err_const=float(min(yb.mean(), 1 - yb.mean())), err_const_animal=float(1 - yb.mean()), regret=reg, cert=cz,
                             covered={k: bool(reg[k] <= cz[k] + 1e-12) for k in ('R0', 'Q')}))
        out.append(dict(src=a, tgt=b, n_target=int(len(yb)), n_target_sites=int(len(set(sb))), reps=reps))
        m = lambda f: round(float(np.mean([f(r) for r in reps])), 3)
        print(a, '->', b, 'AUC', m(lambda r: r['auc']), 'prior dev/tgt', m(lambda r: r['prior_dev']), m(lambda r: r['prior_target']),
              'regret R0/Q/Otsu', [m(lambda r, k=k: r['regret'][k]) for k in ('R0', 'Q', 'Otsu')], 'errR0', m(lambda r: r['err']['R0']), 'const-animal', m(lambda r: r['err_const_animal']),
              'cert R0/Q', m(lambda r: r['cert']['R0']), m(lambda r: r['cert']['Q']), 'covered R0/Q', sum(r['covered']['R0'] for r in reps), sum(r['covered']['Q'] for r in reps), flush=True)
    p = cr.OUT / 'result_px.json'
    p.write_text(json.dumps(out, indent=1, default=float))
    print('wrote', p, hashlib.sha256(p.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
