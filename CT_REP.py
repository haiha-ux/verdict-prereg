"""Amendment 15: R=20 independent site-split replicates of the locked ct_run pipeline (same sampled frames/features; fold and audit/eval assignment redrawn by seed [SEED, 77, r]).
Per replicate: C1 regrets + constant-prediction error (P0), C2 (fold-pools PH/PL with |prior shift|>=.2), C3 coverage, C4 coverage of betting bound and frame-level CP.
usage: python -I ct_rep.py   -> experiments/out/camtrap/result_rep.json"""
import json, sys, hashlib
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ct_run as cr  # noqa: E402  (functions unchanged: run_set, audit_validity, load)

R = 20


def reassign(rows, seed):
    rng = np.random.default_rng(seed)
    sites = sorted({r['site'] for r in rows})
    perm = rng.permutation(len(sites))
    fold = {sites[p]: k % cr.K for k, p in enumerate(perm)}
    role = {}
    for f in range(cr.K):
        fs = [s for s in sites if fold[s] == f]
        for j, s in enumerate(rng.permutation(fs)):
            role[s] = f'f{f}_' + ('audit' if j < len(fs) // 3 else 'eval')
    return [dict(r, role=role[r['site']]) for r in rows]


def main():
    out = dict(R=R, reps=[])
    data = {}
    for name in cr.SETS:
        rows, keep = cr.load(name)
        z = np.load(cr.OUT / f'feat_{name}.npz')
        assert (z['keep'] == keep).all()
        data[name] = (rows, z['F'])
    for r in range(R):
        rep = dict(r=r, sets={})
        allcases, allframes = [], []
        for i, name in enumerate(cr.SETS):
            rows, F = data[name]
            c, f = cr.run_set(name, reassign(rows, [cr.SEED, 77, r, i]), F)
            allcases += c; allframes += f
            y = np.array([x['label_animal'] for x in rows])
            p0 = [x for x in c if x['pool'] == 'P0']
            rep['sets'][name] = dict(
                regret={k: float(np.mean([x['regret'][k] for x in p0])) for k in ('R0', 'Q', 'Otsu')},
                err_R0=float(np.mean([x['err']['R0'] for x in p0])), err_oracle=float(np.mean([x['err']['oracle'] for x in p0])),
                err_const_animal=float(1 - y.mean()), auc=float(np.mean([x['auc'] for x in p0])),
                audit=cr.audit_validity(f, name, np.random.default_rng([cr.SEED, 78, r, i])))
        shifted = [x for x in allcases if x['pool'] in ('PH', 'PL') and abs(x['prior_shift']) >= .20]
        sh_frames = [next(f for f in allframes if (f['set'], f['pool'], f['fold']) == (x['set'], x['pool'], x['fold'])) for x in shifted]
        diffs = [float(np.mean(f['e']['Q'] - f['e']['R0'])) for f in sh_frames]
        rep['c2'] = dict(n_pools=len(shifted), sets=sorted({x['set'] for x in shifted}),
                         mean_Q_minus_R0=float(np.mean(diffs)) if diffs else None)
        cov = [(x['regret'][k] <= x['cert'][k] + 1e-12, abs(x['prior_shift']) >= .20, x['cert'][k], x['regret'][k]) for x in allcases for k in ('R0', 'Q')]
        rep['c3'] = dict(n=len(cov), covered=int(sum(c[0] for c in cov)), n_shifted=int(sum(c[1] for c in cov)),
                         covered_shifted=int(sum(c[0] and c[1] for c in cov)))
        out['reps'].append(rep)
        print('rep', r, rep['c2'], rep['c3'], flush=True)
    p = cr.OUT / 'result_rep.json'
    p.write_text(json.dumps(out, indent=1, default=float))
    print('wrote', p, hashlib.sha256(p.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
