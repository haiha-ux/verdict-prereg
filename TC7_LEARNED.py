"""theory/102 (Amendment 23): age-discounted margin vs LEARNED confidence baselines on a second set of fresh mutex / sirius episodes (disjoint from every manifest and from idx_<ds>F.json).
stages: idx <ds> (venv_dec) | decode <ds> (venv_dec) | feat <ds> (.venv) | run (.venv) -> experiments/out/tc/result_tc7.json
All rules decide the side of the last node (same decision); they differ only in the confidence used to rank frames. Learned rankers are fit on dev_calib frames of z_<ds>.npz only."""
import json, hashlib, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent
OUT = HERE / 'out' / 'tc'
SEED, N_FRESH, TAG = 20261030, 120, 'G'
DS = ('mutex', 'sirius')
MS = (8, 16)
NB = 2000
NAMES = ('cert_k1', 'hold_c', 'hold', 'lr', 'gbt')


def stage_idx(ds):
    import lerobot_prep as lp
    used = {int(r['episode']) for r in json.load(open(ROOT / 'data' / f'lr_{ds}_manifest.json'))}
    used |= set(json.load(open(OUT / f'idx_{ds}F.json'))['episodes'])
    info, ep, fi, g, e, key = lp.load(ds)
    fps = info['fps']; ref = lp.grip_ref(g, g[fi == 0])
    e = e[(e[f'videos/{key}/file_index'] == 0) & (e[f'videos/{key}/chunk_index'] == 0)]
    fresh = sorted(int(x) for x in e['episode_index'] if int(x) not in used)
    pick = sorted(int(x) for x in np.random.RandomState(SEED).choice(fresh, N_FRESH, replace=False))
    start = {int(r['episode_index']): (float(r[f'videos/{key}/from_timestamp']), int(r['length'])) for _, r in e.iterrows()}
    rows = []
    for epi in pick:
        gi = g[ep == epi]; ts, L = start[epi]
        assert len(gi) == L
        lab = lp.label_of(lp.u_of(gi, ref))
        rows += [dict(episode=epi, frame=k, vframe=int(round((ts + k / fps) * fps)), label=int(lab[k])) for k in range(L)]
    (OUT / f'idx_{ds}{TAG}.json').write_text(json.dumps(dict(ds=ds, fps=fps, n_fresh=len(fresh), episodes=pick, rows=rows)))
    print(ds, 'fresh', len(fresh), 'picked', len(pick), 'frames', len(rows))


def stage_decode(ds):
    import av, lerobot_prep as lp
    rows = json.load(open(OUT / f'idx_{ds}{TAG}.json'))['rows']
    want = {r['vframe']: i for i, r in enumerate(rows)}; last = max(want)
    arr = np.lib.format.open_memmap(ROOT / 'data' / f'tc_{ds}{TAG}_dense.npy', mode='w+', dtype=np.uint8, shape=(len(rows), 224, 224, 3))
    c = av.open(str(lp.raw(ds) / 'videos' / lp.DS[ds][1] / 'chunk-000' / 'file-000.mp4')); st = c.streams.video[0]; st.thread_type = 'AUTO'
    fps = json.load(open(lp.raw(ds) / 'meta/info.json'))['fps']
    for fr in c.decode(st):
        n = int(round(float(fr.pts * st.time_base) * fps))
        if n in want:
            arr[want.pop(n)] = fr.to_ndarray(width=224, height=224, format='rgb24')
        if n >= last and not want:
            break
    c.close(); arr.flush()
    if want:
        sys.exit(f'STOP {ds}: {len(want)} frames not decoded')
    print(ds, 'decoded', arr.shape)


def stage_feat(ds):
    import h113_run as h13, t2_run as t2
    from PIL import Image
    idx = json.load(open(OUT / f'idx_{ds}{TAG}.json')); rows = idx['rows']
    P = json.load(open(HERE / 'out' / 'lerobot' / f'fit_{ds}-{ds}.json'))['probe']
    arr = np.load(ROOT / 'data' / f'tc_{ds}{TAG}_dense.npy', mmap_mode='r')
    Z = [t2.logits(P, h13.embed([Image.fromarray(np.asarray(arr[i])) for i in range(a, min(a + 1000, len(rows)))]).astype(np.float32))
         for a in range(0, len(rows), 1000)]
    np.savez(OUT / f'z_{ds}{TAG}.npz', z=np.concatenate(Z), ep=np.array([r['episode'] for r in rows]), label=np.array([r['label'] for r in rows]),
             fps=idx['fps'], tS=P['tS'])
    print(ds, 'feat', len(rows))


def frames(eps_list, z, ep, label, fps, tau, m, e, L, c):
    """between-node labelled frames: features X = (a, age, d) with a = |z_last - tau|, age in frames, d = |z_last - z_prev node|;
    hand margins; err vs reader and label of the hold decision; episode id."""
    X, H, R, B, G = [], [], [], [], []
    for e_i in eps_list:
        s = z[ep == e_i]; l = label[ep == e_i]; n = len(s)
        btw = (np.arange(n) % m != 0) & (l >= 0)
        if not btw.any():
            continue
        nodes = np.arange(0, n, m); j = np.searchsorted(nodes, np.arange(n), side='right') - 1
        v = s[nodes]; last = v[j]; prev = v[np.maximum(j - 1, 0)]
        a = np.abs(last - tau); age = (np.arange(n) - nodes[j]).astype(float); d = np.abs(last - prev)
        dec = (last > tau).astype(int)
        X.append(np.c_[a, age, d][btw])
        H.append(np.c_[a / (e + L * age / fps), a / (age + c), a][btw])
        R.append((dec != (s > tau))[btw]); B.append((dec != l)[btw]); G.append(np.full(int(btw.sum()), e_i))
    return tuple(np.concatenate(x) for x in (X, H, R, B, G))


def design(X):
    a, age, d = X.T
    return np.c_[a, age, d, a * age, a * d, age * d, np.log1p(a), np.log1p(age)]


def margins(X, H, lr, gbt):
    F = design(X)
    return dict(cert_k1=H[:, 0], hold_c=H[:, 1], hold=H[:, 2], lr=-lr.predict_proba(F)[:, 1], gbt=-gbt.predict_proba(F)[:, 1])


def run():
    import tc_cert as t1, tc2_cert as t2, tc3_rc as t3, tc4_rc as t4
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rng = np.random.RandomState(SEED); res = {}
    for ds in DS:
        d = np.load(OUT / f'z_{ds}.npz'); z, ep, label, split, fps, tau = d['z'], d['ep'], d['label'], d['split'], float(d['fps']), float(d['tS'])
        cal = split == 'calib'
        f = np.load(OUT / f'z_{ds}{TAG}.npz'); zf, epf, lf = f['z'], f['ep'], f['label']
        assert abs(float(f['tS']) - tau) < 1e-9 and float(f['fps']) == fps
        e_q = t1.calib(z[cal], ep[cal], fps, 2, 1.0)[0]; L = t2.calib_M(z[cal], ep[cal], fps, 1, 1.0, 'V1')
        cal_eps = list(np.unique(ep[cal]))
        res[ds] = dict(fps=fps, e=e_q, L=L, n_eval_episodes=len(np.unique(epf)), cells=[])
        for m in MS:
            dummy = lambda c: frames(cal_eps, z, ep, label, fps, tau, m, e_q, L, c)  # noqa: E731
            best = min(t4.CGRID, key=lambda c: t3.aurc(dummy(c)[1][:, 1], dummy(c)[2]))
            Xd, Hd, Rd, _, _ = frames(cal_eps, z, ep, label, fps, tau, m, e_q, L, best)
            lr = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(design(Xd), Rd)
            gbt = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=SEED).fit(design(Xd), Rd)
            Xe, He, Re, Be, Ge = frames(list(np.unique(epf)), zf, epf, lf, fps, tau, m, e_q, L, best)
            Me = margins(Xe, He, lr, gbt)
            ids = np.unique(Ge); pos = {i: np.flatnonzero(Ge == i) for i in ids}
            bs = [np.concatenate([pos[i] for i in rng.choice(ids, len(ids), replace=True)]) for _ in range(NB)]
            def diff(a, b, err):
                pt = t3.aurc(Me[a], err) - t3.aurc(Me[b], err)
                return float(pt), float(np.quantile([t3.aurc(Me[a][s], err[s]) - t3.aurc(Me[b][s], err[s]) for s in bs], 0.95))
            cell = dict(m=m, c_star=best, n_dev=int(len(Rd)), n_eval=int(len(Re)), dev_err_rate=float(Rd.mean()),
                        aurc_reader={k: t3.aurc(Me[k], Re) for k in NAMES}, aurc_label={k: t3.aurc(Me[k], Be) for k in NAMES},
                        Q1={b: diff('cert_k1', b, Re) for b in ('lr', 'gbt')}, Q2={b: diff('hold_c', b, Re) for b in ('lr', 'gbt')},
                        Q3={b: diff('cert_k1', b, Be) for b in ('lr', 'gbt')})
            cell['Q1_pass'] = all(cell['Q1'][b][1] < 0.10 * cell['aurc_reader'][b] for b in ('lr', 'gbt'))
            res[ds]['cells'].append(cell)
            print(ds, 'm', m, 'c*', best, {k: round(v, 5) for k, v in cell['aurc_reader'].items()}, 'Q1', cell['Q1'], cell['Q1_pass'], flush=True)
    res['verdict'] = dict(Q1=all(c['Q1_pass'] for ds in DS for c in res[ds]['cells']))
    p = OUT / 'result_tc7.json'; p.write_text(json.dumps(res, indent=1))
    print('Q1', res['verdict']['Q1'], 'wrote', p, hashlib.sha256(p.read_bytes()).hexdigest())


if __name__ == '__main__':
    {'idx': stage_idx, 'decode': stage_decode, 'feat': stage_feat}.get(sys.argv[1], lambda *_: run())(*sys.argv[2:])
