"""theory/101 (Amendment 22): confirmatory test of the sparse-frame gain on FRESH mutex / sirius episodes (never in any manifest).
stages: idx <ds> (venv_dec) | decode <ds> (venv_dec, streaming to data/tc_<ds>F_dense.npy) | feat <ds> (.venv) | run (.venv) -> experiments/out/tc/result_tc6.json
All constants (probe, tau, e, L, c*, theta*) come from the earlier dev_calib episodes in experiments/out/tc/z_<ds>.npz; nothing is fit on fresh episodes."""
import json, hashlib, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent
OUT = HERE / 'out' / 'tc'
SEED, N_FRESH = 20261029, 120
DS = ('mutex', 'sirius')
MS = (2, 4, 8, 16)
PRIMARY_MS = (8, 16)
NB = 2000


def stage_idx(ds):
    import lerobot_prep as lp
    used = {int(r['episode']) for r in json.load(open(ROOT / 'data' / f'lr_{ds}_manifest.json'))}
    info, ep, fi, g, e, key = lp.load(ds)
    fps = info['fps']; ref = lp.grip_ref(g, g[fi == 0])
    e = e[(e[f'videos/{key}/file_index'] == 0) & (e[f'videos/{key}/chunk_index'] == 0)]
    fresh = sorted(int(x) for x in e['episode_index'] if int(x) not in used)
    pick = sorted(int(x) for x in np.random.RandomState(SEED).choice(fresh, N_FRESH, replace=False))
    start = {int(r['episode_index']): (float(r[f'videos/{key}/from_timestamp']), int(r['length'])) for _, r in e.iterrows()}
    rows = []
    for epi in pick:
        m = ep == epi; gi = g[m]; ts, L = start[epi]
        assert len(gi) == L
        lab = lp.label_of(lp.u_of(gi, ref))
        rows += [dict(episode=epi, frame=k, vframe=int(round((ts + k / fps) * fps)), label=int(lab[k])) for k in range(L)]
    (OUT / f'idx_{ds}F.json').write_text(json.dumps(dict(ds=ds, fps=fps, n_fresh=len(fresh), episodes=pick, rows=rows)))
    print(ds, 'fresh', len(fresh), 'picked', len(pick), 'frames', len(rows))


def stage_decode(ds):
    """streaming copy of lerobot_prep.decode: frames go straight to the memmap (RAM bound)."""
    import av, lerobot_prep as lp
    rows = json.load(open(OUT / f'idx_{ds}F.json'))['rows']
    want = {r['vframe']: i for i, r in enumerate(rows)}; last = max(want)
    arr = np.lib.format.open_memmap(ROOT / 'data' / f'tc_{ds}F_dense.npy', mode='w+', dtype=np.uint8, shape=(len(rows), 224, 224, 3))
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
    idx = json.load(open(OUT / f'idx_{ds}F.json')); rows = idx['rows']
    P = json.load(open(HERE / 'out' / 'lerobot' / f'fit_{ds}-{ds}.json'))['probe']
    arr = np.load(ROOT / 'data' / f'tc_{ds}F_dense.npy', mmap_mode='r')
    Z = [t2.logits(P, h13.embed([Image.fromarray(np.asarray(arr[i])) for i in range(a, min(a + 1000, len(rows)))]).astype(np.float32))
         for a in range(0, len(rows), 1000)]
    np.savez(OUT / f'z_{ds}F.npz', z=np.concatenate(Z), ep=np.array([r['episode'] for r in rows]), label=np.array([r['label'] for r in rows]),
             fps=idx['fps'], tS=P['tS'])
    print(ds, 'feat', len(rows))


def per_episode(eps_list, z, ep, label, fps, tau, m, e, L, c, mu, sd):
    """like tc4_rc.collect but keeps the episode id of every between-node labelled frame."""
    import tc4_rc as t4
    acc = {k: ([], [], [], []) for k in t4.NAMES}
    for e_i in eps_list:
        s = z[ep == e_i]; l = label[ep == e_i]; n = len(s)
        btw = (np.arange(n) % m != 0) & (l >= 0)
        if not btw.any():
            continue
        r = (s > tau).astype(int)
        for k, (marg, state, _) in t4.episode(s, m, fps, tau, e, L, c, mu, sd).items():
            dec = state.astype(int)
            acc[k][0].append(marg[btw]); acc[k][1].append((dec != r)[btw]); acc[k][2].append((dec != l)[btw]); acc[k][3].append(np.full(int(btw.sum()), e_i))
    return {k: tuple(np.concatenate(v) for v in vs) for k, vs in acc.items()}


def boot_diff(E, a, b, col, rng):
    """episode bootstrap of AURC(a) - AURC(b); returns (point, one-sided 95% upper bound)."""
    import tc3_rc as t3
    epi = E[a][3]; ids = np.unique(epi); pos = {i: np.flatnonzero(epi == i) for i in ids}
    pt = t3.aurc(E[a][0], E[a][col]) - t3.aurc(E[b][0], E[b][col])
    d = []
    for _ in range(NB):
        sel = np.concatenate([pos[i] for i in rng.choice(ids, len(ids), replace=True)])
        d.append(t3.aurc(E[a][0][sel], E[a][col][sel]) - t3.aurc(E[b][0][sel], E[b][col][sel]))
    return float(pt), float(np.quantile(d, 0.95))


def run():
    import tc_cert as t1, tc2_cert as t2, tc3_rc as t3, tc4_rc as t4
    rng = np.random.RandomState(SEED); res = {}
    for ds in DS:
        d = np.load(OUT / f'z_{ds}.npz'); z, ep, label, split, fps, tau = d['z'], d['ep'], d['label'], d['split'], float(d['fps']), float(d['tS'])
        cal = split == 'calib'
        f = np.load(OUT / f'z_{ds}F.npz'); zf, epf, lf = f['z'], f['ep'], f['label']
        assert abs(float(f['tS']) - tau) < 1e-9 and float(f['fps']) == fps
        e_q = t1.calib(z[cal], ep[cal], fps, 2, 1.0)[0]; L = t2.calib_M(z[cal], ep[cal], fps, 1, 1.0, 'V1')
        zc, lc = z[cal], label[cal]; mu = [float(zc[lc == c].mean()) for c in (0, 1)]; sd = [float(zc[lc == c].std()) + 1e-6 for c in (0, 1)]
        cal_eps = list(np.unique(ep[cal])); fr_eps = list(np.unique(epf))
        res[ds] = dict(fps=fps, e=e_q, L=L, n_fresh_episodes=len(fr_eps), cells=[])
        for m in MS:
            best = min(t4.CGRID, key=lambda c: t3.aurc(*[per_episode(cal_eps, z, ep, label, fps, tau, m, e_q, L, c, mu, sd)['hold_c'][i] for i in (0, 1)]))
            D = per_episode(cal_eps, z, ep, label, fps, tau, m, e_q, L, best, mu, sd)
            E = per_episode(fr_eps, zf, epf, lf, fps, tau, m, e_q, L, best, mu, sd)
            cell = dict(m=m, c_star=best, n_frames=int(len(E['hold'][0])), methods={}, P1={}, P3={})
            for k in t4.NAMES:
                cell['methods'][k] = dict(aurc_reader=t3.aurc(E[k][0], E[k][1]), aurc_label=t3.aurc(E[k][0], E[k][2]),
                                          op002=t3.op_point(D[k][0], D[k][1], E[k][0], E[k][1], 0.002))
            if m in PRIMARY_MS:
                for b in ('hold', 'ema'):
                    cell['P1'][b] = boot_diff(E, 'cert_k1', b, 1, rng)
                    cell['P3'][b] = boot_diff(E, 'cert_k1', b, 2, rng)
            res[ds]['cells'].append(cell)
            print(ds, 'm', m, 'c*', best, {k: round(v['aurc_reader'], 5) for k, v in cell['methods'].items()}, 'P1', cell['P1'], flush=True)
    p1 = all(v[1] < 0 for ds in DS for c in res[ds]['cells'] if c['m'] in PRIMARY_MS for v in c['P1'].values())
    p2 = all((c['methods']['cert_k1']['op002']['cov_eval'] - c['methods']['hold']['op002']['cov_eval'] >= 0.02)
             and (c['methods']['cert_k1']['op002']['risk_eval'] or 0) <= 0.004 for ds in DS for c in res[ds]['cells'] if c['m'] in PRIMARY_MS)
    res['verdict'] = dict(P1=p1, P2=p2)
    p = OUT / 'result_tc6.json'; p.write_text(json.dumps(res, indent=1))
    print('P1', p1, 'P2', p2, 'wrote', p, hashlib.sha256(p.read_bytes()).hexdigest())


if __name__ == '__main__':
    {'idx': stage_idx, 'decode': stage_decode, 'feat': stage_feat}.get(sys.argv[1], lambda *_: run())(*sys.argv[2:])
