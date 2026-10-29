"""theory/103 (Amendment 24): age-discounted margins on Cataract-101 surgical video with expert phase labels.
stages: labels (.venv; csv only) | decode (venv_dec; one mp4 at a time from the zip, every 5th native frame -> data/cat101_raw/fr/<vid>.npy, mp4 deleted) |
        embed (.venv; DINOv2 CLS+mean -> experiments/out/cat/F_<vid>.npy, frames deleted) | run (.venv) -> experiments/out/cat/result_tc8.json
Untrusted archive: read with zipfile, member names checked, nothing bundled is run."""
import json, hashlib, sys, zipfile
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent
RAW = ROOT / 'data' / 'cat101_raw'
ZIP = RAW / 'cataract-101.zip'
OUT = HERE / 'out' / 'cat'
SEED, STEP, PHASE = 20261031, 5, 5          # base grid = every 5th native frame; positive phase = 5 (phacoemulsification)
N_TR, N_CAL = 40, 20
MS = (2, 4, 8, 16)
PRIMARY_MS = (8, 16)


def csv(name):
    rows = zipfile.ZipFile(ZIP).read(name).decode().strip().splitlines()
    return [r.split(';') for r in rows[1:]]


def stage_labels():
    OUT.mkdir(parents=True, exist_ok=True)
    vids = {int(r[0]): dict(frames=int(r[1]), fps=float(r[2]), surgeon=int(r[3]), exp=int(r[4])) for r in csv('videos.csv')}
    ann = {}
    for v, f, p in csv('annotations.csv'):
        ann.setdefault(int(v), []).append((int(f), int(p)))
    order = np.random.RandomState(SEED).permutation(sorted(vids)).tolist()
    split = {v: ('train' if i < N_TR else 'calib' if i < N_TR + N_CAL else 'eval') for i, v in enumerate(order)}
    lab = {}
    for v, a in ann.items():
        a = sorted(a); starts = np.array([f for f, _ in a]); ph = np.array([p for _, p in a])
        g = np.arange(0, vids[v]['frames'], STEP); k = np.searchsorted(starts, g, side='right') - 1
        lab[v] = np.where(k < 0, -1, (ph[np.maximum(k, 0)] == PHASE).astype(int)).tolist()
    (OUT / 'labels.json').write_text(json.dumps(dict(videos={str(v): d for v, d in vids.items()}, split={str(v): s for v, s in split.items()},
                                                     labels={str(v): l for v, l in lab.items()})))
    for s in ('train', 'calib', 'eval'):
        vs = [v for v in vids if split[v] == s]
        y = np.concatenate([np.array(lab[v]) for v in vs]); y = y[y >= 0]
        print(s, 'videos', len(vs), 'surgeons', sorted({vids[v]['surgeon'] for v in vs}), 'frames', len(y), 'pos share', round(float(y.mean()), 3))


def stage_decode():
    import av
    zf = zipfile.ZipFile(ZIP); d = RAW / 'fr'; d.mkdir(exist_ok=True); tmp = RAW / 'tmp.mp4'
    names = [n for n in zf.namelist() if n.startswith('videos/case_') and n.endswith('.mp4') and '..' not in n and n.count('/') == 1]
    for n in names:
        v = int(n[len('videos/case_'):-4]); dst = d / f'{v}.npy'
        if dst.exists() or (OUT / f'F_{v}.npy').exists():
            continue
        with zf.open(n) as src, open(tmp, 'wb') as f:
            while b := src.read(1 << 24):
                f.write(b)
        c = av.open(str(tmp)); st = c.streams.video[0]; st.thread_type = 'AUTO'
        fr = [x.to_ndarray(width=224, height=224, format='rgb24') for i, x in enumerate(c.decode(st)) if i % STEP == 0]
        c.close(); tmp.unlink()
        np.save(dst, np.stack(fr)); print(v, len(fr), flush=True)


def stage_embed():
    import h113_run as h13
    from PIL import Image
    for p in sorted((RAW / 'fr').glob('*.npy')):
        a = np.load(p)
        F = np.concatenate([h13.embed([Image.fromarray(x) for x in a[i:i + 1000]]) for i in range(0, len(a), 1000)]).astype(np.float32)
        np.save(OUT / f'F_{p.stem}.npy', F); p.unlink(); print(p.stem, F.shape, flush=True)


def run():
    import t2_run as t2r, tc_cert as t1, tc2_cert as t2, tc3_rc as t3, tc4_rc as t4, tc6_fresh as t6, tc7_learned as t7
    from sklearn.linear_model import LogisticRegression
    meta = json.load(open(OUT / 'labels.json'))
    split = {int(v): s for v, s in meta['split'].items()}
    F, Y, V = [], [], []
    for v in sorted(split):
        f = np.load(OUT / f'F_{v}.npy'); y = np.array(meta['labels'][str(v)])
        n = min(len(f), len(y)); assert abs(len(f) - len(y)) <= 2, (v, len(f), len(y))
        F.append(f[:n]); Y.append(y[:n]); V.append(np.full(n, v))
    F, Y, V = np.concatenate(F), np.concatenate(Y), np.concatenate(V)
    sp = np.array([split[v] for v in V]); ok = Y >= 0
    fit = t2r.fit_probe(F[ok & (sp == 'train')], Y[ok & (sp == 'train')], F[ok & (sp == 'calib')], Y[ok & (sp == 'calib')])
    P = dict(zip(('mu', 'sd', 'w', 'b'), (np.asarray(a).tolist() for a in fit['pred'].params)), C=fit['C'], tS=fit['tS'])
    z = t2r.logits(P, F); tau = P['tS']; fps = 25.0 / STEP
    ev = sp == 'eval'
    res = dict(probe=dict(C=P['C'], tS=tau, auc_eval=t2r.auc(z[ev & ok], Y[ev & ok]), err_eval=float(((z[ev & ok] > tau) != Y[ev & ok]).mean()),
                          pos_eval=float(Y[ev & ok].mean())), cells=[])
    cal = sp == 'calib'
    zc_ok = cal & ok
    e_q = t1.calib(z[cal], V[cal], fps, 2, 1.0)[0]; L = t2.calib_M(z[cal], V[cal], fps, 1, 1.0, 'V1')
    mu = [float(z[zc_ok & (Y == c)].mean()) for c in (0, 1)]; sd = [float(z[zc_ok & (Y == c)].std()) + 1e-6 for c in (0, 1)]
    res.update(e=e_q, L=L, fps_grid=fps)
    cal_v, ev_v = sorted(set(V[cal])), sorted(set(V[ev]))
    rng = np.random.RandomState(SEED)
    for m in MS:
        best = min(t4.CGRID, key=lambda c: t3.aurc(*[t6.per_episode(cal_v, z, V, Y, fps, tau, m, e_q, L, c, mu, sd)['hold_c'][i] for i in (0, 1)]))
        E = t6.per_episode(ev_v, z, V, Y, fps, tau, m, e_q, L, best, mu, sd)
        Xd, _, Rd, _, _ = t7.frames(cal_v, z, V, Y, fps, tau, m, e_q, L, best)
        lr = LogisticRegression(C=1.0, max_iter=2000).fit(t7.design(Xd), Rd)
        Xe, _, Re, Be, Ge = t7.frames(ev_v, z, V, Y, fps, tau, m, e_q, L, best)
        assert np.array_equal(Ge, E['hold'][3]) and np.array_equal(Re, E['hold'][1])
        E['lr'] = (-lr.predict_proba(t7.design(Xe))[:, 1], Re, Be, Ge)
        cell = dict(m=m, c_star=best, n_frames=int(len(Re)), aurc_reader={k: t3.aurc(E[k][0], E[k][1]) for k in E},
                    aurc_label={k: t3.aurc(E[k][0], E[k][2]) for k in E}, H1={}, H2={}, H3={})
        if m in PRIMARY_MS:
            for b in ('hold', 'ema'):
                cell['H1'][b] = t6.boot_diff(E, 'cert_k1', b, 1, rng)
                cell['H2'][b] = t6.boot_diff(E, 'cert_k1', b, 2, rng)
            d = t6.boot_diff(E, 'cert_k1', 'lr', 1, rng); cell['H3'] = dict(lr=d, pass_=d[1] < 0.10 * cell['aurc_reader']['lr'])
        res['cells'].append(cell)
        print('m', m, 'c*', best, {k: round(v, 5) for k, v in cell['aurc_reader'].items()}, 'H1', cell['H1'], 'H2', cell['H2'], flush=True)
    pc = [c for c in res['cells'] if c['m'] in PRIMARY_MS]
    res['verdict'] = dict(H1=all(v[1] < 0 for c in pc for v in c['H1'].values()), H2=all(v[1] < 0 for c in pc for v in c['H2'].values()),
                          H3=all(c['H3']['pass_'] for c in pc))
    p = OUT / 'result_tc8.json'; p.write_text(json.dumps(res, indent=1))
    print(res['probe'], res['verdict'], 'wrote', p, hashlib.sha256(p.read_bytes()).hexdigest())


if __name__ == '__main__':
    dict(labels=stage_labels, decode=stage_decode, embed=stage_embed, run=run)[sys.argv[1]]()
