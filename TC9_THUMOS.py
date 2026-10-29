"""theory/104 (Amendment 25): label-free regime selector (drift -> age discount, flicker -> smoothing) tested on THUMOS14 (human temporal annotations).
stages: idx (.venv; annotation zips only) | fetch (venv_dec; one mp4 at a time over HTTP, frames at 5 per second by timestamp -> data/th14_raw/fr/<i>.npy, mp4 deleted) |
        embed (.venv; -> experiments/out/th14/F_<i>.npy, frames deleted) | run (.venv) -> experiments/out/th14/result_tc9.json
Untrusted data: annotation zips read with zipfile; videos decoded by PyAV in venv_dec only."""
import json, hashlib, sys, zipfile, urllib.request
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent
META = ROOT / 'data' / 'vid_meta' / 'thumos14'
RAW = ROOT / 'data' / 'th14_raw'
OUT = HERE / 'out' / 'th14'
SEED, N_VID, N_TR, N_CAL, GRID_FPS, CAP = 20261101, 100, 40, 20, 5.0, 11e9
URL = {'validation': 'http://crcv.ucf.edu/THUMOS14/Validation_set/videos/{}.mp4', 'test': 'http://crcv.ucf.edu/THUMOS14/test_set/TH14_test_set_mp4/{}.mp4'}
MS = (2, 4, 8, 16)
PRIMARY_MS = (8, 16)
LAGS, RHO_CUT = (0.2, 1.6), 2.0          # selector: rho = P(flip at 1.6 s) / P(flip at 0.2 s) of the reader decision on calib videos; >= 2 -> hold_c, else ema


def stage_idx():
    OUT.mkdir(parents=True, exist_ok=True)
    seg = {}
    for zname in ('TH14_Temporal_annotations_validation.zip', 'TH14_Temporal_annotations_test.zip'):
        z = zipfile.ZipFile(META / zname)
        for n in z.namelist():
            if not n.endswith('.txt'):
                continue
            amb = n.rsplit('/', 1)[-1].startswith('Ambiguous')
            for line in z.read(n).decode().splitlines():
                p = line.split()
                if len(p) == 3 and p[0].startswith('video_'):
                    seg.setdefault(p[0], []).append((float(p[1]), float(p[2]), amb))
    labelled = sorted(v for v, s in seg.items() if any(not a for _, _, a in s))
    pick = [labelled[i] for i in np.random.RandomState(SEED).permutation(len(labelled))[:N_VID]]
    (OUT / 'idx.json').write_text(json.dumps(dict(n_labelled=len(labelled), order=pick, segments={v: seg[v] for v in pick})))
    print('labelled videos', len(labelled), 'picked', len(pick))


def stage_fetch():
    import av
    idx = json.load(open(OUT / 'idx.json')); d = RAW / 'fr'; d.mkdir(parents=True, exist_ok=True); tmp = RAW / 'tmp.mp4'
    log = OUT / 'fetch.json'; got = json.load(open(log)) if log.exists() else {}
    total = sum(g['bytes'] for g in got.values())
    for v in idx['order']:
        if v in got:
            continue
        if total > CAP:
            print('STOP cap reached', total); break
        try:
            with urllib.request.urlopen(URL[v.split('_')[1]].format(v), timeout=600) as r, open(tmp, 'wb') as f:
                while b := r.read(1 << 22):
                    f.write(b)
        except Exception as e:  # noqa: BLE001
            got[v] = dict(ok=False, bytes=0, err=str(e)[:200]); log.write_text(json.dumps(got)); print(v, 'FAIL', e); continue
        nb = tmp.stat().st_size; total += nb
        c = av.open(str(tmp)); st = c.streams.video[0]; st.thread_type = 'AUTO'; fr, k = [], 0
        for x in c.decode(st):      # grid point k/GRID_FPS takes the first decoded frame at or after it
            t = float(x.pts * st.time_base); a = None
            while k / GRID_FPS <= t + 1e-6:
                a = x.to_ndarray(width=224, height=224, format='rgb24') if a is None else a
                fr.append(a); k += 1
        c.close(); tmp.unlink()
        i = len([g for g in got.values() if g['ok']])
        np.save(d / f'{i}.npy', np.stack(fr))
        got[v] = dict(ok=True, bytes=nb, idx=i, n_grid=len(fr)); log.write_text(json.dumps(got)); print(v, i, nb, len(fr), flush=True)


def stage_embed():
    import h113_run as h13
    from PIL import Image
    for p in sorted((RAW / 'fr').glob('*.npy')):
        a = np.load(p)
        F = np.concatenate([h13.embed([Image.fromarray(x) for x in a[i:i + 1000]]) for i in range(0, len(a), 1000)]).astype(np.float32)
        np.save(OUT / f'F_{p.stem}.npy', F); p.unlink(); print(p.stem, F.shape, flush=True)


def labels(segs, n):
    t = np.arange(n) / GRID_FPS; y = np.zeros(n, int)
    for a, b, amb in segs:
        if not amb:
            y[(t >= a) & (t <= b)] = 1
    for a, b, amb in segs:
        if amb:
            y[(t >= a) & (t <= b)] = -1
    return y


def rho(z, ep, tau):
    p = []
    for lag in LAGS:
        k = max(1, int(round(lag * GRID_FPS)))
        p.append(float(np.concatenate([(z[ep == e][k:] > tau) != (z[ep == e][:-k] > tau) for e in np.unique(ep) if (ep == e).sum() > k]).mean()))
    return p, p[1] / max(p[0], 1e-9)


def run():
    import t2_run as t2r, tc_cert as t1, tc2_cert as t2, tc3_rc as t3, tc4_rc as t4, tc6_fresh as t6
    idx = json.load(open(OUT / 'idx.json')); got = json.load(open(OUT / 'fetch.json'))
    ok = [v for v in idx['order'] if got.get(v, {}).get('ok')]
    split = {v: ('train' if i < N_TR else 'calib' if i < N_TR + N_CAL else 'eval') for i, v in enumerate(ok)}
    F, Y, V, S = [], [], [], []
    for v in ok:
        f = np.load(OUT / f"F_{got[v]['idx']}.npy"); F.append(f); Y.append(labels(idx['segments'][v], len(f)))
        V.append(np.full(len(f), got[v]['idx'])); S.append(np.full(len(f), split[v]))
    F, Y, V, S = np.concatenate(F), np.concatenate(Y), np.concatenate(V), np.concatenate(S); lab = Y >= 0
    fit = t2r.fit_probe(F[lab & (S == 'train')], Y[lab & (S == 'train')], F[lab & (S == 'calib')], Y[lab & (S == 'calib')])
    P = dict(zip(('mu', 'sd', 'w', 'b'), (np.asarray(a).tolist() for a in fit['pred'].params)), C=fit['C'], tS=fit['tS'])
    z = t2r.logits(P, F); tau = P['tS']; fps = GRID_FPS; cal, ev = S == 'calib', S == 'eval'
    pr, r = rho(z[cal], V[cal], tau); sel, oth = ('hold_c', 'ema') if r >= RHO_CUT else ('ema', 'hold_c')
    res = dict(n_videos=len(ok), split_counts={s: int(sum(1 for v in ok if split[v] == s)) for s in ('train', 'calib', 'eval')},
               probe=dict(C=P['C'], tS=tau, auc_eval=t2r.auc(z[ev & lab], Y[ev & lab]), err_eval=float(((z[ev & lab] > tau) != Y[ev & lab]).mean()),
                          pos_eval=float(Y[ev & lab].mean())), flip=pr, rho=r, selected=sel, other=oth, cells=[])
    e_q = t1.calib(z[cal], V[cal], fps, 2, 1.0)[0]; L = t2.calib_M(z[cal], V[cal], fps, 1, 1.0, 'V1')
    mu = [float(z[cal & lab & (Y == c)].mean()) for c in (0, 1)]; sd = [float(z[cal & lab & (Y == c)].std()) + 1e-6 for c in (0, 1)]
    cal_v, ev_v = sorted(set(V[cal])), sorted(set(V[ev])); rng = np.random.RandomState(SEED)
    res.update(e=e_q, L=L)
    for m in MS:
        best = min(t4.CGRID, key=lambda c: t3.aurc(*[t6.per_episode(cal_v, z, V, Y, fps, tau, m, e_q, L, c, mu, sd)['hold_c'][i] for i in (0, 1)]))
        E = t6.per_episode(ev_v, z, V, Y, fps, tau, m, e_q, L, best, mu, sd)
        cell = dict(m=m, c_star=best, n_frames=int(len(E['hold'][0])), aurc_reader={k: t3.aurc(E[k][0], E[k][1]) for k in E},
                    aurc_label={k: t3.aurc(E[k][0], E[k][2]) for k in E})
        if m in PRIMARY_MS:
            cell['S1_label'] = t6.boot_diff(E, sel, oth, 2, rng); cell['S1_reader'] = t6.boot_diff(E, sel, oth, 1, rng)
            cell['S2_label'] = t6.boot_diff(E, sel, 'hold', 2, rng)
        res['cells'].append(cell)
        print('m', m, 'c*', best, {k: round(v, 5) for k, v in cell['aurc_label'].items()}, flush=True)
    pc = [c for c in res['cells'] if c['m'] in PRIMARY_MS]
    res['verdict'] = dict(S1=all(c['S1_label'][1] < 0 and c['S1_reader'][1] < 0 for c in pc), S2=all(c['S2_label'][1] < 0 for c in pc))
    p = OUT / 'result_tc9.json'; p.write_text(json.dumps(res, indent=1))
    print(res['probe'], 'rho', r, 'selected', sel, res['verdict'], 'wrote', p, hashlib.sha256(p.read_bytes()).hexdigest())


if __name__ == '__main__':
    dict(idx=stage_idx, fetch=stage_fetch, embed=stage_embed, run=run)[sys.argv[1]]()
