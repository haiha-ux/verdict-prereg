"""theory/105 (Amendment 26): same Cataract-101 and THUMOS14 tests with a stronger frozen reader (DINOv2-large CLS + mean patch, 2048-d).
stages: cat_decode (venv_dec) | th_fetch (venv_dec) | embed (.venv) | claims (.venv) -> experiments/out/large/result_tc10.json
Videos, splits, labels, rules and selector are those of tc8_cataract.py / tc9_thumos.py; only the encoder changes."""
import json, hashlib, shutil, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE / 'out' / 'large'
MODEL = 'facebook/dinov2-large'
SEED = 20261102
PRIMARY_MS = (8, 16)
_M = None


def _mods():
    import tc8_cataract as t8, tc9_thumos as t9
    t8.OUT = OUT / 'cat'; t9.OUT = OUT / 'th14'
    for d, src, f in ((t8.OUT, HERE / 'out' / 'cat', 'labels.json'), (t9.OUT, HERE / 'out' / 'th14', 'idx.json')):
        d.mkdir(parents=True, exist_ok=True)
        if not (d / f).exists():
            shutil.copy(src / f, d / f)
    return t8, t9


def embed(ims):
    """as h113_run.embed (bicubic 224, ImageNet normalisation, CLS + mean patch) with DINOv2-large."""
    import torch
    from PIL import Image
    global _M
    if _M is None:
        from transformers import AutoModel
        _M = AutoModel.from_pretrained(MODEL).eval().cuda()
    mean, std = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda(), torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
    out = []
    with torch.no_grad():
        for i in range(0, len(ims), 64):
            x = torch.stack([torch.from_numpy(np.asarray(im.convert('RGB').resize((224, 224), Image.BICUBIC), np.float32) / 255).permute(2, 0, 1) for im in ims[i:i + 64]]).cuda()
            h = _M(pixel_values=(x - mean) / std).last_hidden_state
            out.append(torch.cat([h[:, 0], h[:, 1:].mean(1)], 1).float().cpu().numpy())
    return np.concatenate(out)


def stage_embed():
    from PIL import Image
    t8, t9 = _mods()
    for raw, out in ((t8.RAW / 'fr', t8.OUT), (t9.RAW / 'fr', t9.OUT)):
        for p in sorted(raw.glob('*.npy')):
            a = np.load(p)
            F = np.concatenate([embed([Image.fromarray(x) for x in a[i:i + 500]]) for i in range(0, len(a), 500)]).astype(np.float32)
            np.save(out / f'F_{p.stem}.npy', F); p.unlink(); print(out.name, p.stem, F.shape, flush=True)


def load(domain):
    t8, t9 = _mods()
    if domain == 'cat':
        meta = json.load(open(t8.OUT / 'labels.json')); split = {int(v): s for v, s in meta['split'].items()}
        items = [(v, np.load(t8.OUT / f'F_{v}.npy'), np.array(meta['labels'][str(v)]), split[v]) for v in sorted(split)]
        fps = 25.0 / t8.STEP
    else:
        idx = json.load(open(t9.OUT / 'idx.json')); got = json.load(open(t9.OUT / 'fetch.json'))
        ok = [v for v in idx['order'] if got.get(v, {}).get('ok')]
        sp = {v: ('train' if i < t9.N_TR else 'calib' if i < t9.N_TR + t9.N_CAL else 'eval') for i, v in enumerate(ok)}
        items = []
        for v in ok:
            f = np.load(t9.OUT / f"F_{got[v]['idx']}.npy"); items.append((got[v]['idx'], f, t9.labels(idx['segments'][v], len(f)), sp[v]))
        fps = t9.GRID_FPS
    F, Y, V, S = [], [], [], []
    for v, f, y, s in items:
        n = min(len(f), len(y)); assert abs(len(f) - len(y)) <= 2
        F.append(f[:n]); Y.append(y[:n]); V.append(np.full(n, v)); S.append(np.full(n, s))
    return np.concatenate(F), np.concatenate(Y), np.concatenate(V), np.concatenate(S), fps


def claims():
    import t2_run as t2r, tc_cert as t1, tc2_cert as t2, tc3_rc as t3, tc4_rc as t4, tc6_fresh as t6, tc9_thumos as t9
    rng = np.random.RandomState(SEED); res = {}
    for dom in ('cat', 'th14'):
        F, Y, V, S, fps = load(dom); lab = Y >= 0; cal, ev = S == 'calib', S == 'eval'
        fit = t2r.fit_probe(F[lab & (S == 'train')], Y[lab & (S == 'train')], F[lab & cal], Y[lab & cal])
        P = dict(zip(('mu', 'sd', 'w', 'b'), (np.asarray(a).tolist() for a in fit['pred'].params)), C=fit['C'], tS=fit['tS'])
        z = t2r.logits(P, F); tau = P['tS']
        pr = []
        for lag in t9.LAGS:
            k = max(1, int(round(lag * fps)))
            pr.append(float(np.concatenate([(z[cal & (V == e)][k:] > tau) != (z[cal & (V == e)][:-k] > tau) for e in np.unique(V[cal])]).mean()))
        rho = pr[1] / max(pr[0], 1e-9); sel = 'hold_c' if rho >= t9.RHO_CUT else 'ema'
        e_q = t1.calib(z[cal], V[cal], fps, 2, 1.0)[0]; L = t2.calib_M(z[cal], V[cal], fps, 1, 1.0, 'V1')
        mu = [float(z[cal & lab & (Y == c)].mean()) for c in (0, 1)]; sd = [float(z[cal & lab & (Y == c)].std()) + 1e-6 for c in (0, 1)]
        cal_v, ev_v = sorted(set(V[cal])), sorted(set(V[ev]))
        r = dict(probe=dict(C=P['C'], auc_eval=t2r.auc(z[ev & lab], Y[ev & lab]), err_eval=float(((z[ev & lab] > tau) != Y[ev & lab]).mean())),
                 flip=pr, rho=rho, selected=sel, cells=[])
        for m in (2, 4, 8, 16):
            best = min(t4.CGRID, key=lambda c: t3.aurc(*[t6.per_episode(cal_v, z, V, Y, fps, tau, m, e_q, L, c, mu, sd)['hold_c'][i] for i in (0, 1)]))
            E = t6.per_episode(ev_v, z, V, Y, fps, tau, m, e_q, L, best, mu, sd)
            cell = dict(m=m, c_star=best, aurc_label={k: t3.aurc(E[k][0], E[k][2]) for k in E}, aurc_reader={k: t3.aurc(E[k][0], E[k][1]) for k in E})
            if m in PRIMARY_MS:
                cell['R1'] = t6.boot_diff(E, sel, 'hold', 2, rng)
                cell['other_vs_hold'] = t6.boot_diff(E, 'ema' if sel == 'hold_c' else 'hold_c', 'hold', 2, rng)
            r['cells'].append(cell)
            print(dom, 'm', m, {k: round(v, 4) for k, v in cell['aurc_label'].items()}, cell.get('R1'), flush=True)
        r['R1'] = all(c['R1'][1] < 0 for c in r['cells'] if c['m'] in PRIMARY_MS)
        res[dom] = r
        print(dom, r['probe'], 'rho', round(rho, 3), sel, 'R1', r['R1'], flush=True)
    p = OUT / 'result_tc10.json'; p.write_text(json.dumps(res, indent=1))
    print('wrote', p, hashlib.sha256(p.read_bytes()).hexdigest())


if __name__ == '__main__':
    t8, t9 = _mods()
    dict(cat_decode=t8.stage_decode, th_fetch=t9.stage_fetch, embed=stage_embed, claims=claims)[sys.argv[1]]()
