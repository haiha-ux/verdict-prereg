"""theory/95 (temporal certificate test): dense (native-rate) frames of the locked dev_calib + eval episodes of mutex / sirius / droid.
stages: idx <ds> (.venv: episode/frame/vframe/label list) | decode <ds> (venv_dec: all frames 224x224 -> data/tc_<ds>_dense.npy) | feat <ds> (.venv: locked DINOv2 embed + locked probe logits -> experiments/out/tc/z_<ds>.npz)
Nothing is tuned; probe = experiments/out/lerobot/fit_<ds>-<ds>.json (locked Amendment 12). Untrusted video: decoded only by experiments/out/h84/venv_dec/bin/python -I."""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent
OUT = HERE / 'out' / 'tc'
OUT.mkdir(parents=True, exist_ok=True)


def stage_idx(ds):
    import lerobot_prep as lp
    fit = json.load(open(HERE / 'out' / 'lerobot' / f'fit_{ds}-{ds}.json'))
    want = {int(g.split(':')[1]): 'calib' for g in fit['splits']['dev_calib']}
    want.update({int(g.split(':')[1]): 'eval' for g in fit['splits']['eval']})
    info, ep, fi, g, e, key = lp.load(ds)
    fps = info['fps']; ref = lp.grip_ref(g, g[fi == 0])
    e = e[(e[f'videos/{key}/file_index'] == 0) & (e[f'videos/{key}/chunk_index'] == 0)]
    start = {int(r['episode_index']): (float(r[f'videos/{key}/from_timestamp']), int(r['length'])) for _, r in e.iterrows()}
    rows = []
    for epi in sorted(want):
        m = ep == epi; gi = g[m]; ts, L = start[epi]
        assert len(gi) == L
        u = lp.u_of(gi, ref); lab = lp.label_of(u)
        for k in range(L):
            rows.append(dict(episode=epi, frame=k, vframe=int(round((ts + k / fps) * fps)), label=int(lab[k]), u=float(u[k]), split=want[epi]))
    (OUT / f'idx_{ds}.json').write_text(json.dumps(dict(ds=ds, fps=fps, rows=rows)))
    print(ds, 'episodes', len(want), 'frames', len(rows), 'fps', fps)


def stage_decode(ds):
    import lerobot_prep as lp
    idx = json.load(open(OUT / f'idx_{ds}.json'))
    rows = idx['rows']
    got = lp.decode(ds, {r['vframe']: i for i, r in enumerate(rows)})
    arr = np.lib.format.open_memmap(ROOT / 'data' / f'tc_{ds}_dense.npy', mode='w+', dtype=np.uint8, shape=(len(rows), 224, 224, 3))
    for i, a in got.items():
        arr[i] = a
    arr.flush()
    print(ds, 'decoded', arr.shape)


def stage_feat(ds):
    import h113_run as h13
    from PIL import Image
    import t2_run as t2
    idx = json.load(open(OUT / f'idx_{ds}.json')); rows = idx['rows']
    P = json.load(open(HERE / 'out' / 'lerobot' / f'fit_{ds}-{ds}.json'))['probe']
    arr = np.load(ROOT / 'data' / f'tc_{ds}_dense.npy', mmap_mode='r')
    Z = []
    for a in range(0, len(rows), 1000):
        F = h13.embed([Image.fromarray(np.asarray(arr[i])) for i in range(a, min(a + 1000, len(rows)))]).astype(np.float32)
        Z.append(t2.logits(P, F))
    z = np.concatenate(Z)
    np.savez(OUT / f'z_{ds}.npz', z=z, ep=np.array([r['episode'] for r in rows]), frame=np.array([r['frame'] for r in rows]), label=np.array([r['label'] for r in rows]),
             split=np.array([r['split'] for r in rows]), fps=idx['fps'], tS=P['tS'])
    print(ds, 'logits', z.shape, 'tS', P['tS'])


if __name__ == '__main__':
    {'idx': stage_idx, 'decode': stage_decode, 'feat': stage_feat}[sys.argv[1]](sys.argv[2])
