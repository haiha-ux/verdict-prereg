"""theory/90 stage `manifest`: seeded site/sequence/frame sample from LILA metadata only (no image is read).
usage: python -I ct_prep.py manifest <set>   (set in cct | wel | ss3)
"""
import json, sys, hashlib, collections
import numpy as np

SEED = 20261028
MIN_SEQ, MAX_SITES, PER_SITE, K = 40, 150, 25, 5
SETS = {
    'cct': dict(meta='data/cct_meta/caltech_images_20210113.json',
                base='https://lilawildlife.blob.core.windows.net/lila-wildlife/caltech-unzipped/cct_images/'),
    'wel': dict(meta='data/ct_meta/wel/wellington_camera_traps.json',
                base='https://lilawildlife.blob.core.windows.net/lila-wildlife/wellington-unzipped/images/'),
    'ss3': dict(meta='data/ct_meta/ss3/SnapshotSerengetiS03.json',
                base='https://lilawildlife.blob.core.windows.net/lila-wildlife/snapshotserengeti-unzipped/'),
}


def build(name):
    cfg = SETS[name]
    d = json.load(open(cfg['meta']))
    cats = {c['id']: c['name'] for c in d['categories']}
    imgs = {i['id']: i for i in d['images']}
    empty = {}
    for a in d['annotations']:
        empty[a['image_id']] = empty.get(a['image_id'], True) and cats[a['category_id']] == 'empty'
    # sequence -> frames; sequence label = empty iff every annotated frame is empty
    seqs = collections.defaultdict(list)
    for i, e in empty.items():
        im = imgs[i]
        seqs[(im['location'], im['seq_id'])].append((i, e))
    site_seqs = collections.defaultdict(list)
    for (loc, sq), fr in seqs.items():
        site_seqs[loc].append((sq, fr))
    site_share = {}
    for loc, L in site_seqs.items():
        site_share[loc] = float(np.mean([all(e for _, e in fr) for _, fr in L]))
    eligible = sorted(l for l, L in site_seqs.items() if len(L) >= MIN_SEQ)
    rng = np.random.RandomState(SEED)
    chosen = sorted(rng.choice(eligible, size=min(MAX_SITES, len(eligible)), replace=False).tolist())
    perm = rng.permutation(len(chosen))
    fold = {chosen[p]: k % K for k, p in enumerate(perm)}  # site-level K-fold cross-fitting: every site is eval once
    aud = {}  # within a held-out fold, 1/3 of its sites are the audit pool, 2/3 the evaluation pool
    for f in range(K):
        fs = [c for c in chosen if fold[c] == f]
        for j, c in enumerate(rng.permutation(fs)):
            aud[c] = 'audit' if j < len(fs) // 3 else 'eval'
    role = {c: f'f{fold[c]}_{aud[c]}' for c in chosen}
    rows = []
    for loc in chosen:
        L = sorted(site_seqs[loc], key=lambda x: str(x[0]))
        for k in rng.choice(len(L), size=min(PER_SITE, len(L)), replace=False):
            sq, fr = L[k]
            fr = sorted(fr, key=lambda x: str(x[0]))
            i, _ = fr[rng.randint(len(fr))]
            rows.append(dict(site=loc, seq=sq, image=i, file=imgs[i]['file_name'], url=cfg['base'] + imgs[i]['file_name'],
                             label_animal=int(not all(e for _, e in fr)), role=role[loc], site_empty_share=site_share[loc]))
    out = dict(set=name, seed=SEED, n_eligible_sites=len(eligible), n_sites=len(chosen), n_images=len(rows),
               roles=dict(collections.Counter(fold.values())), meta_file=cfg['meta'],
               meta_sha256=hashlib.sha256(open(cfg['meta'], 'rb').read()).hexdigest(), rows=rows)
    p = f'data/ct_{name}_manifest.json'
    json.dump(out, open(p, 'w'))
    print(name, 'eligible', len(eligible), 'sites', len(chosen), 'images', len(rows), out['roles'],
          'animal share', round(float(np.mean([r['label_animal'] for r in rows])), 3))
    return p


if __name__ == '__main__':
    if sys.argv[1] == 'manifest':
        build(sys.argv[2])
