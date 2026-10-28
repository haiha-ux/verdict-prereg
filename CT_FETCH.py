"""theory/90 stage `fetch`: download each manifest image by URL, decode in memory (PIL), resize 224x224 RGB, keep only the array.
usage: python -I ct_fetch.py <set>.  Originals are never written to disk. Failures are recorded, not retried beyond 2 attempts."""
import json, sys, io, urllib.request, concurrent.futures as cf
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = 50_000_000


def get(row):
    for _ in range(3):
        try:
            with urllib.request.urlopen(row['url'], timeout=60) as r:
                b = r.read(20_000_000)
            im = Image.open(io.BytesIO(b)).convert('RGB').resize((224, 224), Image.BILINEAR)
            return np.asarray(im, dtype=np.uint8), len(b)
        except Exception as e:  # noqa: BLE001
            err = repr(e)[:120]
    return None, err


def main(name):
    m = json.load(open(f'data/ct_{name}_manifest.json'))
    rows = m['rows']
    out = np.zeros((len(rows), 224, 224, 3), np.uint8)
    ok = np.zeros(len(rows), bool)
    nbytes, errs = 0, {}
    with cf.ThreadPoolExecutor(8) as ex:
        for k, (a, b) in enumerate(ex.map(get, rows)):
            if a is None:
                errs[k] = b
            else:
                out[k], ok[k] = a, True
                nbytes += b
            if k % 500 == 0:
                print(name, k, len(rows), round(nbytes / 1e6), 'MB', len(errs), 'fail', flush=True)
    np.save(f'data/ct_{name}_frames.npy', out)
    np.save(f'data/ct_{name}_ok.npy', ok)
    json.dump(dict(set=name, n=len(rows), ok=int(ok.sum()), fail=len(errs), bytes=nbytes, errs=list(errs.values())[:20]),
              open(f'experiments/out/camtrap/fetch_{name}.json', 'w'))
    print('done', name, int(ok.sum()), 'ok', len(errs), 'fail', round(nbytes / 1e6), 'MB')


if __name__ == '__main__':
    main(sys.argv[1])
