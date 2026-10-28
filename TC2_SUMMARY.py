"""Scores predictions (a)-(e) of theory/96 from result_tc2.json."""
import json, numpy as np
d = json.load(open('experiments/out/tc/result_tc2.json'))
cells = [(ds, c) for ds, r in d.items() for c in r['cells']]
# (a) V1 vs V0 abstention, k in {2,3}
rel, ok_err = [], []
for ds, r in d.items():
    for c0 in r['cells']:
        if c0['var'] != 'V0' or c0['k'] not in (2, 3): continue
        c1 = next(c for c in r['cells'] if (c['k'], c['q'], c['mode'], c['m'], c['var']) == (c0['k'], c0['q'], c0['mode'], c0['m'], 'V1'))
        a0, a1 = c0['cert']['abstain'], c1['cert']['abstain']
        rel.append((a0 - a1) / max(a0, 1e-9) >= .30); ok_err.append(c1['cert']['err_reader'] <= .005)
print('(a) cells with V1 abstention >=30% lower than V0:', sum(rel), '/', len(rel), '| of these with reader err<=.005:', sum(r and e for r, e in zip(rel, ok_err)), '| share', round(np.mean(rel), 3))
# (b) causal >= retro abstention
b = []
for ds, r in d.items():
    for c in r['cells']:
        if c['mode'] != 'causal': continue
        o = next(x for x in r['cells'] if (x['k'], x['q'], x['var'], x['m'], x['mode']) == (c['k'], c['q'], c['var'], c['m'], 'retro'))
        b.append(c['cert']['abstain'] >= o['cert']['abstain'] - 1e-12)
print('(b) causal abstains >= retro:', sum(b), '/', len(b), round(np.mean(b), 3))
# (c) cert reader-error <= every matched baseline ; (d) label-error closeness
c_ok, d_ok, per_base = [], [], {}
for ds, c in cells:
    ce = c['cert']['err_reader']
    c_ok.append(all(ce <= v['err_reader'] + 1e-12 for v in c['base'].values()))
    d_ok.append(all(abs(c['cert']['err_label'] - v['err_label']) <= .003 for v in c['base'].values()))
    for k, v in c['base'].items():
        per_base.setdefault(k, []).append(ce <= v['err_reader'] + 1e-12)
print('(c) cert reader-err <= all matched baselines:', sum(c_ok), '/', len(c_ok), round(np.mean(c_ok), 3), '| per baseline', {k: round(float(np.mean(v)), 3) for k, v in per_base.items()})
print('(d) |label err diff| <= .003 for all baselines:', sum(d_ok), '/', len(d_ok), round(np.mean(d_ok), 3))
# (e) causal V1 q=1 m=2: k=1 abstention <= k=2
e = []
for ds, r in d.items():
    g = lambda k: next(c for c in r['cells'] if (c['k'], c['q'], c['var'], c['mode'], c['m']) == (k, 1.0, 'V1', 'causal', 2))['cert']['abstain']
    e.append(g(1) <= g(2)); print('  (e)', ds, 'k=1', round(g(1), 3), 'k=2', round(g(2), 3), 'k=3', round(g(3), 3))
print('(e) datasets where k=1 abstains <= k=2:', sum(e), '/ 3')
# headline cells: q=1.0, V1, causal, m=2/4, k=1,2
for ds, r in d.items():
    for c in r['cells']:
        if c['q'] == 1.0 and c['var'] == 'V1' and c['mode'] == 'causal' and c['m'] in (2, 4) and c['k'] in (1, 2):
            print(ds, 'k', c['k'], 'm', c['m'], 'M', f"{c['M']:.3g}", 'cert err_reader', round(c['cert']['err_reader'], 4), 'abst', round(c['cert']['abstain'], 3), '|', ' '.join(f"{n} {v['err_reader']:.4f}" for n, v in c['base'].items()))
