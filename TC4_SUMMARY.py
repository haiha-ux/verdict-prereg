"""Scores predictions (a)-(e) of theory/98 from result_tc4.json."""
import json
d = json.load(open('experiments/out/tc/result_tc4.json'))
cells = [(ds, c) for ds, r in d.items() for c in r['cells']]
A = lambda c, k: c['methods'][k]['aurc_reader']
a = [A(c, 'cone_all') <= A(c, 'hold_c') + 1e-12 for _, c in cells if c['m'] in (4, 8)]
b = [A(c, 'cone_all') < A(c, 'cert_k1') for _, c in cells]
c1 = [A(c, 'median5') > A(c, 'hold') for _, c in cells]; c2 = [A(c, 'sprt5') > A(c, 'hold') for _, c in cells]
dd = [c['empty_frac_cone_all'] <= .01 for _, c in cells]
e = [(c['methods']['cone_all']['op']['0.002']['cov_eval'] - c['methods']['hold_c']['op']['0.002']['cov_eval']) >= .02 for _, c in cells]
print('(a) cone_all AURC <= hold_c, m in {4,8}:', sum(a), '/ 6'); print('(b) cone_all < cert_k1:', sum(b), '/ 9')
print('(c) median5 > hold:', sum(c1), '/ 9 ; sprt5 > hold:', sum(c2), '/ 9'); print('(d) empty interval <= 1%:', sum(dd), '/ 9'); print('(e) cov(cone_all) - cov(hold_c) >= .02 at eps .002:', sum(e), '/ 9')
for ds, r in d.items():
    for c in r['cells']:
        print(ds, 'm', c['m'], 'c*', c['c_star'], 'empty', round(c['empty_frac_cone_all'], 4), '| AURC reader', {k: round(v['aurc_reader'], 5) for k, v in c['methods'].items()})
        print('    cov@.002 (risk)', {k: (round(v['op']['0.002']['cov_eval'], 3), None if v['op']['0.002']['risk_eval'] is None else round(v['op']['0.002']['risk_eval'], 4)) for k, v in c['methods'].items()})
