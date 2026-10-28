"""Scores predictions (a)-(d) of theory/99 from result_tc5.json."""
import json
d = json.load(open('experiments/out/tc/result_tc5.json'))
cells = [(ds, c) for ds, r in d.items() for c in r['cells']]
A = lambda c, k: c['methods'][k]['aurc_reader']
a_win = [A(c, 'cone_max') < A(c, 'cert_k1') - 1e-12 for _, c in cells]; a_ok = [A(c, 'cone_max') <= A(c, 'cert_k1') + 1e-12 for _, c in cells]
a48 = [A(c, 'cone_max') < A(c, 'cert_k1') - 1e-12 for _, c in cells if c['m'] in (4, 8)]
b = [A(c, 'cone_max') <= A(c, 'hold_c') + 1e-12 for _, c in cells if c['m'] in (4, 8)]
cc = [(c['methods']['cone_max']['op']['0.002']['cov_eval'] - c['methods']['hold_c']['op']['0.002']['cov_eval']) >= .02 for _, c in cells]
dd = [abs(A(c, 'cone_max_w5') - A(c, 'cone_max')) <= .05 * A(c, 'cone_max') for _, c in cells]
print('(a) cone_max AURC <= cert_k1:', sum(a_ok), '/ 9 ; strict wins in m in {4,8}:', sum(a48), '/ 6'); print('(b) cone_max AURC <= hold_c, m in {4,8}:', sum(b), '/ 6'); print('(c) coverage gain >= .02 vs hold_c at eps .002:', sum(cc), '/ 9'); print('(d) w5 within 5% of cone_max:', sum(dd), '/ 9')
for ds, r in d.items():
    for c in r['cells']:
        print(ds, 'm', c['m'], {k: round(v['aurc_reader'], 5) for k, v in c['methods'].items()}, '| cov@.002', {k: round(v['op']['0.002']['cov_eval'], 3) for k, v in c['methods'].items()})
