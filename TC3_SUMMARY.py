"""Scores predictions (a)-(d) of theory/97 from result_tc3.json."""
import json
d = json.load(open('experiments/out/tc/result_tc3.json'))
cells = [(ds, c) for ds, r in d.items() for c in r['cells']]
a = [c['methods']['cert_k1']['aurc_reader'] < c['methods']['hold']['aurc_reader'] for _, c in cells]
b = [c['methods']['cert_k2']['aurc_reader'] > c['methods']['cert_k1']['aurc_reader'] for _, c in cells]
cc = [(c['methods']['cert_k1']['op']['0.002']['risk_eval'] is not None and c['methods']['cert_k1']['op']['0.002']['risk_eval'] <= .005) for _, c in cells]
dd = [(c['methods']['cert_k1']['op']['0.002']['cov_eval'] - c['methods']['hold']['op']['0.002']['cov_eval']) >= .02 for _, c in cells]
print('(a) cert_k1 AURC_reader < hold:', sum(a), '/ 9'); print('(b) cert_k2 AURC > cert_k1:', sum(b), '/ 9'); print('(c) eps=.002 cert_k1 eval risk <= .005:', sum(cc), '/ 9'); print('(d) eps=.002 cov(cert_k1) - cov(hold) >= .02:', sum(dd), '/ 9')
for ds, r in d.items():
    for c in r['cells']:
        mm = c['methods']
        print(ds, 'm', c['m'], 'n_eval', c['n_eval_frames'], '| AURC reader', {k: round(v['aurc_reader'], 4) for k, v in mm.items()}, '| AURC label', {k: round(v['aurc_label'], 4) for k, v in mm.items()})
        for eps in ('0.0', '0.002', '0.01'):
            print('    eps', eps, {k: (round(v['op'][eps]['cov_eval'], 3), None if v['op'][eps]['risk_eval'] is None else round(v['op'][eps]['risk_eval'], 4)) for k, v in mm.items()})
