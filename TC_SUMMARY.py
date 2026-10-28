"""Summary of result_tc.json for Amendment 17b."""
import json
d = json.load(open('experiments/out/tc/result_tc.json'))
for ds, r in d.items():
    print(ds, 'eval episodes', r['n_eval_eps'], 'fps', r['fps'], 'tau', round(r['tau'], 3))
    for k in (2, 3, 4):
        for q in (0.9, 1.0):
            row = [c for c in r['cells'] if c['k'] == k and c['q'] == q]
            print(f'  k={k} q={q} e={row[0]["e"]:.2f} M={row[0]["M"]:.3g} | ' + ' | '.join(f'm={c["m"]}: err {c["err"]:.3f} ab {c["abstain"]:.2f} (base {c["base_err"]:.3f})' for c in row))
