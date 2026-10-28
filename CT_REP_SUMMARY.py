"""Summary of result_rep.json (Amendment 15): prints the numbers quoted in Amendment 15b."""
import json, numpy as np
d = json.load(open('experiments/out/camtrap/result_rep.json')); reps = d['reps']
for s in ('cct', 'wel', 'ss3'):
    g = lambda f: [f(r['sets'][s]) for r in reps]
    print(s, 'regret R0/Q/Otsu', [round(float(np.mean(g(lambda x, k=k: x['regret'][k]))), 3) for k in ('R0', 'Q', 'Otsu')],
          'errR0', round(float(np.mean(g(lambda x: x['err_R0']))), 3), 'oracle', round(float(np.mean(g(lambda x: x['err_oracle']))), 3),
          'const', round(float(np.mean(g(lambda x: x['err_const_animal']))), 3), 'auc', round(float(np.mean(g(lambda x: x['auc']))), 3))
    for k in ('5', '10', '20'):
        b = [r['sets'][s]['audit'][k]['betting_cover'] for r in reps]; c = [r['sets'][s]['audit'][k]['cp_cover'] for r in reps]
        u = [r['sets'][s]['audit'][k]['betting_U'] / r['sets'][s]['audit'][k]['truth'] for r in reps]
        print('  k', k, 'betting mean', round(float(np.mean(b)), 3), '#rep<.95', sum(x < .95 for x in b), '| CP mean', round(float(np.mean(c)), 3), '#rep<.95', sum(x < .95 for x in c), '| median U/truth', round(float(np.median(u)), 1))
c2 = [r['c2'] for r in reps]; v = [x['mean_Q_minus_R0'] for x in c2]
print('C2 pools/rep', [x['n_pools'] for x in c2], 'sets', sorted({tuple(x['sets']) for x in c2}), 'Q-R0 mean', round(float(np.mean(v)), 3), 'min/max', round(min(v), 3), round(max(v), 3))
c3 = [r['c3'] for r in reps]; n = sum(x['n'] for x in c3); cv = sum(x['covered'] for x in c3); ns = sum(x['n_shifted'] for x in c3); cs = sum(x['covered_shifted'] for x in c3)
print('C3', cv, '/', n, '| shifted', cs, '/', ns, '| unshifted', cv - cs, '/', n - ns)
