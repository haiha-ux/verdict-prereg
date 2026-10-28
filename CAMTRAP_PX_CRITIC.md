# Critic report: Amendment 16 / 16b (PX cross-set pools), 2026-10-28
Mode code + gia_thuyet. Read: theory/93_amendment16.md, experiments/ct_px.py, ct_run.py (fit_fold, thresholds, errs, cert), t2_run.py (fit_probe, la_threshold, logits, auc), out/camtrap/{result_px.json, px_summary.txt, locks.txt, result_ct.json}, previous critics (camtrap, camtrap-rep). Checks run (one process, venv -I, MemoryMax 8G): re-ran the 6 pairs x 10 seeds with the same rng and fit_fold calls plus extra diagnostics (predicted-positive rates, FP/FN), script in scratchpad (diag.py). Re-run reproduces result_px.json exactly (max abs diff 0 on err R0, err Q, AUC for all 60 cells): the run is deterministic given the saved features. Also compared the public repo (gh api, read only).

## Provenance
- Public repo haiha-ux/verdict-prereg: 58bfba4 (13:13:52Z) holds CT_PX.py sha 0e92de9d... (= ct_px.py in tree, = locks.txt), AMENDMENT_16.md sha 0c1e99c3... (design, = locks.txt line), CT_RUN.py fc65e8a7... (= ct_run.py in tree). bceadb5 (13:15:47Z) adds result json/summary and AMENDMENT_16.md sha 6adf9c6c... (= tree). result_px.json mtime 13:15:23Z (local 20:15:23, +0700) is after the lock and before 16b; sha ab3e8a37... matches. One caveat: ct_px.py mtime is 13:13:52.15Z, the same second as the lock commit, so the mtime proves nothing about when it was written; only the pushed hash does (adequate).

## Verdict per claim
| # | claim | verdict | one line |
|---|---|---|---|
| 1 | Run faithful to the lock; 16b table numbers = result_px.json | VALID | all 6 rows re-derived (AUC, priors, regrets, cert, coverage); re-run identical |
| 2 | No target data in fitting; R0/Q/Otsu label-free; cert from A only | VALID | see section 1 |
| 3 | (a) "Q worse than R0 where shift >= .30" scored FALSE | VALID as a mean statement; NOT ROBUST at seed level | Q<R0 in 9/10, 1/10, 5/10, 9/10 seeds (CCT->Ser, Wel->Ser, Ser->CCT, Ser->Wel) |
| 4 | "Q better in 3 of 4 pairs" | VALID-BUT-TRIVIAL | arithmetic of rate pinning; only 2 of the 3 are consistent across seeds; one is a coin flip |
| 5 | Interpretation "R0 threshold badly off, Q partly corrects it" | UNVERIFIED / mostly speculative | diagnostics below support it for Ser->Wel only |
| 6 | Q meaningful when dev prior .85 and target .46? | INVALID as a test of Q | Q cannot reach the target prior by construction (section 2) |
| 7 | Cert coverage 0/10, 1/10, 9/10 (R0 Ser->Wel), 10/10 | VALID numbers | figures match; interpretation needs correction (section 3) |
| 8 | "width .43 certificate is vacuous / larger than the whole error range" | INVALID wording | .43 is above the plan's .30 bar, but it is not vacuous: it tracks the realised regret (.40) |
| 9 | (b) holds in direction | VALID-BUT-TRIVIAL | only 2 pairs inside the band; oracle ~ constant rule there, regret <= .03 |
| 10 | (c) AUC drop in 6/6 | VALID (6/6 vs target set, 4/6 vs source set); comparability caveat | see section 4 |
| 11 | (d) constant rule competitive | VALID; baseline column mis-framed for Serengeti targets | see section 4 |
| 12 | Otsu "prevalence-dependent heuristic, not a competitor" | UNVERIFIED (set vs prevalence confounded) | Otsu good only on Serengeti targets (prior .46) |
| 13 | Reading "consistent with Prop C', no claim any rule/certificate works" | VALID-BUT-TRIVIAL | the failure is arithmetic of prior shift; wording is correctly modest |

## 1. Fidelity and leakage (file:line)
- ct_px.py:28 `cr.fit_fold(Fa, ya, sa, set(sa), rng)`: tr_sites = all sites of A. ct_run.py:70-76: sites shuffled, 25% (CCT 83 train/27 calibration sites, Wellington and Serengeti 113/37) go to calibration, 75% to training; `m = isin(site, s)` is all of A. fit_probe (t2_run.py:139-152) picks C and tS on the calibration split only; rho = (zcal > tS).mean(), i.e. the PREDICTED-POSITIVE rate on A's calibration, not the prior. No B data enters fit_fold (Fb is touched only at ct_px.py:29).
- ct_run.py:58-61 `thresholds(z, y, tS, rho)`: R0 = tS, Q = quantile(z, 1-rho), Otsu(z); y is used only in `la_threshold(z, y)` for the oracle (in-sample on target; at n = 2750-3750 its optimism is <= ~.005). Label-free rules confirmed. cert (ct_run.py:82-97) uses P['zcal'], P['ycal'] (A) and target z only for r = (z <= t).mean(); no target label.
- The 10 seeds (ct_px.py:27) change only the site split of A into train/calibration. The target frames are identical in all 10 seeds, so the seed SD is calibration-split variability, not a CI over target sites. Calibration prior ("prior dev") is the mean over splits of the calibration-site prior, range .77-.92 (CCT), .80-.84 (Wel), .45-.56 (Ser).

## 2. Q vs R0 (claims 3-6): what the diagnostics show
Mean predicted-positive rate on the target (z > t) vs target prior, from my re-run:
| pair | target prior | rho (Q rate) | R0 pos-rate | err R0 | err Q | const-min err |
|---|---|---|---|---|---|---|
| CCT->Ser | .459 | .943 | .987 | .529 | .490 | .459 |
| Wel->Ser | .459 | .979 | .953 | .518 | .530 | .459 |
| Ser->CCT | .861 | .462 | .441 | .448 | .432 | .139 |
| Ser->Wel | .821 | .461 | .295 | .581 | .469 | .179 |
- Q simply imposes the source predicted-positive rate (about the source prior: .94/.98/.46/.46) on a target whose prior differs by .35-.50. Its error is therefore bounded below by |rho - pi_T| (.48, .52, .40, .36); observed Q errors are .49, .53, .43, .47. In the 4 shifted pairs Q is a mis-specified rule by construction; comparing it with R0 is comparing two prior-blind rules, and "who is less wrong" is set by whether the target's score shift moves R0's positive rate toward or away from the target prior.
- CCT->Ser: Q is .04 better because R0 predicts .987 positive and Q .943 (both about "all animal"); this is not a correction toward the target prior (.459). Wel->Ser: R0 .953 vs Q .979, Q worse. Ser->CCT: .441 vs .462, difference .016 with seed SD .042 (Q<R0 in 5/10): no evidence of any difference. Ser->Wel: R0 falls to .295 (score distributions of Wellington shifted down, AUC .63) and Q restores .461; this is the only pair where "Q partly corrects R0" is supported (regret .402 -> .290, 9/10 seeds), and even there both are .29-.40 above the oracle and .29-.40 above the constant "animal" rule.
- Not in the table: CCT<->Wel. Q pos-rate .93-.97 and R0 .93-.94: both are near "always animal", which is why regret R0 ~ Q ~ .01-.03 there. "Q close to R0 for CCT<->Wel" (second half of (a)) is TRUE and trivial; 16b does not score it.
- Oracle = "always animal" (pos-rate .98-1.0) in the 4 pairs with target CCT or Wellington, so "regret" there is the excess over the constant rule; only the 2 Serengeti-target pairs have a real oracle threshold (pos-rate .40, .35).

## 3. Certificate (claims 7-9)
- Coverage (reg <= cert) re-derived: CCT->Wel 10/10 both; CCT->Ser 0/0; Wel->CCT 10/10 both; Wel->Ser 1/1; Ser->CCT R0 4, Q 2; Ser->Wel R0 9, Q 0. 16b's prose omits Ser->CCT (4/10 and 2/10) from the "failed or vacuous" list.
- The failures are mostly the band assumption (ct_run.py:21 BAND = .30 around A's calibration prior; ct_run.py:91): the true target prior lies outside [cal prior +- .30] in 10/10 seeds for CCT->Ser and Wel->Ser, 9/10 for Ser->CCT and Ser->Wel, 0/10 for the two near-identity pairs. So "certificate failed beyond .30" is the identified-set assumption being false, as the prediction itself said; it is not a test of the certificate inside its assumptions, and the only pairs inside the band are the 2 trivial ones (regret <= .03 vs cert .11-.15, oracle = constant rule). Ser->Wel R0 covers 9/10 although the prior is outside the band, because the max over p sits at the band edge (.78) close to .82.
- Ser->Wel R0: cert .432 vs regret .402; per seed cert - regret = [.084, -.036, .036, .002, .031, .044, .045, .022, .026, .049]: seed 3 covered by .002, seed 1 not covered. So 9/10 is a thin margin, and the bound is nearly tight, not vacuous. Spearman cert vs regret within this pair is .90 (all 60 cells: R0 .65, Q .48), i.e. the label-free certificate tracks the realised regret here. Reported as an observation, not a claim (6 dependent pairs, no CI).
- Pre-registered informativeness bar (ct_run.py:212) is mean width <= .30; .43 fails it. That is the correct sentence. "Larger than the whole error range" is imprecise (the R0-oracle gap is .40; a regret of .43 is not >= any attainable maximum, max regret would be about .82). Do not call it vacuous.

## 4. Predictions (a)-(d), scoring
- (a) as scored FALSE is arithmetically right on means (.327<.366, .293<.310, .290<.402; .170>.158). But 3 of 4 is carried by 2 consistent pairs; Ser->CCT is 5/10 and Wel->Ser reversed. Recommend scoring (a) "first conjunct not supported (Q-R0 sign unstable); second conjunct (CCT<->Wel close) true". Do not say "Q better than R0".
- (b) TRUE in direction, trivial (20/20 coverage in 2 pairs where both regrets are ~0.01-0.03).
- (c) "drop relative to the within-set cross-fits": vs the TARGET set's within-set AUC (.849/.729/.984) all 6 drop (.655<.729, .905<.984, .799<.849, .675<.984, .756<.849, .631<.729). vs the SOURCE set's value only 4 of 6 (CCT->Ser .905 > .849, Wel->CCT .799 > .729). Either reading gives >= 4/6, so TRUE; state which reading. Comparability caveat: within-set AUC is the mean of 5 per-fold AUCs on 375-500 frames (CCT folds .806-.887, Wel .643-.770, Ser .973-.993); PX AUC is pooled over 2750-3750 frames of 110-150 heterogeneous sites. CCT->Wel .655 (seed range .636-.673) lies inside the Wellington per-fold range (.643-.770), Ser->Wel .631 just below; so for Wellington targets the "drop" is within within-set fold variability.
- (d) TRUE: constant-animal error <= R0 error in all 4 pairs with CCT/Wel target (.179<.208, .139<.148, .139<.448, .179<.581). The 16b column "error R0 vs constant-animal" is mis-framed for Serengeti targets: R0 .529/.518 vs "animal" .541 looks like a win, but the best trivial rule there is "always empty" (.459 = err_const, stored in result_px.json); R0 is worse than it by .07 and .06 despite AUC .905 / .675.

## 5. Confounds (claim 12-13 and general)
- n_eff = 3 sets; 6 pairs share sets; the 4 shifted pairs are really one contrast (Serengeti ~ .47 vs CCT/Wel ~ .82-.86) in two directions with two near-duplicate partners. Prior shift is fully confounded with set identity (species, region, camera, season, label protocol) and with large covariate shift (AUC .98 -> .63-.91). No site bootstrap, no CI; seeds do not vary the target.
- Serengeti prior .46 is the prior of the successfully-fetched sampled frames; earlier critic (D6) found that all 404 failed fetches were animal frames, so the .46 is label-dependent on missingness. Otsu's success only on Serengeti targets cannot separate "prevalence" from "set".
- CCT and Wellington are near-constant label (.86/.82 animal): any rule predicting about 93-100% positive is within .03 of the oracle there; results for those targets carry no information on threshold quality.
- `regret` is defined against an in-sample target oracle that collapses to a constant in 4 of 6 pairs.

## Required corrections to Amendment 16b (before any text uses this)
1. Replace "Q is better than R0 in 3 of the 4 such pairs" by the seed counts (Q<R0 in 9/10, 1/10, 5/10, 9/10) and say that the sign is not stable; score (a) second conjunct explicitly.
2. Delete or soften "R0's source threshold is badly off and the quantile rule partly corrects it". Replace by: "Q imposes the source predicted-positive rate (.94-.98 or .46); it moves toward the target only when R0's own rate falls (Ser->Wel, .295 -> .461); in no pair does either rule approach the target prior".
3. Replace "vacuous"/"larger than the whole error range" by "mean width .43 exceeds the plan's .30 informativeness bar; it covered R0 in 9/10 seeds with margins down to .002 and missed Q in 10/10".
4. Add Ser->CCT coverage (R0 4/10, Q 2/10) to the list; add that the target prior lies outside the +-.30 band in 9-10/10 seeds for the four shifted pairs, so these are band-assumption violations, not certificate tests; and that inside the band (2 pairs) the oracle equals the constant rule.
5. Add a "best constant (err_const)" column; state R0 is worse than "always empty" (.459) in both Serengeti-target pairs and note that oracle = "always animal" in the 4 other pairs.
6. (c): say vs which within-set AUC; give both counts (6/6, 4/6) and the pooled-vs-per-fold caveat.
7. Otsu sentence: "good only on the two Serengeti-target pairs (prior .46); prevalence and set are confounded", not "so it is a prevalence-dependent heuristic".
8. Record: 10 seeds vary only A's train/calibration split; no CI over target sites; n_eff = 3 sets.

## Allowed wording
- "On 6 ordered cross-set pairs (3 camera-trap sets), a probe fit on all sites of one set was evaluated on all frames of another. AUC fell below the within-set value of the target set in 6/6 pairs. Where the animal prior differed by .35-.50 (4 pairs, all involving Serengeti) the source threshold and quantile matching gave errors .43-.58, above the best constant rule in all 4 pairs and above the oracle by .16-.40; which of R0 and Q is better varied with the pair and with the calibration split; Otsu was near the oracle only for the two Serengeti targets."
- "The label-free identified-set certificate (band +-.30) covered the realised regret in the 2 pairs where the target prior was within the band (regret <= .03) and covered it in only 17 of 80 seed-rule cases (4 pairs x 2 rules x 10 seeds) where the target prior was outside it; its mean width in Ser->Wel (.43) exceeded the .30 bar."
- "Exploratory, single locked run, deterministic given features; 3 sets, no CI."
## Disallowed wording
- "Q works/fails in general", "quantile matching corrects a bad threshold", "Q better than R0 under prior shift".
- "the certificate is validated / calibrated / vacuous / works under shift".
- "the constant rule beats the probe" without stating it is the best trivial rule only for CCT/Wel targets.
- "Otsu is prevalence-dependent" as a finding; any "Tier-1 reached", "A*", "novelty/first", "generalises across datasets" (3 sets, WCS unconfirmed, PX dependent).

## Summary
Run is faithful and reproducible (re-run diff 0; lock hashes verified on the public repo; no target leakage). All table numbers in 16b are right. The problems are interpretive: the "Q beats R0 in 3/4" claim is not stable across seeds and is mostly arithmetic (Q imposes the source predicted-positive rate, so it cannot track a prior shift of .35-.50); the "R0 badly off, Q partly corrects" mechanism holds only for Ser->Wel; the ".43 width is vacuous" wording is wrong (it tracks regret, is above the .30 bar); certificate failures are band-assumption violations, and the two pairs inside the band are trivial because the oracle is the constant rule; the constant-rule column hides that "always empty" beats R0 on Serengeti targets. Predictions (c) and (d) hold; (b) holds trivially; (a) needs the seed-level qualification. Apply the 8 corrections as an append-only 16c.
