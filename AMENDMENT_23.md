# 102. Age-discounted margin vs learned confidence rankers on a second fresh sample (Amendment 23, 2026-10-29)
STATUS: written before any frame of the episodes below was decoded. Motivation: the critic of Amendment 22 (.ai/runs/2026-10-29-critic-tc6.md) noted that no learned baseline was tried; a ranker fit on (|z_last - tau|, age, |z_last - z_prev|) could make the hand-made margin pointless.

## Data
Same two datasets and video files as theory/101. Episodes: 120 per dataset, RandomState(20261030).choice over the episodes in neither data/lr_<ds>_manifest.json nor experiments/out/tc/idx_<ds>F.json (mutex 1230 left, sirius 289). Native rate, locked label rule.

## Rankers (all decide the side of the last node; they differ only in the confidence that orders frames)
cert_k1: |z_last - tau| / (e + L age / fps) (e, L from dev_calib, as theory/101). hold_c: |z_last - tau| / (age + c*) (c* from dev_calib). hold: |z_last - tau|. lr: logistic regression, gbt: gradient-boosted trees (HistGradientBoosting, 200 iterations, lr .05), both on 8 features of (a, age, d) (`tc7_learned.design`), target = the hold decision disagrees with the reader, fit on dev_calib frames only (the same episodes used for e, L, c*). Confidence = minus predicted error probability.

## Claims (m in {8, 16}; reader target; episode bootstrap 2000, seed 20261030)
Q1 (primary, non-inferiority): in every (dataset, m) cell and for both learned rankers, the one-sided 95% upper bound of AURC(cert_k1) - AURC(learned) is below 10% of AURC(learned). All 8 must pass.
Q2 (secondary): same comparison for hold_c. Q3: label target, descriptive.
Prediction stated now: Q1 FAILS in at least one cell (gbt has more information and hold_c already beat cert_k1 in Amendment 22c). A fail means the hand rule is not competitive with a learned ranker and the paper must say so; a pass means the one-constant rule is as good as a fitted ranker on these data.

## Procedure
Script experiments/tc7_learned.py (stages idx | decode | feat | run -> experiments/out/tc/result_tc7.json). sha256 of this file and the script pushed publicly as Amendment 23 BEFORE stage idx. One run; independent critic afterwards. Limits as theory/101 (same robots, cameras, probe).
Disclosure: before locking, tc7 frames/design/margins were smoke-tested on dev_calib and 5 already-analysed mutex eval episodes (shapes only, no AURC computed); no fresh episode was read.
