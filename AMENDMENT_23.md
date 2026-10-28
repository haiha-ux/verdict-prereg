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

## Addendum 23b (2026-10-29, after the single run; result_tc7.json sha256 c3b035a91a85fb4f41b27ba99870c16c6876cd9e3cc11a7b2a8c8dffa089ebd1)
Run: mutex 120 fresh episodes (27,725 frames), sirius 120 (60,818); no code change after locking.
Q1 FAILED (as predicted): 2 of 4 cells fail, both against lr. AURC reader cert_k1 / hold_c / hold / lr / gbt: mutex m=8 .00307/.00290/.00366/.00301/.00483 (upper bound vs lr .00042 > .10 x .00301, fail); mutex m=16 .01169/.01121/.01827/.01195/.01481 (pass); sirius m=8 .00270/.00261/.00377/.00259/.00437 (pass); sirius m=16 .01098/.01060/.01667/.01023/.01214 (upper bound vs lr .00110 > .00102, fail).
Descriptive (not claimed): cert_k1 point AURC is within -2% to +7% of lr and below gbt in all 4 cells (gbt upper bounds < 0 in 3, ~0 in 1); gbt fit on ~5.8k dev frames (2-5% positives) ranks worse than every hand rule. All age-discounted rankers (cert_k1, hold_c, lr) are 16-37% below plain hold in all 4 cells, which replicates the direction of P1 of Amendment 22 on a second fresh sample (no bootstrap against hold was pre-stated here).
Reading allowed: a one-constant age-discounted margin is roughly as good as a fitted logistic ranker on (|z - tau|, age, change) and better than a boosted-tree ranker at this dev size, but it is NOT shown non-inferior to the logistic ranker within 10%. What helps is discounting by frame age; the specific form (Lipschitz or learned) matters little.
