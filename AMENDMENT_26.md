# 105. Stronger frozen reader on the two human-label video domains (Amendment 26, 2026-10-29)
STATUS: written before any DINOv2-large feature exists. The researcher asked to test a stronger reader after Amendments 24-25 (DINOv2-small: cataract error .099, THUMOS14 reader failed, AUC .67).
Change: encoder facebook/dinov2-large (Apache-2.0), CLS + mean patch (2048-d), same preprocessing. EVERYTHING else identical to Amendments 24 and 25: same videos, splits, grid (5 per second), labels, probe fitting (t2_run.fit_probe), constants from calib, rules, selector (rho >= 2 -> hold_c else ema). Cataract frames are re-decoded from the archive on disk; the same 100 THUMOS14 videos are re-fetched (frames were deleted). Note: the authors know the DINOv2-small results on these exact splits.

## Claims (m in {8, 16}; eval videos; video bootstrap 2000, seed 20261102)
R1 (primary, per domain): the rule picked by the selector on calib has lower AURC than plain hold against the HUMAN label (one-sided 95% upper bound < 0 at both m). Reported per domain; the branch claim 'a between-frame rule beats plain hold against human labels with a stronger reader' needs R1 in both domains.
Reported: probe AUC/error (prediction: both improve), rho and the selection, the non-selected rule against hold, all rules at m in {2, 4, 8, 16}, reader target.
Predictions: cataract: rho < 2, ema selected, R1 passes (with the small reader ema already beat hold on the label). THUMOS14: reader improves but stays weak (AUC < .80); rho < 2; R1 uncertain.

## Procedure
Script experiments/tc10_large.py (stages cat_decode | th_fetch | embed | claims), reusing locked tc8/tc9 functions with output directories redirected to experiments/out/large/. sha256 pushed publicly as Amendment 26 BEFORE any stage. One run; independent critic afterwards. Not Tier-1 (40-41 eval videos per domain).
