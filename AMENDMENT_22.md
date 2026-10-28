# 101. Confirmatory test of the sparse-frame gain on FRESH episodes (Amendment 22, 2026-10-29)
STATUS: written before any frame of the episodes below was decoded or scored. Purpose: confirm finding B3 of theory/100 (found in tests 3-4 on the same eval episodes, so far only exploratory) on episodes never touched by any analysis, with every constant fixed beforehand.

## Data
lerobot/utaustin_mutex and lerobot/austin_sirius_dataset, the wrist video file already on disk (data/lr_<ds>_raw, sha256 verified in theory/85). FRESH = episodes not in data/lr_<ds>_manifest.json (mutex 1350, sirius 409; checked from metadata only). Sample: 120 fresh episodes per dataset, numpy RandomState(20261029).choice without replacement over the sorted fresh ids. Labels: the locked gripper rule of theory/85 (u <= .15 open, u >= .40 closed, band excluded). Native rate (20 fps), all frames.
Unit of inference = episode (120 per dataset, >= 100).

## Everything fixed from the earlier dev_calib episodes (nothing is fit on fresh episodes)
Reader: locked probe experiments/out/lerobot/fit_<ds>-<ds>.json and its threshold tS. e (q = 1.0) and L (= M_1, q = 1.0, V1) from dev_calib as in tc4_rc.py. c* of hold_c tuned on dev_calib over the same grid as tc4. Operating points theta* (risk eps = .002, reader target) from dev_calib. droid has no fresh episodes and is not in this test.

## Methods (causal, between-node frames, as tc4_rc.episode)
cert_k1 (PRIMARY): margin |z_last - tau| / (e + L age/fps); this is the k = 1 instance of the hypothesis and equals an age-discounted hold whose constant c = e fps / L comes from theory, not tuning. Comparators: hold (plain |z_last - tau|), ema. Secondary: hold_c, cone_all, median5, sprt5.

## Claims (stated before the run; m = nodes every m-th frame)
P1 (primary). For each dataset and each m in {8, 16}: AURC (reader target) of cert_k1 is lower than that of hold AND lower than that of ema; one-sided episode-bootstrap (2000 draws, seed 20261029) 95% upper bound of the difference < 0. Passes only if all 2 x 2 x 2 = 8 comparisons pass (intersection-union; no multiplicity correction needed).
P2. At eps = .002 with dev theta*, eval coverage of cert_k1 minus that of hold is >= .02 in every (dataset, m in {8, 16}) cell, and the eval risk of cert_k1 is <= .004.
P3 (label target, secondary, descriptive). Same comparison as P1 against the sensor label.
Reported, not claimed: m in {2, 4}; all secondary methods.
Predictions: P1 passes; P2 passes for m = 8 and probably 16; at m = 2 all rules tie (as before). A fail of P1 means B3 does not replicate, and theory/100 B3 is withdrawn.

## Procedure and locks
Script experiments/tc6_fresh.py: stages idx (venv_dec; metadata only) | decode (venv_dec, streaming into data/tc_<ds>F_dense.npy) | feat (.venv) | run (.venv) -> experiments/out/tc/result_tc6.json. sha256 of this file and of tc6_fresh.py pushed to the public prereg repo as Amendment 22 BEFORE stage idx. One run; any code fix after the run is recorded as an addendum with the defect and both results. Independent critic afterwards.
Limits: same two datasets, same robots, cameras and probes as before (fresh episodes, not new domains); the reader target measures how well a rule predicts the reader's own decision on unobserved frames, not correctness of the scene label (P3 covers the label).
Disclosure: before locking, tc6_fresh.per_episode / boot_diff were smoke-tested on 10 already-analysed mutex eval episodes (m = 8, 20 draws) only to check that the code runs; no fresh episode was read.

## Addendum 22b (2026-10-29, after the single run; result_tc6.json sha256 a8a6e84dd0d1a1191eec242729c6de098cf78928b21dd6eb54a3dab7fb7307f5)
Run: mutex 120 fresh episodes (28,648 frames), sirius 120 (59,394). No code change after locking; one launch failed before reading any data (zsh did not split a command variable) and was relaunched unchanged.
P1 PASSED: 8/8 one-sided 95% upper bounds < 0. AURC reader (cert_k1 / hold / ema): mutex m=8 .00296/.00373/.00853, m=16 .00923/.01389/.03054; sirius m=8 .00238/.00333/.00934, m=16 .01044/.01721/.03938.
P3 (label target) also 8/8 upper bounds < 0.
P2 FAILED in 1 of 4 cells. Coverage at eps .002 with dev theta* (cert_k1 vs hold, eval risk in brackets): mutex m=8 .751 (.0031) vs .602 (.0025); mutex m=16 .451 (.0033) vs .451 (.0084) -> gain 0 < .02, fail; sirius m=8 .704 (.0023) vs .465 (.0024); sirius m=16 .342 (.0019) vs .111 (.0046). Descriptive, not claimed: in the failing cell hold reaches equal coverage only by exceeding its target risk about 4x; cert_k1 eval risk stays <= .0033 in all four cells. The eval risk of cert_k1 exceeds the .002 target in 2 cells (.0031, .0033), within the .004 bound of P2.
Reading allowed: on fresh episodes the k = 1 smoothness margin ranks between-node frames better than plain hold and EMA at 2.5 Hz and 1.25 Hz (P1), against both the reader and the sensor label. B3 of theory/100 is confirmed in the AURC form; the fixed-coverage form (P2) is not fully confirmed. Same two datasets, cameras and probes as before.
