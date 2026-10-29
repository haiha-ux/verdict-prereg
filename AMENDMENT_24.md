# 103. Age-discounted decisions between frames on surgical video with expert labels (Cataract-101; Amendment 24, 2026-10-29)
STATUS: design drafted before the archive was opened; completed after reading ONLY README.md, phases.csv, videos.csv, annotations.csv (archive md5 227678f260243c82be36a3f45a4695c1 = Zenodo). Locked publicly BEFORE any video is decoded.
Data: Cataract-101 (Schoeffmann et al., MMSys 2018; Zenodo record 1220951, CC BY-NC 4.0, research use only, attribution). 101 cataract surgeries, 4 surgeons (2 experience levels), one clinic; phase start points annotated by ophthalmic surgeons (human expert labels). New domain: not robot, not camera trap, never used by the authors.

## Binary state (fixed now, by name, from the published phase list)
y(t) = 1 if frame t lies in the phase 'phacoemulsification' (from its annotated start to the start of the next annotated phase), else 0. Chosen before any data: the longest and visually distinctive phase (phaco tip in the eye). Frames before the first annotated start are excluded.

## Sampling and reader
Base grid: every 5th native frame (5 fps at 25 fps native; exact fps to be read from the files). Videos split by video id, numpy RandomState(20261031): dev_train 40, dev_calib 20, eval 41. Reader: frozen DINOv2-small CLS + mean-patch (as all VERDICT work), logistic probe (C fixed as theory/68) trained on dev_train base-grid frames; tau = the probe's dev_train threshold (t2_run convention). Decode -> embed streaming (no frame cache on disk).

## Rules and constants (as theory/101-102; all from dev_calib)
Nodes every m-th base frame, m in {2, 4, 8, 16} (2.5, 1.25, .625, .31 Hz). cert_k1 = |z_last - tau| / (e + L age), e and L from dev_calib (q = 1.0); hold; ema (alpha .5); hold_c (c* tuned on dev_calib); lr = logistic ranker on (|z_last - tau|, age, |z_last - z_prev|) features of tc7_learned.design, fit on dev_calib.

## Claims (stated before any data; unit = video, 41 eval videos; bootstrap by video, 2000 draws)
H1 (co-primary, reader target): at m in {8, 16}, AURC(cert_k1) < AURC(hold) and < AURC(ema), one-sided 95% upper bound < 0, all 4 comparisons.
H2 (co-primary, EXPERT LABEL target): same 4 comparisons against the human phase label.
H3 (secondary): non-inferiority of cert_k1 to lr within 10% (as Q1 of Amendment 23), m in {8, 16}.
Predictions: H1 passes (age discounting helped on robots); H2 is uncertain (on robots the label result was mixed); H3 fails in at least one cell (as on robots).
Limits declared now: 41 eval videos (< 100 independent groups in the eval pool; the full set has 101 videos, 4 surgeons, 1 clinic), so this is NOT Tier-1 by our own definition; surgeon is a cluster (reported per surgeon, descriptive).

## Completed from the metadata (no video seen)
Phase 5 = 'Phacoemulsificiation' (spelling of phases.csv). Phases can recur in a video (e.g. phase 2 twice); the phase of a frame is that of the latest annotated start <= frame. All videos are 25 fps; base grid = every 5th decoded frame (5 per second). Split (experiments/out/cat/labels.json): train 40 videos (90,345 labelled grid frames, positive share .302), calib 20 (56,996, .323), eval 41 (99,427, .303); every split contains all 4 surgeons. The probe uses t2_run.fit_probe (C and tau chosen on calib), so e, L, c*, the lr ranker and the probe threshold all come from the 20 calib videos.
Script experiments/tc8_cataract.py (stages labels | decode | embed | run). One run; independent critic afterwards. Per-surgeon results are not computed in the locked script (descriptive follow-up only, flagged as such).

## Addendum 24b (2026-10-29, after the single run; result_tc8.json sha256 5ef0f3583460e8ff6c3aecefd688727789fdf0d8e00a9fb17ee4474d814c1afd)
Run: 101 videos decoded (every 5th frame) and embedded, no code change after locking. Note: line 'C fixed as theory/68 ... streaming (no frame cache)' of the draft section is superseded by the completed section and the locked script (C and tau chosen on calib by t2_run.fit_probe; one video's frames cached at a time, deleted after embedding).
Reader: C .01, eval AUC .956, eval error .099, positive share .303 (a much noisier reader than on the robot sets, error < .01).
H1 FAILED, H2 FAILED, H3 FAILED, all in the OPPOSITE direction: cert_k1 has HIGHER AURC than hold and ema in all 8 primary comparisons (95% upper bounds of the difference > 0 and point estimates > 0). AURC reader cert_k1 / hold / ema / lr: m=8 .0182/.0171/.0130/.0160; m=16 .0232/.0199/.0161/.0190. Expert-label target: m=8 .0281/.0250/.0160/.0221; m=16 .0318/.0255/.0163/.0229. The best ranker is the exponential average at every m in {2..16} on the label target and at every m on the reader target; median5 and sprt5 are also better than cert_k1 on the label target.
Reading allowed: on surgical video with expert labels and a noisy frozen reader, discounting the last observation by its age does not help and is worse than smoothing; the age-discounted rule found on two robot datasets (Amendments 22-23) does NOT generalise to this domain. A plausible explanation (not tested): when the reader's frame-level output flickers, averaging nodes removes noise, while age discounting only ranks by recency. The predictions (H1 pass) were wrong; H2 and H3 failed as uncertain/predicted.
