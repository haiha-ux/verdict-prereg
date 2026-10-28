# 90. Camera-trap sets with human labels: natural site shift, many independent sites (design only, 2026-10-28)
STATUS: design written before any image was downloaded and before any feature, probe, threshold or audit was computed on these data. Only metadata JSON files (labels + site ids, no images) were downloaded and summarised (`experiments/ct_meta_stats3.py`, `experiments/cct_meta_stats.py`). theory/68 is NOT edited. Nothing here is evidence. Proposed for a public amendment (Amendment 14) to be pushed BEFORE the first image is fetched.
Scout reports: `.ai/runs/2026-10-28-scout-tier1-datasets.md`, `...-camtrap-verify.md`, `...-camtrap-verify2.md`.
Purpose: first test of the VERDICT estimands on data that meets, as far as verified, the Tier-1 definition (>=4 datasets, >=100 independent groups, ground truth not AI-produced, natural shift, group split). It changes the scope of the paper from robot/door perception to threshold certification under site shift in a frozen-feature reader; the robot results stay as they are.
DISCLOSURE: the authors saw only the metadata (counts, per-site empty fractions) in section 1. No author has looked at images of these sets for this study.

## 1. Datasets (all CDLA-Permissive-1.0 via LILA; labels = animal present vs `empty`)
| set | sites in metadata | sites >=60 sequences | empty share | spread of per-site empty share | label source (verified?) |
|---|---|---|---|---|---|
| CCT | 140 | see manifest | ~.70 | median .08, p90 .36 | expert biologists, sequence level (Beery 2018, ar5iv) yes |
| Wellington | 215 ids (LILA says 187: unexplained) | 182 with >=50 imgs | .169 | .10-.26 deciles | citizen scientists + experts, "no automated or AI classifier" (paper) yes |
| Snapshot Serengeti, season 3 | 160 | 160 | .69 | .16-.94 deciles | volunteer consensus, 96.6% vs expert gold set (secondary sources; paper full text NOT read) conditional |
| WCS Camera Traps / iWildCam | 3911 | 1747 | .43 | 0-.79 deciles | expert annotations per iWildCam 2020 paper, but identity of LILA-WCS with that data is NOT confirmed; MegaDetector only for boxes: INCLUDED ONLY IF the identity is confirmed from a primary source before the first image is fetched, else Tier-1 is declared NOT met (3 sets) |
Labels are per sequence in CCT and Serengeti (frame labels inherit the sequence label; noise at image level). Group (independent unit) = site; sequence is nested in site; ONE random frame per sequence is used, so frames in the sample are never from the same sequence.
All metadata and images are untrusted: each set in its own new directory, images are JPEG only, decoded by PIL in a venv, nothing bundled is run.

## 2. Sample (fixed by seed 20261028, from metadata only, before any image; `experiments/ct_prep.py manifest <set>`)
CHANGE OF DESIGN (made after the first manifest build showed only 34-40 eval sites per set under a single 50/50 split, which is below the 100-group requirement; metadata only, no image seen): site-level 5-fold CROSS-FITTING replaces the single dev/eval split. Per set: eligible sites = >=40 sequences (CCT 110, Wellington 181, Serengeti S3 156); seeded sample of up to 150 sites (CCT 110, Wellington 150, Serengeti 150); per site 25 sequences, one seeded random frame each (CCT 2750, Wellington 3750, Serengeti 3750 images). Sites are assigned to 5 folds BY SITE; for fold f the probe is trained and calibrated on the other 4 folds (dev) and evaluated on the sites of fold f; inside fold f one third of its sites are the audit pool, two thirds the evaluation pool. Every site is evaluated exactly once, so the evaluation groups are 110 / 150 / 150 sites. Sites in the other 4 folds are not independent of each other regarding region, but no image of an evaluated site is in its own training set.
Download budget: per-set cap 2.5 GB (approved by the user on 2026-10-28; the earlier 500 MB rule would allow ~1000 images). If the cap is hit, frames per site are reduced uniformly for that set and recorded. Images are fetched individually by URL (LILA unzipped bases, Azure) and resized to 224x224 (aspect not preserved) once; originals are deleted after feature extraction. Fetch failures (404) are counted per set and dropped; if more than 5% of a set fails the set is reported as incomplete.
Eval pools (defined before any image): (P0) all evaluated sites of the set; (PH) evaluated sites with metadata empty share >= .60; (PL) evaluated sites with empty share <= .20 (a prevalence-shift manipulation chosen from labels, declared as such; pools with fewer than 15 sites are skipped); (PX) cross-set: a probe trained on all sites of set A is evaluated on all sites of set B (natural between-region/camera shift).

## 3. Reader and rules (unchanged from the main protocol)
Frozen DINOv2-small, CLS + mean patch feature, linear probe (logistic regression, C fixed as in theory/68, trained on dev sites only). Positive class = animal present. Threshold rules: R0 (source threshold), Q (quantile matching to the dev positive rate), Otsu, oracle (best threshold on the evaluation pool). Audit rules QA/R0A/LA/BA as in Amendment 10. No tuning on eval sites.

## 4. Claims (all pre-stated; Holm over C1-C4; the unit of inference is the SITE, site bootstrap, 2000 draws)
C1 (descriptive, no direction): per pool, excess error over the oracle of R0, Q, Otsu; report the share of pools where Q beats R0. No test.
C2 (prediction of a FAILURE, from Prop C' of theory/84): in prevalence-shifted pools (|pi_eval - pi_dev| >= .20) the mean of excess(Q) - excess(R0) is >= +.02 with site-bootstrap lower 95% bound > 0. Q equalises the positive rate to the dev prior, which is wrong there; a pass confirms the limitation, a fail (Q not worse) falsifies Prop C' as a practical warning.
C3 (T2 certificate, label-free): rho(r) computed with a prior band [pi_dev - .30, pi_dev + .30] and source class CDFs from dev sites covers the realised regret of R0 and of Q in >= 95% of site-bootstrap pools; report mean width. If width is > .30 the certificate is declared uninformative.
C4 (validity of the cluster audit): betting bound on site-cluster error sums has coverage >= .95 of the true pool error for every set and m in {50, 100, 200} sites*frames; frame-level Clopper-Pearson is reported as a comparison and expected to undercover.
Readers that fail (clean error >= .35) are reported, not dropped. A pool is not removed after the fact.
Predictions stated now: (a) Q is worse than R0 in PH and PL (C2); (b) in P0 within-set pools all rules are within .03 of the oracle (little shift from a random site split), so C1 shows small differences; (c) PX pools show larger R0 excess than P0, with Q competitive where priors are similar (Wellington vs CCT) and bad where they are not (Serengeti vs Wellington).

## 5. Known confounds and limits
1. PREVALENCE: the shift between sites is mostly in the empty share; this is exactly what Q cannot identify (Prop C'), so a Q failure here says little about Q under same-prior shift. PH/PL make this explicit; P0 is the near-same-prior control.
2. Label level: sequence labels on frames (CCT, Serengeti); Wellington likewise; label noise at image level; some frames of an animal sequence show no animal.
3. Only 3 verified sets (4 conditional on WCS); Serengeti only season 3 (one season, same sites across seasons if more were added); Wellington site count mismatch 215 vs 187 (probably several cameras per site; units = `location` field as in metadata).
4. CCT and Serengeti/WCS metadata were not independent of the sampler: sites are drawn from them by seed, not by outcome (except PH/PL by design).
5. DINOv2 is trained on web images that may include camera-trap-like data; no leakage check.
6. Day/night, IR images, season and species mix change with site; no stratification. Sequence length differs by site.
7. Effective sample size is sites (110 / 150 / 150 evaluated sites per set), not frames (2750 / 3750 / 3750). Frames per site (25) are few, so per-site thresholds are not used; thresholds are set per pool.
8. Prior exploratory use: none of these images has been looked at; metadata summaries only.
9. Reading of the Serengeti labelling accuracy (96.6%) comes from secondary web pages, not the paper text.

## 6. Locks (to be written after user approval of the download budget)
sha256 of this file, `experiments/ct_prep.py` (to be written), the seeds, and the site lists will be pushed to the public prereg clone as Amendment 14 before any image fetch.
