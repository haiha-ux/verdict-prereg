# 104. A label-free regime selector on a third domain: THUMOS14 (Amendment 25, 2026-10-29)
STATUS: written before any THUMOS14 video was downloaded; only the two public temporal-annotation zips were read. Locked publicly before stage fetch.
Motivation (post hoc, disclosed): on robot video the age-discounted hold beat smoothing (Amendments 22-23); on surgical video smoothing beat it (Amendment 24). A label-free statistic computed on development data separated the domains: rho = P(reader decision flips over 1.6 s) / P(flips over 0.2 s) was 6.6 / 7.7 / 3.7 on mutex / sirius / droid calib and 1.4 on cataract calib (computed after those results were known). The selector and its cut-off 2 were chosen from these four values; THUMOS14 is the first test.

## Data
THUMOS14 temporal action localisation, validation + test videos with temporal annotations (annotation zips public; videos fetched one by one from crcv.ucf.edu without login). Licence: no terms found on the site (UNVERIFIED); research use only, no redistribution. Annotation source: the THUMOS14 challenge temporal annotations (human annotators per the challenge; primary-source quote NOT yet obtained, UNVERIFIED). 100 videos drawn by RandomState(20261101) from the labelled videos (download cap 11 GB; if reached, fewer videos, recorded). Split in draw order among successfully fetched videos: train 40, calib 20, eval the rest (<= 40).
Binary state: 1 if the time point lies inside any annotated action interval (any of the 20 classes), -1 (excluded) inside an 'Ambiguous' interval, else 0. Base grid 5 points per second by timestamp (first decoded frame at or after each grid time).

## Reader and rules
As theory/103 (frozen DINOv2-small CLS + mean, t2_run.fit_probe on train with C, tau on calib; e, L, c*, mu, sd from calib; rules of tc4_rc.episode, each with its own decision as in Amendments 22 and 24).
Selector (fixed now): rho on the calib videos' reader decisions at lags .2 s and 1.6 s; rho >= 2 selects hold_c (age discount), else ema (smoothing).

## Claims (m in {8, 16}; eval videos; video bootstrap 2000, seed 20261101)
S1 (primary): AURC(selected) < AURC(other) with one-sided 95% upper bound < 0, against the human label AND against the reader, at both m (4 comparisons, all must pass).
S2 (secondary): AURC(selected) < AURC(plain hold) on the label target at both m.
Reported: all rules at m in {2, 4, 8, 16}; rho and flip rates; probe AUC/error.
Prediction: sports broadcast video has shot cuts and camera motion, so the reader probably flickers; rho < 2 and ema selected; S1 passes. Not Tier-1 (<= 40 eval videos); one more domain only.

## Procedure
Script experiments/tc9_thumos.py (stages idx | fetch | embed | run). Plan and script sha256 pushed publicly as Amendment 25 BEFORE stage fetch. One run; independent critic afterwards.

## Addendum 25a (2026-10-29, before any video was read)
The first fetch failed for all 100 videos with SSL CERTIFICATE_VERIFY_FAILED (the http URLs redirect to https and the decoding venv has no CA bundle). No video byte was saved, nothing was decoded, embedded or run (run stopped on an empty list). Fix: urlopen with ssl.create_default_context(cafile='/etc/ssl/certs/ca-certificates.crt'); certificate verification stays on. The empty fetch log was deleted. Nothing else changed; new script sha256 below.

## Addendum 25b (2026-10-29, before any video was read)
The 25a fix was insufficient: www.crcv.ucf.edu does not send its intermediate certificate (openssl: verify return code 21). Fix: the intermediate 'InCommon RSA OV SSL CA 3' was fetched from the AIA URL in the server certificate (http://crt.sectigo.com/InCommonRSAOVSSLCA3.crt; issuer Sectigo Public Server Authentication Root R46, which is in the system store) and appended to a copy of the system CA bundle (data/th14_raw/cabundle.pem); `openssl verify` of the server certificate against it returns OK. Verification stays on. Again no video byte was saved; the fetch log was deleted.

## Addendum 25c (2026-10-29, after the single run; result_tc9.json sha256 5d843bbbc801dfd4a219ce939c96e5ad4a59607b1ba52cbc1202f476d1c23d8c)
Fetch: 100/100 videos, no failure, cap not reached; frames and videos deleted after embedding. Split 40/20/40.
Reader FAILED by the theory/90 criterion (clean error >= .35): eval AUC .672, error .387, positive share .395 (action-vs-background over 20 sports classes is not read by a frozen DINOv2 CLS+mean linear probe). Reported, not dropped.
Selector: flip .106 (0.2 s) / .222 (1.6 s), rho 2.08 >= 2 -> hold_c selected (prediction 'rho < 2, ema' WRONG; rho is barely above the cut).
S1 FAILED: reader target passes at both m (hold_c - ema: m=8 -.0118, upper bound -.0043; m=16 -.0210, ub -.0087), label target point estimates favour hold_c but upper bounds > 0 (m=8 ub +.0031, m=16 ub +.0065). S2 FAILED: on the label target plain hold is better than hold_c (m=8 .2812 vs .2921; m=16 .3145 vs .3234; ub > 0); plain hold is the best ranker on the label at m in {4, 8, 16}.
Reading allowed: with a failed reader (AUC .67) the label-target comparisons are dominated by reader error (label AURC .26-.39 vs reader AURC .04-.13); THUMOS14 with this reader is not an informative test of the selector against human labels. Against the reader's own output the selected rule beat the other one. The selector is NOT confirmed. Across the four domains, no between-frame rule beat plain hold against human labels (cataract: ema best; THUMOS: hold best); the age-discount gain is confirmed only against the reader / sensor-label on two robot datasets.
