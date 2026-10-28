Câu hỏi: Is the "deterministic temporal enclosure + certified side-of-threshold / abstain + abstention-time minimax bound" for a frozen perception reader sampled at frame times (C^k latent score, |s^(k)|<=M, noise<=e, Lagrange remainder + Lebesgue amplification, Landau-Kolmogorov tightening, optimal-recovery lower bound) already in the literature (items 1-7 of brief)?

Scope note: ~20 searches/fetches on 2026-10-28. Several claims come from search snippets only (fetch failed or 403); marked "snippet". Tier per the rules: A = primary paper/docs, B = secondary/classic reference.

## Findings

S1 | tier A | [2019-04] | Abbas & Fainekos, "Temporal Logic Robustness for General Signal Classes": STL monitoring error on reconstructed signals is tightly upper-bounded by the L-infinity signal reconstruction error; spline reconstruction gives smaller robustness error than piecewise-linear at same complexity | https://repository.upenn.edu/mlab_papers/117 | "monitoring error is tightly upper-bounded by the L signal reconstruction error" (snippet, full text 403)

S2 | tier A | [2015-06] | Deshmukh et al., "Robust Online Monitoring of STL" (RV 2015; FMSD 2017): assumes constant (step) interpolation between samples; robust satisfaction interval RoSI = [inf, sup] of robustness over all completions of the partial signal; no derivative bound used | https://arxiv.org/abs/1506.08234 ; https://ar5iv.labs.arxiv.org/html/1506.08234 | "we assume constant interpolation to reconstruct the signal x(t), i.e., given a sequence of time-value pairs" (ar5iv, verified)

S3 | tier A | [2007, year UNSURE] | Fainekos & Pappas, "Robust sampling for MITL specifications": test MITL on continuous signals using discrete-time analysis when sampled signal robustly satisfies the spec; sampling-density error bound | https://repository.upenn.edu/entities/publication/ab29fbf6-806c-4d51-969c-a31c9a0ef78c/full | snippet only; exact bound NOT retrieved

S4 | tier A | [1990] | Osipenko & Stessin preprint: optimal recovery with inaccurate (noisy) data; optimal algorithm = worst-case error equals intrinsic error; optimality conditions originally due to Micchelli & Rivlin | https://osipenko.rstu.ru/Trudy/pR.pdf | snippet: "an optimal algorithm as one whose worst-case error equals the intrinsic error"

S5 | tier B | [1977] | Micchelli & Rivlin optimal recovery survey (general framework; noisy-data conditions) | cited via https://nakamura-u.repo.nii.ac.jp/records/1908 | snippet only; original not retrieved

S6 | tier A | [2018] | Arutyunov & Osipenko: recovery of linear operators from noisy information via Lagrange-function minimality condition | https://osipenko.rstu.ru/aros1.pdf | snippet only

S7 | tier A | [2023] | Foucart, practical linear optimal recovery map when approximation space is Chebyshev | doi 10.1007/s10444-023-10063-x | snippet only

S8 | tier B | [classical] | Landau-Kolmogorov: Hadamard ||f'||<=sqrt(2)*sqrt(||f||*||f''||) on whole line (sharp); Landau 1913 half-line constant 2; Schoenberg general order; C(3,1)=C(3,2)~1.844; explicit constants for k>1 not known in general | https://mathworld.wolfram.com/Landau-KolmogorovConstants.html | snippet only

S9 | tier A | [2022] | Boursinos & Koutsoukos, "Selective Classification of Sequential Data Using Inductive Conformal Prediction" (ICAA 2022): reject untrustworthy predictions by combining p-values from several consecutive inputs to handle time-correlated inputs; tested on GTSRB | https://cs.vuse.vanderbilt.edu/koutsoxd/www/Publications/Selective_Classification_of_Sequential_Data_Using_Inductive_Conformal_Prediction.pdf | snippet only (PDF fetch refused)

S10 | tier A | [2026-09] | arXiv 2609.22632 "Classification with Abstention Under Class-Conditional Error Constraints": minimax lower bound via hard family indexed by VC dimension; excess-ambiguity lower bound of order sqrt(d/n) | https://arxiv.org/pdf/2609.22632 | snippet only; NOT a time-of-decision bound

S11 | tier A | [2019-06] | arXiv 1906.00303: active learning with abstention (membership, pool, stream settings); upper bounds with matching lower bounds claimed; author attribution UNSURE | https://web3.arxiv.org/abs/1906.00303 | snippet only

S12 | tier A | [2015] | Hanneke & Yang, minimax label complexity of active learning under Tsybakov noise (JMLR 16) | https://jmlr2020.csail.mit.edu/papers/volume16/hanneke15a/hanneke15a.pdf | snippet only

S13 | tier A | [2020] | Wu et al., "Robustness Guarantees for Deep Neural Networks on Videos" (CVPR 2020): maximum safe radius over optical flow, Lipschitz assumption, finite optimisation with approximation guarantees; per-clip radius, not temporal interpolation | https://mlanthology.org/cvpr/2020/wu2020cvpr-robustness/ | snippet only

S14 | tier A | [2023-04] | arXiv 2304.03849: every STL specification of its form has a Lipschitz-continuous robustness measure (w.r.t. signal distance), by structural induction | https://arxiv.org/pdf/2304.03849 | snippet only

S15 | tier A | [2025-02] | arXiv 2502.21308 "State-Dependent Conformal Perception Bounds for Neuro-Symbolic Verification": regional perception error added as an interval to measurement model; Taylor-model reachable sets; reachable sets crossing several regions handled separately | https://arxiv.org/pdf/2502.21308 | snippet only

S16 | tier A | [2025-12] | arXiv 2512.02893 "Statistical-Symbolic Verification of Perception-Based Autonomous Systems using State-Dependent Conformal Prediction": Taylor-model union enclosure (Sec. 6.1) | https://arxiv.org/pdf/2512.02893 | snippet only

S17 | tier A | [2026-07] | arXiv 2607.01189 TERA: Taylor-model reachability; reachable set = Taylor polynomial + interval enclosing the true solution; ODE/hybrid/stochastic | https://arxiv.org/pdf/2607.01189 | snippet only

S18 | tier A | [2020-06] | arXiv 2006.05587 Sequential density ratio estimation (ICLR 2021), SPRT-style sequential classification; no abstention or certified guarantee in the snippets; SPRT-TANDEM tested on UCF101 | https://arxiv.org/pdf/2006.05587 | snippet only

S19 | tier B | [2004] | Milanese & Novara: set-membership identification of nonlinear dynamic systems using gradient bounds instead of a functional form; finite-sample accuracy guarantees | URL NOT retrieved | snippet only

S20 | tier B | [2021-09] | arXiv 2109.14299: Lebesgue function gives conditioning of the Lagrange representation; collocation condition number governs noise amplification | https://arxiv.org/pdf/2109.14299 | snippet only

S21 | tier A (internal) | [2026-10] | In-house prior art, not external evidence: deterministic interval pi_hat +- m with m = fraction of frames with |h_hat| <= 2 L eps_max (Prop 1 of theory/34); H71 coverage/width result PARTIAL, width vacuous; H74 sharp interval PARTIAL | /home/sw-haiha/tag/verdict/.ai/runs/INDEX.md (rows 2026-10-22-builder-h71, -h74) | run files listed in INDEX

## Per-item verdict (brief items 1-7)

| item | closest known work | has enclosure+abstain rule? | has abstention-time bound? | has minimax lower bound? | verdict |
|---|---|---|---|---|---|
| 1 optimal recovery | Micchelli-Rivlin (S5), Osipenko-Stessin (S4), Arutyunov-Osipenko (S6), Foucart (S7), Lebesgue/noise amplification (S20) | Recovery theory has the worst-case/intrinsic-error notion; no sign/threshold/abstain version found | no | no for decision time | Recovery machinery KNOWN. Threshold-side abstention as optimal-recovery decision: NOT FOUND (search weak). |
| 2 certified video | Wu et al. 2020 (S13) | no (per-clip radius) | no | no | Video certification via Lipschitz known for clip-level radius. Per-frame temporal enclosure: NOT FOUND. |
| 3 STL sampled monitoring | Abbas-Fainekos 2019 (S1), Deshmukh RoSI (S2), Fainekos-Pappas 2007 (S3), Lipschitz STL robustness (S14) | RoSI is a sound 3-valued (interval) enclosure for unseen suffix, i.e. enclose-then-decide; Abbas gives reconstruction-error bound | RoSI decides early when interval is one-sided; no time bound for abstention | no | Enclose-and-abstain for sampled STL is ESSENTIALLY KNOWN (RoSI). Derivative-based C^k Taylor remainder + noise Lebesgue budget in this form: NOT FOUND. |
| 4 interval/Taylor models | TERA (S17), Taylor-model perception bounds (S15, S16) | Perception interval in reachability, not temporal monitor | no | no | Taylor-model enclosure machinery KNOWN; applied to sequential perception abstain monitor NOT FOUND. |
| 5 Landau-Kolmogorov | Hadamard/Landau/Schoenberg (S8) | not in control/ML certificates in snippets (searched 2 queries) | n/a | n/a | Inequality KNOWN, sharp constants known for n=2,k=1 only; use in certificate: NOT FOUND in search. |
| 6 selective + temporal + SPRT | Boursinos-Koutsoukos 2022 (S9), Wald SPRT (classic, no URL retrieved), SPRT-TANDEM (S18) | statistical (conformal p-value aggregation with reject), not deterministic enclosure | statistical sequential stopping only (SPRT) | no | Statistical counterpart KNOWN; deterministic enclosure counterpart NOT FOUND. |
| 7 set-membership (bounded noise) | Milanese-Vicino UBB (S19; no URL retrieved), Bravo et al. feasible solution set (snippet) | yes for parameter sets (feasible set = enclosure); not for classification of dynamical signals | no | no | Bounded-noise enclosure machinery KNOWN; classification/abstain application NOT FOUND. |

Combined claim: the individual pieces (Lagrange remainder, Lebesgue noise amplification, Landau-Kolmogorov, 3-valued STL enclosure) are classical. The closest published object is RoSI (S2) plus Abbas-Fainekos (S1). The specific combination (C^k + |s^(k)|<=M + bounded noise e, Lebesgue-amplified enclosure, certified argmax across readers, abstention-time and minimax lower bound on decision time) was not found. Minimax abstention bounds found (S10, S11, S12) are on sample/label complexity, not continuous-time decision time.

## Suggested verdict (state uncertainty)

- KNOWN (cite, do not claim): Lagrange remainder + Lebesgue noise amplification (classical interpolation); Landau-Kolmogorov (classical); enclose-then-decide for sampled STL via RoSI (S2); bounded-noise feasible-set enclosures (S19); statistical selective classification over consecutive frames (S9).
- POSSIBLY NEW (state uncertainty, search not exhaustive): the deterministic C^k enclosure of the latent score between frames with explicit noise budget used for certified abstain; certified argmax across readers; the minimax lower bound on abstention time from optimal recovery. Need a targeted search on "optimal recovery" + "sign"/"decision" + "time to decide" and on "Lebesgue function" + "abstain".
- Sanity note: the stated bound is standard. Read from y_i = s(t_i)+eta_i, |s(t)-p(t)| <= M/k! |omega(t)| + Lambda(t) e, with omega(t)=prod(t-t_j). Correctness not checked beyond this.

## MÂU THUẪN

- None between sources on the same claim. Different settings only: Hadamard sqrt(2) (whole line) vs Landau constant 2 (half line) (S8); Fainekos-Pappas (S3) sampling-density error vs Abbas (S1) reconstruction error are different error sources (S3 is by the sampling bound, S1 by reconstruction L-inf).
- Assumption mismatch (not a contradiction): RoSI (S2) assumes step interpolation and bounds nothing about derivatives; the brief's idea assumes C^k with M.

## Could NOT verify

- Full text of S1, S3, S5, S9, S10, S11, S13, S14 (fetch 403/refused or snippet only). Quotes marked snippet.
- Fainekos-Pappas 2007 exact sampling bound and year.
- Milanese-Novara 2004 URL; Wald SPRT URL (classic, not retrieved).
- Whether any paper on certified perception uses Landau-Kolmogorov (2 queries, nothing).
- Coverage is limited (~20 queries, no Golomb-Weinberger, Smolyak, Traub-Wozniakowski papers found). Absence is not proof.
- arXiv 2609.22632 and 2607.01189 dates from ID only (plausible, not checked on page).

THEO DẤU: S2 (Deshmukh RoSI, arXiv 1506.08234 refs), S1 (Abbas-Fainekos 2019 refs for reconstruction bound), S4 (Osipenko-Stessin optimal recovery with noise, refs to Micchelli-Rivlin).
