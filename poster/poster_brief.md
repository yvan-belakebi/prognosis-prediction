# Poster brief: prognosis prediction from multiple stains in kidney biopsies

Facts, methods and a figure plan collected from the development work. It is meant
to be handed to whoever makes the figures, Claude included. Numbers marked
**[preliminary]** come from runs on the old labels or a test split, so they must
be regenerated before the poster is printed (see "Before making the results
figures").

---

## 1. Context frame

- **Clinical question:** predict kidney outcome from the biopsy itself. Can the
  whole-slide images of the different stains of one biopsy predict time to
  kidney failure, and do several stains together beat one?
- **Why several stains:**
  - A kidney biopsy is routinely cut and stained in several ways (HE, PAS, PASM,
    Congo, trichrome, Sirius red, immunostains).
  - Each stain shows different lesions: PAS and PASM show glomerular basement
    membranes and sclerosis, trichrome and Sirius red show fibrosis, Congo shows
    amyloid, and the immunostains show deposits.
  - Pathologists read the stains together; a model usually reads one.
- **Why the disease matters as context:** registry biopsies span about 70
  diagnoses with very different outcomes. Event rates by diagnosis:

  | Diagnosis | Biopsies with an event |
  |---|---|
  | Minimal change | 10% |
  | Tubulointerstitial nephritis | 19% |
  | IgA nephropathy | 24% |
  | ANCA glomerulonephritis | 31% |
  | Benign nephrosclerosis | 49% |
  | Diabetic nephropathy | 62% |

  So a model can score well by recognising the disease, and the poster should
  show what the images add beyond diagnosis and kidney function (section 4).

## 2. Data frame

**Source:** the Norwegian kidney biopsy registry, linked to the national renal
registry for dialysis, transplantation and CKD5 events, and to death records.
Whole-slide images come from the registry biopsies.

**Cohort after the per-biopsy rebuild** (`followup_data/scripts/registry_events.py`):

| Step | Count |
|---|---|
| Registry biopsies | 6,719 |
| Excluded: kidney-replacement therapy before the biopsy | 199 |
| Excluded: under 60 days of follow-up | 269 |
| Kept biopsies (5,907 patients) | 6,257 |
| Biopsies with slides (5,464 patients, 59,381 slides) | 5,764 |
| IgA nephropathy biopsies with slides | 637 |

**Endpoint:** time from biopsy to the first of dialysis, CKD stage 5, kidney
transplant or death.
- Events among kept biopsies: dialysis 595, CKD5 579, death 500, transplant 67.
- Deaths are about 29% of events. Many are probably unrelated to the kidney, which
  caps how well any biopsy-based model can rank patients. Show kidney failure
  alone as a secondary endpoint.
- Censoring is at the date of the registry extract.

**Stain availability** (biopsies with features, from the multi-stain run on
5,754 biopsies):

| Stain | Biopsies | Stain | Biopsies |
|---|---|---|---|
| HE/HES | 98% | IgA | 29% |
| PAS | 98% | IgG | 28% |
| Congo | 81% | C3 | 28% |
| Trichrome | 78% | IgM | 25% |
| PASM | 69% | C1q | 25% |
|  |  | C5-9 | 20% |
|  |  | Kappa | 13% |
|  |  | Lambda | 9% |

Many biopsies have several slides of the same stain; for example, in the IgA
subset 91% of PAS biopsies have two or more PAS slides (recount for the full
cohort).

**Clinical variables:**
- **Diagnosis:** 70 categories, missing for about 39% of biopsies.
- **eGFR at biopsy:** available for about 95% of biopsies.

**Split:** by patient, so all slides of a patient stay in the same split.
- **Train / validation / test:** 60 / 20 / 20%.
- **Test set:** stratified by event and held out entirely; it is scored once, at
  the end.
- **Size** (multi-stain run): 3,451 / 1,139 / 1,164 biopsies.

## 3. Methods frame

**Preprocessing (TRIDENT):**
1. Tissue segmentation (HEST).
2. Tiling into 224 px patches at 20x, without overlap.
3. Optional stain normalisation per stain, using a reference fitted per stain
   (Macenko or Vahadane).
4. Patch features from the **UNI-v2** foundation model (1,536 dimensions per patch).

**Single-stain model** (`MIL.py`), one model per stain:
- **Architecture:** DeepGraphSurv, a graph neural network over the patches with
  each patch linked to its 8 neighbours, followed by attention pooling and a Cox
  risk head. ABMIL, TransMIL and PatchGCN are also available.
- **Training:**
  - Cox partial-likelihood loss;
  - one random slide per biopsy per epoch;
  - batches with the same share of events (event-stratified);
  - AdamW with weight decay, and random patch subsampling.
- **Model selection:** the checkpoint with the best validation C-index, computed
  per biopsy.
- **Biopsy score:** the mean of its slides' risk scores.

**Four fusion strategies** (one poster panel each, side by side):

| Strategy | Where stains are combined | How | Script |
|---|---|---|---|
| **Early fusion** | patches | One shared projection plus a learned stain embedding per patch; one attention pooling over the patches of all stains | `multistain_MIL.py --fusion early` |
| **UNICORN-style fusion** | stain tokens | One transformer encoder per stain, pooled into one token per stain; a transformer combines the stain tokens; Cox head. Optionally each slide is encoded separately and weighted by attention (`--per_slide`) | `multistain_MIL.py` |
| **Late fusion** | model outputs (learned) | A ridge-penalised Cox regression on each stain model's normalised risk plus stain-presence indicators. Fitted on validation, penalty chosen by cross-validation (one-standard-error rule) | `fuse_late.py` |
| **Voting** | model outputs (fixed) | Each stain model's risk becomes a percentile of its training scores, averaged over the biopsy's available stains; optionally weighted by validation C-index | `fuse_votes.py` |

All four handle missing stains: missing stains are masked in the models and
simply don't vote. Stain dropout during training teaches the models to predict
from incomplete stain panels.

**Evaluation**, all on the held-out test set and on biopsies every method scored:
- Harrell's C-index with bootstrap 95% CI, and paired comparisons between methods
  (`compare_cindex.py`).
- **Added value over the clinic:** a Cox model on diagnosis + eGFR, with and
  without the model's risk score, giving the C-index gain with a CI and the
  adjusted hazard ratio (`analyze_model_signal.py`).
- **C-index within diagnosis:** only pairs of patients with the same diagnosis.
- **Kidney-failure-only and death-only endpoints.**
- **Per-stain contributions and leave-one-stain-out ablation** for the
  multi-stain models (`analyze_stain_weights.py`).
- **Risk groups:** Kaplan-Meier curves of predicted low, intermediate and high
  risk groups; n-year survival probability from Breslow baseline recalibration.

## 4. Results frame

All numbers below come from the **held-out test set**, scored once, on labels from
the per-biopsy registry rebuild (runs of 26 Oct). Unless stated otherwise, they use
the **1,188 test biopsies that every method scored (360 events)**. Source files are
under `results/`; `results/model_signal/` holds the cross-method analysis
(`analyze_model_signal.py`).

**Naming:** in `results/model_signal/*.csv`:
- `average` is unweighted rank voting (`voting_rank_none`);
- `voting` is C-index-weighted voting (`voting_rank_cindex`);
- `late` is late fusion;
- `unicorn` and `early` are the two multi-stain models;
- `PAS` is the best single-stain model.

The two voting variants differ by 0.0005, so the poster shows one "Voting" (the
simpler unweighted one).

### Headline

> Biopsy images add about 5 C-index points to diagnosis + eGFR (0.751 → 0.80).
> The PAS model alone already captures this gain. The four fusion strategies tie
> with PAS; they help mainly when a biopsy has to be read through a weaker stain.
> The signal concerns kidney failure much more than death.

### Key numbers

**C-index on the test set (1,188 biopsies, 360 events)**, from
`model_signal/cindex_overview.csv` and `clinical_added_value.csv`:

| Predictor | C-index (95% CI) | Diagnosis + eGFR + predictor | Gain (95% CI) | HR per SD |
|---|---|---|---|---|
| Diagnosis alone | 0.558 | | | |
| eGFR alone | 0.745 | | | |
| Diagnosis + eGFR | 0.751 | | | |
| PAS (single stain) | 0.802 (0.780–0.826) | 0.808 | +0.057 (0.039–0.077) | 2.9 |
| Voting | 0.802 (0.782–0.824) | 0.807 | +0.056 (0.035–0.078) | 3.3 |
| Late fusion | 0.801 (0.780–0.825) | 0.807 | +0.056 (0.036–0.078) | 3.4 |
| Early fusion | 0.794 (0.773–0.815) | 0.802 | +0.051 (0.031–0.072) | 3.0 |
| UNICORN | 0.793 (0.771–0.814) | 0.803 | +0.052 (0.034–0.071) | 3.5 |

- The clinical models are fitted on validation and scored on test. The gain CI is
  a bootstrap CI, and every likelihood-ratio p-value is below 1e-40.
- **Paired differences** (bootstrap on the same biopsies, `results/compare/*_test.txt`):

  | Comparison | Difference (95% CI) | p |
  |---|---|---|
  | Late fusion − PAS | −0.002 (−0.010 to +0.006) | 0.69 |
  | Voting − PAS | −0.000 (−0.009 to +0.008) | 0.96 |
  | Late fusion − Voting | −0.002 | 0.52 |

  UNICORN and early fusion against PAS have **not been run yet** (see "Missing
  before printing").
- **Why fusion can't beat PAS here:** PAS is present in 1,188 of the 1,202 test
  biopsies, so every fused prediction is anchored on a PAS model.
- **Stain presence alone predicts little:** a Cox model on stain-presence
  indicators reaches C 0.54 (`eval/late_fusion/summary.csv`). Which stains a
  biopsy has does not leak the outcome.

**Fusion rescues weak stains.** Paired test C-index of voting minus the single
stain, on the biopsies that have that stain
(`results/compare/voting_none_vs_<stain>_test.txt`). Late fusion shows the same
pattern.

| Stain | n (events) | Single stain | Gain from voting (95% CI) |
|---|---|---|---|
| Lambda | 114 (34) | 0.614 | +0.118 (0.052–0.185) |
| Sirius red | 188 (46) | 0.739 | +0.057 (0.020–0.097) |
| AFOG | 540 (149) | 0.772 | +0.034 (0.019–0.051) |
| Masson trichrome | 377 (130) | 0.764 | +0.028 (0.012–0.045) |
| IgM | 306 (85) | 0.759 | +0.028 (0.008–0.048) |
| C3 | 352 (95) | 0.772 | +0.028 (0.003–0.052) |
| PASM | 835 (255) | 0.774 | +0.022 (0.011–0.034) |
| IgA | 377 (100) | 0.792 | +0.018 (0.002–0.035) |
| HE | 718 (208) | 0.788 | +0.010 (−0.001 to 0.021) |
| Congo | 989 (299) | 0.798 | +0.007 (−0.002 to 0.017) |
| HES | 468 (154) | 0.803 | +0.000 (−0.010 to 0.010) |
| PAS | 1,188 (360) | 0.802 | −0.000 (−0.009 to 0.008) |

- C1q, C5-9, IgG and Kappa gain +0.015 to +0.033, with CIs that touch 0.
- Each single-stain C-index is computed on its own subset of biopsies. **Never plot
  them side by side as if they were comparable**; only the paired gains are.

**Which stains carry the signal.** Two independent views agree: the morphology
stains carry the signal, and the immunostains add almost nothing.
- **Late-fusion weights** per SD of a stain's vote (`eval/late_fusion/coefficients.csv`):

  | Stain | Weight |
  |---|---|
  | PAS | 1.41 |
  | Congo | 1.13 |
  | HES | 0.82 |
  | HE | 0.75 |
  | Masson trichrome | 0.70 |
  | PASM | 0.46 |
  | AFOG | 0.44 |
  | Sirius red | −0.05 |
  | Immunostains (all) | −0.08 to +0.15 |
- **Leave-one-stain-out on UNICORN** (`eval/test/unicorn/stain_ablation.csv`):
  the C-index drop on biopsies that have the stain is 0.017 for PAS, 0.014 for HE,
  0.009 for Congo, 0.006 for HES and 0.006 for Sirius red. Every other stain stays
  within ±0.003.
- **Early fusion** (`eval/test/early/stain_ablation.csv`) leans on HE (drop 0.021).
  It is *hurt* by Sirius red: removing it raises the C-index by 0.013 (n = 188).

**Endpoints** (`model_signal/event_type_cindex.csv`):
- Kidney failure (dialysis, CKD5, transplant): C 0.820–0.826 for every method.
- Death: C 0.713–0.740.

Deaths are 29% of events, so they pull the all-cause C-index down to about 0.80.

**Within diagnosis** (`model_signal/stratified_cindex.csv`): when only pairs with
the same diagnosis are ranked, the C-index is 0.743–0.781, against 0.79–0.80
overall. So most of the signal is not "recognising the disease".

| Method | C within diagnosis |
|---|---|
| PAS | 0.781 |
| Voting | 0.778 |
| Late fusion | 0.769 |
| UNICORN | 0.748 |
| Early fusion | 0.743 |

- **Caveat:** diagnosis is missing for 437 of the 1,188 test biopsies, and only
  8,160 within-diagnosis pairs are comparable.
- **Per diagnosis** (`per_diagnosis_cindex.csv`; n = 51–123 each, so CIs are wide):

  | Diagnosis | C-index across methods |
  |---|---|
  | IgA nephropathy | 0.79–0.83 |
  | ANCA glomerulonephritis | 0.75–0.78 |
  | Benign nephrosclerosis | 0.66–0.72 |
  | Diabetic nephropathy | 0.62–0.75 (UNICORN weakest) |

**The models make the same mistakes** (`model_signal/risk_correlation.csv`,
`discordance_overlap.csv`, `discordance_by_gap.csv`, `hard_biopsies.csv`):
- Risk rank correlation between methods is 0.88–1.00. It is still 0.79–0.94 after
  removing what diagnosis + eGFR explain.
- 52–98% of misranked pairs are shared between methods (Jaccard).
- Errors concentrate in pairs with close outcome times. Share of pairs misranked,
  by time gap:

  | Time gap | Misranked |
  |---|---|
  | < 6 months | 40% |
  | 1–2 years | 31% |
  | > 5 years | 13–15% |
- The most consistently misranked biopsies are high-risk predictions that never
  had an event:
  - acute tubular necrosis and myeloma kidney with eGFR around 5;
  - benign nephrosclerosis;
  - crescentic diseases.

  These kidneys look very damaged but recovered, or were followed without an
  event. Describe them in words only; do not print biopsy IDs.

**n-year risk** (`survival_probability.py`, outputs in `results/survival_probability/`
and `results/survival_probability_kidney/`). Each risk score becomes a probability
through a Cox model + Breslow baseline fitted on validation. IPA is 1 − Brier /
Brier of Kaplan-Meier: 0 means no better than giving everyone the cohort average.

| 5-year risk, any event (292 events by 5 y) | AUC(5 y) (95% CI) | IPA(5 y) (95% CI) |
|---|---|---|
| Diagnosis + eGFR | 0.781 (0.751–0.810) | 0.19 (0.15–0.23) |
| PAS | 0.847 (0.820–0.874) | 0.31 (0.26–0.37) |
| Voting | 0.845 (0.822–0.870) | 0.30 (0.24–0.35) |
| Late fusion | 0.843 (0.818–0.869) | 0.30 (0.24–0.36) |
| UNICORN | 0.837 (0.812–0.862) | 0.28 (0.24–0.33) |
| Early fusion | 0.833 (0.806–0.859) | 0.27 (0.22–0.33) |

- **Against the clinic:** at 5 years every model beats diagnosis + eGFR, by +0.05
  to +0.07 AUC and +0.08 to +0.12 IPA (all CIs exclude 0).
- **Against PAS:** voting and late fusion are not different from PAS. UNICORN
  (IPA −0.030, p = 0.04) and early fusion (−0.039, p = 0.004) give slightly worse
  probabilities.
- **Calibration:** all are well calibrated on test.
  - Calibration slope 0.98–1.14 (1 = ideal).
  - Mean error by decile 0.015–0.040 at 5 years; late fusion is best (0.015).
- **Short term vs long term:** at 1 year, eGFR does as well as the images for
  kidney failure (AUC 0.89 for clinical vs 0.85–0.88). Imminent dialysis is
  visible in kidney function. The images pull ahead from 3 years on, and the gap
  grows with the horizon: for kidney failure at 10 years, AUC is 0.84–0.85 vs
  0.76.
- **Kidney failure at 5 years** (death as a competing risk): AUC 0.84–0.85 for
  the models against 0.81 for the clinic.
- **Decision curve at 5 years:** every model has a higher net benefit than the
  clinic for risk thresholds from 10% to 60%.

**Supporting results** (small, or for a "methods notes" box):
- **Architecture** (validation, same split, `results/eval_*` logs): DeepGraphSurv
  matches or beats ABMIL on 14 of 16 stains.

  | Stain | DeepGraphSurv | ABMIL |
  |---|---|---|
  | PAS | 0.800 | 0.799 |
  | Congo | 0.819 | 0.794 |
  | Masson trichrome | 0.806 | 0.769 |
  | Kappa | 0.775 | 0.658 |
  | Lambda | 0.726 | 0.608 |
- **Overfitting:** early fusion's validation loss rises from epoch 3, while its
  validation C-index plateaus around 0.80 (`results/losses_early/loss_curves.png`).
  The checkpoint is chosen on validation C-index.
- **Late-fusion penalty:** the CV C-index is flat (0.803–0.807) for λ from 0.001
  to 0.03 (`eval/late_fusion/cv.csv`), so the fused weights are stable.

### Panels, in reading order

**Style:**
- Use one fixed colour per method in every panel, with the clinical models in
  grey (dashed for lines):
  ```python
  METHOD = {
      "PAS": "#5C3FA8",
      "Voting": "#E0609A",
      "Late fusion": "#1B8A5A",
      "Early fusion": "#2A78D6",
      "UNICORN": "#C57600",
  }
  MORPH, IMMUNO = "#3B4A5C", "#B5BCC6"
  ```
- `MORPH` and `IMMUNO` colour stains by kind (R2, R3): morphology stains (HE, HES,
  PAS, PASM, Masson trichrome, AFOG, Congo, Sirius red) and immunostains.
- Every caption gives "test set, 1,188 biopsies, 360 events", or the panel's own n.

| # | Panel | Type | Data | Status |
|---|---|---|---|---|
| R1 | **Images add to the clinic** (headline) | Forest plot of C-index with CI for PAS and the 4 fusions, with vertical reference lines at eGFR (0.745) and diagnosis + eGFR (0.751). A second column of points shows "diagnosis + eGFR + model", labelled with the gain and its CI | `cindex_overview.csv`, `clinical_added_value.csv` | **New** |
| R2 | **Fusion rescues weak stains** | Dot-and-whisker plot of the paired gain per stain, sorted, with a zero line and n under each label; PAS and HES sit at about 0, at the bottom | `compare/voting_none_vs_<stain>_test.txt` for the 16 current stains. Ignore the `HE_HES` and `Trichrome` files, dated 22 Oct (old labels) | **New**, parsed from the txt files |
| R3 | **Which stains matter** | Two aligned horizontal bar charts in the same stain order: late-fusion weight, and UNICORN leave-one-out drop. Colour morphology stains apart from immunostains | `late_fusion/coefficients.csv` (`vote_*` rows), `test/unicorn/stain_ablation.csv` | **New**. `test/unicorn/stain_ablation.png` exists but needs restyling |
| R4 | **Kidney failure vs death** | Grouped dots per method over 3 columns: all events, kidney failure, death | `event_type_cindex.csv` | **New** |
| R5 | **Beyond the diagnosis** | Small table or heatmap, method × {overall, within diagnosis, IgA, ANCA, nephrosclerosis, diabetic} | `stratified_cindex.csv`, `per_diagnosis_cindex.csv` | **New** |
| R6 | **Risk groups** | KM curves of voting-risk tertiles on test, with the log-rank p-value and an at-risk table | `eval/voting_rank_none/test/risk_scores.csv` | **New**. Don't reuse `eval/test/<stain>/km_curves.png`: they were drawn with the old "time > 730 days" filter, removed in commit c1acc5b |
| R7 | **Where the models fail** | Bar chart of the share of misranked pairs by time gap, one line per method (they overlap, which is the point), plus a 2-line text box on the hard cases | `discordance_by_gap.csv`, `hard_biopsies.csv` (aggregate only) | **New**; optional if space is short |
| R9 | **From score to n-year risk** | AUC(t) curve, models against the clinic (the gap widens with t), next to a 5-year calibration plot of the best fused model | `results/survival_probability/auc_over_time.png`, `calibration_5y.png` (restyle), or `horizon_metrics.csv`, `calibration.csv` | Made by `survival_probability.py`; restyle |
| R8 | **One case** | Attention heatmaps of a high-risk and a low-risk biopsy across 2–3 stains | `visualize_attention.py --save_as_tif` | Not made yet |

If the frame fits only four panels, use R1, R2, R3 and R4, and add R5's
within-diagnosis number as a line under R1.

**Existing figures worth reusing** after restyling to the poster palette:
- `results/cohort_survival/km_curve_all.png` (for the data frame, not the results);
- `results/eval/test/unicorn/stain_ablation.png`;
- `results/eval/test/unicorn/stain_contribution_by_stain.png`;
- `results/losses_early/loss_curves.png` (supplement).

Ignore `results/model_signal_old_run/` and `results/logs/`: they use the old labels.

### Missing before printing

1. **Paired tests of UNICORN and early fusion** against PAS and voting: run
   `compare_cindex.py` on `results/eval/test/unicorn/risk_scores.csv` and
   `results/eval/test/early/risk_scores.csv` against
   `results/eval/test/PAS/risk_scores.csv`.
2. **Clinical-baseline CIs:** the CSV gives point estimates only. Either bootstrap
   CIs, or draw the baselines as reference lines, as in R1.
3. **Remaining figures:** the risk-group KM for the fused model (R6) and the
   attention maps (R8).
4. **Run `survival_probability.py` into `results/`**, once for each endpoint (see
   the README, step 12). The numbers above come from a run on the current
   `results/eval` scores.

## 5. Figure plan and tools

| Figure | Make with | Input |
|---|---|---|
| Cohort flow chart | SVG or draw.io diagram | counts in section 2 |
| Stain availability | matplotlib horizontal bars | `stain availability` line of the `multistain_MIL.py` log |
| Cohort KM curve | `plot_train_survival.py --split all` | labels CSV |
| Pipeline and 4 fusion panels | SVG diagram, one colour per strategy | section 3 |
| Forest plot / added value | matplotlib, from `cindex_overview.csv`, `clinical_added_value.csv` | `analyze_model_signal.py` output |
| Within-diagnosis heatmap | matplotlib, from `per_diagnosis_cindex.csv` | same |
| Stain contributions / ablation | `analyze_stain_weights.py` PNGs, restyled | its CSVs |
| Risk-group KM | matplotlib + lifelines, tertiles of the voting risk (see R6) | `eval/voting_rank_none/test/risk_scores.csv` |
| Attention maps | `visualize_attention.py --save_as_tif` | best model checkpoint |

**Style rules for every figure:**
- vector output (PDF/SVG) at the final printed size;
- fonts of at least 24 pt at A0;
- one fixed colour per method across all figures;
- the test-set n and number of events in every caption.

## 6. Before making the results figures

1. **Regenerate the registry table:** `registry_events.py --censor_date <extract
   date>`. It fixes per-biopsy biopsy dates and outcomes, and writes no national
   ID.
2. **Regenerate the labels CSVs**, including the super-stain version, from it.
3. **Retrain** every single-stain and multi-stain model on the new labels.
4. **Select on validation only:** checkpoints, settings and late-fusion weights.
5. **Score test once:** `vote_across_stains.sh`, `analyze_stain_weights.py --split
   test`, then `analyze_model_signal.py` with every method.
6. **Privacy check before sharing:** the poster and its source files must not
   contain identifiers. Use `AnonymizedID` or no patient ID at all.
