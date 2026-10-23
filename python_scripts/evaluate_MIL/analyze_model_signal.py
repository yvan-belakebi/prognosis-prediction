"""
analyze_model_signal.py — What do the survival models capture, beyond the clinic?

When very different models (single stains, voting, late fusion, UNICORN, early
fusion) land within a few C-index points of each other, they probably share one
dominant signal. This script compares them on the *same* test biopsies and asks
what that signal is and what is left once it is accounted for:

  1. Overview        C-index of every model on the biopsies all of them scored,
                     with bootstrap 95% CIs.
  2. Event type      C-index for kidney failure only (dialysis, CKD5, transplant;
                     deaths censored) and for death only — deaths unrelated to
                     the kidney cap any biopsy-based model.
  3. Clinical value  Cox models on diagnosis, eGFR at biopsy and both, and the
                     same clinical model + each model's risk: adjusted hazard
                     ratio per SD of risk, likelihood-ratio test, and the C-index
                     gain over the clinical model (paired bootstrap CI). Fitted on
                     --fit_split (val: the models never trained on it), scored on
                     --eval_split (test).
  4. Within diagnosis C-index counting only pairs of patients with the same
                     diagnosis (stratified), and per diagnosis for the larger
                     groups — prognosis within a disease, not disease recognition.
  5. Agreement       Spearman correlation of the models' risks, raw and after
                     removing what diagnosis + eGFR explain: highly correlated
                     models make the same errors, so fusing them cannot help.
  6. Discordant pairs  For each comparable pair (earlier event vs later), whether a
                     model ranks it wrongly: discordance by time gap, overlap of the
                     discordant pairs between models, the pairs all models get
                     wrong, and the biopsies most often involved (hard_biopsies.csv,
                     for case review with attention maps).

Caveats printed with the results: diagnosis is made from the same biopsy, so
controlling for it also removes image signal the pathologist already read, and
it is missing for many registry biopsies (kept as its own "missing" category).

Inputs: evaluate_survival.py-style per-biopsy risk_scores.csv (biopsy, risk,
time, event) per model and split, and the registry CSV for diagnosis, eGFR and
event type.

Usage:
    python python_scripts/evaluate_MIL/analyze_model_signal.py \\
        --model PAS=results/eval/{split}/PAS/risk_scores.csv \\
        --model HE=results/eval/{split}/HE_HES/risk_scores.csv \\
        --model voting=results/eval/voting_rank_none/{split}/risk_scores.csv \\
        --model late=results/eval/late_fusion/{split}/risk_scores.csv \\
        --output_dir results/model_signal
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import chi2, rankdata, spearmanr

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
from evaluate_survival import concordance_index  # noqa: E402
from fuse_late import _cox_loss_grad, fit_cox  # noqa: E402
from fuse_votes import load_risks  # noqa: E402

YEAR = 365.25


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def parse_models(specs, parser):
    models = {}
    for spec in specs:
        if "=" not in spec or "{split}" not in spec:
            parser.error(f"--model {spec!r}: expected NAME=path/with/{{split}}/risk_scores.csv")
        name, pattern = spec.split("=", 1)
        models[name] = pattern
    return models


def load_clinical(args):
    """Per-biopsy diagnosis, eGFR and event type from the registry CSV."""
    cols = [args.biopsy_col, args.diagnosis_col, args.egfr_col, args.event_type_col]
    reg = pd.read_csv(args.registry_csv, usecols=cols, dtype=str, low_memory=False)
    reg[args.biopsy_col] = reg[args.biopsy_col].str.strip()
    reg = reg.dropna(subset=[args.biopsy_col]).drop_duplicates(args.biopsy_col)
    egfr = pd.to_numeric(reg[args.egfr_col], errors="coerce")
    # Same plausibility cut as define_regression_labels.py: > 200 is a bad value.
    egfr = egfr.where(egfr <= args.max_egfr)
    return pd.DataFrame({
        "diagnosis": reg[args.diagnosis_col].str.strip().fillna("missing").to_numpy(),
        "egfr": egfr.to_numpy(),
        "event_type": reg[args.event_type_col].fillna("").str.strip().to_numpy(),
    }, index=reg[args.biopsy_col].to_numpy())


def common_table(models, split, clinical, error):
    """Biopsies every model scored on ``split``: risks, time, event and clinical."""
    risks, labels = {}, []
    for name, pattern in models.items():
        path = pattern.format(split=split)
        if not os.path.isfile(path):
            error(f"Missing {path} (model {name}, split {split}).")
        df = load_risks(path)
        risks[name] = df["risk"]
        labels.append(df[["time", "event"]])
    risk = pd.DataFrame(risks).dropna()
    lab = pd.concat(labels).groupby(level=0).first().loc[risk.index]
    missing = ~risk.index.isin(clinical.index)
    if missing.any():
        print(f"  [{split}] {missing.sum()} biopsies not in the registry CSV — dropped")
    keep = risk.index[~missing]
    return risk.loc[keep], lab.loc[keep], clinical.loc[keep]


# ---------------------------------------------------------------------------
# Pairs
# ---------------------------------------------------------------------------
def comparable_pairs(time, event):
    """comp[i, j]: i had an event strictly before j's time (Harrell-comparable)."""
    return (event[:, None] == 1) & (time[:, None] < time[None, :])


def pair_cindex(comp, risk, within=None):
    """C-index over the comparable pairs, optionally only pairs inside ``within``."""
    if within is not None:
        comp = comp & within
    n = comp.sum()
    if n == 0:
        return np.nan, 0
    conc = (comp & (risk[:, None] > risk[None, :])).sum()
    tied = (comp & (risk[:, None] == risk[None, :])).sum()
    return (conc + 0.5 * tied) / n, int(n)


def bootstrap(fn, n, n_boot, seed):
    """Percentile 95% CI of fn(idx) over bootstrap resamples of n samples."""
    rng = np.random.default_rng(seed)
    vals = [fn(rng.integers(0, n, n)) for _ in range(n_boot)]
    vals = np.array([v for v in vals if not np.isnan(v)])
    return (np.percentile(vals, 2.5), np.percentile(vals, 97.5)) if len(vals) else (np.nan, np.nan)


def cidx(risk, time, event):
    if event.sum() == 0 or len(risk) < 2:
        return np.nan
    return concordance_index(np.asarray(risk, float), np.asarray(time, float),
                             np.asarray(event, int))


# ---------------------------------------------------------------------------
# Clinical covariates
# ---------------------------------------------------------------------------
class ClinicalDesign:
    """Diagnosis dummies + standardised eGFR (+ missing flag), fixed on the fit split."""

    def __init__(self, clin_fit, min_group):
        counts = clin_fit["diagnosis"].value_counts()
        # Largest group is the reference category (no dummy).
        self.levels = [d for d in counts.index if counts[d] >= min_group][1:]
        self.reference = counts.index[0]
        e = clin_fit["egfr"]
        self.mean, self.sd = e.mean(), e.std() or 1.0

    def __call__(self, clin, parts=("diagnosis", "egfr")):
        cols = {}
        if "diagnosis" in parts:
            for lvl in self.levels:
                cols[f"dx={lvl}"] = (clin["diagnosis"] == lvl).astype(float)
        if "egfr" in parts:
            e = clin["egfr"]
            cols["egfr_z"] = ((e - self.mean) / self.sd).fillna(0.0)
            cols["egfr_missing"] = e.isna().astype(float)
        return pd.DataFrame(cols, index=clin.index)


def cox_loglik(X, time, event, beta):
    """Unpenalised Cox log partial likelihood at beta."""
    loss, _ = _cox_loss_grad(beta, X, time, event, 0.0)
    return -loss * max(event.sum(), 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Compare survival models on shared test biopsies, against "
                    "diagnosis and eGFR, and through their discordant pairs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", action="append", required=True, metavar="NAME=PATTERN",
                   help="A model's risk_scores.csv path with {split}; repeat per model.")
    p.add_argument("--fit_split", default="val",
                   help="Split the clinical / added-value Cox models are fitted on.")
    p.add_argument("--eval_split", default="test", help="Split everything is reported on.")
    p.add_argument("--registry_csv",
                   default="followup_data/derived/renamed/registry_anonymized.csv")
    p.add_argument("--biopsy_col", default="biop_number")
    p.add_argument("--diagnosis_col", default="Diagnosis")
    p.add_argument("--egfr_col", default="eGFR")
    p.add_argument("--event_type_col", default="Event")
    p.add_argument("--max_egfr", type=float, default=200.0,
                   help="eGFR above this is treated as missing (implausible).")
    p.add_argument("--kidney_events", nargs="+", default=["Dialyse", "CKD5", "NyreTx"],
                   help="Event types counted as kidney failure.")
    p.add_argument("--death_events", nargs="+", default=["Death"])
    p.add_argument("--min_group", type=int, default=30,
                   help="Smallest diagnosis group given its own category / C-index.")
    p.add_argument("--ridge", type=float, default=1e-2,
                   help="Ridge strength of the clinical Cox models (stabilises rare dummies).")
    p.add_argument("--n_boot", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_hard", type=int, default=30, help="Rows in hard_biopsies.csv.")
    p.add_argument("--output_dir", default="results/model_signal")
    args = p.parse_args()
    models = parse_models(args.model, p)
    names = list(models)
    os.makedirs(args.output_dir, exist_ok=True)
    out = lambda f: os.path.join(args.output_dir, f)  # noqa: E731

    clinical = load_clinical(args)
    risk_f, lab_f, clin_f = common_table(models, args.fit_split, clinical, p.error)
    risk, lab, clin = common_table(models, args.eval_split, clinical, p.error)
    t, e = lab["time"].to_numpy(float), lab["event"].to_numpy(int)
    n = len(t)
    print(f"[{args.eval_split}] {n} biopsies scored by all {len(names)} models, "
          f"{e.sum()} events | [{args.fit_split}] {len(risk_f)} for fitting")

    # Event types should explain every event; report mismatches instead of guessing.
    kidney = np.isin(clin["event_type"], args.kidney_events)
    death = np.isin(clin["event_type"], args.death_events)
    unknown = (e == 1) & ~kidney & ~death
    if unknown.any():
        print(f"  Warning: {unknown.sum()} events have no recognised event type "
              f"({sorted(set(clin['event_type'][unknown]))}); counted in neither endpoint.")

    # --- 1. Overview on the common biopsies ------------------------------------------
    print("\n1. C-index on the common biopsies (bootstrap 95% CI)")
    rows = []
    for m in names:
        r = risk[m].to_numpy()
        c = cidx(r, t, e)
        lo, hi = bootstrap(lambda i: cidx(r[i], t[i], e[i]), n, args.n_boot, args.seed)
        rows.append(dict(model=m, n=n, events=int(e.sum()), c_index=c, ci_low=lo, ci_high=hi))
        print(f"  {m:16s} C={c:.4f}  [{lo:.4f}, {hi:.4f}]")
    pd.DataFrame(rows).to_csv(out("cindex_overview.csv"), index=False)

    # --- 2. Event type ------------------------------------------------------------------
    print("\n2. By event type (the other event type censored at its time)")
    e_k, e_d = (e == 1) & kidney, (e == 1) & death
    print(f"  {e_k.sum()} kidney-failure events {sorted(set(args.kidney_events))}, "
          f"{e_d.sum()} deaths")
    rows = []
    for m in names:
        r = risk[m].to_numpy()
        ck, cd = cidx(r, t, e_k.astype(int)), cidx(r, t, e_d.astype(int))
        rows.append(dict(model=m, c_all=cidx(r, t, e), c_kidney_failure=ck, c_death=cd))
        print(f"  {m:16s} all={rows[-1]['c_all']:.4f}  kidney failure={ck:.4f}  death={cd:.4f}")
    pd.DataFrame(rows).to_csv(out("event_type_cindex.csv"), index=False)

    # --- 3. Clinical baselines and added value -------------------------------------------
    print(f"\n3. Clinical baselines and added value (Cox fitted on {args.fit_split}, "
          f"C-index on {args.eval_split})")
    design = ClinicalDesign(clin_f, args.min_group)
    tf, ef = lab_f["time"].to_numpy(float), lab_f["event"].to_numpy(int)
    print(f"  diagnosis categories: {len(design.levels) + 1} (reference "
          f"{design.reference!r}; groups < {args.min_group} on {args.fit_split} share "
          f"the reference); missing diagnosis: "
          f"{(clin['diagnosis'] == 'missing').mean():.0%} of {args.eval_split}")
    rows, clin_scores = [], {}
    for label, parts in (("diagnosis", ("diagnosis",)), ("eGFR", ("egfr",)),
                         ("diagnosis+eGFR", ("diagnosis", "egfr"))):
        Xf, Xe = design(clin_f, parts).to_numpy(), design(clin, parts).to_numpy()
        beta = fit_cox(Xf, tf, ef, args.ridge)
        clin_scores[label] = Xe @ beta
        c = cidx(clin_scores[label], t, e)
        rows.append(dict(model=f"[clinical] {label}", c_index=c))
        print(f"  {'[clinical] ' + label:28s} C={c:.4f}")

    Xf_c, Xe_c = design(clin_f).to_numpy(), design(clin).to_numpy()
    beta_c = fit_cox(Xf_c, tf, ef, args.ridge)
    ll_c = cox_loglik(Xf_c, tf, ef, beta_c)
    s_clin = Xe_c @ beta_c
    for m in names:
        mu, sd = risk_f[m].mean(), risk_f[m].std() or 1.0
        zf = ((risk_f[m] - mu) / sd).to_numpy()[:, None]
        ze = ((risk[m] - mu) / sd).to_numpy()[:, None]
        Xf, Xe = np.hstack([Xf_c, zf]), np.hstack([Xe_c, ze])
        beta = fit_cox(Xf, tf, ef, args.ridge)
        lr = 2 * (cox_loglik(Xf, tf, ef, beta) - ll_c)
        s_full = Xe @ beta
        c_full = cidx(s_full, t, e)
        gain = c_full - cidx(s_clin, t, e)
        lo, hi = bootstrap(
            lambda i: cidx(s_full[i], t[i], e[i]) - cidx(s_clin[i], t[i], e[i]),
            n, args.n_boot, args.seed,
        )
        rows.append(dict(model=f"diagnosis+eGFR+{m}", c_index=c_full, c_gain=gain,
                         gain_ci_low=lo, gain_ci_high=hi,
                         hr_per_sd=float(np.exp(beta[-1])),
                         lr_stat=lr, lr_p=float(chi2.sf(max(lr, 0), 1))))
        print(f"  {'diagnosis+eGFR+' + m:28s} C={c_full:.4f}  gain={gain:+.4f} "
              f"[{lo:+.4f}, {hi:+.4f}]  HR/SD={np.exp(beta[-1]):.2f}  "
              f"LR p={rows[-1]['lr_p']:.2g}")
    pd.DataFrame(rows).to_csv(out("clinical_added_value.csv"), index=False)

    # --- 4. Within diagnosis -------------------------------------------------------------
    print("\n4. Within diagnosis (only pairs of patients with the same diagnosis)")
    comp = comparable_pairs(t, e)
    dx = clin["diagnosis"].to_numpy()
    same = dx[:, None] == dx[None, :]
    known = (dx != "missing")[:, None] & (dx != "missing")[None, :]
    rows = []
    for m in names:
        r = risk[m].to_numpy()
        c_all, n_all = pair_cindex(comp, r)
        c_in, n_in = pair_cindex(comp, r, same & known)
        c_mis, _ = pair_cindex(comp, r, same & ~known)
        rows.append(dict(model=m, c_all=c_all, c_within_diagnosis=c_in,
                         pairs_within=n_in, pairs_all=n_all, c_within_missing_dx=c_mis))
        print(f"  {m:16s} all pairs C={c_all:.4f}  within diagnosis C={c_in:.4f} "
              f"({n_in} of {n_all} pairs)  within missing-diagnosis C={c_mis:.4f}")
    pd.DataFrame(rows).to_csv(out("stratified_cindex.csv"), index=False)

    rows = []
    for d, idx in clin.groupby("diagnosis").indices.items():
        if len(idx) < args.min_group or e[idx].sum() < 5:
            continue
        row = dict(diagnosis=d, n=len(idx), events=int(e[idx].sum()))
        for m in names:
            row[m] = cidx(risk[m].to_numpy()[idx], t[idx], e[idx])
        rows.append(row)
    per_dx = pd.DataFrame(rows).sort_values("n", ascending=False)
    per_dx.to_csv(out("per_diagnosis_cindex.csv"), index=False)
    print(f"  per-diagnosis C-index ({len(per_dx)} groups with >= {args.min_group} "
          f"biopsies and >= 5 events) -> per_diagnosis_cindex.csv")
    if not per_dx.empty:
        print(per_dx.head(8).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # --- 5. Agreement between models ------------------------------------------------------
    print("\n5. Spearman correlation of risks (lower triangle: after removing "
          "diagnosis + eGFR)")
    ranks = risk.apply(rankdata)
    Xc = np.hstack([np.ones((n, 1)), Xe_c])
    resid = ranks - Xc @ np.linalg.lstsq(Xc, ranks.to_numpy(), rcond=None)[0]
    raw = ranks.corr()
    adj = pd.DataFrame(np.corrcoef(resid.T.to_numpy()), index=names, columns=names)
    corr = raw.where(np.triu(np.ones_like(raw, bool)), adj)
    corr.to_csv(out("risk_correlation.csv"))
    print(corr.to_string(float_format=lambda v: f"{v:.2f}"))

    # --- 6. Discordant pairs --------------------------------------------------------------
    print("\n6. Discordant pairs (earlier event ranked lower)")
    gap = (t[None, :] - t[:, None]) / YEAR
    bins = [0, 0.5, 1, 2, 5, np.inf]
    labels = ["<0.5y", "0.5-1y", "1-2y", "2-5y", ">5y"]
    disc = {m: comp & (risk[m].to_numpy()[:, None] < risk[m].to_numpy()[None, :])
            for m in names}
    rows = []
    for lo_b, hi_b, lbl in zip(bins[:-1], bins[1:], labels):
        sel = comp & (gap >= lo_b) & (gap < hi_b)
        row = dict(time_gap=lbl, pairs=int(sel.sum()))
        for m in names:
            row[m] = (disc[m] & sel).sum() / max(sel.sum(), 1)
        rows.append(row)
    by_gap = pd.DataFrame(rows)
    by_gap.to_csv(out("discordance_by_gap.csv"), index=False)
    print("  share of comparable pairs ranked wrongly, by time between the two outcomes:")
    print(by_gap.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    n_comp = comp.sum()
    all_wrong = np.logical_and.reduce([disc[m] for m in names])
    none_wrong = comp & ~np.logical_or.reduce([disc[m] for m in names])
    print(f"  of {n_comp} comparable pairs: {all_wrong.sum() / n_comp:.1%} wrong for every "
          f"model, {none_wrong.sum() / n_comp:.1%} right for every model")
    early_kidney = kidney[:, None] & all_wrong
    early_death = death[:, None] & all_wrong
    print(f"  pairs every model gets wrong, by the earlier patient's event: kidney failure "
          f"{early_kidney.sum() / max(all_wrong.sum(), 1):.0%}, death "
          f"{early_death.sum() / max(all_wrong.sum(), 1):.0%} (all events: kidney "
          f"{(kidney & (e == 1)).sum() / max(e.sum(), 1):.0%}, death "
          f"{(death & (e == 1)).sum() / max(e.sum(), 1):.0%})")
    jac = pd.DataFrame(
        [[(disc[a] & disc[b]).sum() / max((disc[a] | disc[b]).sum(), 1) for b in names]
         for a in names], index=names, columns=names,
    )
    jac.to_csv(out("discordance_overlap.csv"))
    print("  overlap of the discordant pairs between models (Jaccard):")
    print(jac.to_string(float_format=lambda v: f"{v:.2f}"))

    # Per biopsy: share of its comparable pairs ranked wrongly, averaged over models.
    involved = comp.sum(1) + comp.sum(0)
    wrong = np.mean([disc[m].sum(1) + disc[m].sum(0) for m in names], axis=0)
    hard = pd.DataFrame({
        "biopsy": risk.index, "time_years": t / YEAR, "event": e,
        "event_type": clin["event_type"].to_numpy(), "diagnosis": dx,
        "egfr": clin["egfr"].to_numpy(), "comparable_pairs": involved,
        "discordant_share": wrong / np.maximum(involved, 1),
        "mean_risk_percentile": ranks.mean(axis=1).to_numpy() / n,
    })
    hard = hard[hard.comparable_pairs >= 10].sort_values("discordant_share", ascending=False)
    hard.head(args.n_hard).to_csv(out("hard_biopsies.csv"), index=False)
    print(f"  {args.n_hard} biopsies most often ranked wrongly -> hard_biopsies.csv; top 10:")
    print(hard.head(10).to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    print("\nCaveats: diagnosis is made from the same biopsy, so 'within diagnosis' and "
          "'added value' also remove image signal the pathologist already read; "
          "a missing diagnosis is its own category.")
    print(f"Results → {args.output_dir}")


if __name__ == "__main__":
    main()
