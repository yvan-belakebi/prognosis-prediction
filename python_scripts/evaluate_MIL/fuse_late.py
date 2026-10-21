"""
fuse_late.py — Late fusion of single-stain survival models (stacked Cox model).

Learned counterpart of fuse_votes.py: instead of averaging the single-stain
models' votes, a ridge-penalised Cox model learns how much each stain counts.

Features per biopsy, for each single-stain model (one per stain):
    vote_<m>     the model's risk as a percentile of its own training-split scores
                 (or a z-score, --method zscore), centred so 0 is the median; 0 when
                 the biopsy has no slide of that stain
    has_<m>      1 when the stain is present, else 0 (--no_presence drops these)

Where it is fitted.  Not on the training split: the single-stain models have
partly memorised their training biopsies, so their training-split scores are
over-confident and a combiner fitted on them would over-trust every stain. The
combiner is fitted on the validation split instead — the models never trained on
it — with the ridge strength chosen by K-fold cross-validation inside validation
(the C-index on the held-out folds). The test split is only scored, once.

Panel signal.  The presence indicators let the combiner use which stains were
ordered, which can be prognostic by itself (Congo is ordered when amyloid is
suspected). A presence-only model is fitted the same way and reported next to the
full one, so the share of the C-index that comes from the panel rather than from
the images is visible.

Inputs and outputs follow fuse_votes.py: evaluate_survival.py's per-biopsy
risk_scores.csv per model and split (--pattern), and in --output_dir
    test/risk_scores.csv   fused test risk (and val/ — in-sample, for reference),
                           same columns as evaluate_survival.py: feed compare_cindex.py
    coefficients.csv       the fitted coefficient of every feature
    cv.csv                 cross-validated C-index per ridge strength
    summary.csv            per-model, fused and presence-only C-index per split

Usage:
    python python_scripts/evaluate_MIL/fuse_late.py \\
        --models HE_HES PAS PASM Congo Trichrome Sirius_red Immune \\
        --pattern "results/eval/{split}/{model}/risk_scores.csv" \\
        --output_dir results/eval/late_fusion
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
from fuse_votes import cindex, load_model_risks, split_votes  # noqa: E402


# ---------------------------------------------------------------------------
# Ridge-penalised Cox model
# ---------------------------------------------------------------------------
def _cox_loss_grad(beta, X, time, event, lam):
    """Breslow negative log partial likelihood per event, + lam/2 ||beta||^2."""
    eta = X @ beta
    eta = eta - eta.max()  # exp stability; the partial likelihood is shift-invariant
    order = np.argsort(-time, kind="stable")  # risk set of i = samples with t >= t_i
    t, e, x, w = time[order], event[order], X[order], np.exp(eta[order])
    s0 = np.cumsum(w)
    s1 = np.cumsum(w[:, None] * x, axis=0)
    # Ties: every sample of a tied time sees the whole tied group in its risk set.
    last = np.r_[np.flatnonzero(np.diff(t) != 0), len(t) - 1]
    group_end = np.repeat(last, np.diff(np.r_[-1, last]))
    s0, s1 = s0[group_end], s1[group_end]
    n_events = max(e.sum(), 1)
    loss = -(e * (eta[order] - np.log(s0))).sum() / n_events
    grad = -(e[:, None] * (x - s1 / s0[:, None])).sum(0) / n_events
    return loss + 0.5 * lam * beta @ beta, grad + lam * beta


def fit_cox(X, time, event, lam):
    """Ridge Cox coefficients (no intercept: the partial likelihood has none)."""
    res = minimize(
        _cox_loss_grad, np.zeros(X.shape[1]), args=(X, time, event, lam),
        jac=True, method="L-BFGS-B",
    )
    return res.x


def stratified_folds(event, k, seed):
    """Fold id per sample, with events spread evenly over the folds."""
    rng = np.random.default_rng(seed)
    folds = np.empty(len(event), dtype=int)
    for flag in (0, 1):
        idx = rng.permutation(np.flatnonzero(event == flag))
        folds[idx] = np.arange(len(idx)) % k
    return folds


def choose_lambda(X, time, event, lams, k, seed):
    """Held-out C-index per ridge strength, and the one-standard-error choice.

    With a validation split of a few hundred biopsies the CV curve is flat near
    its maximum, and the arg-max lands on an almost unpenalised fit that gives
    weight to noise. The one-standard-error rule picks the strongest penalty
    whose CV C-index is within one standard error of the best one.
    """
    folds = stratified_folds(event, k, seed)
    rows = []
    for lam in lams:
        cs = []
        for f in range(k):
            tr, te = folds != f, folds == f
            beta = fit_cox(X[tr], time[tr], event[tr], lam)
            df = pd.DataFrame({"risk": X[te] @ beta, "time": time[te], "event": event[te]})
            cs.append(cindex(df, "risk"))
        rows.append({"lambda": lam, "cv_cindex": np.nanmean(cs), "cv_sd": np.nanstd(cs)})
    cv = pd.DataFrame(rows)
    top = cv.loc[cv.cv_cindex.idxmax()]
    threshold = top.cv_cindex - top.cv_sd / np.sqrt(k)
    best = cv[cv.cv_cindex >= threshold]["lambda"].max()
    return best, cv


def design(votes, method, presence):
    """Feature matrix: centred votes (0 where absent) + presence indicators."""
    centre = 0.5 if method == "rank" else 0.0
    feats = (votes - centre).fillna(0.0).add_prefix("vote_")
    if presence:
        feats = feats.join(votes.notna().astype(float).add_prefix("has_"))
    return feats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Late fusion: a ridge Cox model stacked on single-stain risks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--models", nargs="+", required=True,
                   help="Model names, substituted for {model} in --pattern.")
    p.add_argument("--pattern", default="results/eval/{split}/{model}/risk_scores.csv",
                   help="Path of a model's risk_scores.csv, with {split} and {model}.")
    p.add_argument("--fit_split", default="val",
                   help="Split the combiner is fitted (and cross-validated) on.")
    p.add_argument("--eval_splits", nargs="+", default=["test"],
                   help="Splits scored with the fitted combiner.")
    p.add_argument("--reference_split", default="train",
                   help="Split whose scores define each model's normalisation.")
    p.add_argument("--method", default="rank", choices=["rank", "zscore"])
    p.add_argument("--no_presence", action="store_true",
                   help="Leave out the stain-presence indicators (image signal only).")
    p.add_argument("--lambdas", nargs="+", type=float,
                   default=[1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0],
                   help="Ridge strengths tried in cross-validation.")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", default="results/eval/late_fusion")
    args = p.parse_args()
    if args.fit_split in args.eval_splits:
        p.error("--fit_split must not be one of --eval_splits (it would be in-sample).")

    splits = [args.fit_split, *args.eval_splits]
    risks = load_model_risks(
        args.models, args.pattern, {args.reference_split, *splits}, p.error
    )
    data = {}
    for s in splits:
        votes, lab = split_votes(risks, args.models, s, args.reference_split, args.method)
        data[s] = (votes, lab)

    def matrix(split, presence):
        votes, lab = data[split]
        feats = design(votes, args.method, presence)
        return feats, lab["time"].to_numpy(float), lab["event"].to_numpy(int)

    fits = {}
    variants = [("fused", not args.no_presence)]
    if not args.no_presence:
        variants.append(("presence_only", None))
    for name, presence in variants:
        if name == "presence_only":
            feats, t, e = matrix(args.fit_split, True)
            feats = feats.filter(like="has_")
        else:
            feats, t, e = matrix(args.fit_split, presence)
        lam, cv = choose_lambda(feats.to_numpy(), t, e, args.lambdas, args.folds, args.seed)
        beta = fit_cox(feats.to_numpy(), t, e, lam)
        fits[name] = (list(feats.columns), beta, lam, cv)
        print(f"[{name}] fitted on {args.fit_split} ({len(t)} biopsies, {int(e.sum())} "
              f"events); ridge lambda={lam:g} "
              f"(CV C-index {cv.set_index('lambda').loc[lam, 'cv_cindex']:.4f})")

    os.makedirs(args.output_dir, exist_ok=True)
    cols, beta, lam, cv = fits["fused"]
    cv.to_csv(os.path.join(args.output_dir, "cv.csv"), index=False)
    coef = pd.DataFrame({"feature": cols, "coefficient": beta})
    coef.to_csv(os.path.join(args.output_dir, "coefficients.csv"), index=False)
    print("\nCoefficients (log hazard ratio; for a vote: median -> top of that model's "
          "training scores is +0.5 units):")
    for c, b in zip(cols, beta):
        print(f"  {c:20s} {b:+.4f}")

    summary = []
    for split in splits:
        votes, lab = data[split]
        risk = {}
        for name, (cols_n, beta_n, _, _) in fits.items():
            feats = design(votes, args.method, True)
            risk[name] = feats.reindex(columns=cols_n, fill_value=0.0).to_numpy() @ beta_n
        out = lab.assign(risk=risk["fused"])[["risk", "time", "event"]]
        tag = " (in-sample: the combiner was fitted here)" if split == args.fit_split else ""
        out_dir = os.path.join(args.output_dir, split)
        os.makedirs(out_dir, exist_ok=True)
        out.rename_axis("biopsy").reset_index().to_csv(
            os.path.join(out_dir, "risk_scores.csv"), index=False
        )
        print(f"\n[{split}]{tag} {len(out)} biopsies, {int(out['event'].sum())} events")
        present = votes.notna()
        for m in args.models:
            sub = out.loc[present[m]].assign(own=risks[m, split]["risk"])
            c_m, c_f = cindex(sub, "own"), cindex(sub, "risk")
            print(f"  {m:14s} n={len(sub):5d}  C={c_m:.4f}  "
                  f"| fused on the same biopsies C={c_f:.4f}")
            summary.append(dict(split=split, model=m, n=len(sub), c_index=c_m,
                                fused_c_index_same_biopsies=c_f))
        for name in fits:
            c = cindex(lab.assign(risk=risk[name]), "risk")
            label = "FUSED" if name == "fused" else "PRESENCE ONLY"
            print(f"  {label:14s} n={len(out):5d}  C={c:.4f}")
            summary.append(dict(split=split, model=label, n=len(out), c_index=c,
                                fused_c_index_same_biopsies=np.nan))
    pd.DataFrame(summary).to_csv(os.path.join(args.output_dir, "summary.csv"), index=False)
    print(f"\nFused risk scores → {args.output_dir}/<split>/risk_scores.csv")


if __name__ == "__main__":
    main()
