"""
survival_probability.py — n-year event probabilities from risk scores, and model comparison at
fixed horizons.

A MIL risk score only ranks biopsies. To turn it into P(event by t years), each model is
recalibrated on --fit_split (val: the models never trained on it):

  1. standardise the risk with the fit-split mean / SD;
  2. fit a Cox model h(t) = h0(t) exp(beta * z) (beta is the calibration slope);
  3. estimate the baseline cumulative hazard H0 with Breslow;
  4. predict on --eval_split (test): P(event by t) = 1 - exp(-H0(t) exp(beta * z)).

The same is done for diagnosis + eGFR ("clinical") and for diagnosis + eGFR + each model.
The Kaplan-Meier curve of the fit split (same probability for everyone) is the null model.

With --endpoint kidney, the event is kidney failure and death is a competing risk, not
censoring: one cause-specific Cox model per cause (each with its own Breslow hazard), combined
into the cumulative incidence of kidney failure. Observed risks then come from the
Aalen-Johansen estimator instead of 1 - Kaplan-Meier.

Metrics at each horizon t (censoring handled by inverse probability of censoring weights,
the censoring distribution estimated by Kaplan-Meier on the evaluated biopsies):

  AUC(t)    cumulative/dynamic: events by t (cases) against biopsies event-free at t (and, for
            --endpoint kidney, those who died first) — how well the model ranks at t. The
            matching IPCW ROC curves are drawn per horizon (roc_curves.png / .csv).
  Brier(t)  mean squared error of the probability; IPA(t) = 1 - Brier / Brier(null): 0 is no
            better than Kaplan-Meier, higher is better.
  Calibration  predicted vs observed risk by decile of predicted risk; cal_error is the
            n-weighted mean |predicted - observed| over deciles; test_cal_slope is the Cox slope
            of the model's linear predictor on the eval split (1 = well calibrated, < 1 =
            predictions too extreme).
  Net benefit  decision curve at --dca_horizon.

CIs are percentile bootstrap over eval biopsies; paired differences (AUC, IPA) against
--reference and against the clinical model use the same resamples.

Caveat: a model itself fitted on --fit_split (late fusion) is recalibrated in-sample. Only
beta and H0 are added, so the bias is small, but its probabilities are slightly optimistic.

Inputs are the same as analyze_model_signal.py: one --model NAME=path/{split}/risk_scores.csv
per model (biopsy, risk, time in days, event) and the registry CSV.

Usage:
    python python_scripts/evaluate_MIL/survival_probability.py \\
        --model PAS=results/eval/{split}/PAS/risk_scores.csv \\
        --model voting=results/eval/voting_rank_none/{split}/risk_scores.csv \\
        --model late=results/eval/late_fusion/{split}/risk_scores.csv \\
        --model unicorn=results/eval/{split}/unicorn/risk_scores.csv \\
        --model early=results/eval/{split}/early/risk_scores.csv \\
        --output_dir results/survival_probability
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
from analyze_model_signal import (  # noqa: E402
    YEAR, ClinicalDesign, common_table, load_clinical, parse_models,
)
from fuse_late import fit_cox  # noqa: E402

METRICS = ("auc", "brier", "ipa")
_trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy < 2.0 has only trapz


# ---------------------------------------------------------------------------
# Step functions and non-parametric estimators
# ---------------------------------------------------------------------------
def step(times, values, t, before=False, start=1.0):
    """Right-continuous step function (jumps at ``times``) evaluated at t, or just before t."""
    idx = np.searchsorted(times, t, side="left" if before else "right") - 1
    return np.where(idx >= 0, values[np.maximum(idx, 0)], start)


def km_curve(time, event):
    """Kaplan-Meier: distinct times and the survival just after each."""
    ut, inv = np.unique(time, return_inverse=True)
    d = np.bincount(inv, weights=event, minlength=len(ut))
    n_risk = np.bincount(inv, minlength=len(ut))[::-1].cumsum()[::-1]
    return ut, np.cumprod(1 - d / n_risk)


def aalen_johansen(time, status, t):
    """Observed P(cause 1 by t), deaths of other causes (status 2) competing.

    Without competing events this is 1 - Kaplan-Meier."""
    if len(time) == 0:
        return np.nan
    ut, inv = np.unique(time, return_inverse=True)
    n_risk = np.bincount(inv, minlength=len(ut))[::-1].cumsum()[::-1]
    d1 = np.bincount(inv, weights=status == 1, minlength=len(ut))
    d_all = np.bincount(inv, weights=status > 0, minlength=len(ut))
    s_prev = np.r_[1.0, np.cumprod(1 - d_all / n_risk)[:-1]]
    return step(ut, np.cumsum(s_prev * d1 / n_risk), t, start=0.0)


# ---------------------------------------------------------------------------
# Cox + Breslow recalibration
# ---------------------------------------------------------------------------
def breslow(lp, time, event):
    """Baseline cumulative hazard at the event times for linear predictor ``lp``."""
    ut, d = np.unique(time[event == 1], return_counts=True)
    order = np.argsort(time, kind="stable")
    ts, tail = time[order], np.exp(lp[order])[::-1].cumsum()[::-1]
    return ut, np.cumsum(d / tail[np.searchsorted(ts, ut, side="left")])


def fit_predictor(Xf, tf, sf, Xe, causes, lam):
    """Per cause: Cox on the fit split, Breslow H0, and the eval-split linear predictor."""
    fitted = []
    for c in causes:
        ev = (sf == c).astype(int)
        beta = fit_cox(Xf, tf, ev, lam)
        ut, H0 = breslow(Xf @ beta, tf, ev)
        fitted.append(dict(beta=beta, times=ut, H0=H0, lp=Xe @ beta))
    return fitted


def event_prob(fitted, t):
    """P(cause 1 by t) for every eval biopsy, shape (n, len(t)).

    One cause: 1 - exp(-H(t)). Competing causes: cumulative incidence
    sum over jumps s <= t of S(s-) dH1(s), with S = exp(-(H1 + H2))."""
    grid = np.unique(np.concatenate([f["times"] for f in fitted]))
    dH = [np.diff(np.r_[0.0, step(f["times"], f["H0"], grid, start=0.0)])[None, :]
          * np.exp(f["lp"])[:, None] for f in fitted]
    if len(dH) == 1:
        cif = 1 - np.exp(-np.cumsum(dH[0], axis=1))
    else:
        H = np.cumsum(sum(dH), axis=1)
        s_prev = np.exp(-np.hstack([np.zeros((H.shape[0], 1)), H[:, :-1]]))
        cif = np.cumsum(s_prev * dH[0], axis=1)
    idx = np.searchsorted(grid, t, side="right") - 1
    return np.where(idx >= 0, cif[:, np.maximum(idx, 0)], 0.0)


# ---------------------------------------------------------------------------
# IPCW metrics
# ---------------------------------------------------------------------------
def ipcw(time, status, t, g_times, g_surv, g_before):
    """Cases (cause 1 by t) and IPCW weights; biopsies censored before t get weight 0."""
    g_t = step(g_times, g_surv, t)
    if g_t <= 0:
        return None, None
    known = (time <= t) & (status > 0)
    w = np.zeros(len(time))
    w[known] = 1 / g_before[known]
    w[time > t] = 1 / g_t
    return known & (status == 1), w


def weighted_auc(score, case, w):
    """Weighted P(score_case > score_control), ties counted 0.5 (sorted, O(n log n))."""
    ctrl = (w > 0) & ~case
    if not case.any() or not ctrl.any():
        return np.nan
    order = np.argsort(score[ctrl])
    cs, cw = score[ctrl][order], w[ctrl][order]
    cum = np.r_[0.0, np.cumsum(cw)]
    lo = cum[np.searchsorted(cs, score[case], side="left")]
    hi = cum[np.searchsorted(cs, score[case], side="right")]
    return (w[case] * (lo + 0.5 * (hi - lo))).sum() / (w[case].sum() * cw.sum())


def horizon_metrics(P, null, time, status, horizons):
    """AUC, Brier and IPA per predictor and horizon: array (predictors, horizons, 3).

    P: (predictors, n, horizons) predicted risks; null: (horizons,) null-model risks."""
    g_times, g_surv = km_curve(time, (status == 0).astype(float))
    g_before = step(g_times, g_surv, time, before=True)
    res = np.full((P.shape[0], len(horizons), 3), np.nan)
    for k, h in enumerate(horizons):
        case, w = ipcw(time, status, h, g_times, g_surv, g_before)
        if case is None:
            continue
        b_null = np.mean(w * (case - null[k]) ** 2)
        for j in range(P.shape[0]):
            brier = np.mean(w * (case - P[j, :, k]) ** 2)
            res[j, k] = (weighted_auc(P[j, :, k], case, w), brier,
                         1 - brier / b_null if b_null > 0 else np.nan)
    return res


def weighted_roc(score, case, w):
    """IPCW ROC at one horizon: (FPR, TPR) with one point per distinct threshold, high to low."""
    keep = w > 0
    order = np.argsort(-score[keep], kind="stable")
    s, c, ww = score[keep][order], case[keep][order], w[keep][order]
    tp, fp = np.cumsum(ww * c), np.cumsum(ww * ~c)
    if tp[-1] == 0 or fp[-1] == 0:
        return None
    last = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]
    return np.r_[0.0, fp[last] / fp[-1]], np.r_[0.0, tp[last] / tp[-1]]


def calibration_table(pred, time, status, h, n_bins):
    """Deciles of predicted risk: mean predicted vs Aalen-Johansen observed risk at h."""
    bins = pd.qcut(pd.Series(pred).rank(method="first"), n_bins, labels=False).to_numpy()
    rows = []
    for b in range(n_bins):
        m = bins == b
        rows.append(dict(bin=b + 1, n=int(m.sum()), predicted=pred[m].mean(),
                         observed=float(aalen_johansen(time[m], status[m], h))))
    return pd.DataFrame(rows)


def net_benefit(pred, time, status, h, thresholds):
    """Decision curve: treat when predicted risk >= threshold (observed risk by Aalen-Johansen)."""
    out = []
    for pt in thresholds:
        sel = pred >= pt
        if not sel.any():
            out.append(0.0)
            continue
        risk = float(aalen_johansen(time[sel], status[sel], h))
        out.append(sel.mean() * (risk - (1 - risk) * pt / (1 - pt)))
    return np.array(out)


def ci(vals):
    vals = vals[~np.isnan(vals)]
    return (np.percentile(vals, 2.5), np.percentile(vals, 97.5)) if len(vals) else (np.nan, np.nan)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="n-year event probabilities (Cox + Breslow recalibration) and model "
                    "comparison by time-dependent AUC, Brier / IPA, calibration and net benefit.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", action="append", required=True, metavar="NAME=PATTERN",
                   help="A model's risk_scores.csv path with {split}; repeat per model.")
    p.add_argument("--reference", default=None,
                   help="Model the others are compared with (default: the first --model).")
    p.add_argument("--fit_split", default="val", help="Split the recalibration is fitted on.")
    p.add_argument("--eval_split", default="test", help="Split everything is reported on.")
    p.add_argument("--endpoint", default="all", choices=["all", "kidney"],
                   help="all: any event; kidney: kidney failure, with death as a competing risk.")
    p.add_argument("--horizons", nargs="+", type=float, default=[1, 3, 5, 10],
                   help="Horizons in years.")
    p.add_argument("--ibs_max", type=float, default=5.0,
                   help="Integrated Brier score over (0, ibs_max] years.")
    p.add_argument("--calib_horizon", type=float, default=5.0)
    p.add_argument("--dca_horizon", type=float, default=5.0)
    p.add_argument("--n_bins", type=int, default=10, help="Calibration groups.")
    p.add_argument("--registry_csv",
                   default="followup_data/derived/renamed/registry_anonymized.csv")
    p.add_argument("--biopsy_col", default="biop_id")
    p.add_argument("--diagnosis_col", default="Diagnosis")
    p.add_argument("--egfr_col", default="eGFR")
    p.add_argument("--event_type_col", default="Event")
    p.add_argument("--max_egfr", type=float, default=200.0)
    p.add_argument("--kidney_events", nargs="+", default=["Dialyse", "CKD5", "NyreTx"])
    p.add_argument("--min_group", type=int, default=30)
    p.add_argument("--ridge", type=float, default=1e-2,
                   help="Ridge strength of the Cox models that include clinical covariates.")
    p.add_argument("--n_boot", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", default="results/survival_probability")
    args = p.parse_args()
    models = parse_models(args.model, p)
    names = list(models)
    reference = args.reference or names[0]
    if reference not in models:
        p.error(f"--reference {reference!r} is not one of the --model names {names}.")
    os.makedirs(args.output_dir, exist_ok=True)
    out = lambda f: os.path.join(args.output_dir, f)  # noqa: E731

    clinical = load_clinical(args)
    risk_f, lab_f, clin_f = common_table(models, args.fit_split, clinical, p.error)
    risk, lab, clin = common_table(models, args.eval_split, clinical, p.error)
    tf, te = lab_f["time"].to_numpy(float) / YEAR, lab["time"].to_numpy(float) / YEAR

    def status_of(lab_, clin_, split):
        event = lab_["event"].to_numpy(int)
        if args.endpoint == "all":
            return event
        kidney = np.isin(clin_["event_type"], args.kidney_events)
        unknown = (event == 1) & ~kidney & (clin_["event_type"].to_numpy() != "Death")
        if unknown.any():
            print(f"  [{split}] {unknown.sum()} events of unknown type counted as competing")
        return np.where(event == 1, np.where(kidney, 1, 2), 0)

    sf, se = status_of(lab_f, clin_f, args.fit_split), status_of(lab, clin, args.eval_split)
    causes = [1] if args.endpoint == "all" else [1, 2]
    n = len(te)
    print(f"[{args.eval_split}] {n} biopsies scored by all {len(names)} models, "
          f"{(se == 1).sum()} events ({args.endpoint}), {(se == 2).sum()} competing | "
          f"[{args.fit_split}] {len(tf)} for recalibration")

    # --- Recalibrate every predictor on the fit split ---------------------------------------
    design = ClinicalDesign(clin_f, args.min_group)
    Xf_c, Xe_c = design(clin_f).to_numpy(), design(clin).to_numpy()
    fits = {"clinical": fit_predictor(Xf_c, tf, sf, Xe_c, causes, args.ridge)}
    z = {}
    for m in names:
        mu, sd = risk_f[m].mean(), risk_f[m].std() or 1.0
        z[m] = (((risk_f[m] - mu) / sd).to_numpy()[:, None], ((risk[m] - mu) / sd).to_numpy()[:, None])
        fits[m] = fit_predictor(z[m][0], tf, sf, z[m][1], causes, 0.0)
    for m in names:
        fits[f"clinical+{m}"] = fit_predictor(np.hstack([Xf_c, z[m][0]]), tf, sf,
                                              np.hstack([Xe_c, z[m][1]]), causes, args.ridge)
    preds = list(fits)

    # Predictions on one time grid: horizons, the IBS grid and the AUC(t) curve.
    ibs_grid = np.arange(0.25, args.ibs_max + 1e-9, 0.25)
    curve_grid = np.arange(0.5, max(args.horizons) + 1e-9, 0.5)
    grid = np.unique(np.r_[args.horizons, ibs_grid, curve_grid,
                           args.calib_horizon, args.dca_horizon])
    P = np.stack([event_prob(fits[k], grid) for k in preds])          # (predictors, n, grid)
    null = np.array([aalen_johansen(tf, sf, g) for g in grid])
    col = {g: i for i, g in enumerate(grid)}
    hz = [col[h] for h in args.horizons]

    # --- Point estimates and paired bootstrap at the horizons --------------------------------
    point = horizon_metrics(P[:, :, hz], null[hz], te, se, args.horizons)
    rng = np.random.default_rng(args.seed)
    boot = np.empty((args.n_boot,) + point.shape)
    for b in range(args.n_boot):
        i = rng.integers(0, n, n)
        boot[b] = horizon_metrics(P[:, i][:, :, hz], null[hz], te[i], se[i], args.horizons)

    rows, diff_rows, cal_rows = [], [], []
    for j, k in enumerate(preds):
        for h_i, h in enumerate(args.horizons):
            pred = P[j, :, hz[h_i]]
            cal = calibration_table(pred, te, se, h, args.n_bins)
            cal_rows.append(cal.assign(predictor=k, horizon_years=h))
            row = dict(predictor=k, horizon_years=h, n=n,
                       events_by_t=int(((te <= h) & (se == 1)).sum()),
                       event_free_at_t=int((te > h).sum()),
                       mean_predicted=pred.mean(),
                       observed=float(aalen_johansen(te, se, h)),
                       cal_error=np.average(np.abs(cal.predicted - cal.observed), weights=cal.n))
            for x, metric in enumerate(METRICS):
                lo, hi = ci(boot[:, j, h_i, x])
                row.update({metric: point[j, h_i, x], f"{metric}_lo": lo, f"{metric}_hi": hi})
            rows.append(row)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out("horizon_metrics.csv"), index=False)

    for base in dict.fromkeys([reference, "clinical"]):
        jb = preds.index(base)
        for j, k in enumerate(preds):
            if k == base:
                continue
            for h_i, h in enumerate(args.horizons):
                for x, metric in enumerate(METRICS):
                    if metric == "brier":
                        continue
                    d = boot[:, j, h_i, x] - boot[:, jb, h_i, x]
                    d = d[~np.isnan(d)]
                    lo, hi = ci(d)
                    p_val = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())) if len(d) else np.nan
                    diff_rows.append(dict(predictor=k, reference=base, horizon_years=h,
                                          metric=metric,
                                          diff=point[j, h_i, x] - point[jb, h_i, x],
                                          ci_low=lo, ci_high=hi, p=p_val))
    pd.DataFrame(diff_rows).to_csv(out("paired_differences.csv"), index=False)
    pd.concat(cal_rows).to_csv(out("calibration.csv"), index=False)

    # --- Recalibration parameters, test calibration slope, integrated Brier ------------------
    ibs_metrics = horizon_metrics(P[:, :, [col[g] for g in ibs_grid]],
                                  null[[col[g] for g in ibs_grid]], te, se, ibs_grid)
    g_times, g_surv = km_curve(te, (se == 0).astype(float))
    g_before = step(g_times, g_surv, te, before=True)
    null_brier = []
    for g in ibs_grid:
        case, w = ipcw(te, se, g, g_times, g_surv, g_before)
        null_brier.append(np.mean(w * (case - null[col[g]]) ** 2) if case is not None else np.nan)
    ibs_null = _trapz(null_brier, ibs_grid) / (ibs_grid[-1] - ibs_grid[0])
    rec_rows = []
    for j, k in enumerate(preds):
        f = fits[k][0]
        ibs = _trapz(ibs_metrics[j, :, 1], ibs_grid) / (ibs_grid[-1] - ibs_grid[0])
        slope = fit_cox(f["lp"][:, None], te, (se == 1).astype(int), 0.0)[0]
        rec_rows.append(dict(
            predictor=k,
            fit_coef=f["beta"][0] if k in models else np.nan,
            test_cal_slope=slope, ibs=ibs, ibs_null=ibs_null, ibs_scaled=1 - ibs / ibs_null,
            **{f"H0_{h:g}y": float(step(f["times"], f["H0"], h, start=0.0))
               for h in args.horizons},
        ))
    rec = pd.DataFrame(rec_rows)
    rec.to_csv(out("recalibration.csv"), index=False)

    # --- Per-biopsy probabilities ----------------------------------------------------------
    prob = pd.DataFrame({"biopsy": risk.index, "time_years": te, "status": se})
    for j, k in enumerate(preds):
        for h_i, h in enumerate(args.horizons):
            prob[f"{k}_risk_{h:g}y"] = P[j, :, hz[h_i]]
    prob.to_csv(out(f"probabilities_{args.eval_split}.csv"), index=False)

    # --- Print -----------------------------------------------------------------------------
    pd.set_option("display.width", 200)
    print(f"\nRisk of {'kidney failure' if args.endpoint == 'kidney' else 'any event'} by t "
          f"(recalibrated on {args.fit_split}, scored on {args.eval_split}; 95% bootstrap CI)")
    for h in args.horizons:
        sub = metrics[metrics.horizon_years == h]
        r0 = sub.iloc[0]
        print(f"\n  {h:g} y: {r0.events_by_t} events, {r0.event_free_at_t} event-free, "
              f"observed risk {r0.observed:.3f}")
        for r in sub.itertuples():
            print(f"    {r.predictor:24s} AUC {r.auc:.3f} [{r.auc_lo:.3f}, {r.auc_hi:.3f}]  "
                  f"IPA {r.ipa:+.3f} [{r.ipa_lo:+.3f}, {r.ipa_hi:+.3f}]  "
                  f"pred {r.mean_predicted:.3f}  cal_error {r.cal_error:.3f}")
    print(f"\n  Integrated Brier (0-{args.ibs_max:g} y) and test calibration slope:")
    for r in rec.itertuples():
        print(f"    {r.predictor:24s} IBS {r.ibs:.4f} (scaled {r.ibs_scaled:+.3f})  "
              f"slope {r.test_cal_slope:.2f}")

    # --- Figures ---------------------------------------------------------------------------
    shown = ["clinical"] + names
    colors = {k: plt.cm.tab10(i) for i, k in enumerate(shown)}
    colors["clinical"] = "0.5"

    cg = [col[g] for g in curve_grid]
    curve = horizon_metrics(P[:, :, cg], null[cg], te, se, curve_grid)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k in shown:
        ax.plot(curve_grid, curve[preds.index(k), :, 0], marker="o", ms=3, label=k,
                color=colors[k], ls="--" if k == "clinical" else "-")
    ax.set(xlabel="Horizon t (years)", ylabel="AUC(t)", title="Time-dependent AUC")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out("auc_over_time.png"), dpi=150)
    plt.close(fig)

    # ROC curves at each horizon (cases: events by t; controls: event-free at t, plus competing
    # events for --endpoint kidney; IPCW-weighted, so the area equals AUC(t) above).
    ncol = min(2, len(args.horizons))
    nrow = int(np.ceil(len(args.horizons) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 4.4 * nrow), squeeze=False)
    roc_rows = []
    for ax, (h_i, h) in zip(axes.ravel(), enumerate(args.horizons)):
        case, w = ipcw(te, se, h, g_times, g_surv, g_before)
        ax.plot([0, 1], [0, 1], color="k", lw=0.8, ls=":")
        if case is not None:
            for k in shown:
                roc = weighted_roc(P[preds.index(k), :, hz[h_i]], case, w)
                if roc is None:
                    continue
                auc = point[preds.index(k), h_i, 0]
                ax.plot(*roc, color=colors[k], lw=1.4, ls="--" if k == "clinical" else "-",
                        label=f"{k} ({auc:.3f})")
                roc_rows.append(pd.DataFrame({"predictor": k, "horizon_years": h,
                                              "fpr": roc[0], "tpr": roc[1]}))
        ax.set(title=f"{h:g} year{'s' if h != 1 else ''}: {int(case.sum()) if case is not None else 0} "
                     f"events", xlabel="False positive rate", ylabel="True positive rate",
               xlim=(0, 1), ylim=(0, 1), aspect="equal")
        ax.legend(fontsize=7, loc="lower right", title="AUC", title_fontsize=7)
    for ax in axes.ravel()[len(args.horizons):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out("roc_curves.png"), dpi=150)
    plt.close(fig)
    if roc_rows:
        pd.concat(roc_rows).to_csv(out("roc_curves.csv"), index=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    width = 0.8 / len(shown)
    for i, k in enumerate(shown):
        sub = metrics[metrics.predictor == k]
        x = np.arange(len(args.horizons)) + (i - (len(shown) - 1) / 2) * width
        ax.errorbar(x, sub.ipa, yerr=[sub.ipa - sub.ipa_lo, sub.ipa_hi - sub.ipa],
                    fmt="o", color=colors[k], label=k, capsize=2)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(np.arange(len(args.horizons)), [f"{h:g} y" for h in args.horizons])
    ax.set(ylabel="IPA (1 - Brier / Brier of Kaplan-Meier)",
           title="Accuracy of the predicted probability")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out("ipa_by_horizon.png"), dpi=150)
    plt.close(fig)

    h = args.calib_horizon
    ncol = min(3, len(shown))
    nrow = int(np.ceil(len(shown) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.4 * nrow), squeeze=False)
    for ax, k in zip(axes.ravel(), shown):
        cal = calibration_table(P[preds.index(k), :, col[h]], te, se, h, args.n_bins)
        lim = max(cal.predicted.max(), cal.observed.max()) * 1.05
        ax.plot([0, lim], [0, lim], color="k", lw=0.8, ls=":")
        ax.plot(cal.predicted, cal.observed, "o-", color=colors[k])
        ax.set(title=k, xlabel=f"Predicted {h:g}-y risk", ylabel=f"Observed {h:g}-y risk",
               xlim=(0, lim), ylim=(0, lim))
    for ax in axes.ravel()[len(shown):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out(f"calibration_{h:g}y.png"), dpi=150)
    plt.close(fig)

    h = args.dca_horizon
    thresholds = np.arange(0.02, 0.61, 0.02)
    dca = {"treat all": net_benefit(np.ones(n), te, se, h, thresholds),
           "treat none": np.zeros(len(thresholds))}
    for k in shown:
        dca[k] = net_benefit(P[preds.index(k), :, col[h]], te, se, h, thresholds)
    pd.DataFrame(dca, index=pd.Index(thresholds, name="threshold")).to_csv(
        out(f"net_benefit_{h:g}y.csv"))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(thresholds, dca["treat all"], color="k", lw=0.8, label="treat all")
    ax.plot(thresholds, dca["treat none"], color="k", lw=0.8, ls=":", label="treat none")
    for k in shown:
        ax.plot(thresholds, dca[k], color=colors[k], label=k, ls="--" if k == "clinical" else "-")
    ax.set_ylim(-0.02, max(v.max() for v in dca.values()) * 1.1 + 1e-3)
    ax.set(xlabel=f"Risk threshold ({h:g}-y)", ylabel="Net benefit", title="Decision curve")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out(f"net_benefit_{h:g}y.png"), dpi=150)
    plt.close(fig)

    print(f"\nWrote horizon_metrics.csv, paired_differences.csv (vs {reference} and clinical), "
          f"calibration.csv, recalibration.csv, probabilities_{args.eval_split}.csv and figures "
          f"to {args.output_dir}")


if __name__ == "__main__":
    main()
