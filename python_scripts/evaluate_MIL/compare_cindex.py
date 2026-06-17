"""
compare_cindex.py — Bootstrap + variance-covariance test for the difference
between two C-indexes computed from evaluate_survival.py outputs.

Merges both models on shared patient IDs and runs two paired tests:

  1. Bootstrap hypothesis test (non-parametric, centered under H0).
  2. Jackknife variance-covariance Z-test (analogous to the DeLong method for
     AUC): computes Var(C_A), Var(C_B), and Cov(C_A, C_B) via leave-one-out
     jackknife, then tests H0: C_A = C_B with a standard-normal Z-statistic.

Usage:
    python python_scripts/MIL/compare_cindex.py \\
        --inputs_a results/model_a/risk_scores.csv \\
        --inputs_b results/model_b/risk_scores.csv

    # From directories:
    python python_scripts/MIL/compare_cindex.py \\
        --input_dirs_a results/model_a/ \\
        --input_dirs_b results/model_b/

    # With external labels and more bootstrap samples:
    python python_scripts/MIL/compare_cindex.py \\
        --inputs_a results/model_a/risk_scores.csv \\
        --inputs_b results/model_b/risk_scores.csv \\
        --label_csv followup_data/labels_combined.csv \\
        --n_boot 50000
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm
from sksurv.metrics import concordance_index_censored


def load_risk_csv(path):
    df = pd.read_csv(path, dtype={"biopsy": str, "bag_name": str})
    if "biopsy" in df.columns:
        df = df.rename(columns={"biopsy": "id"})
    elif "bag_name" in df.columns:
        df = df.rename(columns={"bag_name": "id"})
    else:
        raise ValueError(
            f"{path}: expected a 'biopsy' or 'bag_name' column, "
            f"found: {list(df.columns)}"
        )
    for col in ("risk", "time", "event"):
        if col not in df.columns:
            raise ValueError(f"{path}: missing required column '{col}'.")
    return df[["id", "risk", "time", "event"]].copy()


def resolve_paths(inputs, input_dirs, csv_name):
    if inputs:
        return list(inputs)
    paths = []
    for d in input_dirs:
        p = os.path.join(d, csv_name)
        if not os.path.isfile(p):
            print(f"Warning: {p} not found — skipping.")
            continue
        paths.append(p)
    return paths


def pool_csvs(paths):
    return pd.concat([load_risk_csv(p) for p in paths], ignore_index=True)


def _cindex(event, time, risk):
    if event.sum() == 0:
        return float("nan")
    c, *_ = concordance_index_censored(event, time, risk)
    return c


def jackknife_loo(event, time, risk):
    """Leave-one-out C-index estimates used for jackknife variance/covariance."""
    n = len(event)
    loo = np.empty(n)
    for k in range(n):
        mask = np.ones(n, dtype=bool)
        mask[k] = False
        e, t, r = event[mask], time[mask], risk[mask]
        if e.sum() == 0:
            loo[k] = np.nan
        else:
            loo[k], *_ = concordance_index_censored(e, t, r)
    return loo


def varcov_test(obs_diff, loo_a, loo_b):
    """
    Jackknife variance-covariance Z-test for H0: C_A = C_B.

    Var(C_A - C_B) = Var(C_A) + Var(C_B) - 2*Cov(C_A, C_B), all estimated
    by the leave-one-out jackknife. Returns (z, p, se, ci_lo, ci_hi) or None
    if the variance estimate is non-positive.
    """
    valid = ~(np.isnan(loo_a) | np.isnan(loo_b))
    n = valid.sum()
    a, b = loo_a[valid], loo_b[valid]
    f = (n - 1) / n  # jackknife factor
    var_a = f * np.sum((a - a.mean()) ** 2)
    var_b = f * np.sum((b - b.mean()) ** 2)
    cov   = f * np.sum((a - a.mean()) * (b - b.mean()))
    var_diff = var_a + var_b - 2 * cov
    if var_diff <= 0:
        return None
    se = np.sqrt(var_diff)
    z  = obs_diff / se
    p  = float(2 * norm.sf(abs(z)))
    ci_lo = obs_diff - 1.96 * se
    ci_hi = obs_diff + 1.96 * se
    return z, p, se, ci_lo, ci_hi


def _sig(p):
    if p < 0.001: return "(p < 0.001 ***)"
    if p < 0.01:  return "(p < 0.01 **)"
    if p < 0.05:  return "(p < 0.05 *)"
    return "(not significant at α = 0.05)"


def bootstrap_diff(event, time, risk_a, risk_b, n_boot, rng):
    n = len(event)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        e, t = event[idx], time[idx]
        if e.sum() == 0:
            diffs[i] = float("nan")
            continue
        ca, *_ = concordance_index_censored(e, t, risk_a[idx])
        cb, *_ = concordance_index_censored(e, t, risk_b[idx])
        diffs[i] = ca - cb
    return diffs


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap test for the difference between two C-indexes."
    )

    grp_a = parser.add_mutually_exclusive_group(required=True)
    grp_a.add_argument("--inputs_a", nargs="+", metavar="CSV",
                       help="Direct paths to risk CSV files for model A.")
    grp_a.add_argument("--input_dirs_a", nargs="+", metavar="DIR",
                       help="Directories containing risk CSV files for model A.")

    grp_b = parser.add_mutually_exclusive_group(required=True)
    grp_b.add_argument("--inputs_b", nargs="+", metavar="CSV",
                       help="Direct paths to risk CSV files for model B.")
    grp_b.add_argument("--input_dirs_b", nargs="+", metavar="DIR",
                       help="Directories containing risk CSV files for model B.")

    parser.add_argument("--csv_name", default="risk_scores.csv",
                        help="Filename to look for when using --input_dirs_* (default: risk_scores.csv).")
    parser.add_argument("--label_csv", default=None,
                        help="Optional external label CSV (biopsy/file_name, time, event) to override labels.")
    parser.add_argument("--n_boot", type=int, default=10_000,
                        help="Number of bootstrap resamples (default: 10 000).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths_a = resolve_paths(args.inputs_a, args.input_dirs_a or [], args.csv_name)
    paths_b = resolve_paths(args.inputs_b, args.input_dirs_b or [], args.csv_name)
    if not paths_a:
        sys.exit("No valid CSV files for model A.")
    if not paths_b:
        sys.exit("No valid CSV files for model B.")

    df_a = pool_csvs(paths_a).rename(columns={"risk": "risk_a"})
    df_b = pool_csvs(paths_b).rename(columns={"risk": "risk_b"})

    # Paired comparison: keep only shared IDs; time/event come from A
    merged = df_a.merge(df_b[["id", "risk_b"]], on="id", how="inner")
    n_a, n_b = len(df_a), len(df_b)
    if len(merged) < n_a or len(merged) < n_b:
        print(
            f"Warning: A has {n_a} rows, B has {n_b} rows; "
            f"{len(merged)} shared IDs used."
        )

    if args.label_csv is not None:
        labels = pd.read_csv(args.label_csv, dtype=str)
        id_col = next((c for c in ("biopsy", "file_name") if c in labels.columns), None)
        if id_col is None:
            sys.exit(f"--label_csv must have 'biopsy' or 'file_name'; found: {list(labels.columns)}")
        labels = labels.rename(columns={id_col: "id"})
        labels["time"] = pd.to_numeric(labels["time"], errors="coerce")
        labels["event"] = pd.to_numeric(labels["event"], errors="coerce")
        labels = labels[["id", "time", "event"]].dropna()
        merged = merged.drop(columns=["time", "event"]).merge(labels, on="id", how="inner")

    merged = merged.dropna(subset=["risk_a", "risk_b", "time", "event"])
    if len(merged) == 0:
        sys.exit("No valid rows after merging.")

    event = merged["event"].astype(bool).to_numpy()
    time = merged["time"].astype(float).to_numpy()
    risk_a = merged["risk_a"].astype(float).to_numpy()
    risk_b = merged["risk_b"].astype(float).to_numpy()

    c_a = _cindex(event, time, risk_a)
    c_b = _cindex(event, time, risk_b)
    obs_diff = c_a - c_b

    print(f"\nN={len(merged)}, events={int(event.sum())}")
    print(f"  C-index A: {c_a:.4f}")
    print(f"  C-index B: {c_b:.4f}")
    print(f"  Observed difference (A − B): {obs_diff:+.4f}")

    # --- Bootstrap test -------------------------------------------------------
    print(f"\nRunning bootstrap (n={args.n_boot})…")
    rng = np.random.default_rng(args.seed)
    boot_diffs = bootstrap_diff(event, time, risk_a, risk_b, args.n_boot, rng)
    valid_boot = boot_diffs[~np.isnan(boot_diffs)]
    if len(valid_boot) < args.n_boot:
        print(f"Warning: {args.n_boot - len(valid_boot)} bootstrap samples skipped (no events).")

    ci_lo, ci_hi = np.percentile(valid_boot, [2.5, 97.5])

    # Shift bootstrap distribution under H0 (diff = 0) before counting extremes
    centered = valid_boot - np.mean(valid_boot)
    p_boot = float(np.mean(np.abs(centered) >= np.abs(obs_diff)))

    print(f"\n[Bootstrap]")
    print(f"  95% CI for (A − B): [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"  Two-tailed p-value: {p_boot:.4f}  {_sig(p_boot)}")

    # --- Variance-covariance (jackknife) test ---------------------------------
    print(f"\nRunning jackknife ({len(event)} LOO iterations)…")
    loo_a = jackknife_loo(event, time, risk_a)
    loo_b = jackknife_loo(event, time, risk_b)
    vc = varcov_test(obs_diff, loo_a, loo_b)

    print(f"\n[Var-Cov / jackknife Z-test]")
    if vc is None:
        print("  Variance estimate non-positive — test not available.")
    else:
        z, p_vc, se, vc_lo, vc_hi = vc
        print(f"  SE(A − B): {se:.4f}   Z = {z:+.3f}")
        print(f"  95% CI for (A − B): [{vc_lo:+.4f}, {vc_hi:+.4f}]")
        print(f"  Two-tailed p-value: {p_vc:.4f}  {_sig(p_vc)}")


if __name__ == "__main__":
    main()
