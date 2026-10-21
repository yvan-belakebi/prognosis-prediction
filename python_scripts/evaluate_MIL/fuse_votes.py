"""
fuse_votes.py — Voting fusion of single-stain survival models.

Combines the per-biopsy risk scores of independently trained single-stain models
(MIL.py, one per stain — see train_per_stain.sh) into one fused risk per biopsy.
No model is trained: this is the decision-level baseline that learned fusion
(late fusion, UNICORN, early fusion) has to beat.

Each model's Cox risk lives on its own arbitrary scale, so scores are first made
comparable, using that model's *training-split* scores as the reference:

    rank    (default) percentile of the score among the model's training scores,
            in [0, 1]; robust to scale and outliers
    zscore  (score - mean) / sd of the model's training scores

A biopsy's fused risk is then the mean over the stains it has a score for —
missing stains simply do not vote. With --weighting cindex each stain votes with
weight max(C_val - 0.5, 0), its validation C-index above chance; the weights are
fixed from the validation split, never from the split being reported.

Inputs: evaluate_survival.py's per-biopsy risk_scores.csv (biopsy, risk, time,
event), one per model and split, found through --pattern, e.g.

    results/eval/{split}/{model}/risk_scores.csv

Outputs, in --output_dir:
    {split}/risk_scores.csv   fused per-biopsy risk, same columns as
                              evaluate_survival.py — feed it to compare_cindex.py
    votes_{split}.csv         the normalised vote of every model per biopsy
    summary.csv               per-model and fused C-index per split, with the
                              number of biopsies and the weights used

Usage (after vote_across_stains.sh has evaluated each model on each split):
    python python_scripts/evaluate_MIL/fuse_votes.py \\
        --models HE_HES PAS PASM Congo Trichrome Sirius_red Immune \\
        --pattern "results/eval/{split}/{model}/risk_scores.csv" \\
        --output_dir results/eval/voting_rank
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
from evaluate_survival import concordance_index  # noqa: E402


def load_risks(path):
    df = pd.read_csv(path, dtype={"biopsy": str})
    missing = {"biopsy", "risk", "time", "event"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing column(s) {sorted(missing)}.")
    return df.set_index("biopsy")[["risk", "time", "event"]]


def normalise(scores, reference, method):
    """Map ``scores`` onto a common scale using the reference (training) scores."""
    ref = np.sort(np.asarray(reference, dtype=float))
    if method == "rank":
        # Mid-rank percentile, so ties with reference scores land in the middle.
        left = np.searchsorted(ref, scores, side="left")
        right = np.searchsorted(ref, scores, side="right")
        return (left + right) / (2.0 * len(ref))
    sd = ref.std()
    return (scores - ref.mean()) / (sd if sd > 0 else 1.0)


def cindex(df, col):
    """C-index of ``df[col]`` against time/event, or NaN without a usable pair."""
    if len(df) < 2 or df["event"].sum() == 0:
        return float("nan")
    return concordance_index(
        df[col].to_numpy(float), df["time"].to_numpy(float), df["event"].to_numpy(int)
    )


def main():
    p = argparse.ArgumentParser(
        description="Voting fusion of single-stain survival models' risk scores.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--models", nargs="+", required=True,
                   help="Model names, substituted for {model} in --pattern.")
    p.add_argument("--pattern", default="results/eval/{split}/{model}/risk_scores.csv",
                   help="Path of a model's risk_scores.csv, with {split} and {model}.")
    p.add_argument("--splits", nargs="+", default=["val", "test"],
                   help="Splits to fuse and report.")
    p.add_argument("--reference_split", default="train",
                   help="Split whose scores define each model's normalisation.")
    p.add_argument("--method", default="rank", choices=["rank", "zscore"])
    p.add_argument("--weighting", default="none", choices=["none", "cindex"],
                   help="cindex: weight each stain by max(C_val - 0.5, 0), from "
                        "--weight_split.")
    p.add_argument("--weight_split", default="val")
    p.add_argument("--output_dir", default="results/eval/voting")
    args = p.parse_args()

    def path(model, split):
        return args.pattern.format(model=model, split=split)

    needed = {args.reference_split, *args.splits}
    if args.weighting == "cindex":
        needed.add(args.weight_split)
    risks = {}
    for m in args.models:
        for s in needed:
            f = path(m, s)
            if not os.path.isfile(f):
                p.error(f"Missing {f} — run evaluate_survival.py --split {s} for {m}.")
            risks[m, s] = load_risks(f)

    # One weight per model, fixed before any reported split is touched.
    weights = {m: 1.0 for m in args.models}
    if args.weighting == "cindex":
        for m in args.models:
            c = cindex(risks[m, args.weight_split], "risk")
            weights[m] = max(c - 0.5, 0.0) if not np.isnan(c) else 0.0
        if not any(weights.values()):
            print("Warning: no model beats chance on validation — equal weights used.")
            weights = {m: 1.0 for m in args.models}
    print("Weights: " + ", ".join(f"{m}={w:.3f}" for m, w in weights.items()))

    summary = []
    for split in args.splits:
        votes, labels = {}, []
        for m in args.models:
            df = risks[m, split]
            ref = risks[m, args.reference_split]["risk"].to_numpy(float)
            votes[m] = pd.Series(
                normalise(df["risk"].to_numpy(float), ref, args.method), index=df.index
            )
            labels.append(df[["time", "event"]])
        votes = pd.DataFrame(votes)  # biopsies x models, NaN where a stain is absent

        # All models see the same labels CSV, so a biopsy's time/event must agree.
        lab = pd.concat(labels)
        clash = lab.groupby(level=0).nunique().gt(1).any(axis=1)
        if clash.any():
            raise ValueError(
                f"{split}: {clash.sum()} biopsies have different time/event across "
                f"models, e.g. {clash[clash].index[0]!r} — were the models evaluated "
                "with the same labels CSV?"
            )
        lab = lab.groupby(level=0).first().loc[votes.index]

        w = pd.Series(weights)[votes.columns]
        present = votes.notna()
        w_present = present.mul(w, axis=1)
        # Biopsies whose present stains all have weight 0 fall back to a plain mean.
        denom = w_present.sum(axis=1)
        fused = votes.fillna(0).mul(w, axis=1).sum(axis=1) / denom.where(denom > 0)
        fused = fused.fillna(votes.mean(axis=1))

        out = lab.assign(risk=fused)[["risk", "time", "event"]]
        out_dir = os.path.join(args.output_dir, split)
        os.makedirs(out_dir, exist_ok=True)
        out.rename_axis("biopsy").reset_index().to_csv(
            os.path.join(out_dir, "risk_scores.csv"), index=False
        )
        votes.join(lab).rename_axis("biopsy").reset_index().to_csv(
            os.path.join(args.output_dir, f"votes_{split}.csv"), index=False
        )

        n_votes = present.sum(axis=1)
        print(
            f"\n[{split}] {len(out)} biopsies, {int(out['event'].sum())} events, "
            f"{n_votes.mean():.1f} stains voting per biopsy (min {n_votes.min()}, "
            f"max {n_votes.max()})"
        )
        for m in args.models:
            # The model's own C-index from its raw risk (the percentile is
            # monotone but can tie), next to the fused one on the same biopsies.
            sub = out.loc[present[m]].assign(own=risks[m, split]["risk"])
            c_model, c_fused = cindex(sub, "own"), cindex(sub, "risk")
            print(
                f"  {m:14s} n={len(sub):5d}  C={c_model:.4f}  "
                f"| fused on the same biopsies C={c_fused:.4f}  weight={weights[m]:.3f}"
            )
            summary.append(dict(split=split, model=m, n=len(sub), c_index=c_model,
                                fused_c_index_same_biopsies=c_fused,
                                weight=weights[m]))
        c_all = cindex(out, "risk")
        print(f"  {'FUSED':14s} n={len(out):5d}  C={c_all:.4f}")
        summary.append(dict(split=split, model="FUSED", n=len(out), c_index=c_all,
                            fused_c_index_same_biopsies=c_all, weight=np.nan))

    os.makedirs(args.output_dir, exist_ok=True)
    pd.DataFrame(summary).to_csv(os.path.join(args.output_dir, "summary.csv"), index=False)
    print(f"\nFused risk scores → {args.output_dir}/<split>/risk_scores.csv")
    print(f"Summary → {os.path.join(args.output_dir, 'summary.csv')}")


if __name__ == "__main__":
    main()
