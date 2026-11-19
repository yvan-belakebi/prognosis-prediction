"""
compare_foundation_models.py — Do two foundation models give the survival models
different information, or the same?

Compares the fusion runs of train_fusion_models.sh trained on features from
different foundation models (FM), on the test biopsies every run scored. Each
run is one (FM, variant, seed), found as

    <eval_root>/<split>/<FM>/<variant>_s<seed>/risk_scores.csv

Two runs of the *same* FM with different seeds already disagree on many pairs
(training noise). So every between-FM number is set against the same number
between seeds of one FM: if the two FMs disagree no more than two seeds of one FM
do, they carry the same information. Train SEEDS="0 1 2" per FM for this baseline.

Run pairs come in four kinds:
    seed              same FM, same variant, another seed   (the baseline)
    fm                other FM, same variant                (the question)
    architecture      same FM, other variant
    fm+architecture   other FM, other variant

  1. Performance      C-index per run; per FM x variant the mean over seeds, with a
                      bootstrap 95% CI, and the paired FM difference.
  2. Error overlap    Over the comparable pairs (earlier event vs later time), each
                      run ranks a pair right or wrong. Per pair of runs: both wrong,
                      only one wrong, Jaccard and overlap coefficient of the wrong
                      pairs, and Cohen's kappa of right/wrong (chance-corrected:
                      two runs with 30% errors share 9% by chance alone). Mean per
                      kind, and kappa(seed) - kappa(fm) with a CI. Also kappa by
                      time between the two outcomes (close outcomes are hard for
                      every model, which inflates overlap).
  3. Complementarity  Spearman correlation of risks. Ensemble gain of two runs
                      (mean percentile rank) over the better of the two: a cross-FM
                      ensemble against a two-seed ensemble of one FM, which gains
                      from variance reduction alone. (Over the members' *mean*, an
                      ensemble with a weaker model always looks like a gain.) Incremental Cox (fitted on
                      --fit_split): likelihood-ratio test of FM B's risk given FM
                      A's, and the reverse, without and with diagnosis + eGFR.
  4. Per biopsy       Share of each biopsy's pairs ranked wrongly, per FM; the
                      biopsies one FM gets wrong and the other right
                      (case_review_*.csv). Kaplan-Meier curves of the groups the
                      two FMs agree / disagree on (above / below median risk).

CIs resample biopsies (the pairs are rebuilt from them), as pairs sharing a
biopsy are not independent: a pair (i, j) counts count_i x count_j times.
Per-FM figures and tables use the risk averaged over seeds (mean percentile
rank) when there are several.

Outputs (in --output_dir):
    runs.csv                   C-index per run
    summary.csv                every quantity above with bootstrap 95% CI
    pairwise_overlap.csv       per pair of runs: kind, kappa, Jaccard, ..., Spearman
    kappa_by_gap.csv           mean kappa per kind and time-gap bin
    ensembles.csv              per two-run ensemble: C-index and gain
    incremental_cox.csv        LR tests of one FM's risk given the other's
    per_biopsy_errors.csv      per biopsy and FM x variant: share of pairs wrong
    case_review_<variant>_<A>_vs_<B>.csv
    km_agreement_<variant>_<A>_vs_<B>.csv
    fig1_cindex_forest.png     fig2_kappa_heatmap.png   fig3_pair_outcomes.png
    fig4_kappa_by_gap.png      fig5_risk_scatter_*.png  fig6_biopsy_errors_*.png
    fig7_ensemble_gain.png     fig9_km_agreement_*.png

Usage:
    python python_scripts/evaluate_MIL/compare_foundation_models.py \\
        --fms uni_v2 virchow2 --variants unicorn early \\
        --output_dir results/compare_fm

    # runs outside the standard layout, named FM/variant_sSEED:
    python python_scripts/evaluate_MIL/compare_foundation_models.py \\
        --model uni_v2/unicorn_s0=results/old/{split}/unicorn/risk_scores.csv ...
"""

import argparse
import glob
import itertools
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.cluster.hierarchy import leaves_list, linkage  # noqa: E402
from scipy.spatial.distance import squareform  # noqa: E402
from scipy.stats import chi2, rankdata  # noqa: E402

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
from analyze_model_signal import (  # noqa: E402
    YEAR, ClinicalDesign, common_table, comparable_pairs, cox_loglik, load_clinical,
)
from fuse_late import fit_cox  # noqa: E402

RUN_RE = re.compile(r"^(?P<fm>[^/]+)/(?P<variant>.+)_s(?P<seed>\d+)$")
KINDS = ["seed", "fm", "architecture", "fm+architecture"]
KIND_LABEL = {"seed": "same FM, other seed", "fm": "other FM",
              "architecture": "same FM, other variant",
              "fm+architecture": "other FM, other variant"}
GAP_BINS = [0, 0.5, 1, 2, 5, np.inf]
GAP_LABELS = ["<0.5y", "0.5-1y", "1-2y", "2-5y", ">5y"]

# ---------------------------------------------------------------------------
# Palette: the validated reference palette (same tokens as
# analyze_stain_weights.py). FMs take the categorical slots in --fms order;
# the within-FM seed baseline is the neutral muted ink, so "other FM" is the
# only colour in the baseline-vs-FM figures.
# ---------------------------------------------------------------------------
THEMES = {
    "light": dict(
        surface="#fcfcfb", ink="#0b0b0b", ink_secondary="#52514e", muted="#898781",
        grid="#e1e0d9", axis="#c3c2b7",
        series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                "#e87ba4", "#008300", "#6250d6", "#e34948"],
        ordinal=["#86b6ef", "#2a78d6", "#104281"],   # blue 250 / 450 / 650
        seq=["#cde2fb", "#2a78d6", "#0d366b"],
    ),
    "dark": dict(
        surface="#1a1a19", ink="#ffffff", ink_secondary="#c3c2b7", muted="#898781",
        grid="#2c2c2a", axis="#383835",
        series=["#3987e5", "#d95926", "#199e70", "#c98500",
                "#d55181", "#0ca30c", "#9085e9", "#e66767"],
        ordinal=["#184f95", "#3987e5", "#9ec5f4"],
        seq=["#104281", "#3987e5", "#cde2fb"],
    ),
}


def style_axes(ax, th, grid_axis="both"):
    ax.set_facecolor(th["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(th["axis"])
    ax.tick_params(colors=th["muted"], labelsize=9, length=0)
    if grid_axis != "none":
        ax.grid(True, axis=grid_axis, color=th["grid"], linewidth=0.8)
        ax.set_axisbelow(True)


def new_figure(th, **kw):
    fig, ax = plt.subplots(**kw)
    fig.patch.set_facecolor(th["surface"])
    return fig, ax


def title(ax, th, text, sub=None):
    ax.set_title(text, color=th["ink"], fontsize=12, loc="left", pad=22 if sub else 10)
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, color=th["ink_secondary"],
                fontsize=9, va="bottom")


def save(fig, th, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=th["surface"], bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def find_runs(args, parser):
    """{FM/variant_sSEED: path pattern with {split}} from the layout and --model."""
    runs = {}
    for path in sorted(glob.glob(os.path.join(
            args.eval_root, args.eval_split, "*", "*_s*", "risk_scores.csv"))):
        run_dir = os.path.dirname(path)
        name = f"{os.path.basename(os.path.dirname(run_dir))}/{os.path.basename(run_dir)}"
        runs[name] = os.path.join(args.eval_root, "{split}", *name.split("/"),
                                  "risk_scores.csv")
    for spec in args.model or []:
        name, _, pattern = spec.partition("=")
        if not RUN_RE.match(name) or "{split}" not in pattern:
            parser.error(f"--model {spec!r}: expected FM/variant_sSEED=path/with/{{split}}/...")
        runs[name] = pattern
    keep = {}
    for name, pattern in runs.items():
        m = RUN_RE.match(name)
        if (m and (not args.fms or m["fm"] in args.fms)
                and (not args.variants or m["variant"] in args.variants)):
            keep[name] = pattern
    return keep


def run_meta(names):
    """FM, variant and seed of each FM/variant_sSEED run name."""
    meta = pd.DataFrame([RUN_RE.match(n).groupdict() for n in names], index=names)
    meta["seed"] = meta["seed"].astype(int)
    return meta


def pair_runs(meta):
    """Every pair of runs (i, j, kind, variant group); see KINDS."""
    pairs = []
    for i, j in itertools.combinations(range(len(meta)), 2):
        a, b = meta.iloc[i], meta.iloc[j]
        same_fm, same_var = a.fm == b.fm, a.variant == b.variant
        if same_var:
            kind = "seed" if same_fm else "fm"
        else:
            kind = "architecture" if same_fm else "fm+architecture"
        pairs.append((i, j, kind, a.variant if same_var else "across variants"))
    return pairs


def pair_scores(r, pi, pj):
    """Per comparable pair (pi had the earlier event): 1 right, 0.5 tie, 0 wrong."""
    d = r[pi] - r[pj]
    return np.where(d > 0, 1.0, np.where(d == 0, 0.5, 0.0)).astype(np.float32)


def agreement(p, both):
    """Agreement of right/wrong between runs, from error rates p (K,) and the
    both-wrong shares (K, K)."""
    pa, pb = p[:, None], p[None, :]
    po = 1 - pa - pb + 2 * both
    pe = pa * pb + (1 - pa) * (1 - pb)
    with np.errstate(invalid="ignore", divide="ignore"):
        return dict(
            kappa=(po - pe) / (1 - pe),
            jaccard=both / (pa + pb - both),
            overlap_coef=both / np.minimum(pa, pb),
            both_wrong=both,
            one_wrong=pa + pb - 2 * both,
            both_right=1 - pa - pb + both,
        )


def weighted_stats(w, S, Wr):
    """C-index per column of S, error rate per run and both-wrong shares."""
    tot = w.sum()
    c = (w @ S) / tot
    p = (w @ Wr) / tot
    both = ((Wr * w[:, None]).T @ Wr) / tot
    return c, agreement(p, both)


def km(time, event):
    """Kaplan-Meier step curve: times (with 0) and survival just after each."""
    order = np.argsort(time)
    t, ev = time[order], event[order]
    u = np.unique(t[ev == 1])
    at_risk = len(t) - np.searchsorted(t, u, side="left")
    d = np.array([(ev[t == x] == 1).sum() for x in u])
    return np.r_[0, u], np.r_[1, np.cumprod(1 - d / at_risk)]


def surv_at(times, surv, x):
    return surv[np.searchsorted(times, x, side="right") - 1]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Compare survival runs on features from different foundation "
                    "models: performance, overlap of misranked pairs, complementarity.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--eval_root", default="results/eval",
                   help="Holds <split>/<FM>/<variant>_s<seed>/risk_scores.csv.")
    p.add_argument("--fms", nargs="+", default=None,
                   help="Foundation models to compare, in this order (default: all found).")
    p.add_argument("--variants", nargs="+", default=None,
                   help="Fusion variants to keep (default: all found).")
    p.add_argument("--model", action="append", metavar="FM/variant_sSEED=PATTERN",
                   help="Extra run outside the layout; PATTERN contains {split}.")
    p.add_argument("--eval_split", default="test")
    p.add_argument("--fit_split", default="val",
                   help="Split the incremental Cox models are fitted on.")
    p.add_argument("--registry_csv",
                   default="followup_data/derived/renamed/registry_anonymized.csv")
    p.add_argument("--biopsy_col", default="biop_id")
    p.add_argument("--diagnosis_col", default="Diagnosis")
    p.add_argument("--egfr_col", default="eGFR")
    p.add_argument("--event_type_col", default="Event")
    p.add_argument("--max_egfr", type=float, default=200.0)
    p.add_argument("--min_group", type=int, default=30,
                   help="Smallest diagnosis group given its own Cox category.")
    p.add_argument("--ridge", type=float, default=1e-2, help="Ridge of the Cox models.")
    p.add_argument("--n_boot", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min_pairs", type=int, default=10,
                   help="Biopsies with fewer comparable pairs are left out of the "
                        "per-biopsy error figures and case review.")
    p.add_argument("--n_cases", type=int, default=20,
                   help="Rows per side in case_review_*.csv.")
    p.add_argument("--theme", default="light", choices=["light", "dark"])
    p.add_argument("--output_dir", default="results/compare_fm")
    args = p.parse_args()
    th = THEMES[args.theme]
    os.makedirs(args.output_dir, exist_ok=True)
    out = lambda f: os.path.join(args.output_dir, f)  # noqa: E731

    # --- Runs ------------------------------------------------------------------------
    models = find_runs(args, p)
    if not models:
        p.error(f"No runs under {args.eval_root}/{args.eval_split}/<FM>/<variant>_s<seed>/.")
    names = sorted(models, key=lambda n: (RUN_RE.match(n)["variant"],
                                          RUN_RE.match(n)["fm"], int(RUN_RE.match(n)["seed"])))
    models = {n: models[n] for n in names}
    meta = run_meta(names)
    fms = [f for f in (args.fms or sorted(meta.fm.unique())) if f in set(meta.fm)]
    variants = [v for v in (args.variants or sorted(meta.variant.unique()))
                if v in set(meta.variant)]
    fm_color = {f: th["series"][i % len(th["series"])] for i, f in enumerate(fms)}
    K = len(names)
    print(f"{K} runs:")
    for (v, f), g in meta.groupby(["variant", "fm"]):
        print(f"  {f}/{v}: seeds {sorted(g.seed)}")
    if meta.fm.nunique() < 2:
        print("  Warning: one FM only - nothing to compare between FMs.")
    if not (meta.groupby(["fm", "variant"]).size() > 1).any():
        print("  Warning: one seed per FM x variant - no within-FM baseline; train "
              "SEEDS=\"0 1 2\" ./train_fusion_models.sh for it.")

    clinical = load_clinical(args)
    risk_f, lab_f, clin_f = common_table(models, args.fit_split, clinical, p.error)
    risk, lab, clin = common_table(models, args.eval_split, clinical, p.error)
    t, e = lab["time"].to_numpy(float), lab["event"].to_numpy(int)
    n = len(t)
    pi, pj = np.nonzero(comparable_pairs(t, e))
    gap = (t[pj] - t[pi]) / YEAR
    print(f"[{args.eval_split}] {n} biopsies scored by every run, {e.sum()} events, "
          f"{len(pi)} comparable pairs | [{args.fit_split}] {len(risk_f)} for fitting")

    pct = risk.apply(rankdata) / n  # percentile rank on the eval split, per run

    # Seed-averaged percentile rank per FM x variant.
    fm_var = {(f, v): list(g.index) for (f, v), g in meta.groupby(["fm", "variant"])}
    avg_pct = {k: pct[idx].mean(axis=1).to_numpy() for k, idx in fm_var.items()}

    # --- Pairs of runs and two-run ensembles -------------------------------------------
    run_pairs = pair_runs(meta)
    ens = [(i, j, kind, var) for i, j, kind, var in run_pairs if kind in ("seed", "fm")]
    ens_names = [f"{names[i]} + {names[j]}" for i, j, _, _ in ens]

    S = np.column_stack(
        [pair_scores(risk[m].to_numpy(), pi, pj) for m in names]
        + [pair_scores((pct.iloc[:, i] + pct.iloc[:, j]).to_numpy(), pi, pj)
           for i, j, _, _ in ens]
    )
    Wr = (S[:, :K] == 0).astype(np.float32)  # wrong = strictly misranked

    groups_kind = {}
    for i, j, kind, var in run_pairs:
        groups_kind.setdefault((kind, var), []).append((i, j))
    groups_ens = {}
    for col, (i, j, kind, var) in enumerate(ens, start=K):
        groups_ens.setdefault((kind, var), []).append((col, i, j))
    run_idx = {k: [names.index(m) for m in idx] for k, idx in fm_var.items()}

    def quantities(c, ag):
        q = {}
        for (f, v), idx in run_idx.items():
            q[("c_index_seed_mean", f, v)] = c[idx].mean()
        for v in variants:
            for fa, fb in itertools.combinations(fms, 2):
                if (fa, v) in run_idx and (fb, v) in run_idx:
                    q[("c_index_diff", f"{fa} - {fb}", v)] = (
                        c[run_idx[(fa, v)]].mean() - c[run_idx[(fb, v)]].mean())
        for (kind, v), prs in groups_kind.items():
            ii, jj = np.array(prs).T
            for stat in ("kappa", "jaccard", "overlap_coef", "both_wrong", "one_wrong",
                         "both_right"):
                q[(stat, kind, v)] = np.nanmean(ag[stat][ii, jj])
        for (kind, v), cols in groups_ens.items():
            cc, ii, jj = np.array(cols).T
            q[("ensemble_gain", kind, v)] = np.mean(c[cc] - np.maximum(c[ii], c[jj]))
        for v in variants + ["across variants"]:
            if ("kappa", "seed", v) in q and ("kappa", "fm", v) in q:
                q[("kappa_diff", "seed - fm", v)] = q[("kappa", "seed", v)] - q[("kappa", "fm", v)]
            if ("ensemble_gain", "seed", v) in q and ("ensemble_gain", "fm", v) in q:
                q[("ensemble_gain_diff", "fm - seed", v)] = (
                    q[("ensemble_gain", "fm", v)] - q[("ensemble_gain", "seed", v)])
        return q

    ones = np.ones(len(pi), np.float32)
    c_all, ag_all = weighted_stats(ones, S, Wr)
    point = quantities(c_all, ag_all)

    print(f"Bootstrap over biopsies ({args.n_boot} resamples)...")
    rng = np.random.default_rng(args.seed)
    boots = []
    for _ in range(args.n_boot):
        cnt = np.bincount(rng.integers(0, n, n), minlength=n).astype(np.float32)
        boots.append(quantities(*weighted_stats(cnt[pi] * cnt[pj], S, Wr)))
    boots = pd.DataFrame(boots)
    summary = pd.DataFrame(
        [dict(quantity=k[0], group=k[1], variant=k[2], estimate=v,
              ci_low=np.nanpercentile(boots[k], 2.5), ci_high=np.nanpercentile(boots[k], 97.5))
         for k, v in point.items()]
    )
    summary.to_csv(out("summary.csv"), index=False)
    sumd = summary.set_index(["quantity", "group", "variant"])

    # --- 1. Performance ------------------------------------------------------------------
    runs_df = meta.assign(c_index=c_all[:K], error_rate=Wr.mean(0))
    runs_df.rename_axis("run").to_csv(out("runs.csv"))
    print("\n1. C-index (mean over seeds, bootstrap 95% CI)")
    for v in variants:
        for f in fms:
            if (f, v) not in run_idx:
                continue
            r = sumd.loc[("c_index_seed_mean", f, v)]
            seeds = ", ".join(f"{c:.3f}" for c in c_all[run_idx[(f, v)]])
            print(f"  {f + '/' + v:28s} C={r.estimate:.4f} [{r.ci_low:.4f}, {r.ci_high:.4f}]"
                  f"  seeds: {seeds}")
    if "c_index_diff" in sumd.index.get_level_values(0):
        for (g, v), r in sumd.loc["c_index_diff"].iterrows():
            print(f"  {v}: {g} = {r.estimate:+.4f} [{r.ci_low:+.4f}, {r.ci_high:+.4f}]")

    # --- 2. Error overlap ----------------------------------------------------------------
    spearman = pct.corr(method="pearson")  # of ranks = Spearman of risks
    spearman.to_csv(out("spearman.csv"))
    rows = []
    for i, j, kind, var in run_pairs:
        rows.append(dict(run_a=names[i], run_b=names[j], kind=kind, variant=var,
                         **{s: ag_all[s][i, j] for s in ("kappa", "jaccard", "overlap_coef",
                                                          "both_wrong", "one_wrong",
                                                          "both_right")},
                         at_least_one_right=1 - ag_all["both_wrong"][i, j],
                         spearman=spearman.iloc[i, j]))
    pair_df = pd.DataFrame(rows)
    pair_df.to_csv(out("pairwise_overlap.csv"), index=False)

    print("\n2. Overlap of misranked pairs, mean per kind of run pair (95% CI)")
    for v in variants + ["across variants"]:
        for kind in KINDS:
            if ("kappa", kind, v) not in point:
                continue
            k = sumd.loc[("kappa", kind, v)]
            jac = point[("jaccard", kind, v)]
            bw = point[("both_wrong", kind, v)]
            print(f"  {v:16s} {KIND_LABEL[kind]:26s} kappa={k.estimate:.3f} "
                  f"[{k.ci_low:.3f}, {k.ci_high:.3f}]  Jaccard={jac:.3f}  both wrong={bw:.1%}")
        if ("kappa_diff", "seed - fm", v) in point:
            d = sumd.loc[("kappa_diff", "seed - fm", v)]
            print(f"  {v:16s} kappa(seed) - kappa(fm) = {d.estimate:+.3f} "
                  f"[{d.ci_low:+.3f}, {d.ci_high:+.3f}]  (> 0: the FMs disagree more "
                  f"than seeds do)")

    rows = []
    for lo, hi, lbl in zip(GAP_BINS[:-1], GAP_BINS[1:], GAP_LABELS):
        sel = (gap >= lo) & (gap < hi)
        if not sel.any():
            continue
        _, ag = weighted_stats(sel.astype(np.float32), S[:, :0], Wr)
        for (kind, v), prs in groups_kind.items():
            ii, jj = np.array(prs).T
            rows.append(dict(time_gap=lbl, pairs=int(sel.sum()), kind=kind, variant=v,
                             kappa=np.nanmean(ag["kappa"][ii, jj]),
                             both_wrong=np.nanmean(ag["both_wrong"][ii, jj])))
    by_gap = pd.DataFrame(rows)
    by_gap.to_csv(out("kappa_by_gap.csv"), index=False)

    # --- 3. Complementarity --------------------------------------------------------------
    ens_df = pd.DataFrame([dict(ensemble=en, kind=kind, variant=var, c_index=c_all[col],
                                c_a=c_all[i], c_b=c_all[j],
                                gain_over_mean=c_all[col] - (c_all[i] + c_all[j]) / 2,
                                gain_over_best=c_all[col] - max(c_all[i], c_all[j]))
                           for col, (en, (i, j, kind, var)) in enumerate(zip(ens_names, ens),
                                                                          start=K)])
    ens_df.to_csv(out("ensembles.csv"), index=False)
    print("\n3. Two-run ensembles (mean percentile rank): gain over the better member")
    for v in variants:
        for kind in ("seed", "fm"):
            if ("ensemble_gain", kind, v) in point:
                g = sumd.loc[("ensemble_gain", kind, v)]
                print(f"  {v:16s} {KIND_LABEL[kind]:26s} gain={g.estimate:+.4f} "
                      f"[{g.ci_low:+.4f}, {g.ci_high:+.4f}]")
        if ("ensemble_gain_diff", "fm - seed", v) in point:
            g = sumd.loc[("ensemble_gain_diff", "fm - seed", v)]
            print(f"  {v:16s} gain(fm) - gain(seed) = {g.estimate:+.4f} "
                  f"[{g.ci_low:+.4f}, {g.ci_high:+.4f}]  (> 0: the FMs complement each other)")

    # Incremental Cox: seed-averaged z-scored risk per FM, z fixed on the fit split.
    design = ClinicalDesign(clin_f, args.min_group)
    Xc_f, Xc_e = design(clin_f).to_numpy(), design(clin).to_numpy()
    tf, ef = lab_f["time"].to_numpy(float), lab_f["event"].to_numpy(int)

    def z_avg(f, v):
        cols = fm_var[(f, v)]
        mu, sd = risk_f[cols].mean(), risk_f[cols].std().replace(0, 1.0)
        return (((risk_f[cols] - mu) / sd).mean(axis=1).to_numpy()[:, None],
                ((risk[cols] - mu) / sd).mean(axis=1).to_numpy()[:, None])

    def fit(Xf, Xe):
        beta = fit_cox(Xf, tf, ef, args.ridge)
        return cox_loglik(Xf, tf, ef, beta), float(pair_scores(Xe @ beta, pi, pj).mean())

    rows = []
    print(f"\n   Incremental Cox (fitted on {args.fit_split}, C-index on {args.eval_split})")
    for v in variants:
        for fa, fb in itertools.permutations(fms, 2):
            if (fa, v) not in fm_var or (fb, v) not in fm_var:
                continue
            (af, ae), (bf, be) = z_avg(fa, v), z_avg(fb, v)
            for adj in ("none", "diagnosis+eGFR"):
                base_f = Xc_f if adj != "none" else np.empty((len(tf), 0))
                base_e = Xc_e if adj != "none" else np.empty((n, 0))
                ll_a, c_a = fit(np.hstack([base_f, af]), np.hstack([base_e, ae]))
                ll_ab, c_ab = fit(np.hstack([base_f, af, bf]), np.hstack([base_e, ae, be]))
                lr = 2 * (ll_ab - ll_a)
                rows.append(dict(variant=v, adjusted_for=adj, given=fa, added=fb,
                                 c_given=c_a, c_both=c_ab, c_gain=c_ab - c_a, lr_stat=lr,
                                 lr_p=float(chi2.sf(max(lr, 0), 1))))
                print(f"  {v:10s} [{adj:14s}] {fb} given {fa}: C {c_a:.4f} -> {c_ab:.4f}"
                      f"  LR={lr:.1f} p={rows[-1]['lr_p']:.2g}")
    pd.DataFrame(rows).to_csv(out("incremental_cox.csv"), index=False)

    # --- 4. Per biopsy -------------------------------------------------------------------
    involved = np.bincount(pi, minlength=n) + np.bincount(pj, minlength=n)
    err = {}
    for k, idx in run_idx.items():
        wrong = sum(np.bincount(pi, Wr[:, c], n) + np.bincount(pj, Wr[:, c], n) for c in idx)
        err[k] = wrong / len(idx) / np.maximum(involved, 1)
    per_b = pd.DataFrame({
        "biopsy": risk.index, "time_years": t / YEAR, "event": e,
        "event_type": clin["event_type"].to_numpy(), "diagnosis": clin["diagnosis"].to_numpy(),
        "egfr": clin["egfr"].to_numpy(), "comparable_pairs": involved,
        **{f"error_{f}/{v}": err[(f, v)] for f, v in err},
        **{f"risk_pct_{f}/{v}": avg_pct[(f, v)] for f, v in avg_pct},
    })
    per_b.to_csv(out("per_biopsy_errors.csv"), index=False)
    fm_pairs = [(v, fa, fb) for v in variants for fa, fb in itertools.combinations(fms, 2)
                if (fa, v) in err and (fb, v) in err]
    enough = involved >= args.min_pairs
    for v, fa, fb in fm_pairs:
        d = per_b[enough].assign(diff=err[(fa, v)][enough] - err[(fb, v)][enough])
        cases = pd.concat([d.nlargest(args.n_cases, "diff").assign(worse=fa),
                           d.nsmallest(args.n_cases, "diff").assign(worse=fb)])
        cases.to_csv(out(f"case_review_{v}_{fa}_vs_{fb}.csv"), index=False)
    print(f"\n4. Per-biopsy errors -> per_biopsy_errors.csv; case_review_*.csv lists the "
          f"{args.n_cases} biopsies (>= {args.min_pairs} pairs) each FM gets most wrong "
          f"relative to the other")

    # KM by agreement (seed-averaged percentile rank above / below the median).
    km_groups = {}
    for v, fa, fb in fm_pairs:
        ha, hb = avg_pct[(fa, v)] > 0.5, avg_pct[(fb, v)] > 0.5
        groups = {"both low": ~ha & ~hb, f"{fa} high, {fb} low": ha & ~hb,
                  f"{fa} low, {fb} high": ~ha & hb, "both high": ha & hb}
        rows = []
        for g, sel in groups.items():
            if sel.sum() == 0:
                continue
            tt, ss = km(t[sel], e[sel])
            rows.append(dict(group=g, n=int(sel.sum()), events=int(e[sel].sum()),
                             surv_5y=surv_at(tt, ss, 5 * YEAR),
                             surv_10y=surv_at(tt, ss, 10 * YEAR)))
        pd.DataFrame(rows).to_csv(out(f"km_agreement_{v}_{fa}_vs_{fb}.csv"), index=False)
        km_groups[(v, fa, fb)] = groups
        print(f"  {v}: {fa} vs {fb} agree on {(ha == hb).mean():.0%} of biopsies "
              f"(above / below median risk)")

    # --- Figures -------------------------------------------------------------------------
    plot_forest(sumd, runs_df, run_idx, fms, variants, fm_color, th, out("fig1_cindex_forest.png"))
    plot_kappa_heatmap(ag_all["kappa"], names, meta, fms, fm_color, th,
                       out("fig2_kappa_heatmap.png"))
    plot_pair_outcomes(point, variants, th, out("fig3_pair_outcomes.png"))
    plot_kappa_by_gap(by_gap, variants, th, out("fig4_kappa_by_gap.png"))
    for v, fa, fb in fm_pairs:
        tag = f"{v}_{fa}_vs_{fb}"
        plot_scatter(avg_pct[(fa, v)], avg_pct[(fb, v)], e, None, fa, fb, v, th,
                     out(f"fig5_risk_scatter_{tag}.png"))
        plot_scatter(err[(fa, v)][enough], err[(fb, v)][enough], e[enough],
                     risk.index[enough], fa, fb, v, th, out(f"fig6_biopsy_errors_{tag}.png"))
        plot_km(km_groups[(v, fa, fb)],
                [th["muted"], fm_color[fa], fm_color[fb], th["ink_secondary"]],
                t, e, v, th, out(f"fig9_km_agreement_{tag}.png"))
    plot_ensemble_gain(sumd, variants, th, out("fig7_ensemble_gain.png"))
    print(f"\nResults -> {args.output_dir}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_forest(sumd, runs_df, run_idx, fms, variants, fm_color, th, path):
    rows = [(f, v) for v in variants for f in fms if (f, v) in run_idx]
    fig, ax = new_figure(th, figsize=(8, 0.5 * len(rows) + 1.8))
    for y, (f, v) in enumerate(rows):
        r = sumd.loc[("c_index_seed_mean", f, v)]
        ax.plot([r.ci_low, r.ci_high], [y, y], color=fm_color[f], linewidth=2,
                solid_capstyle="round")
        ax.scatter(runs_df[(runs_df.fm == f) & (runs_df.variant == v)].c_index,
                   [y] * len(run_idx[(f, v)]), s=28, facecolor=th["surface"],
                   edgecolor=fm_color[f], linewidth=1.5, zorder=3)
        ax.scatter([r.estimate], [y], s=70, color=fm_color[f], zorder=4,
                   edgecolor=th["surface"], linewidth=2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{v} · {f}" for f, v in rows], color=th["ink_secondary"])
    ax.invert_yaxis()
    ax.set_xlabel("C-index", color=th["ink_secondary"])
    style_axes(ax, th, "x")
    handles = [Line2D([], [], marker="o", linestyle="", color=fm_color[f], label=f)
               for f in fms]
    handles += [Line2D([], [], marker="o", linestyle="", markerfacecolor=th["surface"],
                       markeredgecolor=th["muted"], label="single seed")]
    ax.legend(handles=handles, frameon=False, fontsize=9, labelcolor=th["ink_secondary"],
              loc="lower right")
    title(ax, th, "C-index per foundation model",
          "filled: mean over seeds, line: bootstrap 95% CI; hollow: each seed")
    save(fig, th, path)


def plot_kappa_heatmap(kappa, names, meta, fms, fm_color, th, path):
    K = len(names)
    if K < 2:
        return
    kap = np.nan_to_num(kappa.copy(), nan=0.0)
    np.fill_diagonal(kap, 1.0)
    order = (leaves_list(linkage(squareform(np.clip(1 - kap, 0, None), checks=False),
                                 "average")) if K > 2 else np.arange(K))
    m = kappa[np.ix_(order, order)].copy()
    np.fill_diagonal(m, np.nan)
    size = 0.45 * K + 3
    fig = plt.figure(figsize=(size + 1.2, size))
    fig.patch.set_facecolor(th["surface"])
    gs = fig.add_gridspec(1, 3, width_ratios=[0.25, K, 0.4], wspace=0.05)
    ax_s, ax, ax_c = (fig.add_subplot(gs[0, i]) for i in range(3))
    cmap = LinearSegmentedColormap.from_list("kappa", th["seq"])
    cmap.set_bad(th["surface"])
    im = ax.imshow(m, cmap=cmap, vmin=np.nanmin(m), vmax=np.nanmax(m), aspect="auto")
    ax.set_xticks(range(K))
    ax.set_xticklabels([names[i] for i in order], rotation=60, ha="right", fontsize=8,
                       color=th["ink_secondary"])
    ax.set_yticks([])
    if K <= 16:
        mid = (np.nanmin(m) + np.nanmax(m)) / 2
        for a in range(K):
            for b in range(K):
                if a != b:
                    ax.text(b, a, f"{m[a, b]:.2f}", ha="center", va="center", fontsize=7,
                            color="#ffffff" if m[a, b] > mid else th["ink"])
    style_axes(ax, th, "none")
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    # FM strip: the colour carries the FM, the label names the run.
    ax_s.imshow(np.arange(K)[:, None], aspect="auto",
                cmap=ListedColormap([fm_color[meta.fm.iloc[i]] for i in order]))
    ax_s.set_xticks([])
    ax_s.set_yticks(range(K))
    ax_s.set_yticklabels([names[i] for i in order], fontsize=8, color=th["ink_secondary"])
    for s in ax_s.spines.values():
        s.set_visible(False)
    ax_s.tick_params(length=0)
    cb = fig.colorbar(im, cax=ax_c)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=th["muted"], labelsize=8, length=0)
    cb.set_label("Cohen's kappa of right / wrong pairs", color=th["ink_secondary"], fontsize=9)
    ax_s.legend(handles=[Line2D([], [], marker="s", linestyle="", color=fm_color[f], label=f)
                         for f in fms],
                frameon=False, fontsize=8, labelcolor=th["ink_secondary"],
                loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(fms))
    ax.set_title("Error agreement between runs (clustered)", color=th["ink"], fontsize=12,
                 loc="left", pad=24)
    fig.savefig(path, dpi=150, facecolor=th["surface"], bbox_inches="tight")
    plt.close(fig)


def plot_pair_outcomes(point, variants, th, path):
    rows = [(v, kind) for v in variants + ["across variants"] for kind in KINDS
            if ("both_wrong", kind, v) in point]
    if not rows:
        return
    fig, ax = new_figure(th, figsize=(8.5, 0.5 * len(rows) + 2))
    parts = [("both_right", "both right"), ("one_wrong", "only one wrong"),
             ("both_wrong", "both wrong")]
    for y, (v, kind) in enumerate(rows):
        left = 0
        for (stat, lbl), col in zip(parts, th["ordinal"]):
            w = point[(stat, kind, v)]
            ax.barh(y, w, left=left, color=col, edgecolor=th["surface"], linewidth=2,
                    height=0.7, label=lbl if y == 0 else None)
            left += w
        ax.text(1.01, y, f"{point[('both_wrong', kind, v)]:.1%} both wrong",
                va="center", fontsize=8, color=th["ink_secondary"])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{v} · {KIND_LABEL[k]}" for v, k in rows], color=th["ink_secondary"])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of comparable pairs (mean over run pairs)", color=th["ink_secondary"])
    style_axes(ax, th, "x")
    ax.legend(frameon=False, fontsize=9, labelcolor=th["ink_secondary"], ncol=3,
              loc="upper left", bbox_to_anchor=(0, -0.15))
    title(ax, th, "Pair outcomes for two runs",
          "compare 'other FM' with the 'same FM, other seed' baseline")
    save(fig, th, path)


def plot_kappa_by_gap(by_gap, variants, th, path):
    vs = [v for v in variants if not by_gap.empty and (by_gap.variant == v).any()]
    if not vs:
        return
    fig, axes = plt.subplots(1, len(vs), figsize=(4.6 * len(vs), 3.8), sharey=True,
                             squeeze=False)
    fig.patch.set_facecolor(th["surface"])
    style = {"seed": dict(color=th["muted"], linestyle="--"),
             "fm": dict(color=th["series"][0], linestyle="-")}
    for ax, v in zip(axes[0], vs):
        for kind, st in style.items():
            d = by_gap[(by_gap.variant == v) & (by_gap.kind == kind)]
            if d.empty:
                continue
            ax.plot(d.time_gap, d.kappa, marker="o", markersize=6, linewidth=2,
                    label=KIND_LABEL[kind], **st)
            ax.annotate(KIND_LABEL[kind], (len(d) - 1, d.kappa.iloc[-1]),
                        xytext=(4, 0), textcoords="offset points", fontsize=8,
                        color=th["ink_secondary"], va="center")
        style_axes(ax, th, "y")
        ax.set_title(v, color=th["ink"], fontsize=11, loc="left")
        ax.set_xlabel("time between the two outcomes", color=th["ink_secondary"])
    axes[0, 0].set_ylabel("kappa of right / wrong", color=th["ink_secondary"])
    axes[0, 0].legend(frameon=False, fontsize=8, labelcolor=th["ink_secondary"])
    fig.suptitle("Error agreement by time gap", color=th["ink"], fontsize=12, x=0.01,
                 ha="left")
    save(fig, th, path)


def plot_scatter(xa, xb, event, ids, fa, fb, v, th, path):
    """Fig 5 (ids None): seed-averaged risk percentiles. Fig 6: per-biopsy error share."""
    fig, ax = new_figure(th, figsize=(5.6, 5.4))
    ev = event == 1
    ax.scatter(xa[~ev], xb[~ev], s=10, color=th["muted"], alpha=0.5, linewidths=0,
               label="censored")
    ax.scatter(xa[ev], xb[ev], s=12, color=th["series"][0], alpha=0.7, linewidths=0,
               label="event")
    lo, hi = min(xa.min(), xb.min()), max(xa.max(), xb.max())
    ax.plot([lo, hi], [lo, hi], color=th["axis"], linewidth=1, zorder=0)
    if ids is None:
        rho = np.corrcoef(rankdata(xa), rankdata(xb))[0, 1]
        what, sub = "risk percentile", f"Spearman {rho:.2f}"
    else:
        what, sub = "share of the biopsy's pairs ranked wrongly", \
            "above the diagonal: worse for the y-axis FM"
        d = xb - xa
        for k in list(np.argsort(d)[:3]) + list(np.argsort(d)[-3:]):
            ax.annotate(str(ids[k]), (xa[k], xb[k]), xytext=(3, 3),
                        textcoords="offset points", fontsize=7, color=th["ink_secondary"])
    ax.set_xlabel(f"{fa}: {what}", color=th["ink_secondary"], fontsize=9)
    ax.set_ylabel(f"{fb}: {what}", color=th["ink_secondary"], fontsize=9)
    style_axes(ax, th)
    ax.legend(frameon=False, fontsize=8, labelcolor=th["ink_secondary"], ncol=2,
              loc="upper center", bbox_to_anchor=(0.5, -0.12))
    title(ax, th, f"{v}: {fb} vs {fa}", sub)
    save(fig, th, path)


def plot_km(groups, colors, t, e, v, th, path):
    """Agreement groups in grays; a disagreement group in the colour of the FM
    that calls the biopsy high risk."""
    fig, ax = new_figure(th, figsize=(7, 4.6))
    for (g, sel), col in zip(groups.items(), colors):
        if sel.sum() == 0:
            continue
        tt, ss = km(t[sel], e[sel])
        ax.step(tt / YEAR, ss, where="post", color=col, linewidth=2,
                label=f"{g} (n={sel.sum()})")
        ax.annotate(g, (tt[-1] / YEAR, ss[-1]), xytext=(4, 0), textcoords="offset points",
                    fontsize=8, color=th["ink_secondary"], va="center")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("years since biopsy", color=th["ink_secondary"])
    ax.set_ylabel("event-free survival", color=th["ink_secondary"])
    style_axes(ax, th)
    ax.legend(frameon=False, fontsize=8, labelcolor=th["ink_secondary"], loc="lower left")
    title(ax, th, f"{v}: outcome by FM agreement",
          "above / below median risk (mean over seeds) for each FM")
    save(fig, th, path)


def plot_ensemble_gain(sumd, variants, th, path):
    if "ensemble_gain" not in sumd.index.get_level_values(0):
        return
    eg = sumd.loc["ensemble_gain"]
    fig, ax = new_figure(th, figsize=(7, 0.9 * len(variants) + 2))
    style = {"seed": th["muted"], "fm": th["series"][0]}
    y, ticks, labelled = 0, [], set()
    for v in variants:
        for kind in ("seed", "fm"):
            if (kind, v) not in eg.index:
                continue
            r = eg.loc[(kind, v)]
            ax.barh(y, r.estimate, color=style[kind], height=0.7,
                    label=None if kind in labelled else KIND_LABEL[kind])
            labelled.add(kind)
            ax.plot([r.ci_low, r.ci_high], [y, y], color=th["ink"], linewidth=1.2)
            ticks.append((y, f"{v} · {KIND_LABEL[kind]}"))
            y += 1
        y += 0.5
    ax.axvline(0, color=th["axis"], linewidth=1)
    ax.set_yticks([a for a, _ in ticks])
    ax.set_yticklabels([b for _, b in ticks], color=th["ink_secondary"])
    ax.invert_yaxis()
    ax.set_xlabel("C-index of the two-run ensemble minus its better member",
                  color=th["ink_secondary"])
    style_axes(ax, th, "x")
    title(ax, th, "Ensemble gain: other FM vs other seed",
          "a cross-FM gain above the seed gain means the FMs complement each other")
    save(fig, th, path)


if __name__ == "__main__":
    main()
