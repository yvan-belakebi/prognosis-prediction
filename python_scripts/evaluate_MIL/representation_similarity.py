"""
representation_similarity.py — Do two foundation models give the fusion models the
same picture of a biopsy?

compare_foundation_models.py compares what the runs *predict*; this script
compares what they *represent*, at three levels:

  1. Model embeddings  (embeddings.npz, from analyze_stain_weights.py
                       --save_embeddings, which train_fusion_models.sh writes)
       - linear CKA between the pooled bag vectors z of every pair of runs, mean
         per kind of run pair with a bootstrap CI over biopsies, and
         CKA(seed) - CKA(fm): > 0 means the FMs lead to more different
         representations than retraining with another seed does;
       - the same per stain, on the per-stain tokens (biopsies with that stain);
       - cross-prediction: out-of-fold ridge R^2 of run B's z, and of run B's risk,
         from run A's z. Asymmetry (A -> B high, B -> A low) means A's
         representation contains B's but not the reverse.
  2. Raw features      (--features FM=folder, two or more; needs --stains)
       - each slide's patch features mean-pooled (NaN patch rows skipped),
         averaged per biopsy and stain, per FM; cached in the output folder;
       - linear CKA and k-nearest-neighbour overlap between the FMs on the eval
         split, for all stains pooled and per stain: no model involved, so this
         separates "the FMs see the same thing" from "the MIL training collapses
         them onto the same signal";
       - a ridge Cox model on the top principal components of [A], [B] and
         [A | B], fitted on --raw_fit_splits and scored on the eval split, with
         the gain of [A | B] over the better FM (bootstrap CI).
  3. Stain reliance    (stain_ablation.csv, stain_contributions.csv of each run)
       - leave-one-stain-out C-index drop per stain per FM (mean over seeds);
       - Spearman correlation of the per-biopsy stain contributions between FMs,
         against the same between seeds of one FM.

Outputs (in --output_dir):
    z_cka.csv               CKA of z per pair of runs (kind, variant)
    token_cka.csv           CKA of the per-stain tokens, mean per stain and kind
    cross_prediction.csv    per ordered pair of runs: R^2 of z_B and risk_B from z_A
    summary.csv             mean per kind with bootstrap CI, and seed - fm
    raw_similarity.csv      raw features: CKA and kNN overlap per unit and FM pair
    raw_cox.csv             raw features: Cox C-index of [A], [B], [A | B]
    raw_means_<FM>_p<N>.npz cache of the pooled raw features (delete it after
                            changing --stains, --raw_fit_splits or the features)
    stain_reliance.csv      ablation delta per stain and FM
    contribution_agreement.csv
    fig8a_cka.png           CKA of z and per-stain tokens, other seed vs other FM
    fig8b_stain_reliance.png  ablation delta per stain, one dot per FM
    fig8c_raw_cox.png       raw-feature Cox C-index, [A], [B], [A | B]

Usage (stain_config.sh gives FEATURES, LABELS, LABELS_CSV, STAINS):
    source stain_config.sh
    python python_scripts/evaluate_MIL/representation_similarity.py \\
        --fms uni_v2 virchow2 --variants unicorn early \\
        --features uni_v2=WSI/.../features_uni_v2_npy virchow2=WSI/.../features_virchow2_npy \\
        --labels_path "$LABELS" --labels_csv "$LABELS_CSV" --stains "${STAINS[@]}" \\
        --output_dir results/representation_fm
"""

import argparse
import itertools
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

_script_dir = os.path.dirname(os.path.abspath(__file__))
for _d in (_script_dir, os.path.join(_script_dir, "..", "MIL")):
    if _d not in sys.path:
        sys.path.insert(0, os.path.abspath(_d))
from analyze_model_signal import comparable_pairs  # noqa: E402
from compare_foundation_models import (  # noqa: E402
    KIND_LABEL, RUN_RE, THEMES, find_runs, new_figure, pair_runs, pair_scores, run_meta,
    save, style_axes, title,
)
from fuse_late import fit_cox  # noqa: E402

MAIN_KINDS = ("seed", "fm")


# ---------------------------------------------------------------------------
# Similarity measures
# ---------------------------------------------------------------------------
def linear_cka(X, Y):
    """Linear CKA (Kornblith et al. 2019); X and Y may differ in width."""
    X = X.astype(np.float64) - X.mean(0)
    Y = Y.astype(np.float64) - Y.mean(0)
    if X.shape[1] + Y.shape[1] > 2 * X.shape[0]:  # the n x n Gram form is cheaper
        K, L = X @ X.T, Y @ Y.T
        return float((K * L).sum() / np.sqrt((K * K).sum() * (L * L).sum()))
    return float(np.linalg.norm(X.T @ Y) ** 2
                 / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)))


def ridge_cv_r2(X, Y, alpha, k=5, seed=0):
    """Out-of-fold R^2 of a ridge regression of Y on X, pooled over Y's columns
    (so a wide Y counts each column by its variance). X is standardised on each
    training fold; the penalty is alpha x n_train."""
    X = X.astype(np.float64)
    Y = np.asarray(Y, np.float64).reshape(len(X), -1)
    folds = np.random.default_rng(seed).permutation(len(X)) % k
    pred = np.empty_like(Y)
    for f in range(k):
        tr, te = folds != f, folds == f
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        ym = Y[tr].mean(0)
        A = Xtr.T @ Xtr + alpha * tr.sum() * np.eye(X.shape[1])
        pred[te] = Xte @ np.linalg.solve(A, Xtr.T @ (Y[tr] - ym)) + ym
    return float(1 - ((Y - pred) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())


def knn_overlap(X, Y, k):
    """Mean Jaccard overlap of each sample's k nearest neighbours (cosine, on
    standardised features) in X-space and Y-space. Chance is about k / n."""
    def neighbours(Z):
        Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-8)
        Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
        S = Z @ Z.T
        np.fill_diagonal(S, -np.inf)
        return np.argpartition(-S, k, axis=1)[:, :k]
    nx, ny = neighbours(X.astype(np.float64)), neighbours(Y.astype(np.float64))
    inter = np.array([len(np.intersect1d(a, b)) for a, b in zip(nx, ny)])
    return float(np.mean(inter / (2 * k - inter)))


def boot_ci(vals):
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    return (np.percentile(vals, 2.5), np.percentile(vals, 97.5)) if len(vals) else (np.nan,) * 2


# ---------------------------------------------------------------------------
# 1. Model embeddings
# ---------------------------------------------------------------------------
def load_embeddings(models, split, error):
    """{run: npz dict} aligned on the biopsies every run has."""
    embs = {}
    for name, pattern in models.items():
        path = os.path.join(os.path.dirname(pattern.format(split=split)), "embeddings.npz")
        if not os.path.isfile(path):
            error(f"Missing {path}: rerun train_fusion_models.sh (it adds the embeddings) "
                  f"or analyze_stain_weights.py --embeddings_only.")
        with np.load(path, allow_pickle=False) as f:
            embs[name] = {k: f[k] for k in f.files}
    common = set.intersection(*(set(e["biopsy"]) for e in embs.values()))
    order = sorted(common)
    for e in embs.values():
        pos = {b: i for i, b in enumerate(e["biopsy"])}
        idx = np.array([pos[b] for b in order])
        for k in ("biopsy", "z", "tokens", "stain_mask", "stain_att"):
            e[k] = e[k][idx]
    return embs, np.array(order)


def model_embeddings(args, models, meta, out):
    names = list(meta.index)
    embs, biopsies = load_embeddings(models, args.eval_split, sys.exit)
    n = len(biopsies)
    print(f"\n1. Model embeddings: {n} biopsies in every run's embeddings.npz")
    pairs = pair_runs(meta)
    Z = [embs[m]["z"].astype(np.float64) for m in names]

    # CKA of z, per pair of runs, and per kind with a bootstrap CI.
    cka = {(i, j): linear_cka(Z[i], Z[j]) for i, j, _, _ in pairs}
    pd.DataFrame([dict(run_a=names[i], run_b=names[j], kind=kind, variant=var,
                       cka=cka[(i, j)]) for i, j, kind, var in pairs]
                 ).to_csv(out("z_cka.csv"), index=False)
    groups = {}
    for i, j, kind, var in pairs:
        groups.setdefault((kind, var), []).append((i, j))

    def means(idx):
        q = {(kind, var): np.mean([linear_cka(Z[i][idx], Z[j][idx]) for i, j in prs])
             for (kind, var), prs in groups.items()}
        for var in {v for _, v in groups}:
            if ("seed", var) in q and ("fm", var) in q:
                q[("seed - fm", var)] = q[("seed", var)] - q[("fm", var)]
        return q

    point = means(np.arange(n))
    rng = np.random.default_rng(args.seed)
    boots = [means(rng.integers(0, n, n)) for _ in range(args.n_boot)]
    rows = []
    for key, est in point.items():
        lo, hi = boot_ci([b[key] for b in boots])
        rows.append(dict(level="z", quantity="cka", kind=key[0], variant=key[1],
                         estimate=est, ci_low=lo, ci_high=hi))
    print("  linear CKA of the pooled vector z, mean per kind of run pair (95% CI):")
    for r in rows:
        label = KIND_LABEL.get(r["kind"], r["kind"])
        print(f"    {r['variant']:16s} {label:26s} {r['estimate']:+.3f} "
              f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]")

    # Per-stain tokens: biopsies that have the stain (the panel is the same in
    # every run, being data), point estimates.
    stain_names = list(embs[names[0]]["stain_names"])
    tok_rows = []
    for s, stain in enumerate(stain_names):
        has = np.logical_and.reduce([embs[m]["stain_mask"][:, s] for m in names])
        if has.sum() < args.min_biopsies:
            continue
        T = [embs[m]["tokens"][has, s].astype(np.float64) for m in names]
        for (kind, var), prs in groups.items():
            tok_rows.append(dict(stain=stain, kind=kind, variant=var, n=int(has.sum()),
                                 cka=np.mean([linear_cka(T[i], T[j]) for i, j in prs])))
    tok = pd.DataFrame(tok_rows)
    tok.to_csv(out("token_cka.csv"), index=False)
    print(f"  per-stain token CKA ({tok.stain.nunique() if len(tok) else 0} stains with "
          f">= {args.min_biopsies} biopsies) -> token_cka.csv")

    # Cross-prediction between runs of the same variant.
    risks = {}
    for m in names:
        df = pd.read_csv(models[m].format(split=args.eval_split), dtype={"biopsy": str})
        risks[m] = df.set_index("biopsy")["risk"].reindex(biopsies).to_numpy()
    xp_rows = []
    for i, j, kind, var in pairs:
        if kind not in MAIN_KINDS:
            continue
        for a, b in ((i, j), (j, i)):
            ok = np.isfinite(risks[names[b]])
            xp_rows.append(dict(
                run_from=names[a], run_to=names[b], fm_from=meta.fm.iloc[a],
                fm_to=meta.fm.iloc[b], kind=kind, variant=var,
                r2_z=ridge_cv_r2(Z[a], Z[b], args.ridge_alpha, seed=args.seed),
                r2_risk=ridge_cv_r2(Z[a][ok], risks[names[b]][ok], args.ridge_alpha,
                                    seed=args.seed)))
    xp = pd.DataFrame(xp_rows)
    xp.to_csv(out("cross_prediction.csv"), index=False)
    if len(xp):
        print("  cross-prediction, out-of-fold ridge R^2 from z of one run (mean):")
        agg = xp.groupby(["variant", "kind", "fm_from", "fm_to"])[["r2_z", "r2_risk"]].mean()
        for (var, kind, fa, fb), r in agg.iterrows():
            print(f"    {var:12s} {KIND_LABEL[kind]:22s} {fa} -> {fb}: "
                  f"R^2 of z {r.r2_z:.3f}, of risk {r.r2_risk:.3f}")
    return rows, tok


# ---------------------------------------------------------------------------
# 2. Raw features
# ---------------------------------------------------------------------------
def pool_raw(fm, folder, args, out):
    """Long table (biopsy, split, stain, time, event) + matrix of per-stain means."""
    # Delete the cache after changing --stains, --raw_fit_splits or the features.
    cache = out(f"raw_means_{fm}_p{args.raw_patches}.npz")
    if os.path.isfile(cache):
        with np.load(cache, allow_pickle=False) as f:
            meta = pd.DataFrame({k: f[k] for k in ("biopsy", "split", "stain", "time",
                                                   "event")})
            print(f"  {fm}: {len(meta)} biopsy x stain means from {cache}")
            return meta, f["vec"]
    from mil_utils import load_labels_table  # noqa: E402 — torch, only when needed
    from multistain_data import (  # noqa: E402
        StainVocabulary, _bag_shape, _choose_indices, _read_rows, index_biopsies,
        split_records,
    )
    vocab = StainVocabulary(args.stains)
    table = load_labels_table(args.labels_csv)
    recs = index_biopsies(folder, args.labels_path, table, vocab, file_ext=args.file_ext)
    tr, va, te = split_records(recs, table)
    split_of = {r["biopsy"]: s for s, rs in (("train", tr), ("val", va), ("test", te))
                for r in rs}
    recs = [r for r in recs if split_of[r["biopsy"]] in set(args.raw_fit_splits)
            | {args.eval_split}]

    def slide_mean(path):
        n = _bag_shape(path)[0]
        rows = np.unique(_choose_indices(n, min(n, args.raw_patches), False))
        with np.errstate(all="ignore"):
            return np.nanmean(_read_rows(path, rows).astype(np.float32), axis=0)

    jobs = [(r, s, p) for r in recs for s, paths in r["slides"].items() for p in paths]
    print(f"  {fm}: pooling {len(jobs)} slides of {len(recs)} biopsies "
          f"(<= {args.raw_patches} patches each)...")
    with ThreadPoolExecutor(args.workers) as ex:
        means = list(ex.map(lambda j: slide_mean(j[2]), jobs))
    acc = {}
    for (r, s, _), m in zip(jobs, means):
        acc.setdefault((r["biopsy"], s), []).append(m)
    lab = {r["biopsy"]: np.asarray(r["label"], float) for r in recs}
    keys = sorted(acc)
    meta = pd.DataFrame({
        "biopsy": np.array([str(b) for b, _ in keys]),
        "split": np.array([split_of[b] for b, _ in keys]),
        "stain": np.array([vocab.names[s] for _, s in keys]),
        "time": np.array([lab[b][0] for b, _ in keys]),
        "event": np.array([lab[b][1] for b, _ in keys]),
    })
    vec = np.stack([np.nanmean(acc[k], axis=0) for k in keys]).astype(np.float32)
    # Strings as fixed-width unicode: object arrays would need pickle to reload.
    np.savez_compressed(cache, vec=vec, **{
        c: (meta[c].to_numpy(float) if pd.api.types.is_numeric_dtype(meta[c])
            else np.asarray(meta[c].astype(str).tolist(), dtype=str)) for c in meta})
    return meta, vec


def raw_units(meta, vec, min_biopsies):
    """{unit: DataFrame indexed by biopsy (split, time, event, features)}: all
    stains averaged, and each stain."""
    feats = pd.DataFrame(vec, index=meta.index)
    units = {"all stains": feats.groupby(meta["biopsy"]).mean()}
    for stain, idx in meta.groupby("stain").groups.items():
        if len(idx) >= min_biopsies:
            units[stain] = feats.loc[idx].set_axis(meta.loc[idx, "biopsy"])
    info = meta.drop_duplicates("biopsy").set_index("biopsy")[["split", "time", "event"]]
    return units, info


def pca_fit(X, k):
    mu, sd = X.mean(0), X.std(0) + 1e-8
    _, _, vt = np.linalg.svd((X - mu) / sd, full_matrices=False)
    comp = vt[:k].T
    proj = lambda Y: ((Y - mu) / sd) @ comp  # noqa: E731
    scale = proj(X).std(0) + 1e-8
    return lambda Y: proj(Y) / scale


def raw_features(args, fms, out):
    folders = dict(spec.split("=", 1) for spec in args.features)
    fms = [f for f in fms if f in folders] or list(folders)
    if len(fms) < 2:
        print("\n2. Raw features: skipped (give --features for two FMs or more).")
        return None, None
    print("\n2. Raw features (mean-pooled patch features per biopsy and stain)")
    pooled = {fm: raw_units(*pool_raw(fm, folders[fm], args, out), args.min_biopsies)
              for fm in fms}
    sim_rows, cox_rows = [], []
    for fa, fb in itertools.combinations(fms, 2):
        (ua, info), (ub, _) = pooled[fa], pooled[fb]
        for unit in [u for u in ua if u in ub]:
            common = ua[unit].index.intersection(ub[unit].index)
            split = info.loc[common, "split"]
            ev = common[split == args.eval_split]
            if len(ev) < args.min_biopsies:
                continue
            A, B = ua[unit].loc[ev].to_numpy(), ub[unit].loc[ev].to_numpy()
            k = min(args.knn, len(ev) - 1)
            sim_rows.append(dict(fm_a=fa, fm_b=fb, unit=unit, n=len(ev),
                                 cka=linear_cka(A, B), knn_overlap=knn_overlap(A, B, k),
                                 knn_chance=k / (len(ev) - 1)))
            print(f"  {unit:16s} {fa} vs {fb}: CKA {sim_rows[-1]['cka']:.3f}, "
                  f"{k}-NN overlap {sim_rows[-1]['knn_overlap']:.3f} "
                  f"(chance {sim_rows[-1]['knn_chance']:.3f}), n={len(ev)}")

            fit = common[split.isin(args.raw_fit_splits)]
            t_f, e_f = info.loc[fit, "time"].to_numpy(), info.loc[fit, "event"].to_numpy()
            t_e, e_e = info.loc[ev, "time"].to_numpy(), info.loc[ev, "event"].to_numpy()
            if e_f.sum() < args.min_events or e_e.sum() < args.min_events:
                continue
            pi, pj = np.nonzero(comparable_pairs(t_e, e_e.astype(int)))
            Xs = {}
            for f, u in ((fa, ua), (fb, ub)):
                proj = pca_fit(u[unit].loc[fit].to_numpy(), args.pca_dim)
                Xs[f] = (proj(u[unit].loc[fit].to_numpy()), proj(u[unit].loc[ev].to_numpy()))
            Xs["both"] = tuple(np.hstack([Xs[fa][s], Xs[fb][s]]) for s in (0, 1))
            S = np.column_stack([
                pair_scores(Xs[c][1] @ fit_cox(Xs[c][0], t_f, e_f, args.ridge), pi, pj)
                for c in (fa, fb, "both")])

            def gain(w):
                c = (w @ S) / w.sum()
                return c[2] - max(c[0], c[1])
            c = S.mean(0)
            rng = np.random.default_rng(args.seed)
            n_ev = len(ev)
            boots = []
            for _ in range(args.n_boot):
                cnt = np.bincount(rng.integers(0, n_ev, n_ev), minlength=n_ev).astype(float)
                boots.append(gain(cnt[pi] * cnt[pj]))
            lo, hi = boot_ci(boots)
            cox_rows.append(dict(fm_a=fa, fm_b=fb, unit=unit, n_fit=len(fit),
                                 n_eval=n_ev, events_eval=int(e_e.sum()), c_a=c[0],
                                 c_b=c[1], c_both=c[2], gain=c[2] - max(c[0], c[1]),
                                 gain_ci_low=lo, gain_ci_high=hi))
            print(f"  {'':16s} ridge Cox on {args.pca_dim} PCs: {fa} {c[0]:.3f}, "
                  f"{fb} {c[1]:.3f}, both {c[2]:.3f} (gain {cox_rows[-1]['gain']:+.3f} "
                  f"[{lo:+.3f}, {hi:+.3f}])")
    sim, cox = pd.DataFrame(sim_rows), pd.DataFrame(cox_rows)
    sim.to_csv(out("raw_similarity.csv"), index=False)
    cox.to_csv(out("raw_cox.csv"), index=False)
    return sim, cox


# ---------------------------------------------------------------------------
# 3. Stain reliance
# ---------------------------------------------------------------------------
def stain_reliance(args, models, meta, fms, out):
    names = list(meta.index)
    print("\n3. Stain reliance")
    abl_rows = []
    for m in names:
        path = os.path.join(os.path.dirname(models[m].format(split=args.eval_split)),
                            "stain_ablation.csv")
        if os.path.isfile(path):
            d = pd.read_csv(path)
            abl_rows.append(d.assign(run=m, fm=meta.fm[m], variant=meta.variant[m],
                                     seed=meta.seed[m]))
    rel = pd.concat(abl_rows) if abl_rows else pd.DataFrame()
    if len(rel):
        rel.to_csv(out("stain_reliance.csv"), index=False)
        print(f"  leave-one-stain-out C-index drop of {rel.run.nunique()} runs "
              f"-> stain_reliance.csv")
    else:
        print(f"  no stain_ablation.csv on {args.eval_split} (made on test only)")

    contrib = {}
    for m in names:
        path = os.path.join(os.path.dirname(models[m].format(split=args.eval_split)),
                            "stain_contributions.csv")
        if os.path.isfile(path):
            contrib[m] = pd.read_csv(path, dtype={"biopsy": str}).set_index("biopsy")
    rows = []
    if len(contrib) == len(names):
        stains = [c[len("contrib_"):] for c in contrib[names[0]].columns
                  if c.startswith("contrib_")]
        for i, j, kind, var in pair_runs(meta):
            if kind not in MAIN_KINDS:
                continue
            a, b = contrib[names[i]], contrib[names[j]]
            common = a.index.intersection(b.index)
            for s in stains:
                x, y = a.loc[common, f"contrib_{s}"], b.loc[common, f"contrib_{s}"]
                ok = x.notna() & y.notna()
                if ok.sum() >= args.min_biopsies:
                    rows.append(dict(variant=var, kind=kind, stain=s, n=int(ok.sum()),
                                     spearman=spearmanr(x[ok], y[ok])[0]))
    agree = pd.DataFrame(rows)
    if len(agree):
        agree = agree.groupby(["variant", "stain", "kind"], as_index=False).agg(
            n=("n", "first"), spearman=("spearman", "mean"))
        agree.to_csv(out("contribution_agreement.csv"), index=False)
        wide = agree.pivot_table(index=["variant", "stain"], columns="kind",
                                 values="spearman")
        print("  Spearman of per-biopsy stain contributions (mean over run pairs):")
        print(wide.rename(columns=KIND_LABEL).to_string(float_format=lambda v: f"{v:.2f}"))
    return rel


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_cka(rows, tok, variants, th, path):
    z = pd.DataFrame(rows)
    cols = [(v, k) for v in variants for k in MAIN_KINDS
            if ((z.variant == v) & (z.kind == k)).any()]
    if not cols:
        return
    stains = list(dict.fromkeys(tok.stain)) if len(tok) else []
    labels = ["pooled z"] + stains
    M = np.full((len(labels), len(cols)), np.nan)
    for c, (v, k) in enumerate(cols):
        M[0, c] = z[(z.variant == v) & (z.kind == k)].estimate.iloc[0]
        for r, s in enumerate(stains, start=1):
            hit = tok[(tok.variant == v) & (tok.kind == k) & (tok.stain == s)]
            if len(hit):
                M[r, c] = hit.cka.iloc[0]
    fig, ax = new_figure(th, figsize=(1.6 * len(cols) + 3.2, 0.42 * len(labels) + 2.2))
    cmap = LinearSegmentedColormap.from_list("cka", th["seq"])
    cmap.set_bad(th["surface"])
    im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=np.nanmin(M),
                   vmax=np.nanmax(M), aspect="auto")
    mid = (np.nanmin(M) + np.nanmax(M)) / 2
    for r in range(len(labels)):
        for c in range(len(cols)):
            ax.text(c, r, "–" if np.isnan(M[r, c]) else f"{M[r, c]:.2f}", ha="center",
                    va="center", fontsize=8,
                    color="#ffffff" if np.isfinite(M[r, c]) and M[r, c] > mid else th["ink"])
    ax.axhline(0.5, color=th["surface"], linewidth=3)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"{v}\n{KIND_LABEL[k]}" for v, k in cols], fontsize=8,
                       color=th["ink_secondary"])
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9, color=th["ink_secondary"])
    style_axes(ax, th, "none")
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=th["muted"], labelsize=8, length=0)
    cb.set_label("linear CKA (mean over run pairs)", color=th["ink_secondary"], fontsize=9)
    ax.set_title("Representation similarity: other seed vs other FM", color=th["ink"],
                 fontsize=12, loc="left", pad=34)
    save(fig, th, path)


def plot_reliance(rel, variants, fms, fm_color, th, path):
    if rel is None or not len(rel):
        return
    vs = [v for v in variants if (rel.variant == v).any()]
    order = rel.groupby("stain").delta.mean().sort_values(ascending=False).index
    fig, axes = plt.subplots(1, len(vs), figsize=(4.8 * len(vs), 0.38 * len(order) + 2),
                             sharey=True, squeeze=False)
    fig.patch.set_facecolor(th["surface"])
    for ax, v in zip(axes[0], vs):
        d = rel[rel.variant == v]
        mean = d.groupby(["stain", "fm"]).delta.mean().unstack()
        for y, s in enumerate(order):
            if s not in mean.index:
                continue
            vals = mean.loc[s].dropna()
            if len(vals) > 1:
                ax.plot([vals.min(), vals.max()], [y, y], color=th["axis"], linewidth=2,
                        zorder=1)
            for f, x in vals.items():
                seeds = d[(d.stain == s) & (d.fm == f)].delta
                ax.scatter(seeds, [y] * len(seeds), s=18, facecolor=th["surface"],
                           edgecolor=fm_color[f], linewidth=1, zorder=2)
                ax.scatter([x], [y], s=55, color=fm_color[f], edgecolor=th["surface"],
                           linewidth=1.5, zorder=3)
        ax.axvline(0, color=th["axis"], linewidth=1)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order, color=th["ink_secondary"], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("C-index drop without the stain", color=th["ink_secondary"])
        ax.set_title(v, color=th["ink"], fontsize=11, loc="left")
        style_axes(ax, th, "x")
    handles = [Line2D([], [], marker="o", linestyle="", color=fm_color[f], label=f)
               for f in fms if f in set(rel.fm)]
    handles += [Line2D([], [], marker="o", linestyle="", markerfacecolor=th["surface"],
                       markeredgecolor=th["muted"], label="single seed")]
    fig.suptitle("Which stains each FM relies on (leave-one-stain-out)", color=th["ink"],
                 fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.legend(handles=handles, frameon=False, fontsize=9, labelcolor=th["ink_secondary"],
               loc="upper right", ncol=len(handles), bbox_to_anchor=(0.99, 1.0))
    fig.savefig(path, dpi=150, facecolor=th["surface"], bbox_inches="tight")
    plt.close(fig)


def plot_raw_cox(cox, fm_color, th, path):
    if cox is None or not len(cox):
        return
    fig, ax = new_figure(th, figsize=(8, 0.75 * len(cox) + 1.8))
    h = 0.26
    for y, r in enumerate(cox.itertuples()):
        for k, (val, col, lbl) in enumerate(((r.c_a, fm_color[r.fm_a], r.fm_a),
                                             (r.c_b, fm_color[r.fm_b], r.fm_b),
                                             (r.c_both, th["ink_secondary"], "both"))):
            ax.barh(y + (k - 1) * h, val - 0.5, left=0.5, height=h * 0.9, color=col,
                    label=lbl if y == 0 else None)
        ax.text(max(r.c_a, r.c_b, r.c_both) + 0.005, y,
                f"gain {r.gain:+.3f} [{r.gain_ci_low:+.3f}, {r.gain_ci_high:+.3f}]",
                va="center", fontsize=8, color=th["ink_secondary"])
    ax.set_yticks(range(len(cox)))
    ax.set_yticklabels([f"{r.unit} ({r.fm_a} / {r.fm_b})" for r in cox.itertuples()],
                       color=th["ink_secondary"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("C-index on the eval split (from 0.5)", color=th["ink_secondary"])
    style_axes(ax, th, "x")
    ax.legend(frameon=False, fontsize=8, labelcolor=th["ink_secondary"], ncol=3,
              loc="upper center", bbox_to_anchor=(0.5, -0.12))
    title(ax, th, "Raw features alone: ridge Cox on principal components",
          "gain = both FMs together minus the better FM (bootstrap 95% CI)")
    save(fig, th, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Compare the representations of fusion runs trained on different "
                    "foundation models, and the raw features themselves.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--eval_root", default="results/eval",
                   help="Holds <split>/<FM>/<variant>_s<seed>/ with embeddings.npz.")
    p.add_argument("--fms", nargs="+", default=None)
    p.add_argument("--variants", nargs="+", default=None)
    p.add_argument("--model", action="append", metavar="FM/variant_sSEED=PATTERN",
                   help="Extra run outside the layout (its folder holds embeddings.npz).")
    p.add_argument("--eval_split", default="test")
    p.add_argument("--features", nargs="+", default=[], metavar="FM=FOLDER",
                   help="Raw feature folders, one per FM, for part 2.")
    p.add_argument("--labels_path", default="WSI/whole_registry/labels")
    p.add_argument("--labels_csv", default="label_csvs/labels.csv")
    p.add_argument("--file_ext", default=".npy", choices=[".h5", ".npy"])
    p.add_argument("--stains", nargs="+", default=None,
                   help="Stain specs as in stain_config.sh (needed for part 2).")
    p.add_argument("--raw_fit_splits", nargs="+", default=["train", "val"],
                   help="Splits the raw-feature Cox models are fitted on.")
    p.add_argument("--raw_patches", type=int, default=1024,
                   help="Patches read per slide for the raw means (evenly spaced).")
    p.add_argument("--workers", type=int, default=8, help="Reader threads for part 2.")
    p.add_argument("--pca_dim", type=int, default=32)
    p.add_argument("--knn", type=int, default=10)
    p.add_argument("--ridge", type=float, default=1e-2, help="Ridge of the Cox models.")
    p.add_argument("--ridge_alpha", type=float, default=0.1,
                   help="Ridge penalty (x n) of the cross-prediction regressions.")
    p.add_argument("--min_biopsies", type=int, default=50,
                   help="Smallest biopsy count for a per-stain number.")
    p.add_argument("--min_events", type=int, default=20,
                   help="Smallest event count (fit and eval) for a raw Cox model.")
    p.add_argument("--n_boot", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--theme", default="light", choices=["light", "dark"])
    p.add_argument("--output_dir", default="results/representation_fm")
    args = p.parse_args()
    if args.features and not args.stains:
        p.error("--features needs --stains (the STAINS of stain_config.sh).")
    th = THEMES[args.theme]
    os.makedirs(args.output_dir, exist_ok=True)
    out = lambda f: os.path.join(args.output_dir, f)  # noqa: E731

    models = find_runs(args, p)
    if not models:
        p.error(f"No runs under {args.eval_root}/{args.eval_split}/<FM>/<variant>_s<seed>/.")
    names = sorted(models, key=lambda n: (RUN_RE.match(n)["variant"], RUN_RE.match(n)["fm"],
                                          int(RUN_RE.match(n)["seed"])))
    models = {n: models[n] for n in names}
    meta = run_meta(names)
    fms = [f for f in (args.fms or sorted(meta.fm.unique())) if f in set(meta.fm)]
    variants = [v for v in (args.variants or sorted(meta.variant.unique()))
                if v in set(meta.variant)]
    fm_color = {f: th["series"][i % len(th["series"])] for i, f in enumerate(fms)}
    for f in dict(spec.split("=", 1) for spec in args.features):
        fm_color.setdefault(f, th["series"][len(fm_color) % len(th["series"])])
    print(f"{len(names)} runs: " + ", ".join(names))

    rows, tok = model_embeddings(args, models, meta, out)
    sim, cox = raw_features(args, fms or list(fm_color), out)
    rel = stain_reliance(args, models, meta, fms, out)
    pd.DataFrame(rows).to_csv(out("summary.csv"), index=False)

    plot_cka(rows, tok, variants, th, out("fig8a_cka.png"))
    plot_reliance(rel, variants, fms, fm_color, th, out("fig8b_stain_reliance.png"))
    plot_raw_cox(cox, fm_color, th, out("fig8c_raw_cox.png"))
    print(f"\nResults -> {args.output_dir}")


if __name__ == "__main__":
    main()
