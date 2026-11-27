"""
rotation_invariance.py — How much do patch features change when the patch is rotated 90°?

Compares the per-patch features of the slides present in BOTH folders (e.g. features_uni_v2/
and features_uni_v2_rot90/ from extract_rotated_feats.py). Patches are paired by their
coords (not by row index), so the two extractions need not list patches in the same order;
patches missing from either file or with NaN features are dropped.

For each paired patch (a = unrotated, b = rotated):
    cos       cosine similarity(a, b)
    l2        ||a - b||
    rel_l2    ||a - b|| / ||a||

A distance means little on its own, so two within-slide references are computed on a random
subset of --max_patches patches per slide:
    cos_random   cosine similarity of a to a random OTHER unrotated patch (unrelated tissue)
    cos_nn       cosine similarity of a to its nearest OTHER unrotated patch (most similar
                 tissue, often the adjacent tile)
    top1         is the nearest unrotated patch to b its own original a? (retrieval)
    rank         rank of a among all unrotated patches by similarity to b (1 = best)
If rotation barely matters, cos is close to 1, well above cos_nn, and top1 is near 100%.

Usage:
    python python_scripts/evaluate_MIL/rotation_invariance.py \\
        --features_dir .../20x_224px_0px_overlap/features_uni_v2 \\
        --rotated_dir  .../20x_224px_0px_overlap/features_uni_v2_rot90 \\
        --output_dir   results/rotation_invariance/uni_v2

Outputs (in --output_dir):
    per_slide.csv   one row per slide: n patches, mean/median of each metric, top1
    summary.txt     the pooled description printed to stdout
    cosine_hist.png distribution of cos vs cos_nn vs cos_random (all slides pooled)

Caveat: if features_dir was extracted with stain normalisation and rotated_dir without, the
difference measured mixes rotation with stain normalisation. extract_rotated_feats.py writes a
matching unrotated baseline to features_{enc}_rot0/ for exactly this reason.
"""

import argparse
import os

import h5py
import numpy as np
import pandas as pd


def load(path):
    """Return (features float32 [N, D], coords int [N, 2]) of a TRIDENT feature .h5."""
    with h5py.File(path, "r") as f:
        return f["features"][:].astype(np.float32), f["coords"][:]


def pair_by_coords(fa, ca, fb, cb):
    """Keep the patches present in both files (matched on coords) with finite features."""
    ia = {tuple(c): i for i, c in enumerate(ca)}
    pairs = [(ia[tuple(c)], j) for j, c in enumerate(cb) if tuple(c) in ia]
    if not pairs:
        return fa[:0], fb[:0]
    i, j = np.array(pairs).T
    a, b = fa[i], fb[j]
    ok = np.isfinite(a).all(1) & np.isfinite(b).all(1)
    return a[ok], b[ok]


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def slide_metrics(a, b, max_patches, rng):
    """Per-patch metrics for one slide (dict of 1-D arrays) — see module docstring."""
    ua, ub = unit(a), unit(b)
    diff = np.linalg.norm(a - b, axis=1)
    m = {
        "cos": (ua * ub).sum(1),
        "l2": diff,
        "rel_l2": diff / np.maximum(np.linalg.norm(a, axis=1), 1e-12),
    }

    # References on a subset: the full N x N similarity is too big for large slides.
    n = len(a)
    if n < 2:
        return m
    idx = rng.choice(n, size=min(n, max_patches), replace=False)
    sa, sb = ua[idx], ub[idx]
    s_aa = sa @ sa.T                      # unrotated vs unrotated
    np.fill_diagonal(s_aa, -np.inf)       # exclude the patch itself
    m["cos_nn"] = s_aa.max(1)
    other = (np.arange(len(idx)) + rng.integers(1, len(idx), len(idx))) % len(idx)
    m["cos_random"] = (sa * sa[other]).sum(1)
    s_ba = sb @ sa.T                      # rotated vs unrotated
    own = np.diag(s_ba)
    m["rank"] = (s_ba > own[:, None]).sum(1) + 1
    m["top1"] = (m["rank"] == 1).astype(float)
    return m


def describe(name, x):
    q = np.percentile(x, [5, 25, 50, 75, 95])
    return (f"  {name:<11} mean {x.mean():.4f}  sd {x.std():.4f}  "
            f"p5 {q[0]:.4f}  p25 {q[1]:.4f}  median {q[2]:.4f}  "
            f"p75 {q[3]:.4f}  p95 {q[4]:.4f}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--features_dir", required=True, help="Unrotated features (.h5 per slide).")
    p.add_argument("--rotated_dir", required=True, help="Rotated features (.h5 per slide).")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_patches", type=int, default=2000,
                   help="Patches per slide used for the nn / random / retrieval references.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    names = lambda d: {f[:-3] for f in os.listdir(d) if f.endswith(".h5")}  # noqa: E731
    slides = sorted(names(args.features_dir) & names(args.rotated_dir))
    print(f"{len(slides)} slide(s) in both folders.")
    if not slides:
        raise SystemExit("nothing to compare")

    rng = np.random.default_rng(args.seed)
    rows, pooled = [], {}
    for sid in slides:
        fa, ca = load(os.path.join(args.features_dir, f"{sid}.h5"))
        fb, cb = load(os.path.join(args.rotated_dir, f"{sid}.h5"))
        if fa.shape[1:] != fb.shape[1:]:
            raise SystemExit(f"{sid}: feature dims differ ({fa.shape} vs {fb.shape})")
        a, b = pair_by_coords(fa, ca, fb, cb)
        if len(a) == 0:
            print(f"  [WARN] {sid}: no patches in common, skipped")
            continue
        if len(a) < max(len(fa), len(fb)):
            print(f"  {sid}: {len(a)} paired patches "
                  f"(unrotated {len(fa)}, rotated {len(fb)})")
        m = slide_metrics(a, b, args.max_patches, rng)
        row = {"slide_id": sid, "n_patches": len(a)}
        for k, v in m.items():
            row[f"{k}_mean"] = float(v.mean())
            if k != "top1":
                row[f"{k}_median"] = float(np.median(v))
            pooled.setdefault(k, []).append(v)
        rows.append(row)

    os.makedirs(args.output_dir, exist_ok=True)
    per_slide = pd.DataFrame(rows)
    per_slide.to_csv(os.path.join(args.output_dir, "per_slide.csv"), index=False)
    pooled = {k: np.concatenate(v) for k, v in pooled.items()}

    lines = [
        f"unrotated: {args.features_dir}",
        f"rotated:   {args.rotated_dir}",
        f"{len(per_slide)} slide(s), {int(per_slide.n_patches.sum())} paired patches",
        "",
        "Per patch, all slides pooled (rotated vs its own unrotated patch):",
        describe("cos", pooled["cos"]),
        describe("l2", pooled["l2"]),
        describe("rel_l2", pooled["rel_l2"]),
    ]
    if "cos_nn" in pooled:
        top1 = pooled["top1"].mean()
        lines += [
            "",
            f"References (up to {args.max_patches} patches per slide):",
            describe("cos_nn", pooled["cos_nn"]),
            describe("cos_random", pooled["cos_random"]),
            describe("rank", pooled["rank"].astype(float)),
            f"  top1 retrieval: {100 * top1:.1f}% of rotated patches retrieve their own "
            f"original (chance ~{100 * np.mean(1 / per_slide.n_patches.clip(upper=args.max_patches)):.2f}%)",
            f"  median cos {np.median(pooled['cos']):.4f} vs median cos_nn "
            f"{np.median(pooled['cos_nn']):.4f}: rotation moves a patch "
            f"{'less' if np.median(pooled['cos']) > np.median(pooled['cos_nn']) else 'MORE'}"
            " than the gap to its most similar other patch",
        ]
    lines += [
        "",
        "Per slide (mean cos): " + describe("", per_slide.cos_mean.to_numpy()).strip(),
        f"Lowest mean cos: " + ", ".join(
            f"{r.slide_id} ({r.cos_mean:.4f})"
            for r in per_slide.nsmallest(5, "cos_mean").itertuples()
        ),
    ]
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.output_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(min(v.min() for k, v in pooled.items() if k.startswith("cos")), 1, 80)
    for key, label in [("cos", "rotated vs own original"),
                       ("cos_nn", "nearest other patch"),
                       ("cos_random", "random other patch")]:
        if key in pooled:
            ax.hist(pooled[key], bins=bins, density=True, histtype="step", lw=1.5, label=label)
    ax.set_xlabel("cosine similarity")
    ax.set_ylabel("density")
    ax.set_title(f"Rotation invariance ({len(per_slide)} slides)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "cosine_hist.png"), dpi=150)
    print(f"\nWrote per_slide.csv, summary.txt, cosine_hist.png to {args.output_dir}")


if __name__ == "__main__":
    main()
