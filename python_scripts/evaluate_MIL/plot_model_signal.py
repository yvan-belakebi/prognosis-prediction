"""Figures for the results of analyze_model_signal.py, in the poster style.

M1  Models on the same biopsies       M1_overview            cindex_overview.csv (+ clinical baselines)
M2  Kidney failure, not death         M2_endpoints           event_type_cindex.csv
M3  Images add to the clinic          M3_added_value         clinical_added_value.csv
M4  Prognosis within a diagnosis      M4_within_diagnosis    stratified_cindex.csv, per_diagnosis_cindex.csv
M5  The models agree on who is at risk  M5_agreement         risk_correlation.csv, discordance_overlap.csv
M6  Where the models go wrong         M6_discordance         discordance_by_gap.csv, hard_biopsies.csv

Same look as poster/PRESIMAL_2026_poster_Yvan_Belakebi/make_all_results_figures.py:
vector output (pdf, svg) at print size, all text >= 24 pt, a small PNG preview,
one fixed colour per method. Every number is read from the CSVs, nothing is typed in.

Run:
    python python_scripts/evaluate_MIL/plot_model_signal.py                       # all six
    python python_scripts/evaluate_MIL/plot_model_signal.py --only M3 M5
    python python_scripts/evaluate_MIL/plot_model_signal.py \\
        --input_dir results/model_signal --output_dir results/model_signal/figures
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.lines import Line2D

# ------------------------------------------------------------------ style (as the poster)
MM = 1 / 25.4
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
    "font.size": 26,
    "axes.titlesize": 26,
    "axes.labelsize": 26,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#8A93A0",
    "axes.linewidth": 1.5,
    "xtick.color": "#5A6270",
    "ytick.color": "#1F2328",
    "xtick.major.width": 1.5,
    "ytick.major.size": 0,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})
INK, MUTED, GRID, LINE, PALE = "#1F2328", "#5A6270", "#E3E7EB", "#8A93A0", "#C9CFD7"
CLIN = "#8A93A0"

# Display name and colour per model key of the CSVs (the poster's method colours).
LABEL = {"PAS": "PAS", "average": "Voting", "voting": "Voting (C-weighted)",
         "late": "Late fusion", "early": "Early fusion", "unicorn": "UNICORN"}
METHOD = {
    "PAS": "#5C3FA8",
    "Voting": "#E0609A",
    "Late fusion": "#1B8A5A",
    "Early fusion": "#2A78D6",
    "UNICORN": "#C57600",
}
EXTRA = ["#3B4A5C", "#B04A3A", "#7A6C58", "#2E8C9A", "#8E5BB5"]  # models not in METHOD
ORDER = list(METHOD)
TWO_LINE = {"Late fusion": "Late\nfusion", "Early fusion": "Early\nfusion",
            "Voting (C-weighted)": "Voting\n(C-weighted)"}
KIDNEY, DEATH = "#3B4A5C", "#B04A3A"

# Neutral ramps (never a method colour): slate for C-index / rho, stone for Jaccard
SLATE = LinearSegmentedColormap.from_list("slate", ["#F1F3F6", "#A9B3C1", "#4A5A6E", "#1A2330"])
STONE = LinearSegmentedColormap.from_list("stone", ["#F5F2ED", "#C4B9A8", "#7A6C58", "#3A3226"])


def label(m):
    return LABEL.get(m, m)


_extra = {}


def color(m):
    lab = label(m)
    if lab in METHOD:
        return METHOD[lab]
    return _extra.setdefault(lab, EXTRA[len(_extra) % len(EXTRA)])


def fmt_p(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def text_on(rgb):
    r, g, b = to_rgb(rgb)
    return "white" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.5 else INK


# ------------------------------------------------------------------ canvas (as the poster)
class Canvas:
    """A figure laid out in mm from its bottom-left corner."""

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.fig = plt.figure(figsize=(w * MM, h * MM))

    def axes(self, x, y, w, h, **kw):
        return self.fig.add_axes([x / self.w, y / self.h, w / self.w, h / self.h], **kw)

    def title(self, tag, title, sub, x=10):
        y = self.h - 22
        f = self.fig
        f.text(x / self.w, y / self.h, tag, fontsize=40, fontweight="bold", color=INK, va="baseline")
        f.text((x + 24) / self.w, y / self.h, title, fontsize=34, fontweight="bold", color=INK,
               va="baseline")
        f.text((x + 24) / self.w, (y - 12) / self.h, sub, fontsize=24, color=MUTED, va="baseline")

    def note(self, x, y, s, **kw):
        kw.setdefault("color", INK)
        self.fig.text(x / self.w, y / self.h, s, fontsize=24, va="baseline", **kw)

    def legend(self, handles, x, y, **kw):
        kw.setdefault("ncol", len(handles))
        self.fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(x / self.w, y / self.h),
                        frameon=False, fontsize=24, handletextpad=0.4, columnspacing=1.5, **kw)

    def save(self, path):
        for ext in ("pdf", "svg"):
            self.fig.savefig(f"{path}.{ext}")
        self.fig.savefig(f"{path}_preview.png", dpi=72)
        plt.close(self.fig)


def xgrid(ax):
    ax.grid(axis="x", color=GRID, lw=1.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)


def ygrid(ax):
    ax.grid(axis="y", color=GRID, lw=1.5, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=6, colors=MUTED)


def method_legend(cv, models, x, y, ncol=None, marker="o"):
    cv.legend([Line2D([], [], marker=marker, ls="none", ms=17, color=color(m), mec="white", mew=2.5,
                      label=label(m)) for m in models], x, y, ncol=ncol or len(models))


def heatmap(ax, M, cmap, norm, mask=None, size=26):
    """Cells of M (NaN or masked left blank) with their value printed."""
    M = np.asarray(M, float)
    hide = np.isnan(M) if mask is None else (mask | np.isnan(M))
    ax.imshow(np.ma.masked_where(hide, M), cmap=cmap, norm=norm, aspect="auto",
              interpolation="nearest")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not hide[i, j]:
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=size,
                        color=text_on(cmap(norm(M[i, j]))))
    ax.set_xticks(np.arange(M.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(M.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", lw=4)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)


def nice_ticks(lo, hi, step):
    """At most three ticks on multiples of ``step`` within [lo, hi]."""
    t = np.arange(np.ceil(lo / step - 1e-9) * step, hi + 1e-9, step)
    while len(t) > 3:
        step *= 2
        t = np.arange(np.ceil(lo / step - 1e-9) * step, hi + 1e-9, step)
    return np.round(t, 2)


def colorbar(cv, x, y, w, h, cmap, norm, ticks, text):
    cax = cv.axes(x, y, w, h)
    cb = cv.fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    cb.set_ticks(ticks)
    cb.outline.set_edgecolor(LINE)
    cax.tick_params(length=4, colors=MUTED)
    cax.set_title(text, fontsize=24, color=INK, pad=8)


# ------------------------------------------------------------------ data
class Results:
    """The CSVs of one analyze_model_signal.py run, restricted to the shown models."""

    def __init__(self, folder, drop, split_name):
        self.dir = folder
        ov = self.csv("cindex_overview")
        keys = [m for m in ov["model"] if m not in drop]
        rank = {lab: i for i, lab in enumerate(ORDER)}
        self.models = sorted(keys, key=lambda m: (rank.get(label(m), len(rank)), keys.index(m)))
        self.overview = ov.set_index("model").loc[self.models]
        n, ev = int(ov["n"].iloc[0]), int(ov["events"].iloc[0])
        self.caption = f"{split_name}, {n:,} biopsies, {ev} events"

    def csv(self, name, **kw):
        path = os.path.join(self.dir, f"{name}.csv")
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        return pd.read_csv(path, **kw)

    def added_value(self):
        """(baselines, combined): clinical-only rows and baseline + model rows."""
        av = self.csv("clinical_added_value")
        is_base = av["model"].str.startswith("[clinical] ")
        base = av[is_base].copy()
        base["baseline"] = base["model"].str.replace("[clinical] ", "", regex=False)
        comb = av[~is_base].copy()
        # Older runs have no baseline column: split "<baseline>+<model>" on a known model key.
        keys = list(self.overview.index) + [m for m in LABEL if m not in self.overview.index]

        def split(name):
            for k in sorted(keys, key=len, reverse=True):
                if name.endswith("+" + k):
                    return name[: -len(k) - 1], k
            return name.rsplit("+", 1)

        parts = comb["model"].map(split)
        if "baseline" not in comb or comb["baseline"].isna().all():
            comb["baseline"] = parts.str[0]
        comb["key"] = parts.str[1]
        comb = comb[comb["key"].isin(self.models)]
        return base.set_index("baseline"), comb

    def adjusted_for(self):
        base, comb = self.added_value()
        last = comb["baseline"].unique()[-1] if len(comb) else "clinical"
        return last.replace("+", " + ")


# ------------------------------------------------------------------ M1
def figure_overview(R, out, min_baseline_c):
    base, _ = R.added_value()
    shown = base[base["c_index"] >= min_baseline_c].sort_values("c_index", ascending=False)
    hidden = base[base["c_index"] < min_baseline_c]
    rows = [(label(m), r.c_index, r.ci_low, r.ci_high, color(m), True)
            for m, r in R.overview.iterrows()]
    rows += [(b, r.c_index, r.get("ci_low", np.nan), r.get("ci_high", np.nan), CLIN, False)
             for b, r in shown.iterrows()]
    n_mod = len(R.overview)
    ys = [-i for i in range(n_mod)] + [-(n_mod + 0.6 + i) for i in range(len(shown))]
    H = 125 + 24 * (len(rows) + 0.6)
    cv = Canvas(440, H)
    cv.title("M1", "Models on the same biopsies", R.caption)
    ax = cv.axes(140, 75, 180, H - 125)
    for y, (name, c, lo, hi, col, is_model) in zip(ys, rows):
        if np.isfinite(lo):
            ax.plot([lo, hi], [y, y], color=col, lw=4, solid_capstyle="round", zorder=3)
        if is_model:
            ax.plot(c, y, "o", ms=20, color=col, mec="white", mew=2.5, zorder=4)
        else:
            ax.plot(c, y, "s", ms=16, mfc="white", mec=col, mew=3, zorder=4)
        ci = f" ({lo:.3f}–{hi:.3f})" if np.isfinite(lo) else ""
        ax.text(1.03, y, f"{c:.3f}{ci}", transform=ax.get_yaxis_transform(), va="center",
                fontsize=24, color=INK if is_model else MUTED)
    if len(shown):
        ax.axhline(-(n_mod - 0.2), color=LINE, lw=1.5, ls=(0, (2, 3)))
    ax.set_yticks(ys, [r[0] for r in rows])
    for t, r in zip(ax.get_yticklabels(), rows):
        t.set_fontweight("bold" if r[5] else "normal")
        t.set_color(INK if r[5] else MUTED)
    ax.set_ylim(min(ys) - 0.6, 0.6)
    lo_x = np.nanmin([r[2] if np.isfinite(r[2]) else r[1] for r in rows])
    hi_x = np.nanmax([r[3] if np.isfinite(r[3]) else r[1] for r in rows])
    ax.set_xlim(np.floor(lo_x * 100 - 0.5) / 100, np.ceil(hi_x * 100 + 0.5) / 100)
    ax.set_xlabel("C-index (95% CI)")
    xgrid(ax)
    cv.legend([Line2D([], [], marker="o", ls="-", lw=4, ms=18, color=INK, mec="white", label="model"),
               Line2D([], [], marker="s", ls="-", lw=4, ms=14, color=CLIN, mfc="white", mec=CLIN, mew=3,
                      label="clinical Cox baseline")], 130, 28)
    if len(hidden):
        cv.note(140, 14, "Not shown: " + ", ".join(f"{b} {r.c_index:.3f}" for b, r in hidden.iterrows()),
                color=MUTED)
    cv.save(os.path.join(out, "M1_overview"))


# ------------------------------------------------------------------ M2
def figure_endpoints(R, out):
    et = R.csv("event_type_cindex").set_index("model").loc[R.models]
    cols = [("c_all", "All events"), ("c_kidney_failure", "Kidney failure"), ("c_death", "Death")]
    cv = Canvas(400, 310)
    cv.title("M2", "The signal is kidney failure, not death",
             f"{R.caption} · the other event type censored at its time")
    ax = cv.axes(60, 95, 250, 160)
    offs = np.linspace(-0.14, 0.14, len(R.models))
    for x, (c, _) in enumerate(cols):
        lo, hi = et[c].min(), et[c].max()
        ax.plot([x, x], [lo, hi], color=PALE, lw=60, solid_capstyle="butt", zorder=1)
        ax.text(x + 0.3, (lo + hi) / 2, f"{lo:.3f}–{hi:.3f}" if hi - lo >= 5e-4 else f"{lo:.3f}",
                va="center", fontsize=24, color=INK)
    for o, m in zip(offs, R.models):
        v = et.loc[m, [c for c, _ in cols]].to_numpy(float)
        ax.plot(np.arange(3) + o, v, color=color(m), lw=1.5, alpha=0.5, zorder=2)
        ax.plot(np.arange(3) + o, v, "o", ms=17, color=color(m), mec="white", mew=2.5, zorder=4)
    ax.set_xticks(range(3), [lab for _, lab in cols], fontweight="bold")
    ax.tick_params(axis="x", length=0, labelcolor=INK, pad=10)
    ax.set_xlim(-0.45, 2.9)
    vals = et[[c for c, _ in cols]].to_numpy(float)
    ax.set_ylim(np.floor(np.nanmin(vals) * 20 - 0.3) / 20, np.ceil(np.nanmax(vals) * 20 + 0.3) / 20)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))
    ax.set_ylabel("C-index")
    ygrid(ax)
    method_legend(cv, R.models, 50, 30, ncol=min(len(R.models), 3))
    cv.note(330, 230, "Grey bar:", color=MUTED)
    cv.note(330, 218, "range across", color=MUTED)
    cv.note(330, 206, "the models", color=MUTED)
    cv.save(os.path.join(out, "M2_endpoints"))


# ------------------------------------------------------------------ M3
def figure_added_value(R, out):
    base, comb = R.added_value()
    baselines = list(dict.fromkeys(comb["baseline"]))
    n = len(R.models)
    PW, TW = 130, 95  # panel width, text column width (mm)
    W, H = max(110 + len(baselines) * (PW + TW), 560), 140 + 26 * n
    cv = Canvas(W, H)
    cv.title("M3", "Images add to the clinic",
             f"C-index gain when the model's risk joins a clinical Cox model · {R.caption}")
    lo_x = min(0, comb["gain_ci_low"].min()) - 0.005
    hi_x = comb["gain_ci_high"].max() + 0.005
    ys = [-i for i in range(n)]
    for k, b in enumerate(baselines):
        ax = cv.axes(110 + k * (PW + TW), 75, PW, 26 * n)
        sub = comb[comb["baseline"] == b].set_index("key").loc[R.models]
        for y, (m, r) in zip(ys, sub.iterrows()):
            col = color(m)
            ax.plot([r.gain_ci_low, r.gain_ci_high], [y, y], color=col, lw=4, solid_capstyle="round", zorder=3)
            ax.plot(r.c_gain, y, "o", ms=20, color=col, mec="white", mew=2.5, zorder=4)
            ax.text(1.04, y + 0.17, f"+{r.c_gain:.3f} → {r.c_index:.3f}",
                    transform=ax.get_yaxis_transform(), va="center", fontsize=24, color=INK)
            ax.text(1.04, y - 0.2, f"HR {r.hr_per_sd:.2f}/SD, {fmt_p(r.lr_p)}",
                    transform=ax.get_yaxis_transform(), va="center", fontsize=24, color=MUTED)
        ax.axvline(0, color=INK, lw=2, zorder=2)
        ax.set_yticks(ys, [label(m) for m in R.models] if k == 0 else [], fontweight="bold")
        ax.set_ylim(-n + 0.4, 0.6)
        ax.set_xlim(lo_x, hi_x)
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.02))
        ax.set_xlabel("C-index gain (95% CI)")
        xgrid(ax)
        bc = base.loc[b, "c_index"] if b in base.index else np.nan
        ax.set_title(f"{b.replace('+', ' + ')}" + (f"  (C {bc:.3f})" if np.isfinite(bc) else ""),
                     fontweight="bold", loc="left", pad=16, fontsize=26)
    cv.note(110, 28, "Cox models fitted on the validation split, scored on the test split; "
                     "HR per SD of the model's risk,", color=MUTED)
    cv.note(110, 15, "likelihood-ratio test against the baseline; gain CI by paired bootstrap.",
            color=MUTED)
    cv.save(os.path.join(out, "M3_added_value"))


# ------------------------------------------------------------------ M4
def figure_within_diagnosis(R, out, max_groups):
    st = R.csv("stratified_cindex").set_index("model").loc[R.models]
    pdx = R.csv("per_diagnosis_cindex").head(max_groups)
    n, g = len(R.models), len(pdx)
    CELL = 30
    HM_X, HM_W = 400, CELL * n
    W = HM_X + HM_W + 20
    H = max(140 + 24 * n, 165 + CELL * g)
    cv = Canvas(W, H)
    cv.title("M4", "Prognosis within a diagnosis, not disease recognition", R.caption)

    # left: all comparable pairs vs pairs of patients with the same diagnosis
    ax = cv.axes(90, max(75, H - 70 - 24 * n), 110, 24 * n)
    ys = [-i for i in range(n)]
    for y, (m, r) in zip(ys, st.iterrows()):
        col = color(m)
        ax.plot([r.c_within_diagnosis, r.c_all], [y, y], color=col, lw=3, alpha=0.6, zorder=2)
        ax.plot(r.c_all, y, "o", ms=20, color=col, mec="white", mew=2.5, zorder=4)
        ax.plot(r.c_within_diagnosis, y, "D", ms=15, mfc="white", mec=col, mew=3.5, zorder=4)
    ax.set_yticks(ys, [label(m) for m in R.models], fontweight="bold")
    ax.set_ylim(-n + 0.4, 0.6)
    vals = st[["c_all", "c_within_diagnosis"]].to_numpy()
    ax.set_xlim(np.floor(vals.min() * 50 - 0.2) / 50, np.ceil(vals.max() * 50 + 0.2) / 50)
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.02))
    ax.set_xlabel("C-index")
    xgrid(ax)
    r0 = st.iloc[0]
    ax.set_title(f"Same-diagnosis pairs: {int(r0.pairs_within):,} of {int(r0.pairs_all):,}",
                 fontsize=24, color=MUTED, loc="left", pad=14)
    cv.legend([Line2D([], [], marker="o", ls="none", ms=18, color=INK, mec="white", label="all pairs"),
               Line2D([], [], marker="D", ls="none", ms=14, mfc="white", mec=INK, mew=3,
                      label="same diagnosis")], 10, 22, ncol=1)

    # right: C-index per diagnosis group
    if g:
        hm_h = CELL * g
        axh = cv.axes(HM_X, H - 70 - hm_h, HM_W, hm_h)
        M = pdx[R.models].to_numpy(float)
        norm = Normalize(np.floor(np.nanmin(M) * 20) / 20, np.ceil(np.nanmax(M) * 20) / 20)
        heatmap(axh, M, SLATE, norm, size=22)
        axh.set_yticks(range(g), [f"{d}  {n_:,} ({e})" for d, n_, e in
                                  zip(pdx["diagnosis"], pdx["n"], pdx["events"])], fontsize=22)
        axh.xaxis.tick_top()
        axh.set_xticks(range(n), [TWO_LINE.get(label(m), label(m)) for m in R.models], fontsize=20,
                       fontweight="bold", linespacing=1.0)
        for t, m in zip(axh.get_xticklabels(), R.models):
            t.set_color(color(m))
        axh.text(-0.01, 1.0, "diagnosis  n (events)", transform=axh.transAxes, ha="right",
                 va="bottom", fontsize=22, color=MUTED)
        colorbar(cv, HM_X, H - 70 - hm_h - 42, HM_W, 6, SLATE, norm,
                 nice_ticks(norm.vmin, norm.vmax, 0.05), "C-index per diagnosis")
    cv.save(os.path.join(out, "M4_within_diagnosis"))


# ------------------------------------------------------------------ M5
def figure_agreement(R, out):
    corr = R.csv("risk_correlation", index_col=0)
    jac = R.csv("discordance_overlap", index_col=0)
    keys = R.models
    # risk_correlation.csv: upper triangle raw, lower triangle adjusted (by CSV position).
    pos = {m: i for i, m in enumerate(corr.index)}
    C = corr.to_numpy()
    n = len(keys)
    raw = np.full((n, n), np.nan)
    for a, ma in enumerate(keys):
        for b, mb in enumerate(keys):
            if a != b:
                i, j = pos[ma], pos[mb]
                hi_ij = (min(i, j), max(i, j))
                # shown upper (b > a) = raw, shown lower = adjusted
                raw[a, b] = C[hi_ij] if b > a else C[hi_ij[::-1]]
    J = jac.loc[keys, keys].to_numpy(float)
    J[np.triu_indices(n)] = np.nan

    S = 34
    W = 110 + 2 * (n * S + 190)
    H = 150 + n * S
    cv = Canvas(W, H)
    cv.title("M5", "The models agree on who is at risk",
             f"Spearman correlation of the predicted risks and overlap of their mistakes · {R.caption}")
    names = [TWO_LINE.get(label(m), label(m)) for m in keys]
    off = [np.nanmin(raw), np.nanmax(raw)]
    rho_norm = Normalize(np.floor(off[0] * 20) / 20, 1.0)
    x1 = 130
    ax1 = cv.axes(x1, 55, n * S, n * S)
    heatmap(ax1, raw, SLATE, rho_norm)
    ax1.plot([-0.5, n - 0.5], [-0.5, n - 0.5], color=LINE, lw=1.5)
    ax1.text(n - 0.4, 0, "raw", ha="left", va="center", fontsize=26, fontweight="bold")
    ax1.text(-0.5, n - 0.35, f"after {R.adjusted_for()}", ha="left", va="top", fontsize=26,
             fontweight="bold")
    jac_vals = J[np.isfinite(J)]
    jac_norm = Normalize(np.floor(jac_vals.min() * 10) / 10, np.ceil(jac_vals.max() * 10) / 10)
    x2 = x1 + n * S + 200
    ax2 = cv.axes(x2, 55 + S, (n - 1) * S, (n - 1) * S)
    heatmap(ax2, J[1:, :-1], STONE, jac_norm)
    ax2.set_title("Shared mistakes", fontsize=30, fontweight="bold", loc="left", pad=14)
    for ax, rk, ck in ((ax1, keys, keys), (ax2, keys[1:], keys[:-1])):
        ax.set_yticks(range(len(rk)), [label(m) for m in rk], fontweight="bold")
        if ax is ax1:
            ax.xaxis.tick_top()
        ax.set_xticks(range(len(ck)), [TWO_LINE.get(label(m), label(m)) for m in ck], fontweight="bold",
                      linespacing=1.0, fontsize=22)
        for t, m in zip(ax.get_yticklabels(), rk):
            t.set_color(color(m))
        for t, m in zip(ax.get_xticklabels(), ck):
            t.set_color(color(m))
    colorbar(cv, x1 + n * S + 15, 55 + n * S - 40, 90, 6, SLATE, rho_norm,
             nice_ticks(rho_norm.vmin, 1.0, 0.05), "Spearman ρ")
    colorbar(cv, x2 + 2 * S + 5, 55 + n * S - 40, 90, 6, STONE, jac_norm,
             nice_ticks(jac_norm.vmin, jac_norm.vmax, 0.1), "Jaccard overlap")
    cv.note(x2 - 60, 25, f"{jac_vals.min():.0%}–{jac_vals.max():.0%} of the pairs one model "
                         "misranks, the other misranks too.", color=INK)
    cv.note(10, 10, "Lower triangle of ρ: rank correlation after regressing out the clinical "
                    "covariates (and diagnosis).", color=MUTED)
    cv.save(os.path.join(out, "M5_agreement"))


# ------------------------------------------------------------------ M6
def figure_discordance(R, out):
    gap = R.csv("discordance_by_gap")
    hard = R.csv("hard_biopsies")
    cv = Canvas(560, 300)
    cv.title("M6", "Where the models go wrong", R.caption)

    # left: share of comparable pairs misranked, by time between the two outcomes
    ax = cv.axes(55, 95, 190, 150)
    x = np.arange(len(gap))
    for m in R.models:
        ax.plot(x, gap[m] * 100, "-o", color=color(m), lw=3, ms=13, mec="white", mew=2, zorder=3)
    ax.set_xticks(x, [f"{g}\n{p / 1000:.0f}k" for g, p in zip(gap["time_gap"], gap["pairs"])],
                  linespacing=1.2)
    ax.tick_params(axis="x", length=0, pad=8)
    ax.set_xlim(-0.3, len(gap) - 0.7)
    ax.set_ylim(0, np.ceil(gap[R.models].to_numpy().max() * 10) * 10)
    ax.set_ylabel("Pairs misranked (%)")
    ax.set_xlabel("Time between the two outcomes (pairs)", labelpad=6)
    ygrid(ax)
    method_legend(cv, R.models, 45, 22, ncol=3)

    # right: the biopsies most often in a misranked pair
    ax = cv.axes(310, 95, 220, 150)
    dead = hard["event_type"].fillna("").eq("Death") & (hard["event"] == 1)
    kid = (hard["event"] == 1) & ~dead
    cens = hard["event"] == 0
    s = 60 + 900 * (hard["discordant_share"] - hard["discordant_share"].min()).clip(lower=0)
    pct = hard["mean_risk_percentile"] * 100
    ax.scatter(hard["time_years"][cens], pct[cens], s=s[cens], facecolor="white", edgecolor=MUTED,
               lw=2.5, zorder=3, clip_on=False)
    ax.scatter(hard["time_years"][kid], pct[kid], s=s[kid], color=KIDNEY, edgecolor="white", lw=1.5,
               zorder=4)
    ax.scatter(hard["time_years"][dead], pct[dead], s=s[dead], color=DEATH, edgecolor="white", lw=1.5,
               zorder=4)
    ax.axhline(50, color=LINE, lw=1.5, ls=(0, (2, 3)))
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlim(0, np.ceil(hard["time_years"].max() / 5) * 5)
    ax.set_xlabel("Follow-up (years)")
    ax.set_ylabel("Mean risk percentile")
    ygrid(ax)
    ax.set_title(f"{len(hard)} biopsies most often misranked", fontsize=24, color=MUTED, loc="left",
                 pad=14)
    cv.legend([Line2D([], [], marker="o", ls="none", ms=15, mfc="white", mec=MUTED, mew=2.5,
                      label="censored"),
               Line2D([], [], marker="o", ls="none", ms=15, color=KIDNEY, label="kidney failure"),
               Line2D([], [], marker="o", ls="none", ms=15, color=DEATH, label="death")], 300, 30)
    cv.note(310, 10, "Marker size: share of its pairs misranked.",
            color=MUTED)
    cv.save(os.path.join(out, "M6_discordance"))


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--input_dir", default="results/model_signal",
                    help="analyze_model_signal.py --output_dir")
    ap.add_argument("--output_dir", default=None, help="default: <input_dir>/figures")
    ap.add_argument("--only", nargs="+", choices=["M1", "M2", "M3", "M4", "M5", "M6"])
    ap.add_argument("--drop", nargs="*", default=["voting"],
                    help="model keys not shown (the C-index-weighted voting is ~identical to "
                         "unweighted 'average')")
    ap.add_argument("--split_name", default="Test set", help="caption prefix")
    ap.add_argument("--min_baseline_c", type=float, default=0.65,
                    help="M1: clinical baselines below this C-index are named in a note, not drawn")
    ap.add_argument("--max_groups", type=int, default=12, help="M4: largest diagnosis groups shown")
    args = ap.parse_args()
    out = args.output_dir or os.path.join(args.input_dir, "figures")
    os.makedirs(out, exist_ok=True)

    R = Results(args.input_dir, set(args.drop), args.split_name)
    makers = {
        "M1": lambda: figure_overview(R, out, args.min_baseline_c),
        "M2": lambda: figure_endpoints(R, out),
        "M3": lambda: figure_added_value(R, out),
        "M4": lambda: figure_within_diagnosis(R, out, args.max_groups),
        "M5": lambda: figure_agreement(R, out),
        "M6": lambda: figure_discordance(R, out),
    }
    for tag in args.only or makers:
        try:
            makers[tag]()
            print(f"{tag} done")
        except FileNotFoundError as err:
            print(f"{tag} skipped: {err} not found")
    print(f"Figures -> {out}")


if __name__ == "__main__":
    main()
