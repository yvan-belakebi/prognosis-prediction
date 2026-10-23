"""
registry_events.py — Build the per-biopsy survival table of the kidney biopsy registry.

Reads the raw registry exports and writes, per *biopsy* (not per patient):

    followup_data/derived/registry/registry_currated.csv   one row per registry biopsy
    followup_data/derived/renamed/registry_anonymized.csv  one row per slide (WSI), with
        the outcome of the slide's own biopsy — the input of define_labels.py

Outcome of a biopsy:
    Event        the first kidney-replacement event *after the biopsy date*
                 (registry_events.xlsx: Dialyse, CKD5, NyreTx), or Death when the
                 patient dies before any such event; empty when censored
    Event_date   date of that event
    time_to_event  Event_date - Biopsy_date, or --censor_date - Biopsy_date when
                 censored, written as "<days> days"

Why per biopsy.  A patient can have up to five biopsies in the registry. Each slide
is matched to its own biopsy through the biopsy number, so every biopsy gets its own
date, diagnosis, eGFR and time to event. (The previous version kept one row per
patient and gave every slide of that patient the same biopsy date and outcome.)

Matching slides to registry biopsies.  The slide tables write the biopsy number as
LAB+yy+nnnnn (e.g. lab letters, year 16, number 12898), the registry also as
"LAB nnnnn/yy" with or without the lab letters, so both are reduced to (year,
number) and matched within the same patient. A slide biopsy that does not match is
still assigned when its patient has exactly one biopsy in both sources. Anything
else is left out and counted.

Excluded biopsies (counted in the summary):
    * a kidney-replacement event on or before the biopsy date (prevalent dialysis /
      transplant: not at risk of a first event) — keep them with --keep_prior_krt
    * time_to_event <= --min_days (default 60, as before)

Privacy.  registry_anonymized.csv identifies patients only by AnonymizedID; the
national ID number is no longer written to it (it was, as ID_diagnosis).
registry_currated.csv is an internal table and keeps the registry's patient ID.

Run from the project root:
    python followup_data/scripts/registry_events.py --censor_date 2025-05-12
"""

import argparse
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

RAW = "followup_data/raw/registry/"


def biopsy_key(number):
    """(two-digit year, number) from either biopsy-number format, or None."""
    s = str(number).strip()
    m = re.fullmatch(r"[A-Za-z]{0,3}\s*(\d{1,6})/(\d{2})", s)  # LAB nnnnn/yy
    if m:
        return m.group(2), int(m.group(1))
    m = re.fullmatch(r"[A-Za-z]{0,3}(\d{2})(\d{3,5})", s)  # LAB yy nnnnn
    if m:
        return m.group(1), int(m.group(2))
    return None


def load_biopsies():
    df = pd.read_excel(RAW + "registry_full_data.xlsx")
    df = df.rename(
        columns={
            "PersNummer": "patient",
            "Biopsidato": "Biopsy_date",
            "Biopsinr": "registry_biopsy_number",
            "Diagnoser P1.2013_konklusiv_diagnose": "Diagnosis",
            "Beregnet_eGFR_felles": "eGFR",
            "TBiopsi": "transplant_biopsy",
        }
    )
    df = df[
        [
            "patient",
            "Biopsy_date",
            "registry_biopsy_number",
            "Diagnosis",
            "eGFR",
            "transplant_biopsy",
        ]
    ]
    df = df.dropna(subset=["patient"]).copy()
    df["Biopsy_date"] = pd.to_datetime(df["Biopsy_date"], errors="coerce")
    df["is_IgA"] = df["Diagnosis"].str.contains("IgA nefropati", case=False, na=False)
    df["key"] = df["registry_biopsy_number"].map(biopsy_key)
    return df.reset_index(drop=True)


def load_treatments():
    ev = pd.read_excel(RAW + "registry_events.xlsx").rename(
        columns={
            "FNR": "patient",
            "BehandlingStart": "Treatment_date",
            "Behandling": "Treatment",
        }
    )
    ev = ev[["patient", "Treatment_date", "Treatment"]].dropna()
    ev["Treatment_date"] = pd.to_datetime(ev["Treatment_date"], errors="coerce")
    return ev.dropna(subset=["Treatment_date"])


def load_deaths():
    de = pd.read_excel(RAW + "registry_death.xlsx").rename(
        columns={
            "FNR": "patient",
            "Morsdate": "Death_date",
            "MorsReasonEDTA": "Death_code",
        }
    )
    de = de[["patient", "Death_date", "Death_code"]].dropna(subset=["Death_date"])
    de["Death_date"] = pd.to_datetime(de["Death_date"], errors="coerce")
    # A patient can appear on several rows; keep the earliest recorded death.
    return de.sort_values("Death_date").drop_duplicates("patient")


def per_biopsy_outcomes(bx, ev, de, censor_date):
    """First event after each biopsy, the prior-KRT flag, and the time to event."""
    pairs = bx[["patient", "Biopsy_date"]].reset_index().merge(ev, on="patient")
    prior = pairs[pairs.Treatment_date <= pairs.Biopsy_date].groupby("index").size()
    after = pairs[
        (pairs.Treatment_date > pairs.Biopsy_date)
        & (pairs.Treatment_date <= censor_date)
    ]
    first = (
        after.sort_values("Treatment_date").drop_duplicates("index").set_index("index")
    )

    out = bx.copy()
    out["prior_krt"] = out.index.isin(prior.index)
    out["Treatment_date"] = first["Treatment_date"]
    out["Treatment"] = first["Treatment"]
    out = out.merge(de, on="patient", how="left")
    death_ok = (out.Death_date > out.Biopsy_date) & (out.Death_date <= censor_date)
    out.loc[~death_ok, "Death_date"] = pd.NaT
    out.loc[~death_ok, "Death_code"] = float("nan")

    out["Event"] = out["Treatment"]
    out["Event_date"] = out["Treatment_date"]
    # Death is the event when it comes first (or there is no treatment event).
    death_first = out.Death_date.notna() & (
        out.Treatment_date.isna() | (out.Death_date < out.Treatment_date)
    )
    out.loc[death_first, "Event"] = "Death"
    out.loc[death_first, "Event_date"] = out.loc[death_first, "Death_date"]
    end = out["Event_date"].fillna(censor_date)
    out["time_days"] = (end - out["Biopsy_date"]).dt.days
    return out


def match_slides(bx, wsi):
    """Registry biopsy row of every slide biopsy: by (patient, year, number), else
    the patient's only biopsy when both sources have exactly one."""
    keys = bx.dropna(subset=["key"]).groupby(["patient", "key"]).groups
    unique = {k: v[0] for k, v in keys.items() if len(v) == 1}
    n_reg = bx.groupby("patient").size()
    slide_bx = wsi.drop_duplicates(["patient", "biop_number"])
    n_wsi = slide_bx.groupby("patient").size()
    only = bx.groupby("patient").apply(lambda g: g.index[0], include_groups=False)

    rows, how = [], []
    for p, b in zip(slide_bx.patient, slide_bx.biop_number):
        k = (p, biopsy_key(b))
        if k in unique:
            rows.append(unique[k])
            how.append("number")
        elif n_reg.get(p, 0) == 1 and n_wsi[p] == 1:
            rows.append(only[p])
            how.append("single biopsy")
        else:
            rows.append(pd.NA)
            how.append(
                "patient not in registry" if p not in n_reg.index else "ambiguous"
            )
    slide_bx = slide_bx.assign(row=rows, registry_match=how)
    counts = slide_bx.registry_match.value_counts()
    print("Slide biopsies matched to a registry biopsy:", counts.to_dict())
    return wsi.merge(
        slide_bx[["patient", "biop_number", "row", "registry_match"]],
        on=["patient", "biop_number"],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument(
        "--censor_date",
        default=None,
        help="End of follow-up for censored biopsies (YYYY-MM-DD): the date "
        "of the registry extract. Default: the latest event or death "
        "date in the exports, with a warning.",
    )
    p.add_argument(
        "--min_days",
        type=int,
        default=60,
        help="Drop biopsies whose time to event or censoring is <= this.",
    )
    p.add_argument(
        "--keep_prior_krt",
        action="store_true",
        help="Keep biopsies taken after a kidney-replacement event.",
    )
    p.add_argument(
        "--curated_csv", default="followup_data/derived/registry/registry_currated.csv"
    )
    p.add_argument(
        "--anonymized_csv",
        default="followup_data/derived/renamed/registry_anonymized.csv",
    )
    p.add_argument(
        "--plot", default="followup_data/derived/registry/times_to_event.png"
    )
    args = p.parse_args()

    bx, ev, de = load_biopsies(), load_treatments(), load_deaths()
    if args.censor_date:
        censor_date = pd.Timestamp(args.censor_date)
    else:
        censor_date = max(ev.Treatment_date.max(), de.Death_date.max())
        print(
            f"Warning: no --censor_date; using the latest event/death date "
            f"{censor_date.date()}. Pass the registry extract date instead."
        )

    out = per_biopsy_outcomes(bx, ev, de, censor_date)
    n = len(out)
    drop_prior = out.prior_krt & ~args.keep_prior_krt
    drop_short = out.time_days <= args.min_days
    print(
        f"Registry biopsies: {n} | prior kidney-replacement event: {out.prior_krt.sum()}"
        f"{' (kept)' if args.keep_prior_krt else ' (dropped)'} | time <= "
        f"{args.min_days} days: {drop_short.sum()} (dropped)"
    )
    out = out[~drop_prior & ~drop_short].copy()
    out["time_to_event"] = out["time_days"].astype(int).astype(str) + " days"
    print(
        f"Kept {len(out)} biopsies of {out.patient.nunique()} patients, events: "
        f"{out.Event.value_counts().to_dict()}, censored at {censor_date.date()}"
    )

    # Histogram of follow-up for the IgA biopsies, as before.
    plt.figure(figsize=(8, 6))
    plt.hist(
        out.loc[out.is_IgA, "time_days"], bins=20, color="#4C72B0", edgecolor="black"
    )
    plt.xlabel("Time to event (days)")
    plt.ylabel("Count")
    plt.title("Distribution of Time to Event for IgA Biopsies")
    plt.tight_layout()
    plt.savefig(args.plot)
    plt.close()

    curated_cols = [
        "patient",
        "Biopsy_date",
        "registry_biopsy_number",
        "Diagnosis",
        "eGFR",
        "is_IgA",
        "transplant_biopsy",
        "prior_krt",
        "Treatment_date",
        "Treatment",
        "Death_date",
        "Death_code",
        "Event",
        "Event_date",
        "time_to_event",
    ]
    out[curated_cols].to_csv(args.curated_csv, index=False)

    # Slides, each joined to its own biopsy.
    wsi = pd.concat(
        [
            pd.read_excel(RAW + f"{name}_with_filenames_new_stain_all_found_fix.xlsx")[
                [
                    "patient_fnr",
                    "biop_number",
                    "ANON_name",
                    "Stain",
                    "patient_id",
                    "biop_id",
                ]
            ]
            for name in ("NKBR", "KB")
        ],
        ignore_index=True,
    ).rename(columns={"patient_fnr": "patient"})
    wsi = match_slides(bx, wsi)
    wsi = wsi.dropna(subset=["row"]).astype({"row": int})
    wsi = wsi[wsi.row.isin(out.index)]  # biopsy kept after the exclusions
    reg = wsi.join(out.drop(columns=["patient"]), on="row")

    cols = [
        "Biopsy_date",
        "Diagnosis",
        "eGFR",
        "is_IgA",
        "transplant_biopsy",
        "Treatment_date",
        "Treatment",
        "Death_date",
        "Death_code",
        "Event",
        "Event_date",
        "time_to_event",
        "biop_id",
        "ANON_name",
        "Stain",
        "registry_match",
        "patient_id",
    ]
    reg[cols].to_csv(args.anonymized_csv, index=False)
    print(
        f"Wrote {len(reg)} slides of {reg.biop_number.nunique()} biopsies of "
        f"{len(reg.patient_id.unique())} patients -> {args.anonymized_csv}"
    )


if __name__ == "__main__":
    main()
