"""
proteinuria.py — Proteinuria (g/24 h) at the first biopsy, as computed in proteinuria.R.

Reads the registry work file (xlsx) and writes a CSV with two columns:
    <patient id column>, Proteinuria_g_d

Steps (same as proteinuria.R):
  1. keep only the patient's first biopsy (Biopsy_no1 != 0)
  2. drop the dipstick value (Protein) for biopsies before 2000 (only yes/no then)
  3. remove extreme values:
        ProtKvan-g/d              > 12    g/day
        ProtKvan-g/l              > 4.7   g/L
        Alb\\kreat ratio mg/mmol   > 800
        Prot\\kreat ratio mg/mmol  > 1000
        Protein (dipstick)        > 4
  4. convert each measure to g/day:
        g/day        as is
        g/L          * 1.7 (expected daily urine volume)
        PCR          -> ACR (Weaver et al. 2020) -> * 0.01
        ACR          * 0.01
        dipstick     0 -> 0, 1 -> 0.65*1.7, 2 -> 2.0*1.7, 3/4 -> 3.5*1.7
  5. take the first available value in that order (g/day, g/L, PCR, ACR, dipstick)

Example:
    python followup_data/scripts/proteinuria.py arbeidsfil.xlsx proteinuria.csv
"""
import argparse

import numpy as np
import pandas as pd

ID_COL = "BiopsiID / PersonData.Lopenr"
G_D, G_L = "ProtKvan-g/d", "ProtKvan-g/l"
ACR, PCR = "Alb\\kreat ratio mg/mmol", "Prot\\kreat ratio mg/mmol"
DIPSTICK, YEAR, FIRST = "Protein", "Biopsiaar", "Biopsy_no1"


def upcr_to_uacr_weaver(pcr_mg_mmol):
    """U-PCR (mg/mmol) -> U-ACR (mg/mmol), Weaver et al. 2020 (PMC7062208)."""
    pcr = np.asarray(pcr_mg_mmol, dtype=float) * 8.84  # mg/mmol -> mg/g
    with np.errstate(divide="ignore", invalid="ignore"):
        log_pcr = np.log(pcr)
        log_acr = np.select(
            [pcr < 40, pcr < 60, pcr < 250, pcr < 1000, pcr >= 1000],
            [0.9518 + 0.1264 * log_pcr,
             -1.2568 + 0.7251 * log_pcr,
             -6.7837 + 2.0751 * log_pcr,
             -2.9649 + 1.3834 * log_pcr,
             -0.0239 + 0.9577 * log_pcr],
            default=np.nan,  # NaN PCR
        )
    return np.exp(log_acr) * 0.113  # mg/g -> mg/mmol


def proteinuria_g_d(df, id_col=ID_COL):
    num = lambda c: pd.to_numeric(df[c], errors="coerce")
    # Biopsy_no1 == 0 marks a rebiopsy; missing values are kept, as in R
    df = df[num(FIRST) != 0]  # num() now reads the filtered df

    g_d = num(G_D).where(lambda s: s <= 12)
    g_l = num(G_L).where(lambda s: s <= 4.7)
    acr = num(ACR).where(lambda s: s <= 800)
    pcr = num(PCR).where(lambda s: s <= 1000)
    dip = num(DIPSTICK).where((num(YEAR) >= 2000) & (num(DIPSTICK) <= 4))

    dip_g_d = dip.map({0: 0.0, 1: 0.65 * 1.7, 2: 2.0 * 1.7, 3: 3.5 * 1.7, 4: 3.5 * 1.7})
    pcr_g_d = pd.Series(upcr_to_uacr_weaver(pcr), index=df.index) * 0.01

    prot = g_d.fillna(g_l * 1.7).fillna(pcr_g_d).fillna(acr * 0.01).fillna(dip_g_d)
    return pd.DataFrame({id_col: df[id_col], "Proteinuria_g_d": prot})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("xlsx", help="registry work file (xlsx)")
    ap.add_argument("out_csv", help="output CSV")
    ap.add_argument("--sheet", default=0, help="sheet name or index (default: first)")
    ap.add_argument("--id_col", default=ID_COL, help=f"patient number column (default: {ID_COL!r})")
    args = ap.parse_args()

    sheet = int(args.sheet) if str(args.sheet).isdigit() else args.sheet
    df = pd.read_excel(args.xlsx, sheet_name=sheet,
                       usecols=[args.id_col, FIRST, YEAR, G_D, G_L, ACR, PCR, DIPSTICK])
    proteinuria_g_d(df, args.id_col).to_csv(args.out_csv, index=False)


if __name__ == "__main__":
    main()
