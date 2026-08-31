#!/usr/bin/env python3
"""
Append whitelist/blacklist columns to unique_residues_filtered.csv.

Input:
- check_residues/unique_residues_filtered.csv
- check_residues/current_blacklist.csv

Merge key:
- resname

Output:
- check_residues/current_blacklist_frequency.csv
"""

from pathlib import Path

import pandas as pd


# =========================
# Configuration
# =========================

PIPELINE_DIR = Path(
    "/mnt/c/Users/gio/Documents/foldseek_nefertari/filter/ligands_pipeline"
)

CHECK_RESIDUES_DIR = PIPELINE_DIR / "check_residues" / "unique_residues"

UNIQUE_RESIDUES_FILTERED_CSV = CHECK_RESIDUES_DIR / "unique_residues_filtered.csv"
CURRENT_BLACKLIST_CSV = CHECK_RESIDUES_DIR / "current_blacklist.csv"

OUTPUT_CSV = CHECK_RESIDUES_DIR / "current_blacklist_frequency.csv"


# =========================
# Helpers
# =========================

def log(message: str) -> None:
    print(message, flush=True)


def normalize_resname(value) -> str:
    return str(value).strip().upper()


def validate_inputs() -> None:
    if not UNIQUE_RESIDUES_FILTERED_CSV.is_file():
        raise SystemExit(
            f"ERROR: UNIQUE_RESIDUES_FILTERED_CSV not found: {UNIQUE_RESIDUES_FILTERED_CSV}"
        )

    if not CURRENT_BLACKLIST_CSV.is_file():
        raise SystemExit(
            f"ERROR: CURRENT_BLACKLIST_CSV not found: {CURRENT_BLACKLIST_CSV}"
        )


# =========================
# Main
# =========================

def main() -> None:
    validate_inputs()

    unique_df = pd.read_csv(UNIQUE_RESIDUES_FILTERED_CSV)
    blacklist_df = pd.read_csv(CURRENT_BLACKLIST_CSV)

    if "resname" not in unique_df.columns:
        raise SystemExit("ERROR: unique_residues_filtered.csv must contain column 'resname'")

    required_blacklist = {"resname", "whitelist", "blacklist"}
    missing_blacklist = required_blacklist - set(blacklist_df.columns)
    if missing_blacklist:
        raise SystemExit(
            f"ERROR: current_blacklist.csv missing columns: {sorted(missing_blacklist)}"
        )

    unique_df = unique_df.copy()
    blacklist_df = blacklist_df.copy()

    unique_df["_resname_norm"] = unique_df["resname"].apply(normalize_resname)
    blacklist_df["_resname_norm"] = blacklist_df["resname"].apply(normalize_resname)

    # Keep one whitelist/blacklist annotation per residue.
    # If duplicated, keep the first occurrence from current_blacklist.csv.
    annotation_df = (
        blacklist_df[["_resname_norm", "whitelist", "blacklist"]]
        .drop_duplicates(subset=["_resname_norm"], keep="first")
    )

    merged_df = unique_df.merge(
        annotation_df,
        on="_resname_norm",
        how="left",
    )

    merged_df = merged_df.drop(columns=["_resname_norm"])

    # Make sure whitelist/blacklist are final columns.
    base_columns = [
        col for col in merged_df.columns
        if col not in {"whitelist", "blacklist"}
    ]
    merged_df = merged_df[base_columns + ["whitelist", "blacklist"]]

    merged_df.to_csv(OUTPUT_CSV, index=False)

    matched = merged_df["whitelist"].notna().sum() + merged_df["blacklist"].notna().sum()
    matched_resnames = merged_df[
        merged_df["whitelist"].notna() | merged_df["blacklist"].notna()
    ]["resname"].nunique()

    log(f"[INPUT] unique_residues_filtered_rows={len(unique_df)}")
    log(f"[INPUT] current_blacklist_rows={len(blacklist_df)}")
    log(f"[MERGED] resnames_with_whitelist_or_blacklist={matched_resnames}")
    log(f"[CSV] {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
