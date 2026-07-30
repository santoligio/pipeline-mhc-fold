#!/usr/bin/env python3
"""
Compare old and new step7_binders.csv files and identify binders that were
added in the new version.

A binder is identified by:
    PDB ID + chain ID

All columns from the new CSV are preserved in the output.

Usage:
    python3 compare_step7_binders.py old_step7_binders.csv new_step7_binders.csv

Optional output name:
    python3 compare_step7_binders.py old.csv new.csv \
        --out newly_added_binders.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"pdb", "chain_id"}


def normalize_pdb_id(value: object) -> str:
    """
    Normalize PDB IDs consistently with Step 7.

    Examples:
        4jrx   -> 4JRX
        4JRX-A -> 4JRX
    """
    if pd.isna(value):
        return ""

    return str(value).split("-")[0].strip().upper()


def normalize_chain_id(value: object) -> str:
    """
    Normalize chain IDs.

    Chain case is preserved because PDB chain IDs are case-sensitive.
    """
    if pd.isna(value):
        return ""

    return str(value).strip()


def load_binders(csv_path: Path, label: str) -> pd.DataFrame:
    """Load and validate one Step 7 binder CSV."""
    df = pd.read_csv(csv_path, keep_default_na=False)

    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            f"{label} CSV is missing required columns: {sorted(missing)}"
        )

    df = df.copy()

    df["_pdb_normalized"] = df["pdb"].map(normalize_pdb_id)
    df["_chain_normalized"] = df["chain_id"].map(normalize_chain_id)

    invalid_rows = (
        (df["_pdb_normalized"] == "")
        | (df["_chain_normalized"] == "")
    )

    if invalid_rows.any():
        count = int(invalid_rows.sum())
        print(
            f"[warning] {label}: ignoring {count} row(s) "
            "with blank PDB or chain ID"
        )
        df = df.loc[~invalid_rows].copy()

    duplicate_rows = df.duplicated(
        subset=["_pdb_normalized", "_chain_normalized"],
        keep="first",
    )

    if duplicate_rows.any():
        count = int(duplicate_rows.sum())
        print(
            f"[warning] {label}: ignoring {count} duplicate binder row(s)"
        )
        df = df.loc[~duplicate_rows].copy()

    return df


def find_added_binders(
    old_df: pd.DataFrame,
    new_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return binders present in the new CSV but absent from the old CSV.
    """
    old_keys = set(
        zip(
            old_df["_pdb_normalized"],
            old_df["_chain_normalized"],
        )
    )

    added_mask = [
        (pdb_id, chain_id) not in old_keys
        for pdb_id, chain_id in zip(
            new_df["_pdb_normalized"],
            new_df["_chain_normalized"],
        )
    ]

    added = new_df.loc[added_mask].copy()

    added = added.drop(
        columns=["_pdb_normalized", "_chain_normalized"]
    )

    if not added.empty:
        added["_sort_pdb"] = added["pdb"].map(normalize_pdb_id)
        added["_sort_chain"] = added["chain_id"].map(
            normalize_chain_id
        )

        added = (
            added.sort_values(["_sort_pdb", "_sort_chain"])
            .drop(columns=["_sort_pdb", "_sort_chain"])
            .reset_index(drop=True)
        )

    return added


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare old and new Step 7 binder CSV files and "
            "export newly added binders."
        )
    )

    parser.add_argument(
        "old_csv",
        help="Old step7_binders.csv file.",
    )

    parser.add_argument(
        "new_csv",
        help="New step7_binders.csv file.",
    )

    parser.add_argument(
        "--out",
        default="newly_added_step7_binders.csv",
        help=(
            "Output CSV name. "
            "Default: newly_added_step7_binders.csv"
        ),
    )

    args = parser.parse_args()

    old_path = Path(args.old_csv)
    new_path = Path(args.new_csv)
    output_path = Path(args.out)

    if not old_path.is_file():
        raise SystemExit(
            f"ERROR: old CSV not found: {old_path}"
        )

    if not new_path.is_file():
        raise SystemExit(
            f"ERROR: new CSV not found: {new_path}"
        )

    try:
        old_df = load_binders(old_path, "old")
        new_df = load_binders(new_path, "new")

        added_df = find_added_binders(
            old_df=old_df,
            new_df=new_df,
        )

    except Exception as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    added_df.to_csv(output_path, index=False)

    print()
    print("Comparison complete.")
    print(f"Old binders:       {len(old_df)}")
    print(f"New binders:       {len(new_df)}")
    print(f"Newly added:       {len(added_df)}")
    print(f"Output CSV:        {output_path}")

    if added_df.empty:
        print("\nNo new binders were found.")
        return

    display_columns = [
        column
        for column in [
            "pdb",
            "chain_id",
            "chain_residue_count",
            "min_distance_to_mhc_helix",
            "min_distance_to_ligand",
            "nearest_ligand_chain",
            "binder_contact_source",
            "classification",
        ]
        if column in added_df.columns
    ]

    print("\nNewly added binders:")
    print(
        added_df[display_columns].to_string(index=False)
    )


if __name__ == "__main__":
    main()
