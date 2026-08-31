#!/usr/bin/env python3
"""
Reorder new PDB MHC annotation files using the old edited annotation order.

Reference order:
- old pdb_mhc_annotations_edited.csv

Files reordered in-place by writing corrected copies:
- new pdb_mhc_annotations.csv
- new pdb_mhc_annotations_edited.csv

Rows whose pdb_id is not present in the old file are sent to the end.
"""

from pathlib import Path

import pandas as pd


# =========================
# Configuration
# =========================

PIPELINE_DIR = Path(__file__).resolve().parents[2]
WORKSPACE_DIR = PIPELINE_DIR.parent.parent

OLD_EDITED_CSV = (
    WORKSPACE_DIR
    / "version_02"
    / "analysis"
    / "functional_annotation"
    / "pdb"
    / "pdb_mhc_annotations_edited.csv"
)

NEW_ANNOTATION_DIR = PIPELINE_DIR / "step2-1" / "pdb"

NEW_FILES = [
    NEW_ANNOTATION_DIR / "pdb_mhc_annotations.csv",
    NEW_ANNOTATION_DIR / "pdb_mhc_annotations_edited.csv",
]

OUTPUT_SUFFIX = "_ordered"


# =========================
# Helpers
# =========================

def log(message: str) -> None:
    print(message, flush=True)


def normalize_pdb_id(value) -> str:
    return str(value).split("-")[0].strip().upper()


def validate_inputs() -> None:
    if not OLD_EDITED_CSV.is_file():
        raise SystemExit(f"ERROR: OLD_EDITED_CSV not found: {OLD_EDITED_CSV}")

    for path in NEW_FILES:
        if not path.is_file():
            raise SystemExit(f"ERROR: new annotation file not found: {path}")


def load_old_order(old_csv: Path) -> dict:
    old_df = pd.read_csv(old_csv)

    if "pdb_id" not in old_df.columns:
        raise SystemExit("ERROR: old edited CSV must contain column 'pdb_id'")

    order = {}

    for idx, pdb_id in enumerate(old_df["pdb_id"].dropna()):
        pdb_norm = normalize_pdb_id(pdb_id)

        # Keep first occurrence only.
        if pdb_norm not in order:
            order[pdb_norm] = idx

    return order


def reorder_annotation_file(path: Path, old_order: dict) -> Path:
    df = pd.read_csv(path)

    if "pdb_id" not in df.columns:
        raise SystemExit(f"ERROR: {path.name} must contain column 'pdb_id'")

    df = df.copy()
    df["_pdb_norm"] = df["pdb_id"].apply(normalize_pdb_id)
    df["_old_order"] = df["_pdb_norm"].map(old_order)
    df["_new_order"] = range(len(df))

    # Rows absent from old order go to the end, preserving their current order.
    missing_old = df["_old_order"].isna().sum()
    df["_old_order"] = df["_old_order"].fillna(len(old_order) + df["_new_order"])

    df = (
        df.sort_values(["_old_order", "_new_order"], kind="stable")
        .drop(columns=["_pdb_norm", "_old_order", "_new_order"])
    )

    out_path = path.with_name(f"{path.stem}{OUTPUT_SUFFIX}{path.suffix}")
    df.to_csv(out_path, index=False)

    log(f"[ORDERED] {path.name} -> {out_path.name}; new_only_rows={missing_old}")

    return out_path


# =========================
# Main
# =========================

def main() -> None:
    validate_inputs()

    old_order = load_old_order(OLD_EDITED_CSV)
    log(f"[REFERENCE] old_pdb_ids={len(old_order)}")

    for path in NEW_FILES:
        reorder_annotation_file(path, old_order)

    log("[DONE] Annotation ordering finished")


if __name__ == "__main__":
    main()
