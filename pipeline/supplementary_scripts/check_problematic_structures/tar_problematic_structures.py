#!/usr/bin/env python3
"""
Create tar.gz archives for problematic structures from step6 and step5.

Reads:
- step6/pdb/summaries/problematic_list.csv

The problematic list must contain:
- pdb

Creates:
- check_problematic_structures/problematic_merged_cavities.tar.gz
- check_problematic_structures/problematic_mhc_configured.tar.gz
"""

from pathlib import Path
from typing import Optional
import tarfile

import pandas as pd


# =========================
# Configuration
# =========================

PIPELINE_DIR = Path(
    "/mnt/c/Users/gio/Documents/foldseek_nefertari/filter/ligands_pipeline"
)

PROBLEMATIC_LIST_CSV = (
    PIPELINE_DIR / "step6" / "pdb" / "summaries" / "problematic_list_current.csv"
)

MERGED_CAVITIES_DIR = PIPELINE_DIR / "step6" / "pdb" / "1_merged_cavities"
MHC_CONFIGURED_DIR = PIPELINE_DIR / "step5" / "pdb" / "1_mhc_configured"

OUT_DIR = PIPELINE_DIR / "check_problematic_structures"

MERGED_TAR_GZ = OUT_DIR / "problematic_merged_cavities.tar.gz"
CONFIGURED_TAR_GZ = OUT_DIR / "problematic_mhc_configured.tar.gz"

MISSING_CSV = OUT_DIR / "missing_problematic_structures.csv"


# =========================
# Helpers
# =========================

def log(message: str) -> None:
    print(message, flush=True)


def normalize_pdb_id(value) -> str:
    return str(value).split("-")[0].strip().upper()


def load_problematic_ids(path: Path) -> list[str]:
    df = pd.read_csv(path)

    if "pdb" not in df.columns:
        raise SystemExit("ERROR: problematic_list.csv must contain column 'pdb'")

    ids = []
    seen = set()

    for value in df["pdb"].dropna():
        pdb_id = normalize_pdb_id(value)

        if not pdb_id or pdb_id in seen:
            continue

        ids.append(pdb_id)
        seen.add(pdb_id)

    return ids


def find_structure(pdb_id: str, folder: Path, suffix: str) -> Optional[Path]:
    pdb_lower = pdb_id.lower()
    exact = folder / f"{pdb_lower}{suffix}"

    if exact.is_file():
        return exact

    matches = sorted(folder.glob(f"{pdb_lower}*.pdb"))
    return matches[0] if matches else None


def create_archive(
    pdb_ids: list[str],
    source_dir: Path,
    suffix: str,
    out_tar_gz: Path,
    missing_rows: list[dict],
    source_label: str,
) -> int:
    added = 0

    with tarfile.open(out_tar_gz, "w:gz") as tar:
        for pdb_id in pdb_ids:
            pdb_path = find_structure(pdb_id, source_dir, suffix)

            if pdb_path is None:
                missing_rows.append({
                    "pdb": pdb_id,
                    "source": source_label,
                    "folder": str(source_dir),
                    "expected_suffix": suffix,
                })
                continue

            tar.add(pdb_path, arcname=pdb_path.name)
            added += 1

    return added


# =========================
# Main
# =========================

def main() -> None:
    if not PROBLEMATIC_LIST_CSV.is_file():
        raise SystemExit(f"ERROR: PROBLEMATIC_LIST_CSV not found: {PROBLEMATIC_LIST_CSV}")

    if not MERGED_CAVITIES_DIR.is_dir():
        raise SystemExit(f"ERROR: MERGED_CAVITIES_DIR not found: {MERGED_CAVITIES_DIR}")

    if not MHC_CONFIGURED_DIR.is_dir():
        raise SystemExit(f"ERROR: MHC_CONFIGURED_DIR not found: {MHC_CONFIGURED_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pdb_ids = load_problematic_ids(PROBLEMATIC_LIST_CSV)
    missing_rows = []

    log(f"[INPUT] problematic_pdbs={len(pdb_ids)}")

    merged_added = create_archive(
        pdb_ids=pdb_ids,
        source_dir=MERGED_CAVITIES_DIR,
        suffix="_merged_cavities.pdb",
        out_tar_gz=MERGED_TAR_GZ,
        missing_rows=missing_rows,
        source_label="merged_cavities",
    )

    configured_added = create_archive(
        pdb_ids=pdb_ids,
        source_dir=MHC_CONFIGURED_DIR,
        suffix="_mhc_configured.pdb",
        out_tar_gz=CONFIGURED_TAR_GZ,
        missing_rows=missing_rows,
        source_label="mhc_configured",
    )

    pd.DataFrame(
        missing_rows,
        columns=["pdb", "source", "folder", "expected_suffix"],
    ).to_csv(MISSING_CSV, index=False)

    log(f"[TAR] {MERGED_TAR_GZ} | files={merged_added}")
    log(f"[TAR] {CONFIGURED_TAR_GZ} | files={configured_added}")
    log(f"[CSV] {MISSING_CSV} | missing={len(missing_rows)}")
    log("[DONE] Problematic structure archives created")


if __name__ == "__main__":
    main()
