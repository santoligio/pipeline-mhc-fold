#!/usr/bin/env python3
"""
Remove selected chains from input CSV, writing new MHC complex structures
and updates step9_binders.csv / step9_ligands.csv.

"""

import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd
from Bio.PDB import PDBIO, PDBParser

# =========================================================
# Configuration
# =========================================================

PIPELINE_DIR = Path(__file__).resolve().parents[1]

DATABASE = "pdb"

STEP7_DIR = PIPELINE_DIR / "step7_fix" / DATABASE
STEP8_REMODELED_DIR = PIPELINE_DIR / "step8" / DATABASE / "mhc_remodeled"
STEP9_DIR = PIPELINE_DIR / "step9_fix" / DATABASE

INPUT_PDB_DIR = STEP7_DIR / "1_filtered_structures"
STEP7_SUMMARY_DIR = STEP7_DIR / "summaries"
STEP7_BINDERS_CSV = STEP7_SUMMARY_DIR / "step7_binders.csv"
STEP7_LIGANDS_CSV = STEP7_SUMMARY_DIR / "step7_ligands.csv"

MODIFIED_PDB_DIR = STEP9_DIR / "modified_pdbs"
MODIFICATION_LOG = MODIFIED_PDB_DIR / "modified_pdbs_log.csv"
EDITED_BINDERS_CSV = MODIFIED_PDB_DIR / "step9_binders.csv"
EDITED_LIGANDS_CSV = MODIFIED_PDB_DIR / "step9_ligands.csv"
BINDER_REMOVALS_CSV = PIPELINE_DIR / "binder_removals.csv"

LOG_COLUMNS = [
    "timestamp",
    "pdb_id",
    "removed_chain_id",
    "reason",
    "source",
    "status",
]

REQUIRED_INPUT_COLUMNS = {"pdb_id", "chain_id", "reason"}


# =========================================================
# Helpers
# =========================================================

def normalize_pdb_id(value) -> str:
    return str(value).strip().split("-")[0].upper()


def normalize_chain_id(value) -> str:
    return str(value).strip()


def find_step7_pdb(pdb_id: str) -> Path:
    prefix = pdb_id.lower()
    candidates = sorted(
        p for p in INPUT_PDB_DIR.glob("*.pdb")
        if p.stem.lower().startswith(prefix)
    )

    if not candidates:
        raise FileNotFoundError(f"no step7 PDB found for {pdb_id}")

    if len(candidates) > 1:
        print(
            f"[WARNING] {pdb_id}: {len(candidates)} step7 files found; "
            f"using {candidates[0].name}"
        )

    return candidates[0]


def resolve_source_pdb(pdb_id: str) -> Tuple[Path, str]:
    """
    Use the normal source priority before manual modifications exist:
      1. step8 remodeled, if present
      2. step7 filtered structure
    """
    prefix = pdb_id.lower()

    if STEP8_REMODELED_DIR.is_dir():
        remodeled = sorted(
            p for p in STEP8_REMODELED_DIR.glob("*.pdb")
            if p.stem.lower().startswith(prefix)
        )
        if remodeled:
            if len(remodeled) > 1:
                print(
                    f"[WARNING] {pdb_id}: {len(remodeled)} remodeled files found; "
                    f"using {remodeled[0].name}"
                )
            return remodeled[0], "remodeled"

    return find_step7_pdb(pdb_id), "step7"


def clean_existing_log_if_needed() -> None:
    """
    If an older log had path columns, permanently rewrite it with only LOG_COLUMNS.
    """
    if not MODIFICATION_LOG.exists():
        return

    old = pd.read_csv(MODIFICATION_LOG)
    for col in LOG_COLUMNS:
        if col not in old.columns:
            old[col] = ""
    old = old[LOG_COLUMNS]
    old.to_csv(MODIFICATION_LOG, index=False)


def append_log_rows(rows: List[dict]) -> None:
    """
    Append path-free log rows. No source_path or output_path is ever written.
    """
    MODIFIED_PDB_DIR.mkdir(parents=True, exist_ok=True)
    clean_existing_log_if_needed()

    write_header = not MODIFICATION_LOG.exists()

    with MODIFICATION_LOG.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_COLUMNS)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in LOG_COLUMNS})


def read_removal_csv(csv_path: Path) -> Dict[str, List[dict]]:
    if not csv_path.is_file():
        raise SystemExit(f"ERROR: input CSV does not exist: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = REQUIRED_INPUT_COLUMNS - set(df.columns)
    if missing:
        raise SystemExit(
            f"ERROR: input CSV missing columns: {sorted(missing)}. "
            f"Required columns: {sorted(REQUIRED_INPUT_COLUMNS)}"
        )

    grouped: Dict[str, List[dict]] = {}

    for row_number, row in df.iterrows():
        pdb_id = normalize_pdb_id(row["pdb_id"])
        chain_id = normalize_chain_id(row["chain_id"])
        reason = str(row["reason"]).strip()

        if not pdb_id or pdb_id == "NAN":
            print(f"[WARNING] row {row_number + 2}: empty pdb_id; skipping")
            continue
        if not chain_id or chain_id.lower() == "nan":
            print(f"[WARNING] row {row_number + 2}: empty chain_id for {pdb_id}; skipping")
            continue
        if not reason or reason.lower() == "nan":
            reason = "not provided"

        grouped.setdefault(pdb_id, [])

        # Avoid duplicate chain requests for the same PDB while preserving first reason.
        if any(item["chain_id"] == chain_id for item in grouped[pdb_id]):
            print(f"[WARNING] {pdb_id}: duplicate chain {chain_id} in input CSV; keeping first occurrence")
            continue

        grouped[pdb_id].append({
            "chain_id": chain_id,
            "reason": reason,
        })

    return grouped



def summary_pdb_column(df: pd.DataFrame, csv_label: str) -> str:
    """
    Find the PDB ID column in a step7 summary CSV.
    The current step7 summaries usually use 'pdb'.
    """
    for col in ("pdb", "pdb_id", "PDB", "PDB_ID"):
        if col in df.columns:
            return col
    raise SystemExit(
        f"ERROR: {csv_label} has no PDB ID column. "
        f"Expected one of: pdb, pdb_id, PDB, PDB_ID"
    )


def removal_pairs(removals_by_pdb: Dict[str, List[dict]]) -> Set[Tuple[str, str]]:
    """
    Return normalized (PDB_ID, chain_id) pairs from the requested removal list.
    """
    pairs: Set[Tuple[str, str]] = set()
    for pdb_id, removals in removals_by_pdb.items():
        for item in removals:
            pairs.add((normalize_pdb_id(pdb_id), normalize_chain_id(item["chain_id"])))
    return pairs


def write_edited_step7_summary(
    input_csv: Path,
    output_csv: Path,
    removals_by_pdb: Dict[str, List[dict]],
    csv_label: str,
) -> Tuple[int, int]:
    """
    Copy a step7 summary CSV while removing rows whose (pdb_id, chain_id)
    matches the removal list. The original columns are preserved.

    Returns:
        (original_row_count, removed_row_count)
    """
    if not input_csv.is_file():
        raise SystemExit(f"ERROR: missing {csv_label} CSV: {input_csv}")

    df = pd.read_csv(input_csv)
    original_count = len(df)

    if df.empty:
        MODIFIED_PDB_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        return original_count, 0

    if "chain_id" not in df.columns:
        raise SystemExit(f"ERROR: {csv_label} CSV has no chain_id column")

    pdb_col = summary_pdb_column(df, csv_label)
    pairs = removal_pairs(removals_by_pdb)

    remove_mask = df.apply(
        lambda row: (
            normalize_pdb_id(row[pdb_col]),
            normalize_chain_id(row["chain_id"]),
        ) in pairs,
        axis=1,
    )

    edited = df.loc[~remove_mask].copy()
    removed_count = int(remove_mask.sum())

    MODIFIED_PDB_DIR.mkdir(parents=True, exist_ok=True)
    edited.to_csv(output_csv, index=False)

    return original_count, removed_count


def write_edited_step7_summaries(removals_by_pdb: Dict[str, List[dict]]) -> None:
    """
    Create edited copies of step7_binders.csv and step7_ligands.csv after
    excluding any manually removed chains.
    """
    binders_total, binders_removed = write_edited_step7_summary(
        STEP7_BINDERS_CSV,
        EDITED_BINDERS_CSV,
        removals_by_pdb,
        "step7_binders",
    )
    ligands_total, ligands_removed = write_edited_step7_summary(
        STEP7_LIGANDS_CSV,
        EDITED_LIGANDS_CSV,
        removals_by_pdb,
        "step7_ligands",
    )

    print("[SUMMARY CSV] edited copies written")
    print(f"[SUMMARY CSV] step7_binders: rows={binders_total}, removed={binders_removed}")
    print(f"[SUMMARY CSV] step7_ligands: rows={ligands_total}, removed={ligands_removed}")

def remove_chains_for_pdb(pdb_id: str, removals: List[dict]) -> Tuple[int, int]:
    """
    Remove all requested chains for one PDB in one pass.

    Returns:
        (removed_chain_requests, missing_chain_requests)
    """
    source_path, source_label = resolve_source_pdb(pdb_id)
    requested_chains = [item["chain_id"] for item in removals]

    print(f"[SOURCE] {pdb_id}: using {source_label} file")
    print(f"[REQUEST] {pdb_id}: remove chains {','.join(requested_chains)}")

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, str(source_path))

    available_by_model = {model.id: [chain.id for chain in model] for model in structure}
    removed_by_chain = {chain_id: [] for chain_id in requested_chains}

    for model in structure:
        for chain_id in requested_chains:
            if chain_id in model:
                model.detach_child(chain_id)
                removed_by_chain[chain_id].append(model.id)

    timestamp = datetime.now().isoformat(timespec="seconds")
    log_rows = []
    removed_count = 0
    missing_count = 0

    for item in removals:
        chain_id = item["chain_id"]
        status = "removed" if removed_by_chain[chain_id] else "not_found"
        if status == "removed":
            removed_count += 1
        else:
            missing_count += 1

        log_rows.append({
            "timestamp": timestamp,
            "pdb_id": pdb_id,
            "removed_chain_id": chain_id,
            "reason": item["reason"],
            "source": source_label,
            "status": status,
        })

    append_log_rows(log_rows)

    if removed_count == 0:
        print(f"[ERROR] {pdb_id}: none of the requested chains were found; no modified PDB written")
        print("Available chains by model:")
        for model_id, chains in available_by_model.items():
            print(f"  model {model_id}: {','.join(chains) if chains else 'none'}")
        return removed_count, missing_count

    MODIFIED_PDB_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MODIFIED_PDB_DIR / f"{pdb_id.lower()}_mhc_complex.pdb"

    io = PDBIO()
    io.set_structure(structure)
    io.save(str(output_path))

    missing_chains = [chain for chain, models in removed_by_chain.items() if not models]
    if missing_chains:
        print(f"[WARNING] {pdb_id}: chains not found: {','.join(missing_chains)}")
        print("Available chains by model:")
        for model_id, chains in available_by_model.items():
            print(f"  model {model_id}: {','.join(chains) if chains else 'none'}")

    print(f"[OK] {pdb_id}: modified full complex written; removed={removed_count}, not_found={missing_count}")
    return removed_count, missing_count


# =========================================================
# Main
# =========================================================

def main() -> None:
    print("=== Prepare modified full complexes for step9 ===")
    print(f"Step7 input folder exists:     {INPUT_PDB_DIR.is_dir()}")
    print(f"Step8 remodeled folder exists: {STEP8_REMODELED_DIR.is_dir()}")
    print(f"Modified output folder:        {MODIFIED_PDB_DIR.name}")
    print(f"Input CSV:                     {BINDER_REMOVALS_CSV}")
    print("")

    if not INPUT_PDB_DIR.is_dir():
        raise SystemExit(f"ERROR: INPUT_PDB_DIR does not exist: {INPUT_PDB_DIR}")
    if not STEP7_BINDERS_CSV.is_file():
        raise SystemExit(f"ERROR: STEP7_BINDERS_CSV does not exist: {STEP7_BINDERS_CSV}")
    if not STEP7_LIGANDS_CSV.is_file():
        raise SystemExit(f"ERROR: STEP7_LIGANDS_CSV does not exist: {STEP7_LIGANDS_CSV}")

    removals_by_pdb = read_removal_csv(BINDER_REMOVALS_CSV)
    if not removals_by_pdb:
        raise SystemExit("ERROR: no valid removal rows found in input CSV")

    write_edited_step7_summaries(removals_by_pdb)
    print("")

    total_removed = 0
    total_missing = 0
    total_pdbs_with_output = 0

    for pdb_id, removals in sorted(removals_by_pdb.items()):
        try:
            removed, missing = remove_chains_for_pdb(pdb_id, removals)
        except FileNotFoundError as exc:
            timestamp = datetime.now().isoformat(timespec="seconds")
            append_log_rows([
                {
                    "timestamp": timestamp,
                    "pdb_id": pdb_id,
                    "removed_chain_id": item["chain_id"],
                    "reason": item["reason"],
                    "source": "not_found",
                    "status": "source_not_found",
                }
                for item in removals
            ])
            print(f"[ERROR] {pdb_id}: {exc}")
            removed, missing = 0, len(removals)

        total_removed += removed
        total_missing += missing
        if removed > 0:
            total_pdbs_with_output += 1

    print("")
    print("=== Pre-step9 summary ===")
    print(f"PDBs requested:        {len(removals_by_pdb)}")
    print(f"PDBs with output:      {total_pdbs_with_output}")
    print(f"Chain removals done:   {total_removed}")
    print(f"Chain requests missing:{total_missing}")
    print(f"Modified PDB folder:   {MODIFIED_PDB_DIR.name}")
    print(f"Path-free log:         {MODIFICATION_LOG.name}")
    print(f"Edited binders CSV:    {EDITED_BINDERS_CSV.name}")
    print(f"Edited ligands CSV:    {EDITED_LIGANDS_CSV.name}")
    print("[DONE] Pre-step9 finished")


if __name__ == "__main__":
    main()
