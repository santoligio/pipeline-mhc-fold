#!/usr/bin/env python3
"""
Filter unique residues using current pipeline structures.

Main source:
- step6/pdb/2_trimmed_mhc

Fallback for missing step6 outputs:
- step5/pdb/1_mhc_configured
- MHC chain A is filtered using tstart/tend
- MHC residues listed in problematic_mhc_contacts.csv are also retained

The input unique_residues.csv is expected to list residue names from step2.
The filtered output includes residue frequency in current structures.
"""

from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa


# =========================
# Configuration
# =========================

PIPELINE_DIR = Path(__file__).resolve().parents[3]

CHECK_DIR = PIPELINE_DIR / "check_residues" / "unique_residues"

REFERENCE_RESIDUES_CSV = CHECK_DIR / "unique_residues.csv"

STEP6_TRIMMED_DIR = PIPELINE_DIR / "step6" / "pdb" / "2_trimmed_mhc"
STEP5_CONFIGURED_DIR = PIPELINE_DIR / "step5" / "pdb" / "1_mhc_configured"

STEP5_REMAP_CSV = PIPELINE_DIR / "step5" / "pdb" / "pdb_assemblies_remapped.csv"
PROBLEMATIC_CONTACTS_CSV = (
    PIPELINE_DIR / "step6" / "pdb" / "summaries" / "problematic_mhc_contacts.csv"
)

OUTPUT_FILTERED_CSV = CHECK_DIR / "unique_residues_filtered.csv"
OUTPUT_REMOVED_CSV = CHECK_DIR / "unique_residues_removed.csv"
OUTPUT_MISSING_STRUCTURES_CSV = CHECK_DIR / "missing_structures.csv"

MHC_CHAIN_ID = "A"


# =========================
# Helpers
# =========================

def log(message: str) -> None:
    print(message, flush=True)


def normalize_pdb_id(value) -> str:
    return str(value).split("-")[0].strip().upper()


def residue_resseq_int(residue) -> Optional[int]:
    try:
        return int(residue.id[1])
    except (TypeError, ValueError):
        return None


def load_reference_residues(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False)

    if "resname" not in df.columns:
        raise SystemExit("ERROR: unique_residues.csv must contain column 'resname'")

    df = df.copy()
    df["resname"] = df["resname"].astype(str).str.strip().str.upper()
    df = df[df["resname"] != ""].copy()

    return df


def load_trim_ranges(path: Path) -> Dict[str, Tuple[int, int]]:
    df = pd.read_csv(path)

    required = {"pdb_id", "tstart", "tend"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"ERROR: STEP5_REMAP_CSV missing columns: {sorted(missing)}")

    ranges = {}

    for _, row in df.iterrows():
        pdb_id = normalize_pdb_id(row["pdb_id"])
        ranges[pdb_id] = (int(row["tstart"]), int(row["tend"]))

    return ranges


def load_problematic_resseqs(path: Path) -> Dict[str, Set[int]]:
    if not path.is_file():
        log(f"[WARNING] problematic contacts file not found: {path}")
        return {}

    df = pd.read_csv(path)

    if df.empty:
        return {}

    required = {"pdb", "resseq"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(
            f"ERROR: problematic_mhc_contacts.csv missing columns: {sorted(missing)}"
        )

    grouped: Dict[str, Set[int]] = {}

    for _, row in df.iterrows():
        pdb_id = normalize_pdb_id(row["pdb"])

        try:
            resseq = int(row["resseq"])
        except (TypeError, ValueError):
            continue

        grouped.setdefault(pdb_id, set()).add(resseq)

    return grouped


def find_step6_structure(pdb_id: str) -> Optional[Path]:
    pdb_lower = pdb_id.lower()
    exact = STEP6_TRIMMED_DIR / f"{pdb_lower}_trimmed_mhc.pdb"

    if exact.is_file():
        return exact

    matches = sorted(STEP6_TRIMMED_DIR.glob(f"{pdb_lower}*.pdb"))
    return matches[0] if matches else None


def find_step5_structure(pdb_id: str) -> Optional[Path]:
    pdb_lower = pdb_id.lower()
    exact = STEP5_CONFIGURED_DIR / f"{pdb_lower}_mhc_configured.pdb"

    if exact.is_file():
        return exact

    matches = sorted(STEP5_CONFIGURED_DIR.glob(f"{pdb_lower}*.pdb"))
    return matches[0] if matches else None


def collect_expected_pdb_ids(trim_ranges: Dict[str, Tuple[int, int]]) -> Set[str]:
    return set(trim_ranges.keys())


def should_keep_step5_residue(
    residue,
    tstart: int,
    tend: int,
    problematic_resseqs: Set[int],
) -> Tuple[bool, str]:
    chain_id = residue.get_parent().id

    if chain_id != MHC_CHAIN_ID:
        return True, "non_mhc_chain"

    resseq = residue_resseq_int(residue)

    if resseq is None:
        return False, "mhc_non_integer_resseq_removed"

    if tstart <= resseq <= tend:
        return True, "mhc_inside_tstart_tend"

    if resseq in problematic_resseqs:
        return True, "mhc_problematic_contact"

    return False, "mhc_outside_tstart_tend_removed"


def collect_residues_from_structure(
    pdb_id: str,
    pdb_path: Path,
    source: str,
    tstart: Optional[int] = None,
    tend: Optional[int] = None,
    problematic_resseqs: Optional[Set[int]] = None,
) -> list:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, str(pdb_path))

    rows = []
    problematic_resseqs = problematic_resseqs or set()

    for model in structure:
        for chain in model:
            chain_id = chain.id

            for residue in chain:
                if is_aa(residue, standard=True):
                    continue

                keep_reason = "present_in_step6_trimmed"

                if source == "step5_fallback":
                    if tstart is None or tend is None:
                        continue

                    keep, keep_reason = should_keep_step5_residue(
                        residue=residue,
                        tstart=tstart,
                        tend=tend,
                        problematic_resseqs=problematic_resseqs,
                    )

                    if not keep:
                        continue

                resseq = residue_resseq_int(residue)
                resname = residue.get_resname().strip().upper()

                rows.append({
                    "resname": resname,
                    "pdb_id": pdb_id,
                    "chain": chain_id,
                    "resseq": resseq if resseq is not None else residue.id[1],
                    "source": source,
                    "keep_reason": keep_reason,
                    "structure_file": pdb_path.name,
                })

    return rows


# =========================
# Main
# =========================

def main() -> None:
    if not REFERENCE_RESIDUES_CSV.is_file():
        raise SystemExit(f"ERROR: REFERENCE_RESIDUES_CSV not found: {REFERENCE_RESIDUES_CSV}")

    if not STEP5_REMAP_CSV.is_file():
        raise SystemExit(f"ERROR: STEP5_REMAP_CSV not found: {STEP5_REMAP_CSV}")

    if not STEP6_TRIMMED_DIR.is_dir():
        raise SystemExit(f"ERROR: STEP6_TRIMMED_DIR not found: {STEP6_TRIMMED_DIR}")

    if not STEP5_CONFIGURED_DIR.is_dir():
        raise SystemExit(f"ERROR: STEP5_CONFIGURED_DIR not found: {STEP5_CONFIGURED_DIR}")

    CHECK_DIR.mkdir(parents=True, exist_ok=True)

    reference_df = load_reference_residues(REFERENCE_RESIDUES_CSV)
    trim_ranges = load_trim_ranges(STEP5_REMAP_CSV)
    problematic_by_pdb = load_problematic_resseqs(PROBLEMATIC_CONTACTS_CSV)

    expected_pdb_ids = collect_expected_pdb_ids(trim_ranges)

    observed_rows = []
    missing_rows = []

    for pdb_id in sorted(expected_pdb_ids):
        step6_pdb = find_step6_structure(pdb_id)

        if step6_pdb is not None:
            observed_rows.extend(
                collect_residues_from_structure(
                    pdb_id=pdb_id,
                    pdb_path=step6_pdb,
                    source="step6_trimmed",
                )
            )
            continue

        step5_pdb = find_step5_structure(pdb_id)

        if step5_pdb is None:
            missing_rows.append({
                "pdb_id": pdb_id,
                "missing_from": "step6_trimmed_and_step5_configured",
            })
            continue

        tstart, tend = trim_ranges[pdb_id]

        observed_rows.extend(
            collect_residues_from_structure(
                pdb_id=pdb_id,
                pdb_path=step5_pdb,
                source="step5_fallback",
                tstart=tstart,
                tend=tend,
                problematic_resseqs=problematic_by_pdb.get(pdb_id, set()),
            )
        )

    observed_df = pd.DataFrame(
        observed_rows,
        columns=[
            "resname",
            "pdb_id",
            "chain",
            "resseq",
            "source",
            "keep_reason",
            "structure_file",
        ],
    )

    if observed_df.empty:
        observed_resnames = set()
    else:
        observed_df["resname"] = observed_df["resname"].astype(str).str.strip().str.upper()
        observed_resnames = set(observed_df["resname"])

    filtered_df = reference_df[reference_df["resname"].isin(observed_resnames)].copy()
    removed_df = reference_df[~reference_df["resname"].isin(observed_resnames)].copy()

    if not observed_df.empty:
        residue_frequency = (
            observed_df
            .groupby("resname", as_index=False)
            .size()
            .rename(columns={"size": "frequency"})
        )

        first_seen = (
            observed_df
            .sort_values(["resname", "pdb_id", "chain", "resseq"], kind="stable")
            .drop_duplicates(subset=["resname"], keep="first")
            [["resname", "pdb_id", "chain", "resseq"]]
        )

        filtered_df = filtered_df.merge(residue_frequency, on="resname", how="left")
        filtered_df = filtered_df.merge(first_seen, on="resname", how="left")
    else:
        filtered_df["frequency"] = 0

    filtered_df.to_csv(OUTPUT_FILTERED_CSV, index=False)
    removed_df.to_csv(OUTPUT_REMOVED_CSV, index=False)
    pd.DataFrame(missing_rows).to_csv(OUTPUT_MISSING_STRUCTURES_CSV, index=False)

    log(f"[REFERENCE] unique_residues={len(reference_df)}")
    log(f"[OBSERVED] residue_instances={len(observed_df)}")
    log(f"[FILTERED] kept_unique_residues={len(filtered_df)}")
    log(f"[FILTERED] removed_unique_residues={len(removed_df)}")
    log(f"[MISSING] structures={len(missing_rows)}")
    log(f"[CSV] {OUTPUT_FILTERED_CSV}")
    log(f"[CSV] {OUTPUT_REMOVED_CSV}")
    log(f"[CSV] {OUTPUT_MISSING_STRUCTURES_CSV}")


if __name__ == "__main__":
    main()
