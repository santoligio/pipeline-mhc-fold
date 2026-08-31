#!/usr/bin/env python3
"""
Create rotated MHC complexes.

Reads original step2 assemblies, keeps chains near the MHC, renames chains,
renumber MHC chain, and applies the step4 alignment matrix.
"""

import string
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from Bio.PDB import MMCIFParser, PDBIO, NeighborSearch


# =========================
# Configuration
# =========================

PIPELINE_DIR = Path(__file__).resolve().parents[1]

INPUT_CIF_DIR = PIPELINE_DIR / "step2" / "pdb" / "1_assemblies"
INPUT_CSV = PIPELINE_DIR / "step2-1" / "pdb" / "filtered" / "pdb_assemblies_filtered.csv"

MATRIX_DIR = PIPELINE_DIR / "step4" / "pdb" / "1_aligned" / "matrices"

OUT_DIR = PIPELINE_DIR / "step5" / "pdb"
CONFIGURED_PDB_DIR = OUT_DIR / "1_mhc_configured"

CHAIN_MAP_CSV = OUT_DIR / "chain_map.csv"
REMAPPED_TABLE_CSV = OUT_DIR / "pdb_assemblies_remapped.csv"

MHC_DISTANCE_CUTOFF = 10.0
MHC_CHAIN_ID = "A"

OUTPUT_SUFFIX = "_mhc_configured.pdb"
MATRIX_PATTERN = "{pdb_id_lower}*_matrix.txt"

AUTH_CHAINS = True
PDB_CHAIN_IDS = string.ascii_uppercase


# =========================
# Setup
# =========================

def log(message: str) -> None:
    print(message, flush=True)


def make_output_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CONFIGURED_PDB_DIR.mkdir(parents=True, exist_ok=True)


def validate_inputs() -> None:
    log("[CONFIG] STEP5")
    log(f"[PATH] INPUT_CIF_DIR={INPUT_CIF_DIR}")
    log(f"[PATH] INPUT_CSV={INPUT_CSV}")
    log(f"[PATH] MATRIX_DIR={MATRIX_DIR}")
    log(f"[PATH] CONFIGURED_PDB_DIR={CONFIGURED_PDB_DIR}")
    log("")

    if not INPUT_CIF_DIR.is_dir():
        raise SystemExit(f"ERROR: INPUT_CIF_DIR does not exist: {INPUT_CIF_DIR}")

    if not INPUT_CSV.is_file():
        raise SystemExit(f"ERROR: INPUT_CSV does not exist: {INPUT_CSV}")

    if not MATRIX_DIR.is_dir():
        raise SystemExit(f"ERROR: MATRIX_DIR does not exist: {MATRIX_DIR}")


# =========================
# Input table
# =========================

def load_primary_rows(csv_file: Path):
    df = pd.read_csv(csv_file)

    required_cols = {"pdb", "chain", "status"}
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"ERROR: CSV is missing required columns: {sorted(missing)}")

    status_norm = df["status"].astype(str).str.strip().str.lower()
    primary_df = df[status_norm == "primary"].copy()

    log(f"[CSV] rows={len(df)} primary={len(primary_df)}")

    if primary_df.empty:
        raise SystemExit("ERROR: no primary rows found in CSV")

    mhc_map = {}
    for _, row in primary_df.iterrows():
        pdb_id = str(row["pdb"]).split("-")[0].upper()
        mhc_map[pdb_id] = row["chain"]

    return df, primary_df, mhc_map


# =========================
# MHC and chain editing
# =========================

def renumber_mhc_chain(chain) -> None:
    residues = list(chain.get_residues())

    for residue in residues:
        chain.detach_child(residue.id)

    for new_resseq, residue in enumerate(residues, start=1):
        hetflag, _, _icode = residue.id
        residue.id = (hetflag, new_resseq, " ")
        chain.add(residue)


def find_far_chains(structure, mhc_chain_id: str, cutoff: float = MHC_DISTANCE_CUTOFF) -> set:
    model = structure[0]

    if mhc_chain_id not in model:
        return set()

    mhc_chain = model[mhc_chain_id]
    mhc_ca_atoms = [atom for atom in mhc_chain.get_atoms() if atom.id == "CA"]

    if not mhc_ca_atoms:
        return set()

    ns = NeighborSearch(list(structure.get_atoms()))
    near_chain_ids = set()

    for atom in mhc_ca_atoms:
        for neighbor in ns.search(atom.coord, cutoff, level="A"):
            near_chain_ids.add(neighbor.get_parent().get_parent().id)

    far_chains = set(chain.id for chain in model) - near_chain_ids
    far_chains.discard(mhc_chain_id)

    return far_chains


def remove_far_chains(structure, mhc_chain_id: str, cutoff: float) -> set:
    model = structure[0]
    far_chains = find_far_chains(structure, mhc_chain_id, cutoff)

    for chain_id in far_chains:
        if chain_id in model:
            model.detach_child(chain_id)

    return far_chains


def rename_chains_to_temporary_ids(structure) -> Dict[str, str]:
    tmp_to_original = {}

    for model in structure:
        for chain in model:
            original_id = chain.id
            tmp_id = f"_TMP_{original_id}"
            chain.id = tmp_id
            tmp_to_original[tmp_id] = original_id

    return tmp_to_original


def pdb_chain_id_generator():
    for chain_id in PDB_CHAIN_IDS:
        yield chain_id


def remap_chain_ids_keep_mhc_as_A(
    structure,
    tmp_to_original: Dict[str, str],
    original_mhc_chain: str,
) -> Dict[str, str]:
    original_ids = sorted(tmp_to_original.values())
    other_ids = [chain_id for chain_id in original_ids if chain_id != original_mhc_chain]

    if len(original_ids) > len(PDB_CHAIN_IDS):
        raise ValueError(
            f"Too many remaining chains for PDB output: {len(original_ids)}"
        )

    gen = pdb_chain_id_generator()

    chain_map = {original_mhc_chain: MHC_CHAIN_ID}

    first = next(gen)
    if first != MHC_CHAIN_ID:
        raise RuntimeError("Unexpected PDB chain ID generator behavior")

    for chain_id in other_ids:
        chain_map[chain_id] = next(gen)

    for model in structure:
        for chain in model:
            if chain.id in tmp_to_original:
                original_id = tmp_to_original[chain.id]
                chain.id = chain_map[original_id]

    return chain_map


# =========================
# Matrix transform
# =========================

def find_matrix_for_pdb(pdb_id: str) -> Optional[Path]:
    pattern = MATRIX_PATTERN.format(pdb_id_lower=pdb_id.lower())
    matches = sorted(MATRIX_DIR.glob(pattern))

    if not matches:
        return None

    if len(matches) > 1:
        log(f"[WARNING] {pdb_id}: multiple matrices found; using {matches[0].name}")

    return matches[0]


def read_tmalign_matrix(matrix_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    lines = matrix_path.read_text().splitlines()

    start = None
    for idx, line in enumerate(lines):
        if "rotation matrix" in line.lower():
            start = idx + 1
            break

    if start is None:
        raise ValueError(f"Rotation matrix header not found: {matrix_path}")

    rows = []

    for line in lines[start:]:
        parts = line.strip().split()

        try:
            nums = [float(x) for x in parts]
        except ValueError:
            continue

        if len(nums) == 5:
            rows.append(nums[1:5])
        elif len(nums) == 4:
            rows.append(nums)

        if len(rows) == 3:
            break

    if len(rows) != 3:
        raise ValueError(f"Failed to read rotation rows: {matrix_path}")

    matrix = np.array(rows, dtype=float)
    t_vec = matrix[:, 0]
    u_mat = matrix[:, 1:4]

    return u_mat, t_vec


def check_rotation(u_mat: np.ndarray) -> Tuple[float, float]:
    det = float(np.linalg.det(u_mat))
    ortho_err = float(np.linalg.norm(u_mat @ u_mat.T - np.eye(3)))
    return det, ortho_err


def apply_tmalign_transform(structure, u_mat: np.ndarray, t_vec: np.ndarray) -> None:
    # TM-align: X = t + U*x. Biopython uses X = x*R + t.
    rotation = u_mat.T

    for atom in structure.get_atoms():
        atom.transform(rotation, t_vec)


# =========================
# Per-structure processing
# =========================

def process_structure(
    cif_path: Path,
    original_mhc_chain: str,
    out_dir: Path,
):
    pdb_id = cif_path.name.split("-")[0].upper()

    matrix_path = find_matrix_for_pdb(pdb_id)
    if matrix_path is None:
        log(f"[SKIPPED] {pdb_id}: no matrix found")
        return None

    parser = MMCIFParser(QUIET=True, auth_chains=AUTH_CHAINS)
    structure = parser.get_structure(pdb_id, str(cif_path))
    model = structure[0]

    if original_mhc_chain not in model:
        log(f"[SKIPPED] {pdb_id}: MHC chain {original_mhc_chain} not found")
        return None

    chain_count_before = len(list(model.get_chains()))

    renumber_mhc_chain(model[original_mhc_chain])

    far_chains = remove_far_chains(structure, original_mhc_chain, MHC_DISTANCE_CUTOFF)
    if far_chains:
        log(f"[FILTERED] {pdb_id}: removed_far_chains={sorted(far_chains)}")

    chain_count_after = len(list(model.get_chains()))

    tmp_to_original = rename_chains_to_temporary_ids(structure)
    chain_map = remap_chain_ids_keep_mhc_as_A(
        structure=structure,
        tmp_to_original=tmp_to_original,
        original_mhc_chain=original_mhc_chain,
    )

    u_mat, t_vec = read_tmalign_matrix(matrix_path)
    det, ortho_err = check_rotation(u_mat)

    if ortho_err > 1e-2:
        log(
            f"[WARNING] {pdb_id}: matrix may be non-orthonormal "
            f"det={det:.6f} ortho_err={ortho_err:.3e}"
        )

    apply_tmalign_transform(structure, u_mat, t_vec)

    out_pdb = out_dir / f"{pdb_id.lower()}{OUTPUT_SUFFIX}"

    io = PDBIO()
    io.set_structure(structure)
    io.save(str(out_pdb))

    log(
        f"[SAVED] {pdb_id}: chains={chain_count_before}->{chain_count_after} "
        f"matrix={matrix_path.name} output={out_pdb.name}"
    )

    return {
        "pdb": pdb_id,
        "chain_map": chain_map,
    }


# =========================
# Output tables
# =========================

def write_output_tables(
    df_csv: pd.DataFrame,
    mapping_rows: list,
) -> None:
    if mapping_rows:
        mapping_df = pd.DataFrame(mapping_rows)
        mapping_df.to_csv(CHAIN_MAP_CSV, index=False)
        log(f"[CSV] {CHAIN_MAP_CSV}")

    updated_rows = []

    for _, row in df_csv.iterrows():
        pdb_id = str(row["pdb"]).split("-")[0].upper()
        old_chain = row["chain"]

        matches = [
            mapping
            for mapping in mapping_rows
            if mapping["pdb"] == pdb_id and mapping["old_chain"] == old_chain
        ]

        if not matches:
            continue

        updated_rows.append({
            "pdb_id": pdb_id[:4].upper(),
            "chain": matches[0]["new_chain"],
            "tstart": row["tstart"],
            "tend": row["tend"],
        })

    if updated_rows:
        pd.DataFrame(
            updated_rows,
            columns=["pdb_id", "chain", "tstart", "tend"],
        ).to_csv(REMAPPED_TABLE_CSV, index=False)
        log(f"[CSV] {REMAPPED_TABLE_CSV}")


# =========================
# Main
# =========================

def main() -> None:
    make_output_dirs()
    validate_inputs()

    df_csv, df_primary, _mhc_map = load_primary_rows(INPUT_CSV)

    mapping_rows = []

    processed = 0
    missing_cif = 0
    skipped_or_failed = 0

    log("")
    log("[START] STEP5")

    for _, row in df_primary.iterrows():
        pdb_id = str(row["pdb"]).split("-")[0].upper()
        cif_path = INPUT_CIF_DIR / f"{pdb_id.lower()}-assembly1.cif"

        log(f"[PROCESSING] {pdb_id}")

        if not cif_path.exists():
            missing_cif += 1
            log(f"[MISSING] {pdb_id}: {cif_path}")
            continue

        try:
            result = process_structure(
                cif_path=cif_path,
                original_mhc_chain=str(row["chain"]),
                out_dir=CONFIGURED_PDB_DIR,
            )
        except Exception as exc:
            skipped_or_failed += 1
            log(f"[FAILED] {pdb_id}: {exc}")
            continue

        if result is None:
            skipped_or_failed += 1
            continue

        processed += 1

        for old_chain, new_chain in result["chain_map"].items():
            mapping_rows.append({
                "pdb": pdb_id,
                "old_chain": old_chain,
                "new_chain": new_chain,
            })

    write_output_tables(df_csv, mapping_rows)

    log("")
    log("[SUMMARY] STEP5")
    log(f"[SUMMARY] processed={processed}")
    log(f"[SUMMARY] missing_cif={missing_cif}")
    log(f"[SUMMARY] skipped_or_failed={skipped_or_failed}")
    log(f"[OUT] {CONFIGURED_PDB_DIR}")


if __name__ == "__main__":
    main()
