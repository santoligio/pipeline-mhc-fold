#!/usr/bin/env python3
"""
Create MHC-only PDBs.

For each primary PDB:
1. Keep only the MHC chain.
2. Renumber the MHC chain from 1.
3. Trim it to tstart/tend.
4. Rename it to chain A.
"""

from pathlib import Path

import pandas as pd
from Bio.PDB import MMCIFParser, PDBIO


# =========================
# Configuration
# =========================

PIPELINE_DIR = Path("/mnt/c/Users/gio/Documents/foldseek_nefertari/filter/ligands_pipeline")

INPUT_CIF_DIR = Path("/mnt/c/Users/gio/Documents/foldseek/version_02/filter/step2/pdb/1_assemblies")
INPUT_CSV = PIPELINE_DIR / "step2_1" / "pdb" / "filtered" / "pdb_assemblies_filtered.csv"

OUT_DIR = PIPELINE_DIR / "step3" / "pdb"
MHC_PDB_DIR = OUT_DIR / "1_mhc_only"

MHC_CHAIN_ID = "A"


# =========================
# Input table
# =========================

def load_primary_rows(csv_file: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_file)
    return df[df["status"].astype(str).str.lower() == "primary"].copy()


# =========================
# Structure editing
# =========================

def keep_only_chain(model, chain_id: str) -> None:
    for chain in list(model):
        if chain.id != chain_id:
            model.detach_child(chain.id)


def renumber_chain(chain) -> None:
    residues = list(chain.get_residues())

    for residue in residues:
        chain.detach_child(residue.id)

    for new_resseq, residue in enumerate(residues, start=1):
        hetflag, _, icode = residue.id
        residue.id = (hetflag, new_resseq, icode if icode else " ")
        chain.add(residue)


def trim_chain(chain, start_res: int, end_res: int) -> int:
    kept = 0

    for residue in list(chain.get_residues()):
        resseq = residue.id[1]

        if start_res <= resseq <= end_res:
            kept += 1
        else:
            chain.detach_child(residue.id)

    return kept


def process_structure(
    cif_path: Path,
    pdb_id: str,
    original_mhc_chain: str,
    tstart: int,
    tend: int,
) -> dict:
    parser = MMCIFParser(QUIET=True, auth_chains=True)
    structure = parser.get_structure(pdb_id, str(cif_path))
    model = structure[0]

    if original_mhc_chain not in model:
        return {
            "status": "skipped",
            "message": f"[SKIP] {pdb_id}: MHC chain {original_mhc_chain} not found",
        }

    keep_only_chain(model, original_mhc_chain)

    mhc_chain = model[original_mhc_chain]
    renumber_chain(mhc_chain)

    kept_residues = trim_chain(mhc_chain, tstart, tend)
    expected_residues = tend - tstart + 1

    if kept_residues == 0:
        return {
            "status": "failed",
            "message": f"[FAILED] {pdb_id}: no residues kept for range {tstart}-{tend}",
        }

    if kept_residues != expected_residues:
        return {
            "status": "failed",
            "message": (
                f"[FAILED] {pdb_id}: range mismatch "
                f"{kept_residues}/{expected_residues} for {tstart}-{tend}"
            ),
        }

    mhc_chain.id = MHC_CHAIN_ID

    out_pdb = MHC_PDB_DIR / f"{pdb_id.lower()}_mhc.pdb"

    io = PDBIO()
    io.set_structure(structure)
    io.save(str(out_pdb))

    return {
        "status": "saved",
        "message": f"[SAVED] {pdb_id}: {out_pdb.name}",
    }


# =========================
# Main
# =========================

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MHC_PDB_DIR.mkdir(parents=True, exist_ok=True)

    primary_df = load_primary_rows(INPUT_CSV)

    saved = 0
    skipped = 0

    for _, row in primary_df.iterrows():
        pdb_id = str(row["pdb"]).split("-")[0].upper()
        cif_path = INPUT_CIF_DIR / f"{pdb_id.lower()}-assembly1.cif"

        if not cif_path.exists():
            skipped += 1
            print(f"[MISSING] {pdb_id}: {cif_path}")
            continue

        result = process_structure(
            cif_path=cif_path,
            pdb_id=pdb_id,
            original_mhc_chain=str(row["chain"]),
            tstart=int(row["tstart"]),
            tend=int(row["tend"]),
        )

        print(result["message"])

        if result["status"] == "saved":
            saved += 1
        else:
            skipped += 1

    print(f"[DONE] saved={saved} skipped={skipped}")
    print(f"[OUT] {MHC_PDB_DIR}")


if __name__ == "__main__":
    main()
