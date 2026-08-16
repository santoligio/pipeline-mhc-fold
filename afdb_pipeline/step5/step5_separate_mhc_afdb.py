#!/usr/bin/env python3
"""
Create MHC-only structures from downloaded AFDB models (step5 of the AFDB
pipeline, final step).

AFDB pipeline order:
    step1 (select) -> step2 (annotate) -> step3 (filter) -> step4 (download)
    -> step5 (this script)

For each primary model:
1. Identify the model's chain.
2. Renumber that chain from 1.
3. Trim it to tstart/tend.
4. Rename it to chain A.

step1 always records "NoChainInfo" as the chain for AFDB rows, since an AFDB
model is already a single-chain prediction -- there's no author chain ID to
match against. This script auto-detects the model's chain instead, and skips
(rather than guesses) if a model unexpectedly has more than one chain.

DATASET SELECTION: this script does NOT define its own
AFDB_DATASET_SELECTION toggle. It imports the value straight from
step4_download_structures_afdb.py, so it always processes whichever option
step4 actually downloaded -- there's only one place to change the toggle,
and step5 can no longer silently drift onto a different subset than step4
used (that mismatch is what caused [MISSING] rows before).

Because different dataset options can overlap in which models they include,
outputs are written under a subfolder named after the selected option
(step5_mhc_only/<option>/), so running step5 once per option keeps each
option's trimmed structures separate instead of mixing them in one folder.
"""

from pathlib import Path
from typing import Optional

import pandas as pd
from Bio.PDB import MMCIFParser, PDBIO

from afdb_dataset_config import DATASET_MODEL_FILENAMES, STEP3_DIR, STEP4_DIR, STEP5_DIR, describe
from step4_download_structures_afdb import AFDB_DATASET_SELECTION


# =========================
# Configuration
# =========================

INPUT_CIF_DIR = STEP4_DIR / "1_models"
INPUT_CSV = STEP3_DIR / DATASET_MODEL_FILENAMES[AFDB_DATASET_SELECTION]

# One subfolder per dataset option, so running step5 for two different
# options never overwrites the other option's trimmed output.
OUT_DIR = STEP5_DIR / AFDB_DATASET_SELECTION.value
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

def resolve_chain_id(model) -> Optional[str]:
    """AFDB models are single-chain, so just take that one chain. Returns
    None (skip) if a model unexpectedly has more than one chain, rather
    than guessing which one is the MHC."""

    chains = list(model)

    if len(chains) == 1:
        return chains[0].id

    return None


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
    model_id: str,
    tstart: int,
    tend: int,
) -> dict:
    parser = MMCIFParser(QUIET=True, auth_chains=True)
    structure = parser.get_structure(model_id, str(cif_path))
    model = structure[0]

    original_chain = resolve_chain_id(model)

    if original_chain is None:
        return {
            "status": "skipped",
            "message": f"[SKIP] {model_id}: expected a single chain, found {len(list(model))}",
        }

    keep_only_chain(model, original_chain)

    mhc_chain = model[original_chain]
    renumber_chain(mhc_chain)

    kept_residues = trim_chain(mhc_chain, tstart, tend)
    expected_residues = tend - tstart + 1

    if kept_residues == 0:
        return {
            "status": "failed",
            "message": f"[FAILED] {model_id}: no residues kept for range {tstart}-{tend}",
        }

    if kept_residues != expected_residues:
        return {
            "status": "failed",
            "message": (
                f"[FAILED] {model_id}: range mismatch "
                f"{kept_residues}/{expected_residues} for {tstart}-{tend}"
            ),
        }

    mhc_chain.id = MHC_CHAIN_ID

    out_pdb = MHC_PDB_DIR / f"{model_id.lower()}_mhc.pdb"

    io = PDBIO()
    io.set_structure(structure)
    io.save(str(out_pdb))

    return {
        "status": "saved",
        "message": f"[SAVED] {model_id}: {out_pdb.name}",
    }


# =========================
# Main
# =========================

def main() -> None:
    if not INPUT_CSV.is_file():
        raise SystemExit(
            f"ERROR: input CSV not found: {INPUT_CSV}\n"
            f"Did step3_filter_mhc_annotations_afdb.py run and write "
            f"AFDB_DATASET_SELECTION={AFDB_DATASET_SELECTION.value}?"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MHC_PDB_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[DATASET] {describe(AFDB_DATASET_SELECTION)}")
    print(f"[INPUT] {INPUT_CSV}")

    primary_df = load_primary_rows(INPUT_CSV)

    saved = 0
    skipped = 0

    for _, row in primary_df.iterrows():
        model_id = str(row["pdb"])
        # matches the filename step4_download_structures_afdb.py wrote:
        # AFDB_MODEL_DIR / f"{model_id}.cif"
        cif_path = INPUT_CIF_DIR / f"{model_id}.cif"

        if not cif_path.exists():
            skipped += 1
            print(f"[MISSING] {model_id}: {cif_path}")
            continue

        result = process_structure(
            cif_path=cif_path,
            model_id=model_id,
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
