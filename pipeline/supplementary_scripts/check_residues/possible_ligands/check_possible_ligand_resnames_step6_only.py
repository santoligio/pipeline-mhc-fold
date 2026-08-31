#!/usr/bin/env python3
"""
List residue names associated with possible ligand chains.

Input:
- step6/pdb/summaries/possible_ligands.csv

Expected possible_ligands.csv format:
- pdb_id, Chain
or compatible variants:
- pdb, chain_id

Structure source:
- step6/pdb/2_trimmed_mhc

Outputs:
- possible_ligand_chain_resnames.csv
- possible_ligand_chain_residue_instances.csv
- possible_ligand_chain_missing.csv
"""

from pathlib import Path
from typing import Optional

import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa


# =========================
# Configuration
# =========================

PIPELINE_DIR = Path(__file__).resolve().parents[3]

CHECK_DIR = PIPELINE_DIR / "check_residues"

POSSIBLE_LIGANDS_CSV = PIPELINE_DIR / "step6" / "pdb" / "summaries" / "possible_ligands.csv"

STEP6_TRIMMED_DIR = PIPELINE_DIR / "step6" / "pdb" / "2_trimmed_mhc"

OUTPUT_RESNAMES_CSV = CHECK_DIR / "possible_ligand_chain_resnames.csv"
OUTPUT_INSTANCES_CSV = CHECK_DIR / "possible_ligand_chain_residue_instances.csv"
OUTPUT_MISSING_CSV = CHECK_DIR / "possible_ligand_chain_missing.csv"


# =========================
# Helpers
# =========================

def log(message: str) -> None:
    print(message, flush=True)


def normalize_pdb_id(value) -> str:
    return str(value).split("-")[0].strip().upper()


def residue_resseq(residue):
    try:
        return int(residue.id[1])
    except (TypeError, ValueError):
        return residue.id[1]


def get_possible_ligand_columns(df: pd.DataFrame) -> tuple[str, str]:
    pdb_col = None
    chain_col = None

    for candidate in ["pdb_id", "pdb"]:
        if candidate in df.columns:
            pdb_col = candidate
            break

    for candidate in ["Chain", "chain", "chain_id"]:
        if candidate in df.columns:
            chain_col = candidate
            break

    if pdb_col is None:
        raise SystemExit(
            "ERROR: possible_ligands.csv must contain 'pdb_id' or 'pdb'"
        )

    if chain_col is None:
        raise SystemExit(
            "ERROR: possible_ligands.csv must contain 'Chain', 'chain', or 'chain_id'"
        )

    return pdb_col, chain_col


def load_possible_ligands(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    if df.empty:
        return pd.DataFrame(columns=["pdb_id", "Chain"])

    pdb_col, chain_col = get_possible_ligand_columns(df)

    out = pd.DataFrame({
        "pdb_id": df[pdb_col].apply(normalize_pdb_id),
        "Chain": df[chain_col].astype(str).str.strip(),
    })

    out = out[(out["pdb_id"] != "") & (out["Chain"] != "")]
    out = out.drop_duplicates(subset=["pdb_id", "Chain"], keep="first")

    return out


def find_step6_structure(pdb_id: str) -> Optional[Path]:
    pdb_lower = pdb_id.lower()
    exact = STEP6_TRIMMED_DIR / f"{pdb_lower}_trimmed_mhc.pdb"

    if exact.is_file():
        return exact

    matches = sorted(STEP6_TRIMMED_DIR.glob(f"{pdb_lower}*.pdb"))
    return matches[0] if matches else None


def find_structure(pdb_id: str) -> tuple[Optional[Path], str]:
    step6_pdb = find_step6_structure(pdb_id)

    if step6_pdb is not None:
        return step6_pdb, "step6_trimmed"

    return None, "missing"


def collect_chain_residues(pdb_id: str, chain_id: str, pdb_path: Path, source: str) -> list[dict]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, str(pdb_path))
    model = structure[0]

    if chain_id not in model:
        return []

    chain = model[chain_id]
    rows = []

    for residue in chain.get_residues():
        resname = residue.get_resname().strip().upper()

        rows.append({
            "pdb_id": pdb_id,
            "Chain": chain_id,
            "resname": resname,
            "resseq": residue_resseq(residue),
            "hetflag": residue.id[0].strip() if residue.id[0].strip() else "ATOM",
            "is_standard_amino_acid": bool(is_aa(residue, standard=True)),
            "source": source,
            "structure_file": pdb_path.name,
        })

    return rows


# =========================
# Main
# =========================

def main() -> None:
    if not POSSIBLE_LIGANDS_CSV.is_file():
        raise SystemExit(f"ERROR: POSSIBLE_LIGANDS_CSV not found: {POSSIBLE_LIGANDS_CSV}")

    if not STEP6_TRIMMED_DIR.is_dir():
        raise SystemExit(f"ERROR: STEP6_TRIMMED_DIR not found: {STEP6_TRIMMED_DIR}")

    CHECK_DIR.mkdir(parents=True, exist_ok=True)

    possible_df = load_possible_ligands(POSSIBLE_LIGANDS_CSV)

    instance_rows = []
    missing_rows = []

    for _, row in possible_df.iterrows():
        pdb_id = row["pdb_id"]
        chain_id = row["Chain"]

        pdb_path, source = find_structure(pdb_id)

        if pdb_path is None:
            missing_rows.append({
                "pdb_id": pdb_id,
                "Chain": chain_id,
                "reason": "structure_missing_from_step6",
            })
            continue

        rows = collect_chain_residues(
            pdb_id=pdb_id,
            chain_id=chain_id,
            pdb_path=pdb_path,
            source=source,
        )

        if not rows:
            missing_rows.append({
                "pdb_id": pdb_id,
                "Chain": chain_id,
                "reason": f"chain_missing_in_{source}",
                "structure_file": pdb_path.name,
            })
            continue

        instance_rows.extend(rows)

    instances_df = pd.DataFrame(
        instance_rows,
        columns=[
            "pdb_id",
            "Chain",
            "resname",
            "resseq",
            "hetflag",
            "is_standard_amino_acid",
            "source",
            "structure_file",
        ],
    )

    if instances_df.empty:
        resnames_df = pd.DataFrame(
            columns=[
                "resname",
                "example_pdb_id",
                "example_Chain",
                "n_chains",
                "n_residue_instances",
                "sources",
            ]
        )
    else:
        resnames_df = (
            instances_df
            .groupby("resname", as_index=False)
            .agg(
                example_pdb_id=("pdb_id", "first"),
                example_Chain=("Chain", "first"),
                n_chains=("Chain", lambda values: len(set(zip(
                    instances_df.loc[values.index, "pdb_id"],
                    values,
                )))),
                n_residue_instances=("resname", "size"),
                sources=("source", lambda values: ";".join(sorted(set(values)))),
            )
            .sort_values("resname", kind="stable")
        )

    missing_df = pd.DataFrame(
        missing_rows,
        columns=["pdb_id", "Chain", "reason", "structure_file"],
    )

    instances_df.to_csv(OUTPUT_INSTANCES_CSV, index=False)
    resnames_df.to_csv(OUTPUT_RESNAMES_CSV, index=False)
    missing_df.to_csv(OUTPUT_MISSING_CSV, index=False)

    log(f"[INPUT] possible_ligand_chains={len(possible_df)}")
    log(f"[OUTPUT] residue_instances={len(instances_df)}")
    log(f"[OUTPUT] unique_resnames={len(resnames_df)}")
    log(f"[MISSING] entries={len(missing_df)}")
    log(f"[CSV] {OUTPUT_RESNAMES_CSV}")
    log(f"[CSV] {OUTPUT_INSTANCES_CSV}")
    log(f"[CSV] {OUTPUT_MISSING_CSV}")


if __name__ == "__main__":
    main()
