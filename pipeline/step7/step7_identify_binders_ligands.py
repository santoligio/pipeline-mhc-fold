#!/usr/bin/env python3
"""
Filter remaining chains and classify them definitively as binders or ligands.

Ligands:
   - chain length < 30 residues
   - chain is within 4 Å of MHC helix atoms
   - chain is listed in possible_ligands.csv

Binders:
   - chain length >= 30 residues
   - chain is within 4 Å of MHC helix atoms OR an already identified ligand chain 

All chains that do not meet the criteria are removed.
Residues in the removal list are also permanently removed.

"""

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, PDBIO, NeighborSearch
from Bio.PDB.DSSP import DSSP
import warnings

# =========================================================
# Configuration
# =========================================================

PIPELINE_DIR = Path(__file__).resolve().parents[1]

INPUT_PDB_DIR = PIPELINE_DIR / "step6" / "pdb" / "2_trimmed_mhc"

POSSIBLE_LIGANDS_CSV = PIPELINE_DIR / "step6" / "pdb" / "summaries" / "possible_ligands.csv"

BLACKLIST_CSV = PIPELINE_DIR / "step6" / "residue_removal_list.csv"

STEP7_DIR = PIPELINE_DIR / "step7" / "pdb"
FILTERED_PDB_DIR = STEP7_DIR / "1_filtered_structures"
SUMMARY_DIR = STEP7_DIR / "summaries"

MHC_CHAIN_ID = "A"
HELIX_CONTACT_CUTOFF = 4.0
BINDER_MIN_RESIDUES_INCLUSIVE = 30
LIGAND_MAX_RESIDUES_EXCLUSIVE = 30

# Change to "dssp" if your system uses that name instead of "mkdssp".
DSSP_EXE = "mkdssp"

INPUT_PDB_PATTERN = "*.pdb"
OUTPUT_SUFFIX = "_mhc_complex.pdb"

warnings.filterwarnings(
    "ignore",
    message=".*This file does not seem to be an mmCIF file.*",
    category=UserWarning,
    module="Bio.PDB.DSSP",
)

# =========================================================
# Logging / setup
# =========================================================

def log(message: str) -> None:
    print(message, flush=True)


def make_output_dirs() -> None:
    FILTERED_PDB_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


def validate_inputs() -> None:
    log("=== STEP7 configuration ===")
    log(f"INPUT_PDB_DIR:         {INPUT_PDB_DIR} | exists={INPUT_PDB_DIR.is_dir()}")
    log(f"POSSIBLE_LIGANDS_CSV:  {POSSIBLE_LIGANDS_CSV} | exists={POSSIBLE_LIGANDS_CSV.is_file()}")
    log(f"BLACKLIST_CSV:         {BLACKLIST_CSV} | exists={BLACKLIST_CSV.is_file()}")
    log(f"FILTERED_PDB_DIR:      {FILTERED_PDB_DIR}")
    log(f"DSSP_EXE:              {DSSP_EXE}")
    log("")

    if not INPUT_PDB_DIR.is_dir():
        raise SystemExit(f"ERROR: INPUT_PDB_DIR does not exist: {INPUT_PDB_DIR}")

    if not POSSIBLE_LIGANDS_CSV.is_file():
        raise SystemExit(f"ERROR: POSSIBLE_LIGANDS_CSV does not exist: {POSSIBLE_LIGANDS_CSV}")

    if not BLACKLIST_CSV.is_file():
        raise SystemExit(f"ERROR: BLACKLIST_CSV does not exist: {BLACKLIST_CSV}")


# =========================================================
# CSV loading
# =========================================================

def normalize_pdb_id(value) -> str:
    return str(value).split("-")[0].upper()


def pdb_id_from_path(path: Path) -> str:
    """
    Extract PDB ID from filenames such as:
      1abc_step6_1_trimmed_mhc.pdb
      1abc_step6_processed.pdb
      1abc_anything.pdb
    """
    return path.name.split("_")[0].upper()


def load_possible_ligands(possible_ligands_csv: Path) -> Set[Tuple[str, str]]:
    """
    Returns set of:
      (PDB_ID, CHAIN_ID)
    """
    df = pd.read_csv(possible_ligands_csv)

    if df.empty:
        return set()

    required = {"pdb", "chain_id"}
    missing = required - set(df.columns)

    if missing:
        raise SystemExit(
            f"ERROR: POSSIBLE_LIGANDS_CSV missing columns: {sorted(missing)}"
        )

    possible = set()

    for _, row in df.iterrows():
        pdb_id = normalize_pdb_id(row["pdb"])
        chain_id = str(row["chain_id"]).strip()

        if chain_id:
            possible.add((pdb_id, chain_id))

    return possible


def load_blacklist_resnames(blacklist_csv: Path) -> Set[str]:
    """
    Load residue names removed before step7 classification.

    keep_default_na=False is required because residue name "NA" is a valid
    PDB residue code for sodium, but pandas otherwise converts it to NaN.
    """
    df = pd.read_csv(blacklist_csv, keep_default_na=False)

    if df.empty:
        return set()

    column = "resname" if "resname" in df.columns else df.columns[0]

    blacklist = {
        str(value).strip().upper()
        for value in df[column]
        if str(value).strip()
    }

    return blacklist


# =========================================================
# Structure helpers
# =========================================================

def is_polymer_residue(residue) -> bool:
    return residue.id[0].strip() == ""


def count_chain_residues(chain) -> int:
    """
    Count remaining PDB residues in a chain.

    Blacklisted residues are removed before this function is called.
    """
    return len(list(chain.get_residues()))


def get_chain_atoms(chain) -> List:
    return list(chain.get_atoms())


def remove_blacklisted_residues(model, blacklist_resnames: Set[str]) -> List[dict]:
    """
    Physically remove blacklisted residues before step7 analysis.

    Empty non-MHC chains are removed after their blacklisted residues are
    detached. MHC chain A is not removed even if it became empty, so the
    existing MHC-chain validation behavior remains controlled elsewhere.
    """
    removed_rows = []

    for chain in list(model.get_chains()):
        removed_from_chain = 0

        for residue in list(chain.get_residues()):
            resname = residue.get_resname().strip().upper()

            if resname not in blacklist_resnames:
                continue

            hetflag, resseq, icode = residue.id

            removed_rows.append({
                "chain_id": chain.id,
                "resname": residue.get_resname().strip(),
                "resseq": resseq,
                "icode": str(icode).strip(),
                "hetflag": hetflag.strip() if hetflag.strip() else "ATOM",
            })

            chain.detach_child(residue.id)
            removed_from_chain += 1

        if chain.id != MHC_CHAIN_ID and removed_from_chain and not list(chain.get_residues()):
            model.detach_child(chain.id)

    return removed_rows


# =========================================================
# DSSP / helix detection
# =========================================================

def run_dssp(model, pdb_path: Path):
    """
    Run DSSP with configured executable.
    """
    return DSSP(model, str(pdb_path), dssp=DSSP_EXE)


def identify_mhc_helix_residues_with_dssp(
    model,
    pdb_path: Path,
    mhc_chain_id: str,
) -> Set[Tuple[str, int]]:
    """
    Identify helix residues on MHC chain using DSSP.

    DSSP helix codes:
      H = alpha helix
      G = 3-10 helix
      I = pi helix
    """
    dssp = run_dssp(model, pdb_path)

    helix_residues: Set[Tuple[str, int]] = set()

    for key in dssp.keys():
        chain_id, res_id = key
        ss = dssp[key][2]

        if chain_id != mhc_chain_id:
            continue

        if ss in {"H", "G", "I"}:
            helix_residues.add((chain_id, res_id[1]))

    return helix_residues


def collect_helix_atoms(model, helix_residues: Set[Tuple[str, int]]) -> List:
    helix_atoms = []

    for chain_id, resseq in helix_residues:
        if chain_id not in model:
            continue

        chain = model[chain_id]

        for residue in chain:
            if residue.id[1] != resseq:
                continue

            helix_atoms.extend(list(residue.get_atoms()))

    return helix_atoms


# =========================================================
# Neighbor-chain detection and classification
# =========================================================

def find_chains_near_helix(
    model,
    helix_atoms: List,
    cutoff: float,
) -> Dict[str, float]:
    """
    Return non-MHC chains near MHC helix atoms and their minimum distance.
    """
    all_atoms = list(model.get_atoms())

    if not all_atoms or not helix_atoms:
        return {}

    ns = NeighborSearch(all_atoms)
    near_chains: Dict[str, float] = {}

    for helix_atom in helix_atoms:
        neighbors = ns.search(helix_atom.coord, cutoff, level="A")

        for neighbor in neighbors:
            chain_id = neighbor.get_parent().get_parent().id

            if chain_id == MHC_CHAIN_ID:
                continue

            distance = float(np.linalg.norm(neighbor.coord - helix_atom.coord))

            if chain_id not in near_chains:
                near_chains[chain_id] = distance
            else:
                near_chains[chain_id] = min(near_chains[chain_id], distance)

    return near_chains


def find_chains_near_ligands(
    model,
    ligand_chain_ids: Set[str],
    cutoff: float,
) -> Tuple[Dict[str, float], Dict[str, str]]:
    """
    Find non-MHC, non-ligand chains near atoms of already identified ligands.

    Returns:
      near_ligand_chains:
          candidate chain ID -> minimum atom-to-atom distance to any ligand
      nearest_ligand_chain:
          candidate chain ID -> ligand chain producing that minimum distance

    The distance calculation and NeighborSearch cutoff are the same as those
    used for the original MHC-helix contact search.
    """
    if not ligand_chain_ids:
        return {}, {}

    all_atoms = list(model.get_atoms())
    if not all_atoms:
        return {}, {}

    ligand_atoms: List[Tuple[str, object]] = []
    for ligand_chain_id in sorted(ligand_chain_ids):
        if ligand_chain_id not in model:
            continue

        for atom in model[ligand_chain_id].get_atoms():
            ligand_atoms.append((ligand_chain_id, atom))

    if not ligand_atoms:
        return {}, {}

    ns = NeighborSearch(all_atoms)
    near_ligand_chains: Dict[str, float] = {}
    nearest_ligand_chain: Dict[str, str] = {}

    for ligand_chain_id, ligand_atom in ligand_atoms:
        neighbors = ns.search(ligand_atom.coord, cutoff, level="A")

        for neighbor in neighbors:
            chain_id = neighbor.get_parent().get_parent().id

            # Do not classify chain A or the ligand chains themselves through
            # the ligand-mediated binder route.
            if chain_id == MHC_CHAIN_ID or chain_id in ligand_chain_ids:
                continue

            distance = float(np.linalg.norm(neighbor.coord - ligand_atom.coord))

            if (
                chain_id not in near_ligand_chains
                or distance < near_ligand_chains[chain_id]
            ):
                near_ligand_chains[chain_id] = distance
                nearest_ligand_chain[chain_id] = ligand_chain_id

    return near_ligand_chains, nearest_ligand_chain


def classify_chains(
    pdb_id: str,
    model,
    near_helix_chains: Dict[str, float],
    possible_ligands: Set[Tuple[str, str]],
) -> Tuple[Set[str], List[dict], List[dict], List[dict]]:
    """
    Classify chains while preserving the original ligand logic.

    Processing order:
      1. Identify ligands using only the original helix-contact rule.
      2. Use atoms from those accepted ligands as an additional contact
         reference for binder-sized chains.

    A binder-sized chain is kept when it contacts:
      - the MHC helix, or
      - an already accepted ligand.
    """
    keep_chains = {MHC_CHAIN_ID}
    binder_rows = []
    ligand_rows = []
    warning_rows = []

    all_non_mhc_chain_ids = [
        chain.id
        for chain in model.get_chains()
        if chain.id != MHC_CHAIN_ID
    ]

    residue_counts = {
        chain_id: count_chain_residues(model[chain_id])
        for chain_id in all_non_mhc_chain_ids
    }

    # -----------------------------------------------------
    # Phase 1: ligand classification 
    # -----------------------------------------------------
    ligand_chain_ids: Set[str] = set()

    for chain_id in all_non_mhc_chain_ids:
        residue_count = residue_counts[chain_id]

        if residue_count >= BINDER_MIN_RESIDUES_INCLUSIVE:
            continue

        if chain_id not in near_helix_chains:
            warning_rows.append({
                "pdb": pdb_id,
                "chain_id": chain_id,
                "reason": "not_within_helix_cutoff",
                "chain_residue_count": residue_count,
                "min_distance_to_mhc_helix": "",
            })
            continue

        min_helix_distance = round(
            float(near_helix_chains[chain_id]),
            3,
        )

        if (pdb_id, chain_id) in possible_ligands:
            keep_chains.add(chain_id)
            ligand_chain_ids.add(chain_id)
            ligand_rows.append({
                "pdb": pdb_id,
                "chain_id": chain_id,
                "chain_residue_count": residue_count,
                "min_distance_to_mhc_helix": min_helix_distance,
                "classification": "ligand",
            })
        else:
            warning_rows.append({
                "pdb": pdb_id,
                "chain_id": chain_id,
                "reason": "small_neighbor_chain_not_in_possible_ligands",
                "chain_residue_count": residue_count,
                "min_distance_to_mhc_helix": min_helix_distance,
            })

    # -----------------------------------------------------
    # Phase 2: binder classification
    # -----------------------------------------------------
    near_ligand_chains, nearest_ligand_chain = find_chains_near_ligands(
        model=model,
        ligand_chain_ids=ligand_chain_ids,
        cutoff=HELIX_CONTACT_CUTOFF,
    )

    for chain_id in all_non_mhc_chain_ids:
        residue_count = residue_counts[chain_id]

        if residue_count < BINDER_MIN_RESIDUES_INCLUSIVE:
            continue

        helix_distance = near_helix_chains.get(chain_id)
        ligand_distance = near_ligand_chains.get(chain_id)

        if helix_distance is None and ligand_distance is None:
            warning_rows.append({
                "pdb": pdb_id,
                "chain_id": chain_id,
                "reason": "not_within_helix_or_ligand_cutoff",
                "chain_residue_count": residue_count,
                "min_distance_to_mhc_helix": "",
                "min_distance_to_ligand": "",
                "nearest_ligand_chain": "",
            })
            continue

        if helix_distance is not None and ligand_distance is not None:
            contact_source = "helix+ligand"
        elif helix_distance is not None:
            contact_source = "helix"
        else:
            contact_source = "ligand"

        keep_chains.add(chain_id)
        binder_rows.append({
            "pdb": pdb_id,
            "chain_id": chain_id,
            "chain_residue_count": residue_count,
            "min_distance_to_mhc_helix": (
                round(float(helix_distance), 3)
                if helix_distance is not None
                else ""
            ),
            "min_distance_to_ligand": (
                round(float(ligand_distance), 3)
                if ligand_distance is not None
                else ""
            ),
            "nearest_ligand_chain": (
                nearest_ligand_chain.get(chain_id, "")
                if ligand_distance is not None
                else ""
            ),
            "binder_contact_source": contact_source,
            "classification": "binder",
        })

    return keep_chains, binder_rows, ligand_rows, warning_rows


def remove_unwanted_chains(model, keep_chains: Set[str]) -> List[str]:
    removed = []

    for chain in list(model.get_chains()):
        if chain.id not in keep_chains:
            removed.append(chain.id)
            model.detach_child(chain.id)

    return removed


# =========================================================
# Saving
# =========================================================

def save_structure(structure, out_pdb: Path, remark_lines: Optional[List[str]] = None) -> None:
    io = PDBIO()
    io.set_structure(structure)

    with out_pdb.open("w") as handle:
        handle.write(f"HEADER    {out_pdb.name}\n")
        for remark in remark_lines or []:
            handle.write(f"REMARK    {remark}\n")
        io.save(handle)


# =========================================================
# Per-file processing
# =========================================================

def process_one(
    pdb_path: Path,
    possible_ligands: Set[Tuple[str, str]],
    blacklist_resnames: Set[str],
) -> Optional[dict]:
    pdb_id = pdb_id_from_path(pdb_path)

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, str(pdb_path))
    model = structure[0]

    if MHC_CHAIN_ID not in model:
        log(f"[ERROR] {pdb_id}: MHC chain {MHC_CHAIN_ID} not found")
        return None

    removed_blacklist_rows = remove_blacklisted_residues(
        model=model,
        blacklist_resnames=blacklist_resnames,
    )

    if removed_blacklist_rows:
        log(f"[BLACKLIST] {pdb_id}: removed_residues={len(removed_blacklist_rows)}")

    try:
        helix_residues = identify_mhc_helix_residues_with_dssp(
            model=model,
            pdb_path=pdb_path,
            mhc_chain_id=MHC_CHAIN_ID,
        )
    except Exception as exc:
        log(f"[ERROR] {pdb_id}: DSSP failed: {exc}")
        return {
            "pdb": pdb_id,
            "status": "dssp_failed",
            "input_pdb": str(pdb_path),
            "output_pdb": "",
            "removed_chains": "",
            "binder_rows": [],
            "ligand_rows": [],
            "warning_rows": [{
                "pdb": pdb_id,
                "chain_id": "",
                "reason": f"dssp_failed: {exc}",
                "chain_residue_count": "",
                "min_distance_to_mhc_helix": "",
            }],
            "blacklist_rows": [],
            "helix_residue_count": 0,
        }

    if not helix_residues:
        log(f"[WARN] {pdb_id}: DSSP found no helices on MHC chain {MHC_CHAIN_ID}")
        return {
            "pdb": pdb_id,
            "status": "no_mhc_helices",
            "input_pdb": str(pdb_path),
            "output_pdb": "",
            "removed_chains": "",
            "binder_rows": [],
            "ligand_rows": [],
            "warning_rows": [{
                "pdb": pdb_id,
                "chain_id": "",
                "reason": "no_mhc_helices_found_by_dssp",
                "chain_residue_count": "",
                "min_distance_to_mhc_helix": "",
            }],
            "blacklist_rows": [],
            "helix_residue_count": 0,
        }

    helix_atoms = collect_helix_atoms(model, helix_residues)

    if not helix_atoms:
        log(f"[WARN] {pdb_id}: no atoms collected for MHC helix residues")
        return {
            "pdb": pdb_id,
            "status": "no_helix_atoms",
            "input_pdb": str(pdb_path),
            "output_pdb": "",
            "removed_chains": "",
            "binder_rows": [],
            "ligand_rows": [],
            "warning_rows": [{
                "pdb": pdb_id,
                "chain_id": "",
                "reason": "no_atoms_for_mhc_helix_residues",
                "chain_residue_count": "",
                "min_distance_to_mhc_helix": "",
            }],
            "blacklist_rows": [],
            "helix_residue_count": len(helix_residues),
        }

    near_helix_chains = find_chains_near_helix(
        model=model,
        helix_atoms=helix_atoms,
        cutoff=HELIX_CONTACT_CUTOFF,
    )

    keep_chains, binder_rows, ligand_rows, warning_rows = classify_chains(
        pdb_id=pdb_id,
        model=model,
        near_helix_chains=near_helix_chains,
        possible_ligands=possible_ligands,
    )

    ligand_mediated_binders = sum(
        row.get("binder_contact_source") in {"ligand", "helix+ligand"}
        for row in binder_rows
    )

    removed_chains = remove_unwanted_chains(model, keep_chains)

    out_pdb = FILTERED_PDB_DIR / f"{pdb_id.lower()}{OUTPUT_SUFFIX}"

    save_structure(
        structure=structure,
        out_pdb=out_pdb,
        remark_lines=[
            "STEP7 FILTERED STRUCTURE",
            f"MHC HELIX RESIDUES FOUND BY DSSP: {len(helix_residues)}",
            f"KEPT CHAINS: {','.join(sorted(keep_chains))}",
            f"REMOVED CHAINS: {','.join(sorted(removed_chains))}",
            f"BLACKLIST RESIDUES REMOVED: {len(removed_blacklist_rows)}",
            f"BINDERS CONTACTING ACCEPTED LIGANDS: {ligand_mediated_binders}",
        ],
    )

    log(
        f"[OK] {pdb_id}: binders={len(binder_rows)}; ligands={len(ligand_rows)}; "
        f"ligand_contact_binders={ligand_mediated_binders}; "
        f"warnings={len(warning_rows)}; removed_chains={removed_chains}; output={out_pdb.name}"
    )

    return {
        "pdb": pdb_id,
        "status": "ok",
        "input_pdb": str(pdb_path),
        "output_pdb": str(out_pdb),
        "removed_chains": ";".join(removed_chains),
        "binder_rows": binder_rows,
        "ligand_rows": ligand_rows,
        "warning_rows": warning_rows,
        "blacklist_rows": [
            {"pdb": pdb_id, **row}
            for row in removed_blacklist_rows
        ],
        "helix_residue_count": len(helix_residues),
    }


# =========================================================
# Summary writing
# =========================================================

def write_summaries(results: List[dict]) -> None:
    binder_rows = []
    ligand_rows = []
    warning_rows = []
    blacklist_rows = []

    for result in results:
        binder_rows.extend(result["binder_rows"])
        ligand_rows.extend(result["ligand_rows"])
        warning_rows.extend(result["warning_rows"])
        blacklist_rows.extend(result.get("blacklist_rows", []))

    pd.DataFrame(binder_rows).to_csv(
        SUMMARY_DIR / "step7_binders.csv",
        index=False,
    )

    pd.DataFrame(ligand_rows).to_csv(
        SUMMARY_DIR / "step7_ligands.csv",
        index=False,
    )

    pd.DataFrame(warning_rows).to_csv(
        SUMMARY_DIR / "step7_warnings_review_chains.csv",
        index=False,
    )

    pd.DataFrame(blacklist_rows).to_csv(
        SUMMARY_DIR / "step7_removed_blacklist_residues.csv",
        index=False,
    )



# =========================================================
# Main
# =========================================================

def main() -> None:
    make_output_dirs()
    validate_inputs()

    possible_ligands = load_possible_ligands(POSSIBLE_LIGANDS_CSV)
    blacklist_resnames = load_blacklist_resnames(BLACKLIST_CSV)

    pdb_files = sorted(INPUT_PDB_DIR.glob(INPUT_PDB_PATTERN))

    if not pdb_files:
        raise SystemExit(f"ERROR: no PDB files found in {INPUT_PDB_DIR}")

    log(f"[INPUT] PDB files:           {len(pdb_files)}")
    log(f"[INPUT] possible ligands:     {len(possible_ligands)}")
    log(f"[INPUT] blacklisted resnames: {len(blacklist_resnames)}")
    log("")
    log("=== STEP7 processing ===")

    results = []
    failed = 0

    for pdb_path in pdb_files:
        try:
            result = process_one(
                pdb_path=pdb_path,
                possible_ligands=possible_ligands,
                blacklist_resnames=blacklist_resnames,
            )
        except Exception as exc:
            failed += 1
            pdb_id = pdb_id_from_path(pdb_path)
            log(f"[ERROR] {pdb_id}: {exc}")
            results.append({
                "pdb": pdb_id,
                "status": f"error: {exc}",
                "input_pdb": str(pdb_path),
                "output_pdb": "",
                "removed_chains": "",
                "binder_rows": [],
                "ligand_rows": [],
                "warning_rows": [{
                    "pdb": pdb_id,
                    "chain_id": "",
                    "reason": f"error: {exc}",
                    "chain_residue_count": "",
                    "min_distance_to_mhc_helix": "",
                }],
                "blacklist_rows": [],
                "helix_residue_count": 0,
            })
            continue

        if result is None:
            failed += 1
            continue

        results.append(result)

    write_summaries(results)

    log("")
    log("=== STEP7 summary ===")
    log(f"Processed records: {len(results)}")
    log(f"Failed hard:       {failed}")
    log(f"Filtered PDBs:     {FILTERED_PDB_DIR}")
    log(f"Summaries:         {SUMMARY_DIR}")


if __name__ == "__main__":
    main()
