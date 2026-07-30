#!/usr/bin/env python3
"""
Map possible ligands and identify problematic MHC contacts.

Uses step5 configured complexes and step4 filtered cavities.
Writes merged cavity files and CSVs; final trimming is done in step6.1.
Manual correction is restricted to amino-acid MHC chain A contacts.
"""

import shutil
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from Bio.PDB import NeighborSearch, PDBIO, PDBParser, Select
from Bio.PDB.Polypeptide import is_aa


# =========================================================
# Configuration
# =========================================================

PIPELINE_DIR = Path(
    "/mnt/c/Users/gio/Documents/foldseek_nefertari/filter/ligands_pipeline"
)

# Inputs
STEP5_PDB_DIR = PIPELINE_DIR / "step5" / "pdb" / "1_mhc_configured"
STEP5_CHAIN_MAPPING_CSV = PIPELINE_DIR / "step5" / "pdb" / "chain_map.csv"
STEP5_REMAP_CSV = PIPELINE_DIR / "step5" / "pdb" / "pdb_assemblies_remapped.csv"

# Used only to identify duplicate chains from the original Foldseek table.
STEP1_CSV = PIPELINE_DIR / "step1" / "pdb" / "pdb_assemblies.csv"

# Residue names to ignore when classifying possible ligands/problematic MHC contacts.
# Expected CSV structure: one column named "resname".
BLACKLIST_CSV = PIPELINE_DIR / "step6" / "residue_removal_list.csv"

FILTERED_CAVITY_DIR = PIPELINE_DIR / "step4" / "pdb" / "3_filtered_cavities"

# Outputs
STEP6_DIR = PIPELINE_DIR / "step6" / "pdb"
MERGED_INTERMEDIATE_DIR = STEP6_DIR / "1_merged_cavities"
SUMMARY_DIR = STEP6_DIR / "summaries"
RESNAME_MAP_CSV = SUMMARY_DIR / "resname_map.csv"

CAVITY_CONTACT_CUTOFF = 1.0
POSSIBLE_LIGAND_MAX_RESIDUES_EXCLUSIVE = 30
MHC_CHAIN_ID = "A"

# Amino-acid problematic contacts are evaluated using backbone atoms only.
BACKBONE_ATOM_NAMES = {"N", "CA", "C", "O"}

WRITE_MERGED_INTERMEDIATE_PDB = True

MERGED_SUFFIX = "_merged_cavities.pdb"


# =========================================================
# Logging / setup
# =========================================================

def log(message: str) -> None:
    print(message, flush=True)


def make_output_dirs() -> None:
    for path in [
        MERGED_INTERMEDIATE_DIR,
        SUMMARY_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def validate_inputs() -> None:
    log("[CONFIG] STEP6")
    log(f"PIPELINE_DIR:             {PIPELINE_DIR}")
    log(f"STEP5_PDB_DIR:            {STEP5_PDB_DIR} | exists={STEP5_PDB_DIR.is_dir()}")
    log(f"STEP5_CHAIN_MAPPING_CSV:  {STEP5_CHAIN_MAPPING_CSV} | exists={STEP5_CHAIN_MAPPING_CSV.is_file()}")
    log(f"STEP5_REMAP_CSV:          {STEP5_REMAP_CSV} | exists={STEP5_REMAP_CSV.is_file()}")
    log(f"STEP1_CSV:                {STEP1_CSV} | exists={STEP1_CSV.is_file()}")
    log(f"BLACKLIST_CSV:            {BLACKLIST_CSV} | exists={BLACKLIST_CSV.is_file()}")
    log(f"FILTERED_CAVITY_DIR:      {FILTERED_CAVITY_DIR} | exists={FILTERED_CAVITY_DIR.is_dir()}")
    log("")

    if not STEP5_PDB_DIR.is_dir():
        raise SystemExit(f"ERROR: STEP5_PDB_DIR does not exist: {STEP5_PDB_DIR}")

    if not STEP5_REMAP_CSV.is_file():
        raise SystemExit(f"ERROR: STEP5_REMAP_CSV does not exist: {STEP5_REMAP_CSV}")

    if not STEP1_CSV.is_file():
        raise SystemExit(f"ERROR: STEP1_CSV does not exist: {STEP1_CSV}")

    if not BLACKLIST_CSV.is_file():
        raise SystemExit(f"ERROR: BLACKLIST_CSV does not exist: {BLACKLIST_CSV}")

    if not STEP5_CHAIN_MAPPING_CSV.is_file():
        raise SystemExit(f"ERROR: STEP5_CHAIN_MAPPING_CSV does not exist: {STEP5_CHAIN_MAPPING_CSV}")

    if not FILTERED_CAVITY_DIR.is_dir():
        raise SystemExit(f"ERROR: FILTERED_CAVITY_DIR does not exist: {FILTERED_CAVITY_DIR}")


# =========================================================
# CSV loading
# =========================================================

def normalize_pdb_id(value) -> str:
    return str(value).split("-")[0].upper()


def is_integer_text(value: str) -> bool:
    try:
        int(str(value).strip())
        return True
    except ValueError:
        return False


def format_atom_name(atom_name: str) -> str:
    atom_name = str(atom_name).strip()

    if len(atom_name) >= 4:
        return atom_name[:4]

    if atom_name and atom_name[0].isdigit():
        return f"{atom_name:<4}"

    return f"{atom_name:>4}"


def parse_occupancy_bfactor(tokens: List[str]) -> Tuple[float, float]:
    """
    Parse occupancy and B-factor, including glued values such as 1.00102.41.
    """
    if len(tokens) >= 11:
        try:
            return float(tokens[9]), float(tokens[10])
        except ValueError:
            pass

    glued = tokens[9] if len(tokens) > 9 else ""
    match = re.match(r"^(-?\d+\.\d{2})(-?\d+\.\d{2})$", glued)

    if match:
        return float(match.group(1)), float(match.group(2))

    raise ValueError(f"Could not parse occupancy/B-factor from tokens: {tokens}")


def split_altloc_resname(raw_resname: str) -> Tuple[str, str, str]:
    """
    Split tokenized residue text into altloc, corrected resname, and logged old resname.

    PDB allows an alternate-location indicator in column 17. When token parsing
    joins altloc + residue name, do not treat the altloc as part of the residue name.
    """
    raw = str(raw_resname).strip()

    if len(raw) <= 3:
        return " ", raw, raw

    # Tokenized altloc + 3-character residue name, e.g. AUNL.
    if len(raw) == 4 and raw[0].isalpha():
        altloc = raw[0]
        resname = raw[1:]
        return altloc, resname, resname

    # True long residue name. Keep the last 3 characters for PDB output.
    return " ", raw[-3:], raw


def atom_line_needs_resname_fix(line: str) -> bool:
    """
    Detect true residue-name overflow while ignoring altloc + 3-letter resname.
    """
    tokens = line.split()

    if len(tokens) < 6:
        return False

    raw = tokens[3]

    if len(raw) <= 3:
        return False

    if len(raw) == 4 and raw[0].isalpha():
        return False

    return True


def ter_line_needs_resname_fix(line: str) -> bool:
    tokens = line.split()

    if len(tokens) < 5:
        return False

    raw = tokens[2]

    if len(raw) <= 3:
        return False

    if len(raw) == 4 and raw[0].isalpha():
        return False

    return True


def rebuild_atom_line_from_tokens(line: str) -> Tuple[str, Optional[dict]]:
    """
    Rebuild ATOM/HETATM lines whose residue name overflows PDB columns.

    PDB residue names have 3 columns. Long names are permanently shortened
    to their last 3 characters in step6 outputs.
    """
    tokens = line.split()

    if len(tokens) < 11:
        return line, None

    record = tokens[0]
    serial = int(tokens[1])
    atom_name = tokens[2]
    altloc, new_resname, old_resname = split_altloc_resname(tokens[3])
    chain_id = tokens[4][0]
    resseq = int(tokens[5])
    x = float(tokens[6])
    y = float(tokens[7])
    z = float(tokens[8])
    occupancy, bfactor = parse_occupancy_bfactor(tokens)

    element = ""
    if tokens[-1].isalpha() and len(tokens[-1]) <= 2:
        element = tokens[-1].upper()
    else:
        element = "".join(ch for ch in atom_name if ch.isalpha())[:1].upper() or "C"

    fixed_line = (
        f"{record:<6}{serial:5d} {format_atom_name(atom_name)}"
        f"{altloc:1}{new_resname:>3} {chain_id:1}{resseq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}"
        f"{occupancy:6.2f}{bfactor:6.2f}          "
        f"{element:>2}\n"
    )

    mapping = None
    if old_resname != new_resname:
        mapping = {
            "old_resname": old_resname,
            "new_resname": new_resname,
            "chain_id": chain_id,
            "resseq": resseq,
        }

    return fixed_line, mapping


def rebuild_ter_line_from_tokens(line: str) -> Tuple[str, Optional[dict]]:
    """
    Rebuild TER lines whose residue name overflows PDB columns.
    """
    tokens = line.split()

    if len(tokens) < 5 or tokens[0] != "TER":
        return line, None

    serial = int(tokens[1])
    _altloc, new_resname, old_resname = split_altloc_resname(tokens[2])
    chain_id = tokens[3][0]
    resseq = int(tokens[4])

    fixed_line = f"TER   {serial:5d}      {new_resname:>3} {chain_id:1}{resseq:4d}\n"

    mapping = None
    if old_resname != new_resname:
        mapping = {
            "old_resname": old_resname,
            "new_resname": new_resname,
            "chain_id": chain_id,
            "resseq": resseq,
        }

    return fixed_line, mapping


def line_has_long_resname_tokens(line: str) -> bool:
    """
    Detect true long residue names, excluding tokenized altloc + resname.
    """
    if line.startswith(("ATOM", "HETATM")):
        return atom_line_needs_resname_fix(line)

    if line.startswith("TER"):
        return ter_line_needs_resname_fix(line)

    return False


def normalize_long_resnames_pdb(
    in_pdb: Path,
    out_pdb: Path,
    pdb_id: str,
) -> List[dict]:
    """
    Write a corrected PDB with long residue names shortened to 3 characters.
    Returns mapping rows for renamed residues.
    """
    mapping_rows = []
    seen = set()

    with in_pdb.open("r") as fin, out_pdb.open("w") as fout:
        for line in fin:
            if not line.startswith(("ATOM", "HETATM", "TER")):
                fout.write(line)
                continue

            if not line_has_long_resname_tokens(line):
                fout.write(line)
                continue

            try:
                if line.startswith("TER"):
                    fixed_line, mapping = rebuild_ter_line_from_tokens(line)
                else:
                    fixed_line, mapping = rebuild_atom_line_from_tokens(line)

                fout.write(fixed_line)

                if mapping is not None:
                    key = (
                        pdb_id,
                        mapping["old_resname"],
                        mapping["new_resname"],
                        mapping["chain_id"],
                        mapping["resseq"],
                    )

                    if key not in seen:
                        mapping_rows.append({
                            "pdb": pdb_id,
                            **mapping,
                        })
                        seen.add(key)

            except Exception:
                fout.write(line)

    return mapping_rows


def load_remapped_rows(remap_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(remap_csv)
    required = {"pdb_id", "chain", "tstart", "tend"}
    missing = required - set(df.columns)

    if missing:
        raise SystemExit(f"ERROR: STEP5_REMAP_CSV missing columns: {sorted(missing)}")

    df = df.copy()
    df["pdb_id"] = df["pdb_id"].astype(str).str[:4].str.upper()

    if df.empty:
        raise SystemExit("ERROR: no rows found in STEP5_REMAP_CSV")

    return df


def load_blacklist_resnames(blacklist_csv: Path) -> Set[str]:
    """
    Load residue names ignored during contact classification.

    keep_default_na=False is required because residue name "NA" is a valid
    PDB residue code for sodium, but pandas otherwise converts it to NaN.
    """
    df = pd.read_csv(blacklist_csv, keep_default_na=False)

    if "resname" not in df.columns:
        raise SystemExit("ERROR: blacklist.csv must contain a 'resname' column")

    blacklist = {
        str(value).strip().upper()
        for value in df["resname"]
        if str(value).strip()
    }

    log(f"[BLACKLIST] resnames= {len(blacklist)}")
    return blacklist


def load_duplicate_old_chains(step1_csv: Path) -> Dict[str, Set[str]]:
    df = pd.read_csv(step1_csv)
    status = df["status"].astype(str).str.strip().str.lower()
    dup_df = df[status == "duplicate"].copy()

    duplicate_old_chains: Dict[str, Set[str]] = {}

    for _, row in dup_df.iterrows():
        pdb_id = normalize_pdb_id(row["pdb"])
        duplicate_old_chains.setdefault(pdb_id, set()).add(str(row["chain"]))

    return duplicate_old_chains


def load_old_to_new_chain_map(mapping_csv: Path) -> Dict[Tuple[str, str], str]:
    df = pd.read_csv(mapping_csv)

    required = {"pdb", "old_chain", "new_chain"}
    missing = required - set(df.columns)

    if missing:
        raise SystemExit(f"ERROR: STEP5 mapping CSV missing columns: {sorted(missing)}")

    chain_map: Dict[Tuple[str, str], str] = {}

    for _, row in df.iterrows():
        pdb_id = normalize_pdb_id(row["pdb"])
        old_chain = str(row["old_chain"])
        new_chain = str(row["new_chain"])
        chain_map[(pdb_id, old_chain)] = new_chain

    return chain_map


def build_duplicate_new_chain_map(
    duplicate_old_chains: Dict[str, Set[str]],
    old_to_new_chain_map: Dict[Tuple[str, str], str],
) -> Dict[str, Set[str]]:
    """Map step1 duplicate chains to step5 renamed chain IDs."""
    duplicate_new_chains: Dict[str, Set[str]] = {}

    for pdb_id, old_chains in duplicate_old_chains.items():
        for old_chain in old_chains:
            new_chain = old_to_new_chain_map.get((pdb_id, old_chain))

            # If the duplicate chain was removed in step5, it will not appear here.
            if new_chain is None:
                continue

            duplicate_new_chains.setdefault(pdb_id, set()).add(new_chain)

    return duplicate_new_chains


# =========================================================
# File discovery
# =========================================================

def find_step5_pdb(pdb_id: str) -> Optional[Path]:
    pdb_id_lower = pdb_id.lower()
    exact = STEP5_PDB_DIR / f"{pdb_id_lower}_mhc_configured.pdb"

    if exact.is_file():
        return exact

    matches = sorted(STEP5_PDB_DIR.glob(f"{pdb_id_lower}*.pdb"))
    return matches[0] if matches else None


def find_filtered_cavity_pdb(pdb_id: str) -> Optional[Path]:
    """
    Singular only:
      {pdb_id}_filtered_cavity.pdb
    """
    pdb_id_lower = pdb_id.lower()
    path = FILTERED_CAVITY_DIR / f"{pdb_id_lower}_filtered_cavity.pdb"

    if path.is_file():
        return path

    return None


# =========================================================
# Structure helpers
# =========================================================

def residue_resseq_int(residue) -> Optional[int]:
    """
    Return integer residue number when possible.

    Some PDB residues can carry non-integer residue IDs after parsing. Those
    should not crash contact mapping.
    """
    try:
        return int(residue.id[1])
    except (TypeError, ValueError):
        return None


def residue_key(residue) -> Tuple[str, str, str, str]:
    chain_id = residue.get_parent().id
    hetflag, resseq, icode = residue.id
    return (chain_id, hetflag, str(resseq).strip(), str(icode).strip())


def residue_label(residue) -> str:
    chain_id, hetflag, resseq, icode = residue_key(residue)
    resname = residue.get_resname().strip()
    icode_text = icode if icode else ""
    het_text = hetflag.strip() if hetflag.strip() else "ATOM"
    return f"{chain_id}:{resname}:{resseq}{icode_text}:{het_text}"


def is_polymer_residue(residue) -> bool:
    return residue.id[0].strip() == ""


def count_chain_residues(chain, blacklist_resnames) -> int:
    residues = [
        residue
        for residue in chain.get_residues()
        if residue.get_resname().strip().upper() not in blacklist_resnames
    ]
    return len(residues)


def remove_duplicate_chains(structure, duplicate_chains: Set[str]) -> List[str]:
    model = structure[0]
    removed = []

    for chain_id in sorted(duplicate_chains):
        if chain_id == MHC_CHAIN_ID:
            log("[WARNING] duplicate chain mapped to MHC chain A; not removing A")
            continue

        if chain_id in model:
            model.detach_child(chain_id)
            removed.append(chain_id)

    return removed


# =========================================================
# Cavity parsing / merged intermediate
# =========================================================

def parse_cavity_points(cavity_pdb: Path) -> List[Dict[str, object]]:
    """Read KVFinder cavity points grouped by residue-name cavity ID."""
    points = []

    with cavity_pdb.open("r") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue

            cavity_id = line[17:20].strip() or "UNK"

            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue

            points.append({
                "cavity_id": cavity_id,
                "coord": np.array([x, y, z], dtype=float),
                "line": line,
            })

    return points


def write_merged_system_plus_cavity_pdb(
    system_pdb: Path,
    cavity_points: List[Dict[str, object]],
    out_pdb: Path,
) -> None:
    """
    Write system atoms followed by cavity points for inspection/debugging.
    """
    with system_pdb.open("r") as fin, out_pdb.open("w") as fout:
        for line in fin:
            if line.startswith("END"):
                continue
            fout.write(line)

        fout.write("REMARK    CAVITY POINTS APPENDED BY STEP6\n")

        for idx, point in enumerate(cavity_points, start=1):
            cavity_id = str(point["cavity_id"])[:3].rjust(3)
            x, y, z = point["coord"]

            # Chain Z is used only in this merged/debug file.
            fout.write(
                f"HETATM{idx % 100000:5d}  CV  {cavity_id} Z{idx % 10000:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           X\n"
            )

        fout.write("END\n")


# =========================================================
# Contact classification
# =========================================================

def find_cavity_contacts(
    structure,
    cavity_points: List[Dict[str, object]],
    cutoff: float,
) -> Dict[Tuple[str, str, str, str], Dict[str, object]]:
    """
    Return contacting residues with all-atom and backbone-only distances.

    For amino-acid problematic contacts, classification later uses only
    backbone atom contacts. Non-amino-acids keep the original all-atom logic.
    """
    atoms = list(structure.get_atoms())

    if not atoms or not cavity_points:
        return {}

    ns = NeighborSearch(atoms)
    contacts: Dict[Tuple[str, str, str, str], Dict[str, object]] = {}

    for point in cavity_points:
        coord = point["coord"]
        cavity_id = str(point["cavity_id"])

        for atom in ns.search(coord, cutoff, level="A"):
            residue = atom.get_parent()
            key = residue_key(residue)
            distance = float(np.linalg.norm(atom.coord - coord))
            atom_name = str(atom.name).strip()

            if key not in contacts:
                contacts[key] = {
                    "residue": residue,
                    "min_distance": distance,
                    "cavity_ids": set([cavity_id]),
                    "atom_names": set([atom_name]),
                    "backbone_min_distance": None,
                    "backbone_cavity_ids": set(),
                    "backbone_atom_names": set(),
                }
            else:
                contacts[key]["min_distance"] = min(
                    float(contacts[key]["min_distance"]),
                    distance,
                )
                contacts[key]["cavity_ids"].add(cavity_id)
                contacts[key]["atom_names"].add(atom_name)

            if atom_name in BACKBONE_ATOM_NAMES:
                current = contacts[key]["backbone_min_distance"]

                if current is None:
                    contacts[key]["backbone_min_distance"] = distance
                else:
                    contacts[key]["backbone_min_distance"] = min(float(current), distance)

                contacts[key]["backbone_cavity_ids"].add(cavity_id)
                contacts[key]["backbone_atom_names"].add(atom_name)

    return contacts


def problematic_contact_distance_info(
    residue,
    info: Dict[str, object],
) -> Optional[Tuple[float, List[str], str]]:
    """
    Return the distance data used for problematic-contact classification.

    Amino-acid contacts are considered problematic only when a backbone atom
    contacts the cavity. Side-chain-only amino-acid contacts are ignored here.
    Non-amino-acid contacts use the original all-atom contact distance.
    """
    resname = residue.get_resname().strip().upper()

    if is_aa(resname, standard=False):
        backbone_distance = info.get("backbone_min_distance")

        if backbone_distance is None:
            return None

        backbone_cavity_ids = sorted(info.get("backbone_cavity_ids", set()))

        return (
            float(backbone_distance),
            backbone_cavity_ids,
            "backbone",
        )

    return (
        float(info["min_distance"]),
        sorted(info["cavity_ids"]),
        "all_atoms",
    )


def classify_contacts(
    structure,
    contacts: Dict[Tuple[str, str, str, str], Dict[str, object]],
    tstart: int,
    tend: int,
    blacklist_resnames: Set[str],
) -> Tuple[List[dict], List[dict], List[dict], bool, Set[str]]:
    """
    Returns:
      possible_ligand_rows
      problematic_mhc_rows
      problematic_binder_rows
      is_problematic
      problematic_reasons
    """
    model = structure[0]

    chain_residue_counts = {
        chain.id: count_chain_residues(chain, blacklist_resnames)
        for chain in model.get_chains()
    }

    possible_ligand_rows = []
    problematic_mhc_rows = []
    problematic_binder_rows = []
    is_problematic = False
    problematic_reasons: Set[str] = set()

    seen_possible = set()
    seen_problematic_mhc = set()
    seen_problematic_binder = set()

    for key, info in contacts.items():
        chain_id, hetflag, resseq, icode = key
        residue = info["residue"]
        resseq_int = residue_resseq_int(residue)
        resname = residue.get_resname().strip().upper()

        # Blacklisted residue names are ignored for both possible-ligand
        # classification and problematic-MHC-contact recognition.
        if resname in blacklist_resnames:
            continue

        if chain_id != MHC_CHAIN_ID:
            chain_len = chain_residue_counts.get(chain_id, 0)

            if chain_len < POSSIBLE_LIGAND_MAX_RESIDUES_EXCLUSIVE:
                # Save possible ligands at chain level only.
                row_key = chain_id

                if row_key not in seen_possible:
                    possible_ligand_rows.append({
                        "chain_id": chain_id,
                    })
                    seen_possible.add(row_key)

                continue

            # Record all accepted binder-sized contacts for review/traceability.
            # Amino-acid binder contacts use the same backbone-only rule as MHC.
            # Non-amino-acid binder contacts use all-atom distance and are the
            # only binder contacts moved automatically in step6.1.
            if chain_len >= POSSIBLE_LIGAND_MAX_RESIDUES_EXCLUSIVE:
                is_amino_acid = is_aa(resname, standard=False)
                distance_info = problematic_contact_distance_info(residue, info)

                if distance_info is None:
                    continue

                min_distance, cavity_ids, contact_basis = distance_info
                row_key = (chain_id, hetflag, resseq, icode)

                if row_key not in seen_problematic_binder:
                    problematic_binder_rows.append({
                        "chain_id": chain_id,
                        "chain_residue_count": chain_len,
                        "residue": residue_label(residue),
                        "resname": residue.get_resname().strip(),
                        "resseq": resseq_int if resseq_int is not None else resseq,
                        "hetflag": hetflag.strip() if hetflag.strip() else "ATOM",
                        "cavity_ids": ";".join(cavity_ids),
                        "min_distance_to_cavity": round(float(min_distance), 3),
                        "contact_basis": contact_basis,
                        "is_amino_acid": is_amino_acid,
                    })
                    seen_problematic_binder.add(row_key)

                if not is_amino_acid:
                    is_problematic = True
                    problematic_reasons.add("binder_non_aminoacid_contact")

            continue

        # MHC chain A contact.
        if resseq_int is not None and tstart <= resseq_int <= tend:
            # Contact inside the final MHC trimmed region: ignore.
            continue

        distance_info = problematic_contact_distance_info(residue, info)

        if distance_info is None:
            continue

        min_distance, cavity_ids, contact_basis = distance_info

        is_problematic = True
        if is_aa(resname, standard=False):
            problematic_reasons.add("mhc_aminoacid_contact")
        else:
            problematic_reasons.add("mhc_non_aminoacid_contact")

        row_key = (chain_id, hetflag, resseq, icode)

        if row_key not in seen_problematic_mhc:
            problematic_mhc_rows.append({
                "chain_id": chain_id,
                "residue": residue_label(residue),
                "resname": residue.get_resname().strip(),
                "resseq": resseq_int if resseq_int is not None else resseq,
                "hetflag": hetflag.strip() if hetflag.strip() else "ATOM",
                "cavity_ids": ";".join(cavity_ids),
                "min_distance_to_cavity": round(float(min_distance), 3),
                "contact_basis": contact_basis,
                "is_amino_acid": is_aa(resname, standard=False),
            })
            seen_problematic_mhc.add(row_key)

    return (
        possible_ligand_rows,
        problematic_mhc_rows,
        problematic_binder_rows,
        is_problematic,
        problematic_reasons,
    )


# =========================================================
# MHC trimming
# =========================================================

class Step6Select(Select):
    """
    Keep all chains and residues, except trim MHC chain A to tstart/tend
    when the structure is not problematic.
    """

    def __init__(self, trim_mhc: bool, tstart: int, tend: int):
        self.trim_mhc = trim_mhc
        self.tstart = tstart
        self.tend = tend

    def accept_residue(self, residue):
        chain_id = residue.get_parent().id

        if not self.trim_mhc:
            return True

        if chain_id != MHC_CHAIN_ID:
            return True

        resseq = residue_resseq_int(residue)
        if resseq is None:
            return False

        return self.tstart <= resseq <= self.tend


def save_processed_structure(
    structure,
    out_pdb: Path,
    tstart: int,
    tend: int,
    is_problematic: bool,
) -> None:
    selector = Step6Select(
        trim_mhc=not is_problematic,
        tstart=tstart,
        tend=tend,
    )

    io = PDBIO()
    io.set_structure(structure)

    with out_pdb.open("w") as handle:
        handle.write(f"HEADER    {out_pdb.name}\n")
        if is_problematic:
            handle.write("REMARK    STEP6 PROBLEMATIC: MHC NOT TRIMMED\n")
        else:
            handle.write(f"REMARK    STEP6 MHC CHAIN A TRIMMED TO {tstart}-{tend}\n")
        io.save(handle, selector)


# =========================================================
# Per-structure processing
# =========================================================

def process_one(
    pdb_id: str,
    tstart: int,
    tend: int,
    duplicate_new_chains: Set[str],
    blacklist_resnames: Set[str],
) -> Optional[dict]:
    system_pdb = find_step5_pdb(pdb_id)
    if system_pdb is None:
        log(f"[MISSING] {pdb_id}: no step5 PDB found")
        return None

    cavity_pdb = find_filtered_cavity_pdb(pdb_id)
    if cavity_pdb is None:
        log(f"[MISSING] {pdb_id}: no filtered cavity PDB found; expected {pdb_id.lower()}_filtered_cavity.pdb")
        return None

    normalized_pdb = MERGED_INTERMEDIATE_DIR / f".{pdb_id.lower()}_normalized_resnames.tmp.pdb"
    resname_mapping_rows = normalize_long_resnames_pdb(
        in_pdb=system_pdb,
        out_pdb=normalized_pdb,
        pdb_id=pdb_id,
    )

    if resname_mapping_rows:
        parser_input_pdb = normalized_pdb
        log(f"[RENAMED] {pdb_id}: long_resnames={len(resname_mapping_rows)}")
    else:
        normalized_pdb.unlink(missing_ok=True)
        parser_input_pdb = system_pdb

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, str(parser_input_pdb))
    model = structure[0]

    if MHC_CHAIN_ID not in model:
        log(f"[ERROR] {pdb_id}: MHC chain {MHC_CHAIN_ID} not found in step5 PDB")
        return None

    removed_duplicate_chains = remove_duplicate_chains(structure, duplicate_new_chains)

    cavity_points = parse_cavity_points(cavity_pdb)

    if WRITE_MERGED_INTERMEDIATE_PDB:
        merged_pdb = MERGED_INTERMEDIATE_DIR / f"{pdb_id.lower()}{MERGED_SUFFIX}"
        write_merged_system_plus_cavity_pdb(parser_input_pdb, cavity_points, merged_pdb)
    else:
        merged_pdb = None

    if parser_input_pdb == normalized_pdb:
        normalized_pdb.unlink(missing_ok=True)

    contacts = find_cavity_contacts(
        structure=structure,
        cavity_points=cavity_points,
        cutoff=CAVITY_CONTACT_CUTOFF,
    )

    (
        possible_ligand_rows,
        problematic_mhc_rows,
        problematic_binder_rows,
        is_problematic,
        problematic_reasons,
    ) = classify_contacts(
        structure=structure,
        contacts=contacts,
        tstart=tstart,
        tend=tend,
        blacklist_resnames=blacklist_resnames,
    )

    log(
        f"[MAPPED] {pdb_id}: contacts={len(contacts)}; "
        f"possible_ligand_chains={len(possible_ligand_rows)}; "
        f"problematic_mhc_contacts={len(problematic_mhc_rows)}; "
        f"binder_sized_contacts={len(problematic_binder_rows)}; "
        f"problematic={is_problematic}; "
        f"removed_duplicates={removed_duplicate_chains}; "
        f"merged={merged_pdb.name if merged_pdb else 'NA'}"
    )

    return {
        "pdb": pdb_id,
        "system_pdb": str(system_pdb),
        "cavity_pdb": str(cavity_pdb),
        "merged_pdb": str(merged_pdb) if merged_pdb else "",
        "removed_duplicate_chains": ";".join(removed_duplicate_chains),
        "is_problematic": is_problematic,
        "n_contacts": len(contacts),
        "n_possible_ligand_chains": len(possible_ligand_rows),
        "n_problematic_mhc_contacts": len(problematic_mhc_rows),
        "n_binder_sized_contacts": len(problematic_binder_rows),
        "problematic_reasons": ";".join(
            reason
            for reason in [
                "mhc_aminoacid_contact",
                "mhc_non_aminoacid_contact",
                "binder_non_aminoacid_contact",
            ]
            if reason in problematic_reasons
        ),
        "possible_ligand_rows": possible_ligand_rows,
        "problematic_mhc_rows": problematic_mhc_rows,
        "problematic_binder_rows": problematic_binder_rows,
        "resname_mapping_rows": resname_mapping_rows,
    }


# =========================================================
# Summary writing
# =========================================================

def write_summary_csvs(results: List[dict]) -> None:
    possible_ligand_rows = []
    problematic_mhc_rows = []
    problematic_binder_rows = []
    resname_mapping_rows = []

    for result in results:
        pdb_id = result["pdb"]

        for row in result["possible_ligand_rows"]:
            possible_ligand_rows.append({"pdb": pdb_id, **row})

        for row in result["problematic_mhc_rows"]:
            problematic_mhc_rows.append({"pdb": pdb_id, **row})

        for row in result["problematic_binder_rows"]:
            problematic_binder_rows.append({"pdb": pdb_id, **row})

        resname_mapping_rows.extend(result.get("resname_mapping_rows", []))

    pd.DataFrame(possible_ligand_rows).to_csv(
        SUMMARY_DIR / "possible_ligands.csv",
        index=False,
    )

    pd.DataFrame(problematic_mhc_rows).to_csv(
        SUMMARY_DIR / "problematic_mhc_contacts.csv",
        index=False,
    )

    pd.DataFrame(problematic_binder_rows).to_csv(
        SUMMARY_DIR / "problematic_binder_contacts.csv",
        index=False,
    )

    if resname_mapping_rows:
        pd.DataFrame(resname_mapping_rows).drop_duplicates().to_csv(
            RESNAME_MAP_CSV,
            index=False,
        )
    else:
        pd.DataFrame(
            columns=["pdb", "old_resname", "new_resname", "chain_id", "resseq"]
        ).to_csv(RESNAME_MAP_CSV, index=False)

    problematic_list = [
        {
            "pdb": result["pdb"],
            "reason": result.get("problematic_reasons", ""),
        }
        for result in results
        if bool(result["is_problematic"])
    ]

    pd.DataFrame(problematic_list, columns=["pdb", "reason"]).to_csv(
        SUMMARY_DIR / "problematic_list.csv",
        index=False,
    )


# =========================================================
# Main
# =========================================================

def main() -> None:
    make_output_dirs()
    validate_inputs()

    remapped_df = load_remapped_rows(STEP5_REMAP_CSV)
    blacklist_resnames = load_blacklist_resnames(BLACKLIST_CSV)

    duplicate_old_chains = load_duplicate_old_chains(STEP1_CSV)
    old_to_new_chain_map = load_old_to_new_chain_map(STEP5_CHAIN_MAPPING_CSV)
    duplicate_new_chain_map = build_duplicate_new_chain_map(
        duplicate_old_chains=duplicate_old_chains,
        old_to_new_chain_map=old_to_new_chain_map,
    )

    log(f"[CSV] remapped= {len(remapped_df)}")
    log(f"[CSV] duplicate_pdbs= {len(duplicate_new_chain_map)}")
    log("")
    log("[START] STEP6")

    results = []
    missing_or_failed = 0

    for _, row in remapped_df.iterrows():
        pdb_id = normalize_pdb_id(row["pdb_id"])
        tstart = int(row["tstart"])
        tend = int(row["tend"])
        duplicate_new_chains = duplicate_new_chain_map.get(pdb_id, set())

        log(f"[START] {pdb_id}")

        try:
            result = process_one(
                pdb_id=pdb_id,
                tstart=tstart,
                tend=tend,
                duplicate_new_chains=duplicate_new_chains,
                blacklist_resnames=blacklist_resnames,
            )
        except Exception as exc:
            missing_or_failed += 1
            log(f"[ERROR] {pdb_id}: {exc}")
            continue

        if result is None:
            missing_or_failed += 1
            continue

        results.append(result)

    write_summary_csvs(results)

    log("")
    log("[SUMMARY] STEP6")
    log(f"[SUMMARY] processed={len(results)}")
    log(f"[SUMMARY] missing_or_failed={missing_or_failed}")
    log(f"[OUT] {MERGED_INTERMEDIATE_DIR}")
    log(f"[OUT] {SUMMARY_DIR}")


if __name__ == "__main__":
    main()
