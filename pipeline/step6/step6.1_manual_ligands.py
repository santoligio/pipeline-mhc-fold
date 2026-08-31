#!/usr/bin/env python3
from pathlib import Path
from typing import Dict, Optional, Set, Tuple, List
import string
import re

import pandas as pd
from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.Polypeptide import is_aa


# =========================================================
# Configuration
# =========================================================

PIPELINE_DIR = Path(__file__).resolve().parents[1]

STEP1_CSV = PIPELINE_DIR / "step1" / "pdb" / "pdb_assemblies.csv"
STEP5_REMAP_CSV = PIPELINE_DIR / "step5" / "pdb" / "pdb_assemblies_remapped.csv"
STEP5_CHAIN_MAP_CSV = PIPELINE_DIR / "step5" / "pdb" / "chain_map.csv"
STEP5_PDB_DIR = PIPELINE_DIR / "step5" / "pdb" / "1_mhc_configured"

STEP6_1_DIR = PIPELINE_DIR / "step6_fix" / "pdb"
TRIMMED_MHC_DIR = STEP6_1_DIR / "2_trimmed_mhc"
SUMMARY_DIR = STEP6_1_DIR / "summaries"

# Original step6 outputs. This script never edits them.
PROBLEMATIC_LIST_ORIGINAL_CSV = SUMMARY_DIR / "problematic_list.csv"
PROBLEMATIC_CONTACTS_ORIGINAL_CSV = SUMMARY_DIR / "problematic_mhc_contacts.csv"
PROBLEMATIC_BINDER_CONTACTS_ORIGINAL_CSV = SUMMARY_DIR / "problematic_binder_contacts.csv"

# Working files updated after each manual CSV run.
PROBLEMATIC_LIST_CURRENT_CSV = SUMMARY_DIR / "problematic_list_current.csv"
PROBLEMATIC_CONTACTS_CURRENT_CSV = SUMMARY_DIR / "problematic_mhc_contacts_current.csv"
PROBLEMATIC_BINDER_CONTACTS_CURRENT_CSV = SUMMARY_DIR / "problematic_binder_contacts_current.csv"

MANUAL_LIGANDS_CSV = SUMMARY_DIR / "manual_problematic_ligands.csv"
POSSIBLE_LIGANDS_CSV = SUMMARY_DIR / "possible_ligands.csv"

MHC_CHAIN_ID = "A"

STEP5_STRUCTURE_SUFFIX = "_mhc_configured.pdb"
OUTPUT_SUFFIX = "_trimmed_mhc.pdb"

PDB_CHAIN_IDS = list(string.ascii_uppercase + string.ascii_lowercase + string.digits)

DUPLICATE_CHAINS_BY_PDB: Dict[str, Set[str]] = {}


# =========================================================
# Logging / setup
# =========================================================

def log(message: str) -> None:
    print(message, flush=True)


def make_output_dirs() -> None:
    TRIMMED_MHC_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


def validate_inputs() -> None:
    log("[CONFIG] STEP6.1")
    log(f"STEP1_CSV:             {STEP1_CSV} | exists={STEP1_CSV.is_file()}")
    log(f"STEP5_REMAP_CSV:       {STEP5_REMAP_CSV} | exists={STEP5_REMAP_CSV.is_file()}")
    log(f"STEP5_CHAIN_MAP_CSV:   {STEP5_CHAIN_MAP_CSV} | exists={STEP5_CHAIN_MAP_CSV.is_file()}")
    log(f"STEP5_PDB_DIR:         {STEP5_PDB_DIR} | exists={STEP5_PDB_DIR.is_dir()}")
    log(f"ORIGINAL_LIST_CSV:     {PROBLEMATIC_LIST_ORIGINAL_CSV} | exists={PROBLEMATIC_LIST_ORIGINAL_CSV.is_file()}")
    log(f"ORIGINAL_CONTACTS_CSV: {PROBLEMATIC_CONTACTS_ORIGINAL_CSV} | exists={PROBLEMATIC_CONTACTS_ORIGINAL_CSV.is_file()}")
    log(f"ORIGINAL_BINDER_CSV:   {PROBLEMATIC_BINDER_CONTACTS_ORIGINAL_CSV} | exists={PROBLEMATIC_BINDER_CONTACTS_ORIGINAL_CSV.is_file()}")
    log(f"MANUAL_LIGANDS_CSV:    {MANUAL_LIGANDS_CSV} | exists={MANUAL_LIGANDS_CSV.is_file()}")
    log(f"POSSIBLE_LIGANDS_CSV:  {POSSIBLE_LIGANDS_CSV} | exists={POSSIBLE_LIGANDS_CSV.is_file()}")
    log(f"Output dir:            {TRIMMED_MHC_DIR}")
    log("")

    required_files = [
        STEP1_CSV,
        STEP5_REMAP_CSV,
        STEP5_CHAIN_MAP_CSV,
        PROBLEMATIC_LIST_ORIGINAL_CSV,
        PROBLEMATIC_CONTACTS_ORIGINAL_CSV,
        PROBLEMATIC_BINDER_CONTACTS_ORIGINAL_CSV,
        MANUAL_LIGANDS_CSV,
    ]

    for path in required_files:
        if not path.is_file():
            raise SystemExit(f"ERROR: required file does not exist: {path}")

    if not STEP5_PDB_DIR.is_dir():
        raise SystemExit(f"ERROR: STEP5_PDB_DIR does not exist: {STEP5_PDB_DIR}")


# =========================================================
# CSV loading
# =========================================================

def normalize_pdb_id(value) -> str:
    return str(value).split("-")[0].strip().upper()


def load_remapped_ranges(remap_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(remap_csv)

    required = {"pdb_id", "tstart", "tend"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"ERROR: STEP5_REMAP_CSV missing columns: {sorted(missing)}")

    df = df.copy()
    df["pdb_id"] = df["pdb_id"].astype(str).str[:4].str.upper()

    if df.empty:
        raise SystemExit("ERROR: no rows found in STEP5_REMAP_CSV")

    return df


def load_duplicate_chains_by_pdb(step1_csv: Path, chain_map_csv: Path) -> Dict[str, Set[str]]:
    """
    Load duplicate MHC chains from step1 and translate them to current step5 chain IDs.
    """
    step1_df = pd.read_csv(step1_csv)
    chain_map_df = pd.read_csv(chain_map_csv)

    required_step1 = {"pdb", "chain", "status"}
    missing_step1 = required_step1 - set(step1_df.columns)
    if missing_step1:
        raise SystemExit(f"ERROR: STEP1_CSV missing columns: {sorted(missing_step1)}")

    required_map = {"pdb", "old_chain", "new_chain"}
    missing_map = required_map - set(chain_map_df.columns)
    if missing_map:
        raise SystemExit(f"ERROR: STEP5_CHAIN_MAP_CSV missing columns: {sorted(missing_map)}")

    chain_lookup = {}
    for _, row in chain_map_df.iterrows():
        pdb_id = normalize_pdb_id(row["pdb"])
        old_chain = str(row["old_chain"]).strip()
        new_chain = str(row["new_chain"]).strip()

        if old_chain:
            chain_lookup[(pdb_id, old_chain)] = new_chain

    duplicate_chains: Dict[str, Set[str]] = {}

    status = step1_df["status"].astype(str).str.strip().str.lower()
    duplicate_df = step1_df[status == "duplicate"].copy()

    for _, row in duplicate_df.iterrows():
        pdb_id = normalize_pdb_id(row["pdb"])
        old_chain = str(row["chain"]).strip()
        mapped_chain = chain_lookup.get((pdb_id, old_chain))

        if not mapped_chain or mapped_chain == MHC_CHAIN_ID:
            continue

        duplicate_chains.setdefault(pdb_id, set()).add(mapped_chain)

    return duplicate_chains


def load_problematic_set(problematic_csv: Path) -> Set[str]:
    df = pd.read_csv(problematic_csv)

    if df.empty:
        return set()

    column = "pdb" if "pdb" in df.columns else df.columns[0]

    problematic = set()
    for value in df[column].dropna():
        problematic.add(normalize_pdb_id(value))

    return problematic


def load_problematic_contacts(problematic_contacts_csv: Path) -> Dict[str, List[dict]]:
    """
    Load MHC problematic contacts grouped by PDB.
    """
    df = pd.read_csv(problematic_contacts_csv)

    if df.empty:
        return {}

    required = {"pdb", "resname", "resseq"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(
            f"ERROR: PROBLEMATIC_CONTACTS_CSV missing columns: {sorted(missing)}"
        )

    grouped: Dict[str, List[dict]] = {}

    for _, row in df.iterrows():
        pdb_id = normalize_pdb_id(row["pdb"])
        chain_id = str(row["chain_id"]).strip() if "chain_id" in df.columns else MHC_CHAIN_ID

        grouped.setdefault(pdb_id, []).append({
            "chain_id": chain_id or MHC_CHAIN_ID,
            "resname": str(row["resname"]).strip(),
            "resseq": int(row["resseq"]),
        })

    return grouped


def load_binder_problematic_contacts(problematic_contacts_csv: Path) -> Dict[str, List[dict]]:
    """
    Load automatic binder non-amino-acid contacts grouped by PDB.
    """
    df = pd.read_csv(problematic_contacts_csv)

    if df.empty:
        return {}

    required = {"pdb", "chain_id", "resname", "resseq"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(
            f"ERROR: PROBLEMATIC_BINDER_CONTACTS_CSV missing columns: {sorted(missing)}"
        )

    grouped: Dict[str, List[dict]] = {}

    for _, row in df.iterrows():
        pdb_id = normalize_pdb_id(row["pdb"])
        grouped.setdefault(pdb_id, []).append({
            "chain_id": str(row["chain_id"]).strip(),
            "resname": str(row["resname"]).strip(),
            "resseq": int(row["resseq"]),
        })

    return grouped


def initialize_current_problematic_files() -> None:
    """
    Create current problematic CSVs from original step6 outputs if needed.
    Original step6 outputs are never modified.
    """
    if not PROBLEMATIC_LIST_CURRENT_CSV.is_file():
        PROBLEMATIC_LIST_CURRENT_CSV.write_bytes(PROBLEMATIC_LIST_ORIGINAL_CSV.read_bytes())
        log(f"[INIT] {PROBLEMATIC_LIST_CURRENT_CSV.name}")

    if not PROBLEMATIC_CONTACTS_CURRENT_CSV.is_file():
        PROBLEMATIC_CONTACTS_CURRENT_CSV.write_bytes(PROBLEMATIC_CONTACTS_ORIGINAL_CSV.read_bytes())
        log(f"[INIT] {PROBLEMATIC_CONTACTS_CURRENT_CSV.name}")

    if not PROBLEMATIC_BINDER_CONTACTS_CURRENT_CSV.is_file():
        PROBLEMATIC_BINDER_CONTACTS_CURRENT_CSV.write_bytes(PROBLEMATIC_BINDER_CONTACTS_ORIGINAL_CSV.read_bytes())
        log(f"[INIT] {PROBLEMATIC_BINDER_CONTACTS_CURRENT_CSV.name}")


def _filter_current_contacts(
    contacts_csv: Path,
    resolved_pdbs: Set[str],
) -> Set[str]:
    """
    Remove resolved PDB rows from a current contact CSV.

    Returns the set of PDBs still present in that CSV after filtering.
    """
    if not contacts_csv.is_file():
        return set()

    df = pd.read_csv(contacts_csv)

    if df.empty or "pdb" not in df.columns:
        df.to_csv(contacts_csv, index=False)
        return set()

    resolved = {normalize_pdb_id(pdb_id) for pdb_id in resolved_pdbs}

    df = df[~df["pdb"].apply(normalize_pdb_id).isin(resolved)].copy()
    df.to_csv(contacts_csv, index=False)

    if df.empty:
        return set()

    return {
        normalize_pdb_id(value)
        for value in df["pdb"].dropna()
    }


def remove_resolved_problematic_pdbs(resolved_pdbs: Set[str]) -> None:
    """
    Remove fully resolved PDBs from current problematic CSVs.

    The current problematic list is rebuilt from the remaining current MHC and
    binder contact CSVs. This prevents auto-resolved non-amino-acid contacts
    from leaving stale PDB IDs in problematic_list_current.csv.
    """
    if not resolved_pdbs:
        return

    resolved = {normalize_pdb_id(pdb_id) for pdb_id in resolved_pdbs}

    remaining_mhc = _filter_current_contacts(
        PROBLEMATIC_CONTACTS_CURRENT_CSV,
        resolved,
    )
    remaining_binder = _filter_current_contacts(
        PROBLEMATIC_BINDER_CONTACTS_CURRENT_CSV,
        resolved,
    )
    remaining_problematic = remaining_mhc | remaining_binder

    if PROBLEMATIC_LIST_CURRENT_CSV.is_file():
        df = pd.read_csv(PROBLEMATIC_LIST_CURRENT_CSV)

        if df.empty:
            df.to_csv(PROBLEMATIC_LIST_CURRENT_CSV, index=False)
        else:
            column = "pdb" if "pdb" in df.columns else df.columns[0]
            df = df[
                df[column].apply(normalize_pdb_id).isin(remaining_problematic)
            ].copy()
            df.to_csv(PROBLEMATIC_LIST_CURRENT_CSV, index=False)

    log(f"[UPDATED] current_problematic_removed={sorted(resolved)}")
    log(f"[UPDATED] current_problematic_remaining={len(remaining_problematic)}")


def parse_single_ligand_group(text: str) -> List[int]:
    value = str(text).strip()

    if not value or value.lower() == "nan":
        return []

    cleaned = value.replace(",", " ")
    values = []

    for item in cleaned.split():
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start = int(start_text)
            end = int(end_text)

            if start > end:
                raise ValueError(f"Invalid ligand range: {item}")

            values.extend(range(start, end + 1))
        else:
            values.append(int(item))

    return list(dict.fromkeys(values))


def parse_ligands_field(value) -> List[List[int]]:
    """
    Parse the ligands column.

    Semicolon separates ligand groups. Each group becomes one new chain.
    """
    text = str(value).strip()

    if not text or text.lower() == "nan":
        return []

    groups = []

    for group_text in text.split(";"):
        group = parse_single_ligand_group(group_text)

        if group:
            groups.append(group)

    return groups


def load_manual_ligand_rows(path: Path) -> Dict[str, dict]:
    """
    Load manual MHC actions keyed by PDB ID.
    """
    df = pd.read_csv(path)

    required = {"pdb_id", "action", "ligands"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"ERROR: manual CSV missing columns: {sorted(missing)}")

    manual: Dict[str, dict] = {}

    for _, row in df.iterrows():
        pdb_id = normalize_pdb_id(row["pdb_id"])
        action = str(row["action"]).strip().lower()

        if action not in {"move", "checked"}:
            log(f"[WARNING] {pdb_id}: unsupported manual action={row['action']}")
            continue

        ligand_groups = parse_ligands_field(row["ligands"])

        if action == "checked" and ligand_groups:
            log(f"[WARNING] {pdb_id}: action=checked ignores non-empty ligands")

        if action == "move" and not ligand_groups:
            log(f"[WARNING] {pdb_id}: action=move has no ligands; treating as checked")
            action = "checked"

        manual[pdb_id] = {
            "action": action,
            "ligand_groups": ligand_groups if action == "move" else [],
        }

    return manual


# =========================================================
# File helpers / parser fixes
# =========================================================

def find_step5_structure(pdb_id: str) -> Optional[Path]:
    pdb_id_lower = pdb_id.lower()
    exact = STEP5_PDB_DIR / f"{pdb_id_lower}{STEP5_STRUCTURE_SUFFIX}"

    if exact.is_file():
        return exact

    matches = sorted(STEP5_PDB_DIR.glob(f"{pdb_id_lower}*.pdb"))
    return matches[0] if matches else None


def output_path_for(pdb_id: str) -> Path:
    return TRIMMED_MHC_DIR / f"{pdb_id.lower()}{OUTPUT_SUFFIX}"


def format_atom_name(atom_name: str) -> str:
    atom_name = str(atom_name).strip()

    if len(atom_name) >= 4:
        return atom_name[:4]

    if atom_name and atom_name[0].isdigit():
        return f"{atom_name:<4}"

    return f"{atom_name:>4}"


def parse_occupancy_bfactor(tokens: List[str]) -> Tuple[float, float]:
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
    raw = str(raw_resname).strip()

    if len(raw) <= 3:
        return " ", raw, raw

    if len(raw) == 4 and raw[0].isalpha():
        return raw[0], raw[1:], raw[1:]

    return " ", raw[-3:], raw


def atom_line_needs_resname_fix(line: str) -> bool:
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


def rebuild_atom_line_from_tokens(line: str) -> str:
    tokens = line.split()

    if len(tokens) < 11:
        return line

    record = tokens[0]
    serial = int(tokens[1])
    atom_name = tokens[2]
    altloc, new_resname, _old_resname = split_altloc_resname(tokens[3])
    chain_id = tokens[4][0]
    resseq = int(tokens[5])
    x = float(tokens[6])
    y = float(tokens[7])
    z = float(tokens[8])
    occupancy, bfactor = parse_occupancy_bfactor(tokens)

    if tokens[-1].isalpha() and len(tokens[-1]) <= 2:
        element = tokens[-1].upper()
    else:
        element = "".join(ch for ch in atom_name if ch.isalpha())[:1].upper() or "C"

    return (
        f"{record:<6}{serial:5d} {format_atom_name(atom_name)}"
        f"{altloc:1}{new_resname:>3} {chain_id:1}{resseq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}"
        f"{occupancy:6.2f}{bfactor:6.2f}          "
        f"{element:>2}\n"
    )


def rebuild_ter_line_from_tokens(line: str) -> str:
    tokens = line.split()

    if len(tokens) < 5 or tokens[0] != "TER":
        return line

    serial = int(tokens[1])
    _altloc, new_resname, _old_resname = split_altloc_resname(tokens[2])
    chain_id = tokens[3][0]
    resseq = int(tokens[4])

    return f"TER   {serial:5d}      {new_resname:>3} {chain_id:1}{resseq:4d}\n"


def write_parser_ready_pdb(in_pdb: Path, out_pdb: Path) -> bool:
    """
    Write a temporary parser-ready PDB only when true long residue names exist.
    """
    has_fix = False

    with in_pdb.open("r") as fin:
        for line in fin:
            if line.startswith(("ATOM", "HETATM")) and atom_line_needs_resname_fix(line):
                has_fix = True
                break
            if line.startswith("TER") and ter_line_needs_resname_fix(line):
                has_fix = True
                break

    if not has_fix:
        return False

    with in_pdb.open("r") as fin, out_pdb.open("w") as fout:
        for line in fin:
            if line.startswith(("ATOM", "HETATM")) and atom_line_needs_resname_fix(line):
                try:
                    fout.write(rebuild_atom_line_from_tokens(line))
                except Exception:
                    fout.write(line)
            elif line.startswith("TER") and ter_line_needs_resname_fix(line):
                try:
                    fout.write(rebuild_ter_line_from_tokens(line))
                except Exception:
                    fout.write(line)
            else:
                fout.write(line)

    return True


def load_structure_for_editing(pdb_id: str, input_pdb: Path):
    temp_pdb = SUMMARY_DIR / f".{pdb_id.lower()}_parser_ready.tmp.pdb"
    has_temp = write_parser_ready_pdb(input_pdb, temp_pdb)
    parser_input = temp_pdb if has_temp else input_pdb

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, str(parser_input))

    if has_temp:
        temp_pdb.unlink(missing_ok=True)

    return structure


def remove_duplicate_chains_from_structure(structure, pdb_id: str) -> List[str]:
    duplicate_chains = DUPLICATE_CHAINS_BY_PDB.get(pdb_id, set())

    if not duplicate_chains:
        return []

    model = structure[0]
    removed = []

    for chain_id in sorted(duplicate_chains):
        if chain_id == MHC_CHAIN_ID:
            continue

        if chain_id in model:
            model.detach_child(chain_id)
            removed.append(chain_id)

    return removed


def load_structure_without_duplicates(pdb_id: str, input_pdb: Path):
    structure = load_structure_for_editing(pdb_id, input_pdb)
    removed = remove_duplicate_chains_from_structure(structure, pdb_id)

    if removed:
        log(f"[DUPLICATES] {pdb_id}: removed_chains={removed}")

    return structure


# =========================================================
# Chain / residue operations
# =========================================================

def used_chain_ids(structure) -> Set[str]:
    return {chain.id for chain in structure[0].get_chains()}


def choose_new_chain_id(structure) -> str:
    used = used_chain_ids(structure)

    for chain_id in PDB_CHAIN_IDS:
        if chain_id not in used:
            return chain_id

    raise ValueError("No available single-character chain ID left for new ligand chain.")


def residue_resseq(residue) -> int:
    return int(residue.id[1])


def move_mhc_residues_to_new_chain(
    structure,
    ligand_resseqs: Set[int],
    new_chain_id: str,
) -> int:
    """
    Move selected MHC chain A residues into a new ligand chain.
    """
    if not ligand_resseqs:
        return 0

    model = structure[0]

    if MHC_CHAIN_ID not in model:
        raise ValueError(f"MHC chain {MHC_CHAIN_ID} not found")

    mhc_chain = model[MHC_CHAIN_ID]
    new_chain = mhc_chain.__class__(new_chain_id)

    residues_to_move = [
        residue
        for residue in list(mhc_chain.get_residues())
        if residue_resseq(residue) in ligand_resseqs
    ]

    if not residues_to_move:
        raise ValueError(
            f"No residues found in chain {MHC_CHAIN_ID} for ligand residues "
            f"{sorted(ligand_resseqs)}"
        )

    for residue in residues_to_move:
        mhc_chain.detach_child(residue.id)
        new_chain.add(residue)

    model.add(new_chain)

    return len(residues_to_move)


def move_chain_residues_to_new_chain(
    structure,
    source_chain_id: str,
    ligand_resseqs: Set[int],
    new_chain_id: str,
) -> int:
    """
    Move selected residues from source_chain_id into a new ligand chain.
    """
    if not ligand_resseqs:
        return 0

    model = structure[0]

    if source_chain_id not in model:
        raise ValueError(f"Source chain {source_chain_id} not found")

    source_chain = model[source_chain_id]
    new_chain = source_chain.__class__(new_chain_id)

    residues_to_move = [
        residue
        for residue in list(source_chain.get_residues())
        if residue_resseq(residue) in ligand_resseqs
    ]

    if not residues_to_move:
        raise ValueError(
            f"No residues found in chain {source_chain_id} for ligand residues "
            f"{sorted(ligand_resseqs)}"
        )

    for residue in residues_to_move:
        source_chain.detach_child(residue.id)
        new_chain.add(residue)

    model.add(new_chain)

    return len(residues_to_move)


def trim_mhc_to_range(structure, tstart: int, tend: int) -> int:
    model = structure[0]

    if MHC_CHAIN_ID not in model:
        raise ValueError(f"MHC chain {MHC_CHAIN_ID} not found")

    mhc_chain = model[MHC_CHAIN_ID]

    removed = 0
    for residue in list(mhc_chain.get_residues()):
        resseq = residue_resseq(residue)

        if not (tstart <= resseq <= tend):
            mhc_chain.detach_child(residue.id)
            removed += 1

    return removed


def append_possible_ligand_chain(pdb_id: str, chain_id: str) -> None:
    new_row = pd.DataFrame([{
        "pdb": pdb_id,
        "chain_id": chain_id,
    }])

    if POSSIBLE_LIGANDS_CSV.is_file():
        old_df = pd.read_csv(POSSIBLE_LIGANDS_CSV)

        for col in new_row.columns:
            if col not in old_df.columns:
                old_df[col] = ""

        for col in old_df.columns:
            if col not in new_row.columns:
                new_row[col] = ""

        combined = pd.concat([old_df, new_row[old_df.columns]], ignore_index=True)
    else:
        combined = new_row

    if {"pdb", "chain_id"}.issubset(combined.columns):
        combined = combined.drop_duplicates(subset=["pdb", "chain_id"], keep="first")

    combined.to_csv(POSSIBLE_LIGANDS_CSV, index=False)


def save_structure(structure, out_pdb: Path, remark_lines=None) -> None:
    io = PDBIO()
    io.set_structure(structure)

    with out_pdb.open("w") as handle:
        handle.write(f"HEADER    {out_pdb.name}\n")
        for remark in remark_lines or []:
            handle.write(f"REMARK    {remark}\n")
        io.save(handle)


# =========================================================
# Processing
# =========================================================

def trim_non_problematic(pdb_id: str, input_pdb: Path, tstart: int, tend: int) -> None:
    structure = load_structure_without_duplicates(pdb_id, input_pdb)
    removed_count = trim_mhc_to_range(structure, tstart, tend)

    out_pdb = output_path_for(pdb_id)

    save_structure(
        structure,
        out_pdb,
        remark_lines=[
            "STEP6.1 NON-PROBLEMATIC TRIM",
            f"MHC CHAIN {MHC_CHAIN_ID} TRIMMED TO {tstart}-{tend}",
        ],
    )

    log(
        f"[SAVED] {pdb_id}: non-problematic trimmed; "
        f"trimmed_removed={removed_count}; output={out_pdb.name}"
    )


def build_auto_groups_from_non_aminoacid_contacts(
    contacts: List[dict],
    default_chain_id: str,
) -> List[dict]:
    """
    Build one automatic ligand group per unique non-amino-acid contact residue.
    """
    groups = []
    seen = set()

    for entry in contacts:
        resname = str(entry["resname"]).strip().upper()

        if is_aa(resname, standard=False):
            continue

        chain_id = str(entry.get("chain_id", default_chain_id)).strip() or default_chain_id
        resseq = int(entry["resseq"])
        key = (chain_id, resseq)

        if not chain_id or key in seen:
            continue

        groups.append({
            "source_chain_id": chain_id,
            "ligand_resseqs": {resseq},
        })
        seen.add(key)

    return groups


def has_aminoacid_contacts(contacts: List[dict]) -> bool:
    return any(
        is_aa(str(entry["resname"]).strip().upper(), standard=False)
        for entry in contacts
    )


def process_problematic(
    pdb_id: str,
    input_pdb: Path,
    tstart: int,
    tend: int,
    manual_rows: Dict[str, dict],
    mhc_contacts: List[dict],
    binder_contacts: List[dict],
) -> dict:
    manual_entry = manual_rows.get(pdb_id)

    mhc_has_aminoacid = has_aminoacid_contacts(mhc_contacts)

    if mhc_has_aminoacid and manual_entry is None:
        log(f"[WAITING] {pdb_id}: missing manual CSV row for amino-acid MHC contact")
        return {
            "pdb": pdb_id,
            "action": "waiting_manual_csv",
            "input_pdb": str(input_pdb),
            "output_pdb": "",
        }

    structure = load_structure_without_duplicates(pdb_id, input_pdb)

    ligand_chain_ids = []
    moved_count = 0

    mhc_auto_groups = build_auto_groups_from_non_aminoacid_contacts(
        contacts=mhc_contacts,
        default_chain_id=MHC_CHAIN_ID,
    )

    for group in mhc_auto_groups:
        new_chain_id = choose_new_chain_id(structure)

        moved_this = move_mhc_residues_to_new_chain(
            structure=structure,
            ligand_resseqs=group["ligand_resseqs"],
            new_chain_id=new_chain_id,
        )

        moved_count += moved_this
        ligand_chain_ids.append(new_chain_id)
        append_possible_ligand_chain(pdb_id, new_chain_id)

        log(
            f"[MHC_AUTO] {pdb_id}: "
            f"{sorted(group['ligand_resseqs'])} -> chain {new_chain_id}"
        )

    if mhc_has_aminoacid:
        if manual_entry["action"] == "move":
            for ligand_group in manual_entry["ligand_groups"]:
                new_chain_id = choose_new_chain_id(structure)

                moved_this = move_mhc_residues_to_new_chain(
                    structure=structure,
                    ligand_resseqs=set(ligand_group),
                    new_chain_id=new_chain_id,
                )

                moved_count += moved_this
                ligand_chain_ids.append(new_chain_id)
                append_possible_ligand_chain(pdb_id, new_chain_id)

            log(f"[MHC_MANUAL] {pdb_id}: moved ligand groups to chains={ligand_chain_ids}")
        else:
            log(f"[MHC_MANUAL] {pdb_id}: manually checked; no ligand chains created")

    binder_groups = build_auto_groups_from_non_aminoacid_contacts(
        contacts=binder_contacts,
        default_chain_id="",
    )

    skipped_binder_aminoacids = len(binder_contacts) - len(binder_groups)

    for group in binder_groups:
        new_chain_id = choose_new_chain_id(structure)

        moved_this = move_chain_residues_to_new_chain(
            structure=structure,
            source_chain_id=group["source_chain_id"],
            ligand_resseqs=group["ligand_resseqs"],
            new_chain_id=new_chain_id,
        )

        moved_count += moved_this
        ligand_chain_ids.append(new_chain_id)
        append_possible_ligand_chain(pdb_id, new_chain_id)

        log(
            f"[BINDER_AUTO] {pdb_id}: {group['source_chain_id']} "
            f"{sorted(group['ligand_resseqs'])} -> chain {new_chain_id}"
        )

    if skipped_binder_aminoacids:
        log(f"[BINDER] {pdb_id}: amino-acid contacts recorded but not moved={skipped_binder_aminoacids}")

    removed_count = trim_mhc_to_range(
        structure=structure,
        tstart=tstart,
        tend=tend,
    )

    out_pdb = output_path_for(pdb_id)

    save_structure(
        structure,
        out_pdb,
        remark_lines=[
            "STEP6.1 MHC MANUAL CSV / BINDER AUTO CORRECTION",
            f"MHC CHAIN {MHC_CHAIN_ID} TRIMMED TO {tstart}-{tend}",
            f"NEW LIGAND CHAINS: {','.join(ligand_chain_ids)}",
        ],
    )

    if manual_entry is None:
        action_label = "auto_non_aminoacid_only"
    else:
        action_label = manual_entry["action"]

    log(
        f"[SAVED] {pdb_id}: action={action_label}; "
        f"new_ligand_chains={ligand_chain_ids}; moved={moved_count}; "
        f"trimmed_removed={removed_count}; output={out_pdb.name}"
    )

    return {
        "pdb": pdb_id,
        "action": "resolved_problematic",
        "input_pdb": str(input_pdb),
        "output_pdb": str(out_pdb),
    }


def main() -> None:
    make_output_dirs()
    validate_inputs()

    initialize_current_problematic_files()

    global DUPLICATE_CHAINS_BY_PDB

    remapped_df = load_remapped_ranges(STEP5_REMAP_CSV)
    DUPLICATE_CHAINS_BY_PDB = load_duplicate_chains_by_pdb(STEP1_CSV, STEP5_CHAIN_MAP_CSV)

    problematic_set = load_problematic_set(PROBLEMATIC_LIST_CURRENT_CSV)
    problematic_contacts_by_pdb = load_problematic_contacts(PROBLEMATIC_CONTACTS_CURRENT_CSV)
    binder_contacts_by_pdb = load_binder_problematic_contacts(PROBLEMATIC_BINDER_CONTACTS_CURRENT_CSV)
    manual_rows = load_manual_ligand_rows(MANUAL_LIGANDS_CSV)

    log(f"[CSV] remapped=      {len(remapped_df)}")
    log(f"[CSV] duplicate_pdbs= {len(DUPLICATE_CHAINS_BY_PDB)}")
    log(f"[CSV] problematic=   {len(problematic_set)}")
    log(f"[CSV] mhc_contacts=  {len(problematic_contacts_by_pdb)}")
    log(f"[CSV] binder_auto=   {len(binder_contacts_by_pdb)}")
    log(f"[CSV] manual_rows=   {len(manual_rows)}")
    log("")
    log("[START] STEP6.1")

    problematic_set = (
        set(problematic_set)
        | set(problematic_contacts_by_pdb)
        | set(binder_contacts_by_pdb)
    )

    resolved_pdbs = set()

    for _, row in remapped_df.iterrows():
        pdb_id = normalize_pdb_id(row["pdb_id"])
        tstart = int(row["tstart"])
        tend = int(row["tend"])

        out_pdb = output_path_for(pdb_id)
        if out_pdb.is_file():
            log(f"[SKIPPED] {pdb_id}: already present in 2_trimmed_mhc")
            continue

        input_pdb = find_step5_structure(pdb_id)

        if input_pdb is None:
            log(f"[MISSING] {pdb_id}: no step5 structure found")
            continue

        if pdb_id not in problematic_set:
            trim_non_problematic(pdb_id, input_pdb, tstart, tend)
            continue

        try:
            action = process_problematic(
                pdb_id=pdb_id,
                input_pdb=input_pdb,
                tstart=tstart,
                tend=tend,
                manual_rows=manual_rows,
                mhc_contacts=problematic_contacts_by_pdb.get(pdb_id, []),
                binder_contacts=binder_contacts_by_pdb.get(pdb_id, []),
            )

            if action["action"] == "resolved_problematic":
                resolved_pdbs.add(pdb_id)
        except Exception as exc:
            log(f"[ERROR] {pdb_id}: {exc}")

    remove_resolved_problematic_pdbs(resolved_pdbs)

    log("")
    log("[SUMMARY] STEP6.1")
    log(f"[OUT] {TRIMMED_MHC_DIR}")


if __name__ == "__main__":
    main()
