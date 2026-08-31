#!/usr/bin/env python3
"""
Separate MHC complex into groups of structures for analysis.

Structure source priority per PDB:
  1. step9/pdb/modified_pdbs/  — manually edited full complex, when present
  2. step8/pdb/1_mhc_remodeled/  — loop-remodeled structure (not all PDBs have one)
  3. step7/pdb/1_filtered_structures/  — fallback when no modified/remodeled file exists

For each structure:
1. MHC chain A only.
2. Each binder chain by itself.
3. MHC + all classified ligand chains.
4. MHC + all classified ligand chains to a ligand-specific folder only
   when ligand chains exist.
5. Full MHC complex files, respecting priority rules.
6. Full MHC complex files only when binder=yes in step9_annotations.csv

step9_residues.csv:
     columns: pdb_id, resname, resid, type, nonstandard, in_ptm_dict
     type:        mhc | binder | ligand
     nonstandard: yes  when Biopython's is_aa(..., standard=True) returns False
     in_ptm_dict: yes  when the residue code is in the curated PTM table
     Only rows where at least one flag is "yes" are written.
"""

import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import pandas as pd
from Bio.PDB import PDBIO, PDBParser, Select, is_aa

# =========================================================
# Configuration
# =========================================================

PIPELINE_DIR = Path(__file__).resolve().parents[1]

DATABASE = "pdb"

STEP7_DIR = PIPELINE_DIR / "step7" / DATABASE
STEP9_DIR = PIPELINE_DIR / "step9" / DATABASE
INPUT_PDB_DIR = STEP7_DIR / "1_filtered_structures"
STEP7_SUMMARY_DIR = STEP7_DIR / "summaries"
BINDERS_CSV = STEP7_SUMMARY_DIR / "step7_binders.csv"
LIGANDS_CSV = STEP7_SUMMARY_DIR / "step7_ligands.csv"

ANNOTATIONS_CSV = (
    PIPELINE_DIR
    / "step2-1"
    / "pdb"
    / "filtered"
    / "pdb_mhc_annotations_filtered.csv"
)
ANNOTATIONS_OUT_CSV = STEP9_DIR / "step9_annotations.csv"


STEP8_REMODELED_DIR = PIPELINE_DIR / "step8" / DATABASE / "1_mhc_remodeled"

MODIFIED_PDB_DIR = STEP9_DIR / "modified_pdbs"

OUTPUT_MHC_ONLY    = STEP9_DIR / "1_mhc_only"
OUTPUT_BINDERS     = STEP9_DIR / "2_binders"
OUTPUT_MHC_ALL     = STEP9_DIR / "3_mhc_all"
OUTPUT_MHC_LIGANDS = STEP9_DIR / "4_mhc_ligands"
OUTPUT_MHC_COMPLEX = STEP9_DIR / "5_mhc_complex"
OUTPUT_MHC_BINDER  = STEP9_DIR / "6_mhc_binder"
RESIDUES_CSV       = STEP9_DIR / "step9_residues.csv"

MHC_CHAIN_ID    = "A"
INPUT_PDB_PATTERN = "*.pdb"

MHC_ONLY_SUFFIX        = "_mhc_groove.pdb"
BINDER_SUFFIX_TEMPLATE = "_binder_chain{chain_id}.pdb"
MHC_ALL_SUFFIX         = "_mhc_all.pdb"
MHC_LIGANDS_SUFFIX     = "_mhc_ligands.pdb"


# =========================================================
# PTM residue table
# Keys are 3-letter PDB residue codes; values are labels
# (stored in the dict but not written to the CSV — they are
# only used internally if you want to extend the output later).
# =========================================================

PTM_RESIDUES: Dict[str, str] = {
    # --- Phosphorylation ---
    "SEP": "phosphorylation",
    "TPO": "phosphorylation",
    "PTR": "phosphorylation",

    # --- Glycosylation / common sugars ---
    "NAG": "glycosylation",
    "NDG": "glycosylation",
    "MAN": "glycosylation",
    "BMA": "glycosylation",
    "FUC": "glycosylation",
    "GAL": "glycosylation",
    "GLC": "glycosylation",
    "SIA": "glycosylation",
    "NAN": "glycosylation",

    # --- Acetylation ---
    "ALY": "acetylation",
    "ACE": "acetylation",

    # --- Methylation ---
    "MLY": "methylation",
    "M3L": "methylation",
    "MLZ": "methylation",
    "AGM": "methylation",
    "DM0": "methylation",

    # --- Hydroxylation ---
    "HYP": "hydroxylation",
    "HYL": "hydroxylation",

    # --- Sulfation ---
    "TYS": "sulfation",

    # --- Citrullination / deimination ---
    "CIR": "citrullination",

    # --- Carboxylation ---
    "KCX": "carboxylation",
    "CGU": "carboxylation",

    # --- Cysteine modifications / oxidation / redox-related ---
    "CSO": "cysteine modification",
    "CME": "cysteine modification",
    "CSD": "cysteine oxidation",
    "CSX": "cysteine oxidation",
    "OCS": "cysteine oxidation",
    "CSU": "cysteine oxidation",
    "SNC": "S-nitrosylation",

    # --- Other common modified amino acids ---
    "PCA": "pyroglutamate formation",
    "LLP": "cofactor-linked lysine modification",
    "MSE": "selenomethionine substitution",
    "KYN": "kynurenine substitution",

    # --- Peptide-like / backbone-risk residues ---
    "3AZ": "peptide-like residue",
    "3X9": "modified/unknown peptide residue",
    "4OG": "modified/unknown peptide residue",
    "ABA": "modified/unknown peptide residue",
    "BAL": "beta-amino acid / peptide-like residue",
    "CDE": "modified/unknown peptide residue",
    "F2F": "modified/unknown peptide residue",
    "GIC": "peptide-like residue",
    "LPH": "modified/unknown peptide residue",
    "NVA": "modified/unknown peptide residue",
    "OSE": "modified/unknown peptide residue",
    "PFF": "modified/unknown peptide residue",
    "PRQ": "modified/unknown peptide residue",
    "PRV": "modified/unknown peptide residue",
    "QM8": "modified/unknown peptide residue",
    "QMB": "modified/unknown peptide residue",
    "TIG": "amino-acid analog",
    "XFW": "peptide-like residue",
}


# =========================================================
# Logging / setup
# =========================================================

def log(message: str) -> None:
    print(message, flush=True)


def make_output_dirs() -> None:
    for path in [
        OUTPUT_MHC_ONLY,
        OUTPUT_BINDERS,
        OUTPUT_MHC_ALL,
        OUTPUT_MHC_LIGANDS,
        OUTPUT_MHC_COMPLEX,
        OUTPUT_MHC_BINDER,
    ]:
        path.mkdir(parents=True, exist_ok=True)
    # RESIDUES_CSV lives directly in STEP9_DIR, which is created above implicitly.
    STEP9_DIR.mkdir(parents=True, exist_ok=True)


def validate_inputs() -> None:
    log("=== STEP9 configuration ===")
    log(f"INPUT_PDB_DIR (step7):  {INPUT_PDB_DIR} | exists={INPUT_PDB_DIR.is_dir()}")
    log(f"STEP8_REMODELED_DIR:    {STEP8_REMODELED_DIR} | exists={STEP8_REMODELED_DIR.is_dir()}")
    log(f"MODIFIED_PDB_DIR:       {MODIFIED_PDB_DIR} | exists={MODIFIED_PDB_DIR.is_dir()}")
    log(f"BINDERS_CSV:            {BINDERS_CSV} | exists={BINDERS_CSV.is_file()}")
    log(f"LIGANDS_CSV:            {LIGANDS_CSV} | exists={LIGANDS_CSV.is_file()}")
    log(f"STEP9_DIR:              {STEP9_DIR}")
    log(f"RESIDUES_CSV:           {RESIDUES_CSV}")
    log(f"ANNOTATIONS_CSV:        {ANNOTATIONS_CSV} | exists={ANNOTATIONS_CSV.is_file()}")
    log(f"ANNOTATIONS_OUT_CSV:    {ANNOTATIONS_OUT_CSV}")
    log("")

    if not INPUT_PDB_DIR.is_dir():
        raise SystemExit(f"ERROR: INPUT_PDB_DIR does not exist: {INPUT_PDB_DIR}")
    if not BINDERS_CSV.is_file():
        raise SystemExit(f"ERROR: BINDERS_CSV does not exist: {BINDERS_CSV}")
    if not LIGANDS_CSV.is_file():
        raise SystemExit(f"ERROR: LIGANDS_CSV does not exist: {LIGANDS_CSV}")
    if not ANNOTATIONS_CSV.is_file():
        raise SystemExit(f"ERROR: ANNOTATIONS_CSV does not exist: {ANNOTATIONS_CSV}")
    if not STEP8_REMODELED_DIR.is_dir():
        log("[INFO] STEP8_REMODELED_DIR not found – structures without manual edits will be taken from step7.")
    if not MODIFIED_PDB_DIR.is_dir():
        log("[INFO] MODIFIED_PDB_DIR not found – no manually edited complexes will be used.")


# =========================================================
# Helpers
# =========================================================

def normalize_pdb_id(value) -> str:
    return str(value).split("-")[0].strip().upper()


def pdb_id_from_path(path: Path) -> str:
    return path.name.split("_")[0].strip().upper()


def resolve_pdb_path(pdb_id: str, step7_path: Path) -> Tuple[Path, str]:
    # Return (path_to_load, source_label).
    #
    # Priority:
    #   1. Any *.pdb in MODIFIED_PDB_DIR whose stem starts with the
    #      lower-cased PDB ID. These are manually edited full complexes.
    #   2. Any *.pdb in STEP8_REMODELED_DIR whose stem starts with the
    #      lower-cased PDB ID (e.g. 1abc_remodeled.pdb, 1abcA_loop.pdb).
    #   3. The original step7 path.
    prefix = pdb_id.lower()

    if MODIFIED_PDB_DIR.is_dir():
        candidates = sorted(
            p for p in MODIFIED_PDB_DIR.glob("*.pdb")
            if p.stem.lower().startswith(prefix)
        )
        if candidates:
            if len(candidates) > 1:
                log(
                    f"[WARNING] {pdb_id}: {len(candidates)} modified files found; "
                    f"using {candidates[0].name}"
                )
            return candidates[0], "modified"

    if STEP8_REMODELED_DIR.is_dir():
        candidates = sorted(
            p for p in STEP8_REMODELED_DIR.glob("*.pdb")
            if p.stem.lower().startswith(prefix)
        )
        if candidates:
            if len(candidates) > 1:
                log(
                    f"[WARNING] {pdb_id}: {len(candidates)} remodeled files found; "
                    f"using {candidates[0].name}"
                )
            return candidates[0], "remodeled"
    return step7_path, "step7"


def load_chain_map(csv_path: Path, label: str) -> Dict[str, List[str]]:
    """
    Load chain classifications from a step7 summary CSV.

    Expected columns: pdb, chain_id, chain_residue_count,
                      min_distance_to_mhc_helix, classification
    """
    df = pd.read_csv(csv_path)
    if df.empty:
        return {}

    missing = {"pdb", "chain_id"} - set(df.columns)
    if missing:
        raise SystemExit(
            f"ERROR: {label} CSV missing columns: {sorted(missing)} in {csv_path}"
        )

    chain_map: Dict[str, List[str]] = {}
    for _, row in df.iterrows():
        pdb_id  = normalize_pdb_id(row["pdb"])
        chain_id = str(row["chain_id"]).strip()
        if not pdb_id or not chain_id or chain_id.lower() == "nan":
            continue
        chain_map.setdefault(pdb_id, [])
        if chain_id not in chain_map[pdb_id]:
            chain_map[pdb_id].append(chain_id)
    return chain_map


def scan_residues_in_chain(chain, pdb_id: str, chain_type: str) -> List[dict]:
    """
    Iterate every residue in a Biopython chain and record any that are
    non-standard, in the PTM dictionary, or both.

    Each unique (resname, resseq, icode) position is reported at most once.

    Returns a list of dicts with keys:
      pdb_id, resname, resid, type, nonstandard, in_ptm_dict
    """
    rows: List[dict] = []
    seen: Set[Tuple[str, int, str]] = set()

    for residue in chain:
        resname = residue.resname.strip().upper()
        is_nonstandard = not is_aa(residue, standard=True)
        in_ptm        = resname in PTM_RESIDUES

        if not is_nonstandard and not in_ptm:
            continue

        resseq: int = residue.id[1]
        icode:  str = residue.id[2].strip()
        key = (resname, resseq, icode)
        if key in seen:
            continue
        seen.add(key)

        rows.append({
            "pdb_id":      pdb_id,
            "resname":     resname,
            "resid":       resseq,
            "type":        chain_type,
            "nonstandard": "yes" if is_nonstandard else "no",
            "in_ptm_dict": "yes" if in_ptm        else "no",
        })

    return rows



def write_annotated_csv(
    binders: Dict[str, List[str]],
    ligands: Dict[str, List[str]],
    present_pdb_ids: Set[str],
) -> None:
    # Read ANNOTATIONS_CSV, add binder/ligand yes/no columns, write ANNOTATIONS_OUT_CSV.
    # PDB ID column is auto-detected (first of: pdb, pdb_id, PDB, PDB_ID).
    df = pd.read_csv(ANNOTATIONS_CSV)

    pdb_col = next(
        (c for c in ("pdb", "pdb_id", "PDB", "PDB_ID") if c in df.columns),
        None,
    )
    if pdb_col is None:
        raise SystemExit(
            f"ERROR: no PDB ID column found in {ANNOTATIONS_CSV}. "
            f"Columns present: {list(df.columns)}"
        )

    original_count = len(df)
    normalized_ids = df[pdb_col].apply(normalize_pdb_id)
    df = df.loc[normalized_ids.isin(present_pdb_ids)].copy()

    binder_ids = set(binders.keys())
    ligand_ids = set(ligands.keys())

    df["binder"] = df[pdb_col].apply(
        lambda v: "yes" if normalize_pdb_id(v) in binder_ids else "no"
    )
    df["ligand"] = df[pdb_col].apply(
        lambda v: "yes" if normalize_pdb_id(v) in ligand_ids else "no"
    )

    df.to_csv(ANNOTATIONS_OUT_CSV, index=False)

    n_binder = (df["binder"] == "yes").sum()
    n_ligand = (df["ligand"] == "yes").sum()
    log(f"[ANNOTATIONS] {len(df)} rows written to {ANNOTATIONS_OUT_CSV}")
    log(f"[ANNOTATIONS]   excluded because not processed: {original_count - len(df)}")
    log(f"[ANNOTATIONS]   binder=yes: {n_binder}  ligand=yes: {n_ligand}")


def copy_mhc_binder_complexes() -> Tuple[int, List[str]]:
    """
    Copy full MHC-complex PDBs with binder=yes into 6_mhc_binder.

    This uses the already written step9_annotations.csv and the already written
    5_mhc_complex files. It does not change the separation logic.
    """
    df = pd.read_csv(ANNOTATIONS_OUT_CSV)

    pdb_col = next(
        (c for c in ("pdb_id", "pdb", "PDB_ID", "PDB") if c in df.columns),
        None,
    )
    if pdb_col is None:
        raise SystemExit(
            f"ERROR: no PDB ID column found in {ANNOTATIONS_OUT_CSV}. "
            f"Columns present: {list(df.columns)}"
        )
    if "binder" not in df.columns:
        raise SystemExit(f"ERROR: binder column not found in {ANNOTATIONS_OUT_CSV}")

    binder_ids = (
        df[df["binder"].astype(str).str.lower() == "yes"][pdb_col]
        .apply(normalize_pdb_id)
        .dropna()
        .drop_duplicates()
        .sort_values()
    )

    copied = 0
    missing: List[str] = []

    for pdb_id in binder_ids:
        pdb_file = OUTPUT_MHC_COMPLEX / f"{pdb_id.lower()}_mhc_complex.pdb"
        if pdb_file.exists():
            shutil.copy2(pdb_file, OUTPUT_MHC_BINDER / pdb_file.name)
            copied += 1
        else:
            missing.append(pdb_file.name)

    log(f"[MHC_BINDER] Copied {copied} binder-positive complexes to {OUTPUT_MHC_BINDER}")
    if missing:
        log(f"[MHC_BINDER] Missing {len(missing)} expected files:")
        for name in missing:
            log(f"[MHC_BINDER]   {name}")

    return copied, missing


class ChainSetSelect(Select):
    def __init__(self, chain_ids: Iterable[str]):
        self.chain_ids: Set[str] = set(chain_ids)

    def accept_chain(self, chain) -> int:
        return int(chain.id in self.chain_ids)


def save_selected_chains(structure, out_path: Path, chain_ids: Iterable[str]) -> None:
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(out_path), ChainSetSelect(chain_ids))


# =========================================================
# Main processing
# =========================================================

def process_one(
    pdb_path: Path,
    binders: Dict[str, List[str]],
    ligands: Dict[str, List[str]],
) -> List[dict] | None:
    """
    Separate one PDB into output folders and return residue scan rows.

    The structure to load is resolved via resolve_pdb_path: a remodeled file
    from step8 is preferred over the original step7 file when one exists.
    """
    pdb_id = pdb_id_from_path(pdb_path)
    resolved_path, structure_source = resolve_pdb_path(pdb_id, pdb_path)

    if structure_source in {"modified", "remodeled"}:
        log(f"[SOURCE] {pdb_id}: using {structure_source} structure → {resolved_path.name}")

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, str(resolved_path))
    model = structure[0]

    residue_rows: List[dict] = []

    if MHC_CHAIN_ID not in model:
        log(f"[WARNING] {pdb_id}: MHC chain {MHC_CHAIN_ID} not found; skipping")
        return None

    # --- Full complex (source file copied as-is) ---
    complex_dest = OUTPUT_MHC_COMPLEX / f"{pdb_id.lower()}_mhc_complex.pdb"
    shutil.copy2(resolved_path, complex_dest)

    # --- MHC chain A ---
    mhc_chain = model[MHC_CHAIN_ID]
    residue_rows.extend(scan_residues_in_chain(mhc_chain, pdb_id, "mhc"))

    mhc_only_path = OUTPUT_MHC_ONLY / f"{pdb_id.lower()}{MHC_ONLY_SUFFIX}"
    save_selected_chains(structure, mhc_only_path, [MHC_CHAIN_ID])

    # --- Binder chains ---
    written_binders: List[str] = []
    for chain_id in binders.get(pdb_id, []):
        if chain_id not in model:
            log(f"[WARNING] {pdb_id}: binder chain {chain_id} not found")
            continue
        residue_rows.extend(scan_residues_in_chain(model[chain_id], pdb_id, "binder"))
        out_path = OUTPUT_BINDERS / f"{pdb_id.lower()}{BINDER_SUFFIX_TEMPLATE.format(chain_id=chain_id)}"
        save_selected_chains(structure, out_path, [chain_id])
        written_binders.append(chain_id)

    # --- Ligand chains ---
    written_ligands: List[str] = []
    for chain_id in ligands.get(pdb_id, []):
        if chain_id not in model:
            log(f"[WARNING] {pdb_id}: ligand chain {chain_id} not found")
            continue
        residue_rows.extend(scan_residues_in_chain(model[chain_id], pdb_id, "ligand"))
        written_ligands.append(chain_id)

    mhc_all_chains = [MHC_CHAIN_ID] + written_ligands

    # MHC + all ligands – always written.
    mhc_all_path = OUTPUT_MHC_ALL / f"{pdb_id.lower()}{MHC_ALL_SUFFIX}"
    save_selected_chains(structure, mhc_all_path, mhc_all_chains)

    # MHC + ligands-only folder – only when ligand chains exist.
    if written_ligands:
        mhc_ligands_path = OUTPUT_MHC_LIGANDS / f"{pdb_id.lower()}{MHC_LIGANDS_SUFFIX}"
        save_selected_chains(structure, mhc_ligands_path, mhc_all_chains)

    log(
        f"[OK] {pdb_id} ({structure_source}): "
        f"binders={len(written_binders)}  ligands={len(written_ligands)}  "
        f"flagged_residues={len(residue_rows)}"
    )
    return residue_rows


def main() -> None:
    make_output_dirs()
    validate_inputs()

    binders = load_chain_map(BINDERS_CSV, "binders")
    ligands = load_chain_map(LIGANDS_CSV, "ligands")

    pdb_files = sorted(INPUT_PDB_DIR.glob(INPUT_PDB_PATTERN))
    if not pdb_files:
        raise SystemExit(f"ERROR: no PDB files found in {INPUT_PDB_DIR}")

    log(f"[INPUT] PDB files:            {len(pdb_files)}")
    log(f"[INPUT] PDBs with binders:    {len(binders)}")
    log(f"[INPUT] PDBs with ligands:    {len(ligands)}")
    log(f"[INPUT] PTM residues tracked: {len(PTM_RESIDUES)}")
    log("")
    log("=== STEP9 separating structures ===")

    all_residue_rows: List[dict] = []
    processed_pdb_ids: Set[str] = set()

    for pdb_path in pdb_files:
        pdb_id = pdb_id_from_path(pdb_path)
        try:
            rows = process_one(pdb_path=pdb_path, binders=binders, ligands=ligands)
        except Exception as exc:
            log(f"[ERROR] {pdb_id}: {exc}")
            rows = None

        if rows is None:
            continue

        processed_pdb_ids.add(pdb_id)
        all_residue_rows.extend(rows)

    # ---------------------------------------------------------
    # Write unified residues CSV.
    # Columns: pdb_id, resname, resid, type, nonstandard, in_ptm_dict
    # Only rows where at least one flag is "yes" are present (guaranteed
    # by scan_residues_in_chain, but we drop_duplicates for safety).
    # ---------------------------------------------------------
    cols = ["pdb_id", "resname", "resid", "type", "nonstandard", "in_ptm_dict"]
    if all_residue_rows:
        (
            pd.DataFrame(all_residue_rows)[cols]
            .drop_duplicates()
            .sort_values(["pdb_id", "type", "resid", "resname"])
            .to_csv(RESIDUES_CSV, index=False)
        )
    else:
        pd.DataFrame(columns=cols).to_csv(RESIDUES_CSV, index=False)

    # ---------------------------------------------------------
    # Write annotated copy of the step2-1 annotations CSV.
    # ---------------------------------------------------------
    write_annotated_csv(
        binders=binders,
        ligands=ligands,
        present_pdb_ids=processed_pdb_ids,
    )

    # ---------------------------------------------------------
    # Copy full complexes with binder=yes to 6_mhc_binder.
    # ---------------------------------------------------------
    copied_mhc_binder, missing_mhc_binder = copy_mhc_binder_complexes()

    log("")
    log("=== STEP9 summary ===")
    log(f"Processed PDBs:        {len(pdb_files)}")
    log(f"Flagged residues:      {len(all_residue_rows)}")
    log(f"Residues CSV:          {RESIDUES_CSV}")
    log(f"MHC-only structures:   {OUTPUT_MHC_ONLY}")
    log(f"Binder-only chains:    {OUTPUT_BINDERS}")
    log(f"MHC-all structures:    {OUTPUT_MHC_ALL}")
    log(f"MHC-ligand structures: {OUTPUT_MHC_LIGANDS}")
    log(f"MHC-complex folder:    {OUTPUT_MHC_COMPLEX}")
    log(f"MHC-binder folder:     {OUTPUT_MHC_BINDER}")
    log(f"MHC-binder copied:     {copied_mhc_binder}")
    log(f"MHC-binder missing:    {len(missing_mhc_binder)}")
    log(f"Annotations CSV:       {ANNOTATIONS_OUT_CSV}")
    log("[DONE] Step9 finished")


if __name__ == "__main__":
    main()
