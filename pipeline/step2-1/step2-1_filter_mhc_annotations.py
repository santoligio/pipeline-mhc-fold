#!/usr/bin/env python3
"""
Filter MHC functional annotations after step2.1 annotation.

Manual-edit requirement:
- PDB uses pdb_mhc_annotations_edited.csv.
- AFDB uses afdb_mhc_annotations.csv directly.

Outputs filtered annotation and step1 tables using ALLOWED_SPECIES.
"""

from pathlib import Path
from typing import List, Set

import pandas as pd


# =========================
# Configuration
# =========================

PIPELINE_DIR = Path(__file__).resolve().parents[1]
STEP2_1_DIR = PIPELINE_DIR / "step2-1"

DATABASE = "pdb"  # "pdb", "afdb", or "both"

PDB_ANN_FILE = STEP2_1_DIR / "pdb" / "pdb_mhc_annotations_edited.csv"
AFDB_ANN_FILE = STEP2_1_DIR / "afdb" / "afdb_mhc_annotations.csv"

PDB_ASSEMBLIES_FILE = STEP2_1_DIR / "step1" / "pdb" / "pdb_assemblies.csv"
AFDB_MODELS_FILE = STEP2_1_DIR / "step1" / "afdb" / "afdb_models.csv"

GENE_MAPPING_FILE = STEP2_1_DIR / "gene_mapping.csv"
AFDB_UNIPROT_REMOVE_FILE = STEP2_1_DIR / "uniprot_proteins_to_remove.txt"

PDB_OUT_DIR = STEP2_1_DIR / "pdb" / "filtered"
AFDB_OUT_DIR = STEP2_1_DIR / "afdb" / "filtered"

PDB_REMOVED_LOG_FILE = PDB_OUT_DIR / "removed_ids.csv"
AFDB_REMOVED_LOG_FILE = AFDB_OUT_DIR / "removed_ids.csv"

ALLOWED_SPECIES = {
    "Homo sapiens",
    "Human cytomegalovirus (strain AD169)",
    "Cowpox virus (strain Brighton Red)",
    "Yaba-like disease virus",
}


EXCLUDED_GENE_NAMES = {
    "HLA",
    "HLA locus",
    "MHC",
    "HLA-DRB1",
    "HLA-DPB1",
    "DKFZp686N10220",
    "DKFZp686P19218",
    "FLJ45422",
    "LOC554223",
    "DKFZp762B162",
}


# =========================
# Setup
# =========================

def log(message: str) -> None:
    print(message, flush=True)


def selected_databases() -> List[str]:
    if DATABASE == "both":
        return ["pdb", "afdb"]

    if DATABASE in {"pdb", "afdb"}:
        return [DATABASE]

    raise ValueError(f"Unknown DATABASE: {DATABASE}")


def make_output_dirs() -> None:
    PDB_OUT_DIR.mkdir(parents=True, exist_ok=True)
    AFDB_OUT_DIR.mkdir(parents=True, exist_ok=True)


def validate_common_inputs() -> None:
    if not GENE_MAPPING_FILE.is_file():
        raise SystemExit(f"ERROR: GENE_MAPPING_FILE not found: {GENE_MAPPING_FILE}")


def validate_database_inputs(database: str) -> None:
    if database == "pdb":
        required = [PDB_ANN_FILE, PDB_ASSEMBLIES_FILE]
    elif database == "afdb":
        required = [AFDB_ANN_FILE, AFDB_MODELS_FILE, AFDB_UNIPROT_REMOVE_FILE]
    else:
        raise ValueError(database)

    for path in required:
        if not path.is_file():
            raise SystemExit(f"ERROR: required file not found: {path}")


# =========================
# Helpers
# =========================

def normalize_pdb_id(value) -> str:
    return str(value).split("-")[0].strip().upper()


def load_uniprot_remove_file(path: Path) -> Set[str]:
    with path.open() as handle:
        return {
            line.strip()
            for line in handle
            if line.strip() and not line.lstrip().startswith("#")
        }


def log_removal(rows: list, pdb, uniprot_id, database: str, reason: str) -> None:
    rows.append({
        "pdb": pdb,
        "uniprot_id": uniprot_id,
        "database": database,
        "reason": reason,
    })


def load_gene_mapping() -> pd.DataFrame:
    gene_mapping = pd.read_csv(GENE_MAPPING_FILE)

    required = {"database", "gene_name", "mapped_gene_name", "mapped_class", "mapped_superclass"}
    missing = required - set(gene_mapping.columns)

    if missing:
        raise SystemExit(f"ERROR: GENE_MAPPING_FILE missing columns: {sorted(missing)}")

    return gene_mapping


def apply_assigned_pdb_edits(pdb_ann: pd.DataFrame) -> pd.DataFrame:
    pdb_ann = pdb_ann.copy()

    for idx, row in pdb_ann.iterrows():
        if "gene_name_assigned" in pdb_ann.columns and pd.notna(row.get("gene_name_assigned")) and str(row.get("gene_name_assigned")).strip():
            pdb_ann.at[idx, "gene_name"] = row["gene_name_assigned"]

        if "grouped_gene_assigned" in pdb_ann.columns and pd.notna(row.get("grouped_gene_assigned")) and str(row.get("grouped_gene_assigned")).strip():
            pdb_ann.at[idx, "gene_name"] = row["grouped_gene_assigned"]

        if "organism_assigned" in pdb_ann.columns and pd.notna(row.get("organism_assigned")) and str(row.get("organism_assigned")).strip():
            pdb_ann.at[idx, "organism"] = row["organism_assigned"]

    return pdb_ann


def resolve_pdb_chimeric_rows(pdb_ann: pd.DataFrame, removed_log: list) -> pd.DataFrame:
    if "possibly_chimeric" not in pdb_ann.columns:
        return pdb_ann

    chimeric = pdb_ann[pdb_ann["possibly_chimeric"] == "yes"]
    to_drop = []

    for (pdb_id, chain), group in chimeric.groupby(["pdb_id", "chain"]):
        if len(group) <= 1:
            continue

        if "grouped_gene_assigned" not in group.columns:
            continue

        preferred = group[group["grouped_gene_assigned"].notna()]

        if preferred.empty:
            continue

        dropped = group.index.difference(preferred.index)

        for idx in dropped:
            log_removal(
                removed_log,
                pdb=pdb_ann.at[idx, "pdb_id"],
                uniprot_id=pdb_ann.at[idx, "uniprot_id"] if "uniprot_id" in pdb_ann.columns else "",
                database="pdb",
                reason="chimeric_mapping_resolution",
            )

        to_drop.extend(dropped)

    return pdb_ann.drop(index=to_drop)


def apply_global_filters(df: pd.DataFrame, database: str, removed_log: list) -> pd.DataFrame:
    df = df.copy()

    mask_species = ~df["organism"].isin(ALLOWED_SPECIES)

    for _, row in df[mask_species].iterrows():
        log_removal(
            removed_log,
            pdb=row["pdb_id"],
            uniprot_id=row["uniprot_id"] if "uniprot_id" in df.columns else "",
            database=database,
            reason="excluded_species",
        )

    df = df[~mask_species].copy()

    mask_gene = df["gene_name"].isin(EXCLUDED_GENE_NAMES)

    for _, row in df[mask_gene].iterrows():
        log_removal(
            removed_log,
            pdb=row["pdb_id"],
            uniprot_id=row["uniprot_id"] if "uniprot_id" in df.columns else "",
            database=database,
            reason="excluded_gene_name",
        )

    return df[~mask_gene].copy()


def apply_gene_mapping(df: pd.DataFrame, database: str, gene_mapping: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    db_map = dict(
        zip(
            gene_mapping[gene_mapping["database"] == database]["gene_name"],
            gene_mapping[gene_mapping["database"] == database]["mapped_gene_name"],
        )
    )

    df["gene_name"] = df["gene_name"].replace(db_map)

    class_map = (
        gene_mapping[["mapped_gene_name", "mapped_class", "mapped_superclass"]]
        .drop_duplicates(subset="mapped_gene_name")
        .set_index("mapped_gene_name")
    )

    return df.join(class_map, on="gene_name")


def remove_missing_gene(df: pd.DataFrame, database: str, removed_log: list) -> pd.DataFrame:
    df = df.copy()
    mask = df["gene_name"] == "Missing entry"

    for _, row in df[mask].iterrows():
        log_removal(
            removed_log,
            pdb=row["pdb_id"],
            uniprot_id=row["uniprot_id"] if "uniprot_id" in df.columns else "",
            database=database,
            reason="missing_gene_name",
        )

    return df[~mask].copy()


def drop_cleanup_columns(df: pd.DataFrame, database: str) -> pd.DataFrame:
    df = df.copy()

    if "uniprot_length" in df.columns:
        df = df.drop(columns="uniprot_length")

    if database == "afdb" and "chain" in df.columns:
        df = df.drop(columns="chain")

    if database == "pdb":
        author_columns = [
            "possibly_chimeric",
            "grouped_gene_assigned",
            "gene_name_assigned",
            "organism_assigned",
            "comments",
        ]
        df = df.drop(columns=[col for col in author_columns if col in df.columns])

    return df


# =========================
# PDB filtering
# =========================

def filter_pdb(gene_mapping: pd.DataFrame, removed_log: list) -> None:
    log("[START] PDB filtering")

    pdb_ann = pd.read_csv(PDB_ANN_FILE)
    pdb_assemblies = pd.read_csv(PDB_ASSEMBLIES_FILE)

    pdb_ann = apply_assigned_pdb_edits(pdb_ann)
    pdb_ann = resolve_pdb_chimeric_rows(pdb_ann, removed_log)
    pdb_ann = apply_global_filters(pdb_ann, "pdb", removed_log)
    pdb_ann = apply_gene_mapping(pdb_ann, "pdb", gene_mapping)
    pdb_ann = remove_missing_gene(pdb_ann, "pdb", removed_log)
    pdb_ann = drop_cleanup_columns(pdb_ann, "pdb")

    valid_pdb_ids = set(pdb_ann["pdb_id"].astype(str).str.upper())

    pdb_assemblies = pdb_assemblies.copy()
    pdb_assemblies["_pdb_base"] = pdb_assemblies["pdb"].apply(normalize_pdb_id)

    pdb_assemblies_filtered = (
        pdb_assemblies[pdb_assemblies["_pdb_base"].isin(valid_pdb_ids)]
        .drop(columns="_pdb_base")
    )

    pdb_ann.to_csv(PDB_OUT_DIR / "pdb_mhc_annotations_filtered.csv", index=False)
    pdb_assemblies_filtered.to_csv(PDB_OUT_DIR / "pdb_assemblies_filtered.csv", index=False)

    log(f"[CSV] {PDB_OUT_DIR / 'pdb_mhc_annotations_filtered.csv'}")
    log(f"[CSV] {PDB_OUT_DIR / 'pdb_assemblies_filtered.csv'}")


# =========================
# AFDB filtering
# =========================

def write_afdb_helper_outputs(afdb_ann_filtered: pd.DataFrame, afdb_models_filtered: pd.DataFrame) -> None:
    models = afdb_models_filtered.copy()

    models["uniprot_id"] = (
        models["pdb"]
        .astype(str)
        .str.replace("AF-", "", regex=False)
        .str.split("-F1")
        .str[0]
    )

    reviewed_ann = afdb_ann_filtered[
        afdb_ann_filtered["entry_status"] == "Uniprotkb reviewed (swiss-prot)"
    ].copy()
    reviewed_models = models[models["uniprot_id"].isin(reviewed_ann["uniprot_id"])].copy()

    length_ann = afdb_ann_filtered[
        (afdb_ann_filtered["target_length"] > 175)
        & (afdb_ann_filtered["target_length"] < 185)
    ].copy()
    length_models = models[models["uniprot_id"].isin(length_ann["uniprot_id"])].copy()

    reviewed_ann.to_csv(AFDB_OUT_DIR / "afdb_mhc_annotations_reviewed.csv", index=False)
    reviewed_models.to_csv(AFDB_OUT_DIR / "afdb_models_reviewed.csv", index=False)

    length_ann.to_csv(AFDB_OUT_DIR / "afdb_mhc_annotations_length_175_185.csv", index=False)
    length_models.to_csv(AFDB_OUT_DIR / "afdb_models_length_175_185.csv", index=False)


def filter_afdb(gene_mapping: pd.DataFrame, removed_log: list) -> None:
    log("[START] AFDB filtering")

    afdb_ann = pd.read_csv(AFDB_ANN_FILE)
    afdb_models = pd.read_csv(AFDB_MODELS_FILE)

    afdb_ann = apply_global_filters(afdb_ann, "afdb", removed_log)
    afdb_ann = apply_gene_mapping(afdb_ann, "afdb", gene_mapping)
    afdb_ann = remove_missing_gene(afdb_ann, "afdb", removed_log)

    if "target_length" in afdb_ann.columns:
        mask_length = afdb_ann["target_length"] < 140

        for _, row in afdb_ann[mask_length].iterrows():
            log_removal(
                removed_log,
                pdb=row["pdb_id"],
                uniprot_id=row["uniprot_id"],
                database="afdb",
                reason="target_length_lt_140",
            )

        afdb_ann = afdb_ann[~mask_length].copy()

    uniprot_to_remove = load_uniprot_remove_file(AFDB_UNIPROT_REMOVE_FILE)

    if "uniprot_id" in afdb_ann.columns:
        mask_remove = afdb_ann["uniprot_id"].isin(uniprot_to_remove)

        for _, row in afdb_ann[mask_remove].iterrows():
            log_removal(
                removed_log,
                pdb=row["pdb_id"],
                uniprot_id=row["uniprot_id"],
                database="afdb",
                reason="explicit_uniprot_removal",
            )

        afdb_ann = afdb_ann[~mask_remove].copy()

    afdb_ann_filtered = drop_cleanup_columns(afdb_ann, "afdb")

    valid_afdb_ids = set(afdb_ann_filtered["pdb_id"])
    afdb_models_filtered = afdb_models[afdb_models["pdb"].isin(valid_afdb_ids)].copy()

    afdb_ann_filtered.to_csv(AFDB_OUT_DIR / "afdb_mhc_annotations_filtered.csv", index=False)
    afdb_models_filtered.to_csv(AFDB_OUT_DIR / "afdb_models_filtered.csv", index=False)

    write_afdb_helper_outputs(afdb_ann_filtered, afdb_models_filtered)

    log(f"[CSV] {AFDB_OUT_DIR / 'afdb_mhc_annotations_filtered.csv'}")
    log(f"[CSV] {AFDB_OUT_DIR / 'afdb_models_filtered.csv'}")


# =========================
# Main
# =========================

def main() -> None:
    make_output_dirs()
    validate_common_inputs()

    removed_log = []
    gene_mapping = load_gene_mapping()

    for database in selected_databases():
        validate_database_inputs(database)

        if database == "pdb":
            filter_pdb(gene_mapping, removed_log)
        elif database == "afdb":
            filter_afdb(gene_mapping, removed_log)

    removed_df = pd.DataFrame(
        removed_log,
        columns=["pdb", "uniprot_id", "database", "reason"],
    )

    pdb_removed = removed_df[removed_df["database"] == "pdb"].copy()
    afdb_removed = removed_df[removed_df["database"] == "afdb"].copy()

    pdb_removed.to_csv(PDB_REMOVED_LOG_FILE, index=False)
    afdb_removed.to_csv(AFDB_REMOVED_LOG_FILE, index=False)

    log(f"[CSV] {PDB_REMOVED_LOG_FILE}")
    log(f"[CSV] {AFDB_REMOVED_LOG_FILE}")
    log("[DONE] Step2.1 annotation filtering")


if __name__ == "__main__":
    main()
