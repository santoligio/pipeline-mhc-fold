#!/usr/bin/env python3
"""
Filter AFDB MHC functional annotations (step3 of the AFDB pipeline).

Takes the annotation table produced by step2_record_mhc_annotations_afdb.py,
applies the same species / gene-name / length / manual-removal filters used
in the PDB pipeline's step2-1_filter_mhc_annotations.py, and writes out
EVERY dataset option defined in afdb_dataset_config.py -- not just one. Each
run produces:

    1. All human AFDB MHC             (species + excluded-gene-name filters only)
    2. Human AFDB MHC, current restrictions   (the full filter pipeline; default)
    3. AFDB subset 1  -- Swiss-Prot reviewed
    4. AFDB subset 2  -- target_length 175-185
    5. Both AFDB subsets  -- union of 3 and 4

step4_download_structures_afdb.py (and any future step5) then just pick one
option via a toggle and look up its filename in afdb_dataset_config.py --
they don't each hardcode their own INPUT_CSV, which is what caused the
subset/full-list mismatch before.
"""

from pathlib import Path
from typing import List, Set

import pandas as pd

from afdb_dataset_config import (
    AfdbDataset,
    DATASET_ANNOTATION_FILENAMES,
    DATASET_MODEL_FILENAMES,
    GENE_MAPPING_FILE,
    STEP1_DIR,
    STEP2_DIR,
    STEP3_DIR,
    UNIPROT_REMOVE_FILE,
    describe,
)


# =========================
# Configuration
# =========================

STEP1_MODELS_FILE = STEP1_DIR / "afdb" / "afdb_models.csv"
STEP2_ANN_FILE = STEP2_DIR / "afdb_mhc_annotations.csv"

OUT_DIR = STEP3_DIR
REMOVED_LOG_FILE = OUT_DIR / "removed_ids.csv"

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

MIN_TARGET_LENGTH = 140


# =========================
# Setup
# =========================

def log(message: str) -> None:
    print(message, flush=True)


def make_output_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def validate_inputs() -> None:
    required = [STEP1_MODELS_FILE, STEP2_ANN_FILE, GENE_MAPPING_FILE, UNIPROT_REMOVE_FILE]
    for path in required:
        if not path.is_file():
            raise SystemExit(f"ERROR: required file not found: {path}")


# =========================
# Helpers
# =========================

def load_uniprot_remove_file(path: Path) -> Set[str]:
    with path.open() as handle:
        return {
            line.strip()
            for line in handle
            if line.strip() and not line.lstrip().startswith("#")
        }


def load_gene_mapping() -> pd.DataFrame:
    gene_mapping = pd.read_csv(GENE_MAPPING_FILE)

    required = {"database", "gene_name", "mapped_gene_name", "mapped_class", "mapped_superclass"}
    missing = required - set(gene_mapping.columns)

    if missing:
        raise SystemExit(f"ERROR: GENE_MAPPING_FILE missing columns: {sorted(missing)}")

    return gene_mapping


def log_removal(rows: list, pdb, uniprot_id, reason: str) -> None:
    rows.append({
        "pdb": pdb,
        "uniprot_id": uniprot_id,
        "database": "afdb",
        "reason": reason,
    })


def apply_global_filters(df: pd.DataFrame, removed_log: list) -> pd.DataFrame:
    """Species + excluded-gene-name filters -- shared by every option."""
    df = df.copy()

    mask_species = ~df["organism"].isin(ALLOWED_SPECIES)

    for _, row in df[mask_species].iterrows():
        log_removal(removed_log, row["pdb_id"], row.get("uniprot_id", ""), "excluded_species")

    df = df[~mask_species].copy()

    mask_gene = df["gene_name"].isin(EXCLUDED_GENE_NAMES)

    for _, row in df[mask_gene].iterrows():
        log_removal(removed_log, row["pdb_id"], row.get("uniprot_id", ""), "excluded_gene_name")

    return df[~mask_gene].copy()


def apply_gene_mapping(df: pd.DataFrame, gene_mapping: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    db_map = dict(
        zip(
            gene_mapping[gene_mapping["database"] == "afdb"]["gene_name"],
            gene_mapping[gene_mapping["database"] == "afdb"]["mapped_gene_name"],
        )
    )

    df["gene_name"] = df["gene_name"].replace(db_map)

    class_map = (
        gene_mapping[["mapped_gene_name", "mapped_class", "mapped_superclass"]]
        .drop_duplicates(subset="mapped_gene_name")
        .set_index("mapped_gene_name")
    )

    return df.join(class_map, on="gene_name")


def remove_missing_gene(df: pd.DataFrame, removed_log: list) -> pd.DataFrame:
    df = df.copy()
    mask = df["gene_name"] == "Missing entry"

    for _, row in df[mask].iterrows():
        log_removal(removed_log, row["pdb_id"], row.get("uniprot_id", ""), "missing_gene_name")

    return df[~mask].copy()


def remove_short_targets(df: pd.DataFrame, removed_log: list) -> pd.DataFrame:
    if "target_length" not in df.columns:
        return df

    df = df.copy()
    mask = df["target_length"] < MIN_TARGET_LENGTH

    for _, row in df[mask].iterrows():
        log_removal(removed_log, row["pdb_id"], row["uniprot_id"], "target_length_lt_140")

    return df[~mask].copy()


def remove_explicit_uniprot(df: pd.DataFrame, uniprot_to_remove: Set[str], removed_log: list) -> pd.DataFrame:
    if "uniprot_id" not in df.columns:
        return df

    df = df.copy()
    mask = df["uniprot_id"].isin(uniprot_to_remove)

    for _, row in df[mask].iterrows():
        log_removal(removed_log, row["pdb_id"], row["uniprot_id"], "explicit_uniprot_removal")

    return df[~mask].copy()


def drop_cleanup_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "uniprot_length" in df.columns:
        df = df.drop(columns="uniprot_length")

    if "chain" in df.columns:
        df = df.drop(columns="chain")

    return df


def build_models_with_uniprot_id(afdb_models: pd.DataFrame) -> pd.DataFrame:
    """afdb_models.csv only has the raw AFDB model id ('pdb' column) --
    this adds a parsed uniprot_id column so subset options can be matched
    against it, the same way the original reviewed/length outputs were."""
    models = afdb_models.copy()
    models["uniprot_id"] = (
        models["pdb"]
        .astype(str)
        .str.replace("AF-", "", regex=False)
        .str.split("-F1")
        .str[0]
    )
    return models


# =========================
# Main
# =========================

def main() -> None:
    make_output_dirs()
    validate_inputs()

    removed_log: List[dict] = []
    gene_mapping = load_gene_mapping()
    uniprot_to_remove = load_uniprot_remove_file(UNIPROT_REMOVE_FILE)

    afdb_ann_raw = pd.read_csv(STEP2_ANN_FILE)
    afdb_models = pd.read_csv(STEP1_MODELS_FILE)

    # ---- Option 1: All human AFDB MHC ----
    # Species + excluded-gene-name filters only -- the broadest candidate list.
    afdb_ann_broad = apply_global_filters(afdb_ann_raw, removed_log)
    afdb_ann_broad = apply_gene_mapping(afdb_ann_broad, gene_mapping)
    afdb_ann_broad_out = drop_cleanup_columns(afdb_ann_broad)
    afdb_models_broad = afdb_models[afdb_models["pdb"].isin(afdb_ann_broad_out["pdb_id"])].copy()

    # ---- Option 2: Human AFDB MHC with current restrictions ----
    # Everything from option 1, plus missing-gene removal, target_length >= 140,
    # and the explicit UniProt removal list. Pipeline's main/default output.
    afdb_ann_restricted = remove_missing_gene(afdb_ann_broad, removed_log)
    afdb_ann_restricted = remove_short_targets(afdb_ann_restricted, removed_log)
    afdb_ann_restricted = remove_explicit_uniprot(afdb_ann_restricted, uniprot_to_remove, removed_log)
    afdb_ann_restricted_out = drop_cleanup_columns(afdb_ann_restricted)
    afdb_models_restricted = afdb_models[afdb_models["pdb"].isin(afdb_ann_restricted_out["pdb_id"])].copy()

    # ---- Options 3-5: narrower subsets of option 2 ----
    models_with_uid = build_models_with_uniprot_id(afdb_models)

    reviewed_ann = afdb_ann_restricted_out[
        afdb_ann_restricted_out["entry_status"] == "Uniprotkb reviewed (swiss-prot)"
    ].copy()
    reviewed_models = models_with_uid[models_with_uid["uniprot_id"].isin(reviewed_ann["uniprot_id"])].copy()

    length_ann = afdb_ann_restricted_out[
        (afdb_ann_restricted_out["target_length"] > 175)
        & (afdb_ann_restricted_out["target_length"] < 185)
    ].copy()
    length_models = models_with_uid[models_with_uid["uniprot_id"].isin(length_ann["uniprot_id"])].copy()

    both_ann = pd.concat([reviewed_ann, length_ann], ignore_index=True).drop_duplicates(subset="pdb_id")
    both_models = models_with_uid[models_with_uid["uniprot_id"].isin(both_ann["uniprot_id"])].copy()

    # ---- Write every option, keyed by the shared filenames in ----
    # ---- afdb_dataset_config.py, so step4/step5 just look one up. ----
    datasets = {
        AfdbDataset.ALL_HUMAN_MHC: (afdb_ann_broad_out, afdb_models_broad),
        AfdbDataset.CURRENT_RESTRICTIONS: (afdb_ann_restricted_out, afdb_models_restricted),
        AfdbDataset.SUBSET_REVIEWED: (reviewed_ann, reviewed_models),
        AfdbDataset.SUBSET_LENGTH_175_185: (length_ann, length_models),
        AfdbDataset.BOTH_SUBSETS: (both_ann, both_models),
    }

    log("[AFDB DATASET OPTIONS]")
    for option, (ann_df, models_df) in datasets.items():
        ann_path = OUT_DIR / DATASET_ANNOTATION_FILENAMES[option]
        models_path = OUT_DIR / DATASET_MODEL_FILENAMES[option]
        ann_df.to_csv(ann_path, index=False)
        models_df.to_csv(models_path, index=False)
        log(f"  {describe(option)}")
        log(f"    -> {len(models_df)} models  [{models_path.name}]")

    removed_df = pd.DataFrame(removed_log, columns=["pdb", "uniprot_id", "database", "reason"])
    removed_df.to_csv(REMOVED_LOG_FILE, index=False)

    log(f"[CSV] {REMOVED_LOG_FILE}")
    log("[DONE] AFDB filtering complete. Set AFDB_DATASET_SELECTION in step4 "
        "(and step5, if you add one) to choose which option gets used downstream.")


if __name__ == "__main__":
    main()
