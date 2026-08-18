#!/usr/bin/env python3
"""
Step 3 of the AFDB pipeline: filter step2's annotations (species, gene
name, length, manual removal list) and write every dataset option defined
in afdb_dataset_config.py:

    1. All human AFDB MHC                      (species + excluded-gene filters only)
    2. Human AFDB MHC, current restrictions    (full filter pipeline; default)
    3. AFDB subset 1 -- Swiss-Prot reviewed
    4. AFDB subset 2 -- target_length 175-185
    5. Both AFDB subsets -- union of 3 and 4

step4/step5 pick one option by name and look up its filename here, so
there's a single source of truth for which file corresponds to which option.
"""

import sys
from pathlib import Path
from typing import List, Set

import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
# Filters
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
    """Species + excluded-gene-name filters, shared by every option."""
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

    afdb_map = gene_mapping[gene_mapping["database"] == "afdb"]
    db_map = dict(zip(afdb_map["gene_name"], afdb_map["mapped_gene_name"]))
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
        log_removal(removed_log, row["pdb_id"], row["uniprot_id"], f"target_length_lt_{MIN_TARGET_LENGTH}")

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


def models_for(annotations: pd.DataFrame, all_models: pd.DataFrame) -> pd.DataFrame:
    """Select the step1 model rows whose model ID appears in a filtered
    annotation table. Matches on the full AFDB model ID (models["pdb"] ==
    annotations["pdb_id"]) -- both refer to the same identifier, so there's
    no need to re-derive a UniProt ID (a previous version of this function
    tried to, by string-splitting on "-F1", which silently dropped any
    model whose AlphaFold fragment wasn't F1)."""
    return all_models[all_models["pdb"].isin(annotations["pdb_id"])].copy()


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
    afdb_ann_broad = apply_global_filters(afdb_ann_raw, removed_log)
    afdb_ann_broad = apply_gene_mapping(afdb_ann_broad, gene_mapping)
    afdb_ann_broad_out = drop_cleanup_columns(afdb_ann_broad)
    afdb_models_broad = models_for(afdb_ann_broad_out, afdb_models)

    # ---- Option 2: Human AFDB MHC, current restrictions (default) ----
    afdb_ann_restricted = remove_missing_gene(afdb_ann_broad, removed_log)
    afdb_ann_restricted = remove_short_targets(afdb_ann_restricted, removed_log)
    afdb_ann_restricted = remove_explicit_uniprot(afdb_ann_restricted, uniprot_to_remove, removed_log)
    afdb_ann_restricted_out = drop_cleanup_columns(afdb_ann_restricted)
    afdb_models_restricted = models_for(afdb_ann_restricted_out, afdb_models)

    # ---- Options 3-5: narrower subsets of option 2 ----
    reviewed_ann = afdb_ann_restricted_out[
        afdb_ann_restricted_out["entry_status"] == "Uniprotkb reviewed (swiss-prot)"
    ].copy()
    reviewed_models = models_for(reviewed_ann, afdb_models)

    length_ann = afdb_ann_restricted_out[
        (afdb_ann_restricted_out["target_length"] > 175)
        & (afdb_ann_restricted_out["target_length"] < 185)
    ].copy()
    length_models = models_for(length_ann, afdb_models)

    both_ann = pd.concat([reviewed_ann, length_ann], ignore_index=True).drop_duplicates(subset="pdb_id")
    both_models = models_for(both_ann, afdb_models)

    # ---- Write every option under the shared filenames from ----
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
