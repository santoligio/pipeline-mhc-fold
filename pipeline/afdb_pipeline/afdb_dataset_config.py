#!/usr/bin/env python3
"""
Shared AFDB dataset-selection config.

step3_filter_mhc_annotations_afdb.py writes one CSV pair (annotations +
model list) per option below, every run.

step4_download_structures_afdb.py -- and step5_separate_mhc_afdb.py, which
imports the toggle straight from step4 -- read whichever option is
selected via AFDB_DATASET_SELECTION in step4, by looking up its filename
here. Keeping the option list and filenames in one shared file (instead of
each script hardcoding its own path) is what keeps step4/step5 from
silently drifting onto different subsets of the data.

Place this file in the same directory as step1-step5 so the plain
`import afdb_dataset_config` in those scripts resolves.
"""

from enum import Enum
from pathlib import Path


# =========================
# Directory layout
# =========================
#
#   afdb_pipeline/
#     step1/afdb/afdb_models.csv        <- step1 (selection)
#     step2/afdb_mhc_annotations.csv    <- step2 (annotate)
#     step3/afdb_models_*.csv           <- step3 (filter, 5 dataset options)
#     step4/1_models/*.cif              <- step4 (download)
#     step5/<option>/1_mhc_only/*.pdb   <- step5 (trim to MHC-only chain)

PIPELINE_DIR = Path("/mnt/c/Users/gio/Documents/foldseek_nefertari/filter/ligands_pipeline")
AFDB_DIR = PIPELINE_DIR / "afdb_pipeline"

STEP1_DIR = AFDB_DIR / "step1"
STEP2_DIR = AFDB_DIR / "step2"
STEP3_DIR = AFDB_DIR / "step3"
STEP4_DIR = AFDB_DIR / "step4"
STEP5_DIR = AFDB_DIR / "step5"

# Curated files shared with the PDB pipeline's step2-1 filter script.
# Kept at the top of PIPELINE_DIR (not under afdb_pipeline/) on purpose --
# both pipelines read the same copy.
GENE_MAPPING_FILE = PIPELINE_DIR / "step2-1" / "gene_mapping.csv"
UNIPROT_REMOVE_FILE = PIPELINE_DIR / "step2-1" / "uniprot_proteins_to_remove.txt"


class AfdbDataset(Enum):
    ALL_HUMAN_MHC = "all_human_mhc"
    CURRENT_RESTRICTIONS = "current_restrictions"
    SUBSET_REVIEWED = "subset_reviewed"
    SUBSET_LENGTH_175_185 = "subset_length_175_185"
    BOTH_SUBSETS = "both_subsets"


# Plain-language description of each option -- printed by step3/step4/step5
# so the active dataset is always visible in the logs.
AFDB_DATASET_DESCRIPTIONS = {
    AfdbDataset.ALL_HUMAN_MHC: (
        "1. All human AFDB MHC -- species + excluded-gene-name filters only. "
        "The broadest candidate list; skips the missing-gene/length/explicit-"
        "removal restrictions applied in option 2."
    ),
    AfdbDataset.CURRENT_RESTRICTIONS: (
        "2. Human AFDB MHC with current restrictions -- the full step3 filter "
        "pipeline (species, gene mapping, missing-gene removal, "
        "target_length >= 140, explicit UniProt removal list). This is the "
        "pipeline's main/default output."
    ),
    AfdbDataset.SUBSET_REVIEWED: (
        "3. AFDB subset 1 -- narrower than option 2: Swiss-Prot reviewed "
        "UniProt entries only."
    ),
    AfdbDataset.SUBSET_LENGTH_175_185: (
        "4. AFDB subset 2 -- narrower than option 2: target_length strictly "
        "between 175 and 185 (typical MHC domain window)."
    ),
    AfdbDataset.BOTH_SUBSETS: (
        "5. Both AFDB subsets -- union of subset 1 (reviewed) and subset 2 "
        "(length 175-185)."
    ),
}

# Model-list filenames written by step3 into afdb_pipeline/step3/, and read
# from there by step4/step5 based on AFDB_DATASET_SELECTION.
DATASET_MODEL_FILENAMES = {
    AfdbDataset.ALL_HUMAN_MHC: "afdb_models_all_human_mhc.csv",
    AfdbDataset.CURRENT_RESTRICTIONS: "afdb_models_filtered.csv",
    AfdbDataset.SUBSET_REVIEWED: "afdb_models_reviewed.csv",
    AfdbDataset.SUBSET_LENGTH_175_185: "afdb_models_length_175_185.csv",
    AfdbDataset.BOTH_SUBSETS: "afdb_models_both_subsets.csv",
}

# Matching annotation-table filenames (same rows, with annotation columns).
DATASET_ANNOTATION_FILENAMES = {
    AfdbDataset.ALL_HUMAN_MHC: "afdb_mhc_annotations_all_human_mhc.csv",
    AfdbDataset.CURRENT_RESTRICTIONS: "afdb_mhc_annotations_filtered.csv",
    AfdbDataset.SUBSET_REVIEWED: "afdb_mhc_annotations_reviewed.csv",
    AfdbDataset.SUBSET_LENGTH_175_185: "afdb_mhc_annotations_length_175_185.csv",
    AfdbDataset.BOTH_SUBSETS: "afdb_mhc_annotations_both_subsets.csv",
}


def describe(selection: "AfdbDataset") -> str:
    """One printable line for logs: which option, and what it means."""
    return AFDB_DATASET_DESCRIPTIONS[selection]
