#!/usr/bin/env python3

import pandas as pd

# ============================================================
# File paths
# ============================================================
ANNOTATIONS_FILE = "/mnt/4TB/giovanna/foldseek/version_02/analysis/functional_annotation/filtered/afdb_mhc_annotations_filtered.csv"
MODELS_FILE = "/mnt/4TB/giovanna/foldseek/version_02/analysis/functional_annotation/filtered/afdb_models_analysis_filtered.csv"

# ============================================================
# Load data
# ============================================================
annotations_df = pd.read_csv(ANNOTATIONS_FILE)
models_df = pd.read_csv(MODELS_FILE)

# ============================================================
# Extract UniProt ID from PDB column
# Format example: AF-Q29757-F1-model_v6
# ============================================================
models_df["uniprot_id"] = (
    models_df["pdb"]
    .str.replace("AF-", "", regex=False)
    .str.split("-F1").str[0]
)

# ============================================================
# Filter 1 — Uniprotkb reviewed
# ============================================================
reviewed_annotations = annotations_df[
    annotations_df["entry_status"] == "Uniprotkb reviewed (swiss-prot)"
].copy()

reviewed_annotations_file = "afdb_mhc_annotations_reviewed.csv"
reviewed_annotations.to_csv(reviewed_annotations_file, index=False)

# Filter corresponding models
reviewed_models = models_df[
    models_df["uniprot_id"].isin(reviewed_annotations["uniprot_id"])
].copy()

reviewed_models_file = "afdb_models_analysis_reviewed.csv"
reviewed_models.to_csv(reviewed_models_file, index=False)

# ============================================================
# Filter 2 — Target length between 175 and 185
# ============================================================
length_annotations = annotations_df[
    (annotations_df["target_length"] > 175) &
    (annotations_df["target_length"] < 185)
].copy()

length_annotations_file = "afdb_mhc_annotations_length_175_185.csv"
length_annotations.to_csv(length_annotations_file, index=False)

# Filter corresponding models
length_models = models_df[
    models_df["uniprot_id"].isin(length_annotations["uniprot_id"])
].copy()

length_models_file = "afdb_models_analysis_length_175_185.csv"
length_models.to_csv(length_models_file, index=False)

# ============================================================
# Summary
# ============================================================
print("Filter 1 — Reviewed:")
print(f"Annotations: {len(reviewed_annotations)}")
print(f"Models: {len(reviewed_models)}\n")

print("Filter 2 — Length 175–185:")
print(f"Annotations: {len(length_annotations)}")
print(f"Models: {len(length_models)}")
