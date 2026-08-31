#!/usr/bin/env python3

import pandas as pd

# =========================
# INPUT FILES
# =========================
ANNOTATIONS_CSV = "binders_annotations_merged.csv"
GENE_MAPPING_CSV = "binders_mapping_gene_name.csv"
GO_MAPPING_CSV = "binders_mapping_go_annotations.csv"

# =========================
# OUTPUT FILES
# =========================
OUTPUT_ANNOTATIONS_CSV = "binders_annotations_filtered.csv"
OUTPUT_DROPPED_CSV = "binders_filter_dropped.csv"

# =========================
# LOAD FILES
# =========================
df_annotations = pd.read_csv(ANNOTATIONS_CSV)
df_gene_map = pd.read_csv(GENE_MAPPING_CSV)
df_go_map = pd.read_csv(GO_MAPPING_CSV)

# =========================
# Prepare mapping dictionaries
# =========================
gene_map_dict = df_gene_map.set_index("original_gene").to_dict("index")
go_map_dict = df_go_map.set_index("binder_name_raw").to_dict("index")

# =========================
# Process rows
# =========================
output_rows = []
dropped_rows = []

for idx, row in df_annotations.iterrows():

    gene_name = str(row["gene_name"]).strip()

    # Exclude B2M
    if gene_name == "B2M":
        dropped_rows.append({
            "pdb_id": row["pdb_id"],
            "new_chain": row["new_chain"],
            "original_chain": row["original_chain"],
            "uniprot_id": row["uniprot_id"],
            "organism": row["organism"],
            "classification": str(row["classification"]).strip(),
            "reason": "excluded: gene_name is B2M",
        })
        continue

    classification = str(row["classification"]).strip()

    mapped_class = None
    mapped_superclass = None

    # -------------------------
    # CASE 1: gene_name present
    # -------------------------
    if gene_name != "Missing entry":

        if classification in gene_map_dict:
            mapped_class = gene_map_dict[classification]["mapped_class"]
            mapped_superclass = gene_map_dict[classification]["mapped_superclass"]
        else:
            msg = f"No gene mapping found for classification '{classification}'"
            print(f"WARNING: {msg} (PDB: {row['pdb_id']}, chain: {row['new_chain']})")
            dropped_rows.append({
                "pdb_id": row["pdb_id"],
                "new_chain": row["new_chain"],
                "original_chain": row["original_chain"],
                "uniprot_id": row["uniprot_id"],
                "organism": row["organism"],
                "classification": classification,
                "reason": msg,
            })

    # -------------------------
    # CASE 2: gene_name missing
    # -------------------------
    else:

        if classification in go_map_dict:
            regex_mapping = go_map_dict[classification]["regex_mapping"]
            manual_mapping = go_map_dict[classification]["manual_mapping"]

            if str(regex_mapping).lower() != "unknown":
                mapped_class = regex_mapping
            else:
                mapped_class = manual_mapping

            mapped_superclass = go_map_dict[classification]["mapped_superclass"]
        else:
            msg = f"No GO mapping found for classification '{classification}'"
            print(f"WARNING: {msg} (PDB: {row['pdb_id']}, chain: {row['new_chain']})")
            dropped_rows.append({
                "pdb_id": row["pdb_id"],
                "new_chain": row["new_chain"],
                "original_chain": row["original_chain"],
                "uniprot_id": row["uniprot_id"],
                "organism": row["organism"],
                "classification": classification,
                "reason": msg,
            })

    # Append row only if mapping was found
    if mapped_class is not None and mapped_superclass is not None:
        output_rows.append({
            "pdb_id": row["pdb_id"],
            "new_chain": row["new_chain"],
            "original_chain": row["original_chain"],
            "uniprot_id": row["uniprot_id"],
            "organism": row["organism"],
            "classification": classification,
            "mapped_class": mapped_class,
            "mapped_superclass": mapped_superclass
        })

# =========================
# Save filtered annotations
# =========================
df_output = pd.DataFrame(output_rows)
df_output.to_csv(OUTPUT_ANNOTATIONS_CSV, index=False)

# =========================
# Save dropped rows (B2M exclusions + unmapped classifications)
# =========================
df_dropped = pd.DataFrame(dropped_rows)
df_dropped.to_csv(OUTPUT_DROPPED_CSV, index=False)

print("\nFiltering complete (annotations).")
print(f"Output written to: {OUTPUT_ANNOTATIONS_CSV}")
print(f"Total entries kept: {len(df_output)}")
print(f"Dropped-rows log written to: {OUTPUT_DROPPED_CSV}")
print(f"Total entries dropped: {len(df_dropped)}")

# Reconciliation check: every input row must be either kept or dropped-and-logged.
total_in = len(df_annotations)
total_accounted = len(df_output) + len(df_dropped)
if total_accounted != total_in:
    print(
        f"WARNING: RECONCILIATION MISMATCH - {total_in} input rows, "
        f"{total_accounted} accounted for ({len(df_output)} kept + {len(df_dropped)} dropped). "
        f"{total_in - total_accounted} row(s) unaccounted for - investigate immediately."
    )
else:
    print(f"Reconciliation OK: {total_in} input rows = {len(df_output)} kept + {len(df_dropped)} dropped.")

print("\nChains analysis filtering complete.")
