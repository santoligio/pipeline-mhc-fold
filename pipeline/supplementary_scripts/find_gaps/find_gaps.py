#!/usr/bin/env python3

import os
import pandas as pd
from Bio.PDB import MMCIFParser


# =========================
# Configuration
# =========================

CIF_DIR = "/media/gio/portgas/gio/mhc/version_02/filter/step2/pdb/1_assemblies"
CSV_FILE = "/media/gio/portgas/gio/mhc/version_02/filter/step1/pdb/pdb_assemblies.csv"

ANNOTATION_FILE = "/media/gio/portgas/gio/mhc/version_02/analysis/functional_annotation/filtered/pdb_mhc_annotations_filtered.csv"

OUT_DIR = "/media/gio/portgas/gio/mhc/version_02/filter/find_gaps/"

MAIN_GAPS_CSV   = os.path.join(OUT_DIR, "mhc_chain_gaps.csv")
FLANK_GAPS_CSV  = os.path.join(OUT_DIR, "mhc_chain_flank_gaps.csv")

os.makedirs(OUT_DIR, exist_ok=True)


# =========================
# GAP CALCULATION CORE
# =========================
def calculate_gaps_from_region(chain, start, end):

    auth_resseqs = []
    for res in chain.get_residues():
        hetflag, auth_resseq, _ = res.id
        if hetflag.strip():
            continue
        auth_resseqs.append(auth_resseq)

    if not auth_resseqs:
        return 0, []

    # Map renumbered → auth
    renum_to_auth = {
        idx + 1: auth
        for idx, auth in enumerate(auth_resseqs)
    }

    # Extract ONLY region
    region_auth = [
        renum_to_auth[i]
        for i in range(start, end + 1)
        if i in renum_to_auth
    ]

    if len(region_auth) < 2:
        return 0, []

    gap_list = []

    for prev, curr in zip(region_auth, region_auth[1:]):
        gap = curr - prev - 1
        if gap > 0:
            gap_list.append(gap)

    return sum(gap_list), gap_list


# =========================
# Formatting helper
# =========================
def format_gap_list(gap_list, cap=None):
    formatted = []

    for g in gap_list:
        if cap is not None and g > cap:
            formatted.append(f"{cap}+")
        else:
            formatted.append(str(g))

    return "[" + ", ".join(formatted) + "]"

# =========================
# Main
# =========================
def main():

    df = pd.read_csv(CSV_FILE)
    df_primary = df[df["status"].str.lower() == "primary"]

    df_annot = pd.read_csv(ANNOTATION_FILE)
    df_annot["pdb_id"] = df_annot["pdb_id"].str.upper()
    class_map = dict(zip(df_annot["pdb_id"], df_annot["mapped_class"]))

    parser = MMCIFParser(QUIET=True, auth_chains=True)

    main_rows = []
    flank_rows = []

    for _, row in df_primary.iterrows():

        pdb_id = row["pdb"].split("-")[0].upper()

        if pdb_id not in class_map:
            continue

        cif_path = os.path.join(CIF_DIR, f"{pdb_id.lower()}-assembly1.cif")
        if not os.path.exists(cif_path):
            continue

        structure = parser.get_structure(pdb_id, cif_path)
        model = structure[0]

        mhc_chain_id = row["chain"]
        if mhc_chain_id not in model:
            continue

        mhc_chain = model[mhc_chain_id]

        tstart = int(row["tstart"])
        tend   = int(row["tend"])

        # =========================
        # MAIN REGION
        # =========================
        total_gaps, gap_list = calculate_gaps_from_region(
            mhc_chain, tstart, tend
        )

        if total_gaps > 0:
            main_rows.append({
                "pdb_id": pdb_id,
                "class": class_map[pdb_id],
                "total_gaps": total_gaps,
                "gaps": format_gap_list(gap_list)
            })

        # =========================
        # FLANK REGIONS (STRICT)
        # =========================

        # LEFT flank: strictly before tstart
        left_start = max(1, tstart - 10)
        left_end   = tstart - 1

        # RIGHT flank: strictly after tend
        right_start = tend + 1
        right_end   = tend + 10

        left_total, left_gaps = calculate_gaps_from_region(
            mhc_chain, left_start, left_end
        )

        right_total, right_gaps = calculate_gaps_from_region(
            mhc_chain, right_start, right_end
        )

        if left_gaps or right_gaps:
            flank_rows.append({
                "pdb_id": pdb_id,
                "class": class_map[pdb_id],
                "left_total_gaps": left_total,
                "left_gaps": format_gap_list(left_gaps, cap=10),
                "right_total_gaps": right_total,
                "right_gaps": format_gap_list(right_gaps, cap=10)
            })
    # =========================
    # WRITE OUTPUTS
    # =========================
    if main_rows:
        pd.DataFrame(main_rows).to_csv(MAIN_GAPS_CSV, index=False)
        print(f"[MAIN GAPS WRITTEN] {MAIN_GAPS_CSV}")
    else:
        print("[INFO] No main-region gaps found.")

    if flank_rows:
        pd.DataFrame(flank_rows).to_csv(FLANK_GAPS_CSV, index=False)
        print(f"[FLANK GAPS WRITTEN] {FLANK_GAPS_CSV}")
    else:
        print("[INFO] No flanking-region gaps found.")


if __name__ == "__main__":
    main()
