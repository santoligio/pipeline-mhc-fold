#!/usr/bin/env python3
"""
Step 4 of the AFDB pipeline: download the AFDB models that survived step3
filtering. Which option gets downloaded is controlled by
AFDB_DATASET_SELECTION below.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

import pandas as pd
import requests
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from afdb_dataset_config import AfdbDataset, DATASET_MODEL_FILENAMES, STEP3_DIR, STEP4_DIR, describe


# =========================
# Configuration
# =========================

# See afdb_dataset_config.py for the full description of each option:
#   AfdbDataset.ALL_HUMAN_MHC          1. All human AFDB MHC
#   AfdbDataset.CURRENT_RESTRICTIONS   2. Human AFDB MHC, current restrictions (default)
#   AfdbDataset.SUBSET_REVIEWED        3. AFDB subset 1 (Swiss-Prot reviewed)
#   AfdbDataset.SUBSET_LENGTH_175_185  4. AFDB subset 2 (target_length 175-185)
#   AfdbDataset.BOTH_SUBSETS           5. Both AFDB subsets
AFDB_DATASET_SELECTION = AfdbDataset.CURRENT_RESTRICTIONS

INPUT_CSV = STEP3_DIR / DATASET_MODEL_FILENAMES[AFDB_DATASET_SELECTION]

OUT_DIR = STEP4_DIR
AFDB_MODEL_DIR = OUT_DIR / "1_models"

THREADS = 8
REQUEST_TIMEOUT = 60

# AFDB model IDs to skip, exactly as they appear in the "pdb" column
# (e.g. "AF-P01911-F1-model_v4").
DOWNLOAD_REMOVAL_LIST: set = set()
_BLACKLIST_UPPER = {str(v).strip().upper() for v in DOWNLOAD_REMOVAL_LIST}


# =========================
# Helpers
# =========================

def download_file(url: str, out_path: Path) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        print(f"[SKIP] {out_path.name}")
        return True

    response = requests.get(url, timeout=REQUEST_TIMEOUT)
    if response.status_code != 200:
        return False

    out_path.write_bytes(response.content)
    print(f"[DOWNLOAD] {out_path.name}")
    return True


def download_afdb_model(model_id: str) -> None:
    out_path = AFDB_MODEL_DIR / f"{model_id}.cif"
    url = f"https://alphafold.ebi.ac.uk/files/{model_id}.cif"

    if not download_file(url, out_path):
        print(f"[FAILED] {model_id}")


def is_blacklisted_structure(structure_id: str) -> bool:
    return str(structure_id).strip().upper() in _BLACKLIST_UPPER


# =========================
# Main
# =========================

def main() -> None:
    if not INPUT_CSV.is_file():
        raise SystemExit(
            f"ERROR: input CSV not found: {INPUT_CSV}\n"
            f"Did step3_filter_mhc_annotations_afdb.py run and write "
            f"AFDB_DATASET_SELECTION={AFDB_DATASET_SELECTION.value}?"
        )

    AFDB_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[DATASET] {describe(AFDB_DATASET_SELECTION)}")
    print(f"[INPUT] {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    primary_df = df[df["status"].astype(str).str.lower() == "primary"].copy()

    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        futures = []

        for _, row in primary_df.iterrows():
            model_id = str(row["pdb"])

            if is_blacklisted_structure(model_id):
                print(f"[SKIPPED] {model_id}: manual blacklist")
                continue

            futures.append(executor.submit(download_afdb_model, model_id))

        for future in futures:
            future.result()

    print(f"[DONE] afdb structures written to: {AFDB_MODEL_DIR}")


if __name__ == "__main__":
    main()
