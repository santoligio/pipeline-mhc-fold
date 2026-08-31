#!/usr/bin/env python3
"""
Download selected PDB assemblies or AFDB models.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import gzip
import shutil

import pandas as pd
import requests


# =========================
# Configuration
# =========================

PIPELINE_DIR = Path(__file__).resolve().parents[1]
DATASET = "pdb"  # "pdb" or "afdb"

INPUT_CSV = {
    "pdb": PIPELINE_DIR / "step1" / "pdb" / "pdb_assemblies.csv",
    "afdb": PIPELINE_DIR / "step1" / "afdb" / "afdb_models.csv",
}

OUT_DIR = PIPELINE_DIR / "step2" / DATASET

PDB_ASSEMBLY_DIR = OUT_DIR / "1_assemblies"
AFDB_MODEL_DIR = OUT_DIR / "1_models"

THREADS = 8
REQUEST_TIMEOUT = 60

# Manual download blacklist.
# For PDB, use the four-character PDB ID, e.g. {"7B5F"}.
# For AFDB, use the model ID exactly as it appears in the input CSV.
DOWNLOAD_REMOVAL_LIST = {
     "7B5F", "6ENY", "8JHV", "1HLA", "1B3J", "9CGS", "1LQV", "3JTC", "7RNO", "4PJ8"
}


# =========================
# File helpers
# =========================

def extract_gzip(gz_path: Path) -> Path:
    out_path = gz_path.with_suffix("")

    with gzip.open(gz_path, "rb") as f_in:
        with out_path.open("wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    gz_path.unlink()
    print(f"[EXTRACT] {out_path.name}")

    return out_path


def download_file(url: str, out_path: Path) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() or out_path.with_suffix("").exists():
        print(f"[SKIP] {out_path.name}")
        return True

    response = requests.get(url, timeout=REQUEST_TIMEOUT)

    if response.status_code != 200:
        return False

    out_path.write_bytes(response.content)
    print(f"[DOWNLOAD] {out_path.name}")
    return True


# =========================
# Download modes
# =========================

def download_pdb_assembly(pdb_id: str, assembly_number: str) -> None:
    pdb_id_lower = pdb_id.lower()
    pdb_id_upper = pdb_id.upper()

    gz_name = f"{pdb_id_lower}-assembly{assembly_number}.cif.gz"
    out_path = PDB_ASSEMBLY_DIR / gz_name

    assembly_url = (
        f"https://files.rcsb.org/download/"
        f"{pdb_id_upper}-assembly{assembly_number}.cif.gz"
    )

    ok = download_file(assembly_url, out_path)

    # Fallback to entry CIF when biological assembly is unavailable.
    if not ok:
        print(f"[FALLBACK] {pdb_id}: trying entry CIF")
        fallback_url = f"https://files.rcsb.org/download/{pdb_id_upper}.cif.gz"
        out_path = PDB_ASSEMBLY_DIR / f"{pdb_id_lower}-assembly1.cif.gz"
        ok = download_file(fallback_url, out_path)

    if not ok:
        print(f"[FAILED] {pdb_id}")
        return

    if out_path.exists() and out_path.suffix == ".gz":
        extract_gzip(out_path)


def download_afdb_model(model_id: str) -> None:
    out_path = AFDB_MODEL_DIR / f"{model_id}.cif"
    url = f"https://alphafold.ebi.ac.uk/files/{model_id}.cif"

    ok = download_file(url, out_path)

    if not ok:
        print(f"[FAILED] {model_id}")


def parse_pdb_and_assembly(value: str) -> tuple[str, str]:
    pdb_id, assembly_part = value.split("-")
    assembly_number = assembly_part.replace("assembly", "").replace(".cif", "")
    return pdb_id, assembly_number


def is_blacklisted_structure(structure_id: str) -> bool:
    blacklist = {str(value).strip().upper() for value in DOWNLOAD_REMOVAL_LIST}
    return str(structure_id).strip().upper() in blacklist


# =========================
# Main
# =========================

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    input_csv = INPUT_CSV[DATASET]
    df = pd.read_csv(input_csv)

    primary_df = df[df["status"].astype(str).str.lower() == "primary"].copy()

    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        futures = []

        if DATASET == "pdb":
            PDB_ASSEMBLY_DIR.mkdir(parents=True, exist_ok=True)

            for _, row in primary_df.iterrows():
                pdb_id, assembly_number = parse_pdb_and_assembly(row["pdb"])

                if is_blacklisted_structure(pdb_id):
                    print(f"[SKIPPED] {pdb_id.upper()}: manual blacklist")
                    continue

                futures.append(
                    executor.submit(download_pdb_assembly, pdb_id, assembly_number)
                )

        elif DATASET == "afdb":
            AFDB_MODEL_DIR.mkdir(parents=True, exist_ok=True)

            for _, row in primary_df.iterrows():
                model_id = str(row["pdb"])

                if is_blacklisted_structure(model_id):
                    print(f"[SKIPPED] {model_id}: manual blacklist")
                    continue

                futures.append(
                    executor.submit(download_afdb_model, model_id)
                )

        else:
            raise ValueError(f"Unknown DATASET: {DATASET}")

        for future in futures:
            future.result()

    print(f"[DONE] {DATASET} structures written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
