#!/usr/bin/env python3
"""
Record MHC functional annotations for AFDB models.

AFDB pipeline order (differs from PDB):
    step1 (select) -> step2 (this script: annotate) -> step3 (filter) -> step4 (download)

Because AFDB is much larger than the PDB, annotation happens directly against
the step1 selection table -- BEFORE anything is downloaded -- so step3 can
discard everything that doesn't pass the MHC / species filters and step4
only has to download the models that survive.

Only UniProt REST lookups are needed here, so no structure files are read.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
import logging
import re
import time

import pandas as pd
import requests

from afdb_dataset_config import STEP1_DIR, STEP2_DIR


# =========================
# Configuration
# =========================

# NOTE: step1 (not part of this edit) needs to write its AFDB output to
# this same path -- afdb_pipeline/step1/afdb/afdb_models.csv -- for this
# to resolve. If step1 still writes to the old top-level "step1/afdb/"
# location, either update step1's output path to match, or share step1
# here so it can be updated too.
STEP1_CSV = STEP1_DIR / "afdb" / "afdb_models.csv"

OUT_DIR = STEP2_DIR

THREADS = 16
REQUEST_SLEEP = 0.2
REQUEST_TIMEOUT = 20
REQUEST_RETRIES = 3


# =========================
# Setup
# =========================

def log(message: str) -> None:
    print(message, flush=True)


def make_output_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("afdb_step2_annotations")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    log_file = OUT_DIR / "afdb_mhc_annotations.log"

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def validate_inputs() -> None:
    if not STEP1_CSV.is_file():
        raise SystemExit(f"ERROR: input CSV not found: {STEP1_CSV}")


# =========================
# Network / UniProt
# =========================

def safe_get_json(url: str, label: str, logger: logging.Logger) -> Optional[dict]:
    for attempt in range(1, REQUEST_RETRIES + 1):
        try:
            time.sleep(REQUEST_SLEEP)
            response = requests.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            logger.warning(f"[REQUEST] {label}: attempt={attempt}/{REQUEST_RETRIES}; {exc}")
            time.sleep(2 ** attempt)

    logger.error(f"[REQUEST_FAILED] {label}")
    return None


def get_uniprot_metadata(uniprot_id: str, logger: logging.Logger) -> Optional[dict]:
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    data = safe_get_json(url, f"UNIPROT:{uniprot_id}", logger)

    if data is None:
        return None

    name = "Missing entry"
    try:
        name = data["proteinDescription"]["recommendedName"]["fullName"]["value"]
    except Exception:
        try:
            name = data["proteinDescription"]["submissionNames"][0]["fullName"]["value"]
            logger.warning(f"[UNIPROT] {uniprot_id}: used submissionNames")
        except Exception:
            logger.warning(f"[UNIPROT] {uniprot_id}: protein name missing")

    uniprot_length = data.get("sequence", {}).get("length", "Missing entry")
    organism = data.get("organism", {}).get("scientificName", "Missing entry")

    gene_name = "Missing entry"
    try:
        gene_name = data["genes"][0]["geneName"]["value"]
    except Exception:
        logger.warning(f"[UNIPROT] {uniprot_id}: gene name missing")

    entry_status = data.get("entryType")
    entry_status = entry_status.capitalize() if entry_status else "Missing entry"

    belongs_to = ""
    for comment in data.get("comments", []):
        for text in comment.get("texts", []):
            value = text.get("value", "")
            if "belongs to" not in value.lower():
                continue

            match = re.search(r"(Belongs to .*?\.)", value)
            belongs_to = match.group(1) if match else value
            break

        if belongs_to:
            break

    return {
        "uniprot_name": name,
        "uniprot_length": uniprot_length,
        "organism": organism,
        "gene_name": gene_name,
        "entry_status": entry_status,
        "belongs_to": belongs_to,
    }


# =========================
# Input helpers
# =========================

def extract_afdb_uniprot_id(model_id: str) -> str:
    value = str(model_id).strip()

    if value.startswith("AF-"):
        parts = value.split("-")
        if len(parts) >= 2:
            return parts[1]

    return value


def load_primary_rows() -> pd.DataFrame:
    df = pd.read_csv(STEP1_CSV)

    required = {"pdb", "chain", "tstart", "tend", "status"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"ERROR: {STEP1_CSV} missing columns: {sorted(missing)}")

    return df[df["status"].astype(str).str.strip().str.lower() == "primary"].copy()


# =========================
# Annotation logic
# =========================

def annotate_afdb_row(row: dict, index: int, total: int, logger: logging.Logger) -> Tuple[List[dict], List[dict]]:
    records = []
    errors = []

    model_id = str(row["pdb"]).strip()
    chain_id = str(row["chain"])
    tstart = int(row["tstart"])
    tend = int(row["tend"])
    target_length = tend - tstart + 1

    logger.info(f"[{index}/{total}] [AFDB] {model_id}")

    if target_length <= 0:
        errors.append({"pdb_id": model_id, "chain": chain_id, "error": "invalid residue range"})
        return records, errors

    uniprot_id = extract_afdb_uniprot_id(model_id)
    meta = get_uniprot_metadata(uniprot_id, logger)

    if meta is None:
        errors.append({"pdb_id": model_id, "chain": chain_id, "error": f"UniProt metadata failed: {uniprot_id}"})
        return records, errors

    records.append({
        "pdb_id": model_id,
        "chain": chain_id,
        "res_start": tstart,
        "res_end": tend,
        "uniprot_id": uniprot_id,
        "target_length": target_length,
        "mapped_length": meta["uniprot_length"],
        **meta,
    })

    return records, errors


# =========================
# Main
# =========================

def main() -> None:
    make_output_dirs()
    validate_inputs()

    logger = setup_logger()
    input_df = load_primary_rows()
    total = len(input_df)

    logger.info(f"[START] AFDB annotations; rows={total}")

    all_records: List[dict] = []
    all_errors: List[dict] = []

    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        futures = [
            executor.submit(annotate_afdb_row, row, index, total, logger)
            for index, row in enumerate(input_df.to_dict("records"), start=1)
        ]

        for future in as_completed(futures):
            records, errors = future.result()
            all_records.extend(records)
            all_errors.extend(errors)

    ann_csv = OUT_DIR / "afdb_mhc_annotations.csv"
    err_csv = OUT_DIR / "afdb_mhc_annotation_errors.csv"

    pd.DataFrame(all_records).to_csv(ann_csv, index=False)
    pd.DataFrame(all_errors, columns=["pdb_id", "chain", "error"]).to_csv(err_csv, index=False)

    logger.info(f"[CSV] {ann_csv}")
    logger.info(f"[CSV] {err_csv}")
    logger.info("[DONE] AFDB annotations")


if __name__ == "__main__":
    main()
