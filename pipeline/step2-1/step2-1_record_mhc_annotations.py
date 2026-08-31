#!/usr/bin/env python3
"""
Record MHC functional annotations.
Only structures actually downloaded in step2 are annotated.

"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import re
import shutil
import time

import pandas as pd
import requests


# =========================
# Configuration
# =========================

PIPELINE_DIR = Path(__file__).resolve().parents[1]
FILTER_DIR = PIPELINE_DIR

DATABASE = "both"  # "pdb", "afdb", or "both"

STEP1_CSV = {
    "pdb": FILTER_DIR / "step1" / "pdb" / "pdb_assemblies.csv",
    "afdb": FILTER_DIR / "step1" / "afdb" / "afdb_models.csv",
}

STEP2_DOWNLOAD_DIR = {
    "pdb": FILTER_DIR / "step2" / "pdb" / "1_assemblies",
    "afdb": FILTER_DIR / "step2" / "afdb" / "1_models",
}

OUT_DIR = FILTER_DIR / "step2-1"

THREADS = 16
REQUEST_SLEEP = 0.2
REQUEST_TIMEOUT = 20
REQUEST_RETRIES = 3


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
    for database in ["pdb", "afdb"]:
        (OUT_DIR / database).mkdir(parents=True, exist_ok=True)


def setup_logger(database: str) -> logging.Logger:
    logger = logging.getLogger(database)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    log_file = OUT_DIR / database / f"{database}_mhc_annotations.log"

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def validate_inputs(database: str) -> None:
    if not STEP1_CSV[database].is_file():
        raise SystemExit(f"ERROR: input CSV not found: {STEP1_CSV[database]}")

    if not STEP2_DOWNLOAD_DIR[database].is_dir():
        raise SystemExit(f"ERROR: step2 download folder not found: {STEP2_DOWNLOAD_DIR[database]}")


# =========================
# Network
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


# =========================
# UniProt / SIFTS
# =========================

def get_sifts_mapping(pdb_id: str, logger: logging.Logger) -> Dict:
    pdb_clean = pdb_id.split("-")[0].lower()
    url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_clean}"

    data = safe_get_json(url, f"SIFTS:{pdb_clean.upper()}", logger)

    if not data or pdb_clean not in data:
        logger.error(f"[SIFTS_FAILED] {pdb_clean.upper()}: no UniProt mapping")
        return {}

    return data[pdb_clean].get("UniProt", {})


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

def normalize_pdb_id(value) -> str:
    return str(value).split("-")[0].strip().upper()


def parse_pdb_and_assembly(value: str) -> Tuple[str, str]:
    pdb_id, assembly_part = str(value).split("-")
    assembly_number = assembly_part.replace("assembly", "").replace(".cif", "")
    return pdb_id, assembly_number


def extract_afdb_uniprot_id(model_id: str) -> str:
    value = str(model_id).strip()

    if value.startswith("AF-"):
        parts = value.split("-")
        if len(parts) >= 2:
            return parts[1]

    return value


def find_downloaded_pdb_assembly(pdb_value: str) -> Optional[Path]:
    pdb_id, assembly_number = parse_pdb_and_assembly(pdb_value)
    pdb_lower = pdb_id.lower()

    exact = STEP2_DOWNLOAD_DIR["pdb"] / f"{pdb_lower}-assembly{assembly_number}.cif"
    if exact.is_file():
        return exact

    # Step2 fallback can write assembly1 even when the original row requested another assembly.
    fallback = STEP2_DOWNLOAD_DIR["pdb"] / f"{pdb_lower}-assembly1.cif"
    if fallback.is_file():
        return fallback

    matches = sorted(STEP2_DOWNLOAD_DIR["pdb"].glob(f"{pdb_lower}*.cif"))
    return matches[0] if matches else None


def find_downloaded_afdb_model(model_id: str) -> Optional[Path]:
    path = STEP2_DOWNLOAD_DIR["afdb"] / f"{model_id}.cif"

    if path.is_file():
        return path

    matches = sorted(STEP2_DOWNLOAD_DIR["afdb"].glob(f"{model_id}*.cif"))
    return matches[0] if matches else None


def load_downloaded_primary_rows(database: str, logger: logging.Logger) -> pd.DataFrame:
    df = pd.read_csv(STEP1_CSV[database])

    required = {"pdb", "chain", "tstart", "tend", "status"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"ERROR: {STEP1_CSV[database]} missing columns: {sorted(missing)}")

    primary_df = df[df["status"].astype(str).str.strip().str.lower() == "primary"].copy()

    kept_rows = []
    skipped_rows = []

    for _, row in primary_df.iterrows():
        if database == "pdb":
            downloaded = find_downloaded_pdb_assembly(row["pdb"])
        else:
            downloaded = find_downloaded_afdb_model(str(row["pdb"]))

        if downloaded is None:
            skipped_rows.append({
                "pdb": row["pdb"],
                "chain": row["chain"],
                "reason": "not_downloaded_in_step2",
            })
            continue

        row = row.copy()
        row["downloaded_file"] = str(downloaded)
        kept_rows.append(row)

    kept_df = pd.DataFrame(kept_rows)
    skipped_df = pd.DataFrame(skipped_rows)

    logger.info(
        f"[INPUT] {database}: primary={len(primary_df)} downloaded={len(kept_df)} skipped={len(skipped_df)}"
    )

    return kept_df


# =========================
# Annotation logic
# =========================

def annotate_pdb_row(row: dict, index: int, total: int, logger: logging.Logger) -> Tuple[List[dict], List[dict]]:
    records = []
    errors = []

    pdb_id = normalize_pdb_id(row["pdb"])
    chain_id = str(row["chain"]).split("-")[0]
    tstart = int(row["tstart"])
    tend = int(row["tend"])
    target_length = tend - tstart + 1

    logger.info(f"[{index}/{total}] [PDB] {pdb_id}:{chain_id}")

    if target_length <= 0:
        errors.append({"pdb_id": pdb_id, "chain": chain_id, "error": "invalid residue range"})
        return records, errors

    mappings = get_sifts_mapping(pdb_id, logger)

    if not mappings:
        errors.append({"pdb_id": pdb_id, "chain": chain_id, "error": "no SIFTS UniProt mapping"})
        return records, errors

    seen_uniprot = set()
    uniprot_hits = []

    for uniprot_key, block in mappings.items():
        uniprot_id = uniprot_key.split("_")[0]

        if uniprot_id in seen_uniprot:
            continue

        for segment in block.get("mappings", []):
            if segment.get("chain_id") != chain_id:
                continue

            seen_uniprot.add(uniprot_id)

            meta = get_uniprot_metadata(uniprot_id, logger)

            if meta is None:
                errors.append({
                    "pdb_id": pdb_id,
                    "chain": chain_id,
                    "error": f"UniProt metadata failed: {uniprot_id}",
                })
                continue

            uniprot_hits.append(uniprot_id)

            records.append({
                "pdb_id": pdb_id,
                "chain": chain_id,
                "res_start": tstart,
                "res_end": tend,
                "uniprot_id": uniprot_id,
                "target_length": target_length,
                "mapped_length": meta["uniprot_length"],
                **meta,
                "possibly_chimeric": "",
                "grouped_gene_assigned": "",
                "gene_name_assigned": "",
                "organism_assigned": "",
                "comments": "",
            })
            break

    if len(set(uniprot_hits)) > 1:
        for record in records:
            record["possibly_chimeric"] = "yes"

    if not records:
        errors.append({"pdb_id": pdb_id, "chain": chain_id, "error": "no compatible UniProt mapping"})

    return records, errors


def annotate_afdb_row(row: dict, index: int, total: int, logger: logging.Logger) -> Tuple[List[dict], List[dict]]:
    records = []
    errors = []

    model_id = str(row["pdb"]).strip()
    chain_id = str(row["chain"])
    tstart = int(row["tstart"])
    tend = int(row["tend"])
    target_length = tend - tstart + 1

    uniprot_id = extract_afdb_uniprot_id(model_id)

    logger.info(f"[{index}/{total}] [AFDB] {model_id}")

    if target_length <= 0:
        errors.append({"pdb_id": model_id, "chain": chain_id, "error": "invalid residue range"})
        return records, errors

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


def write_pdb_edited_template_if_missing(annotation_csv: Path) -> None:
    edited_csv = annotation_csv.with_name("pdb_mhc_annotations_edited.csv")

    if not edited_csv.exists():
        shutil.copy2(annotation_csv, edited_csv)


def annotate_database(database: str) -> None:
    logger = setup_logger(database)
    validate_inputs(database)

    input_df = load_downloaded_primary_rows(database, logger)
    total = len(input_df)

    logger.info(f"[START] {database.upper()} annotations; rows={total}")

    all_records = []
    all_errors = []

    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        futures = []

        for index, row in enumerate(input_df.to_dict("records"), start=1):
            if database == "pdb":
                futures.append(executor.submit(annotate_pdb_row, row, index, total, logger))
            elif database == "afdb":
                futures.append(executor.submit(annotate_afdb_row, row, index, total, logger))

        for future in as_completed(futures):
            records, errors = future.result()
            all_records.extend(records)
            all_errors.extend(errors)

    ann_csv = OUT_DIR / database / f"{database}_mhc_annotations.csv"
    err_csv = OUT_DIR / database / f"{database}_mhc_annotation_errors.csv"
    pd.DataFrame(all_records).to_csv(ann_csv, index=False)
    pd.DataFrame(all_errors, columns=["pdb_id", "chain", "error"]).to_csv(err_csv, index=False)

    if database == "pdb":
        write_pdb_edited_template_if_missing(ann_csv)

    logger.info(f"[CSV] {ann_csv}")
    logger.info(f"[CSV] {err_csv}")
    logger.info(f"[DONE] {database.upper()} annotations")


# =========================
# Main
# =========================

def main() -> None:
    make_output_dirs()

    for database in selected_databases():
        annotate_database(database)


if __name__ == "__main__":
    main()
