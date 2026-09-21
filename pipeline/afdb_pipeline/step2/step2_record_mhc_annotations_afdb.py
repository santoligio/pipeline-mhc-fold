#!/usr/bin/env python3
"""
Step 2 of the AFDB pipeline: annotate step1's primary models with UniProt
metadata (name, organism, gene, length, review status) via REST lookups.
No structure files are read here -- filtering and download happen later
(step3, step4).

NOVO em relacao a versao anterior:
  - Escrita incremental (append, thread-safe via lock) em vez de acumular
    tudo em memoria e escrever soh no final. Cada linha processada (sucesso
    ou erro) eh gravada no CSV assim que termina.
  - Ao iniciar, le os CSVs de output existentes (se houver) e pula
    pares (pdb_id, chain) ja processados -- permite retomar apos uma
    interrupcao (queda de rede, hibernacao, fechar terminal, etc) sem
    perder o trabalho ja feito e sem reprocessar linhas ja feitas.
  - Motivo: a execucao completa (64668 linhas) foi estimada em ~14h
    rodando local; sem checkpointing, qualquer interrupcao no meio
    perderia TODO o trabalho, ja que a versao anterior soh escrevia o
    CSV final depois do ThreadPoolExecutor inteiro terminar.

Recomendado rodar com nohup/tmux para sobreviver a fechar o terminal:
  nohup python3 step2_record_mhc_annotations_afdb.py > run.log 2>&1 &
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import csv
import logging
import re
import sys
import threading
import time

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from afdb_dataset_config import STEP1_DIR, STEP2_DIR


# =========================
# Configuration
# =========================

STEP1_CSV = STEP1_DIR / "afdb" / "afdb_models.csv"
OUT_DIR = STEP2_DIR

ANN_CSV = OUT_DIR / "afdb_mhc_annotations.csv"
ERR_CSV = OUT_DIR / "afdb_mhc_annotation_errors.csv"

ANN_FIELDNAMES = [
    "pdb_id", "chain", "res_start", "res_end", "uniprot_id",
    "target_length", "mapped_length", "uniprot_name", "uniprot_length",
    "organism", "gene_name", "entry_status", "belongs_to",
]
ERR_FIELDNAMES = ["pdb_id", "chain", "error"]

THREADS = 16
REQUEST_SLEEP = 0.2
REQUEST_TIMEOUT = 20
REQUEST_RETRIES = 3

write_lock = threading.Lock()


# =========================
# Setup
# =========================

def make_output_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("afdb_step2_annotations")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    log_file = OUT_DIR / "afdb_mhc_annotations.log"

    # NOVO: modo "a" (append), nao "w" -- preserva o log de execucoes
    # anteriores caso o script seja retomado apos interrupcao.
    file_handler = logging.FileHandler(log_file, mode="a")
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
# NOVO: Checkpointing helpers
# =========================

def load_done_keys() -> Set[Tuple[str, str]]:
    """Le os CSVs de output existentes (se houver) e retorna o conjunto de
    (pdb_id, chain) ja processados -- sucesso OU erro. Uma linha que ja
    falhou 3x nao eh reprocessada automaticamente; ver nota no main()."""
    done: Set[Tuple[str, str]] = set()

    for path, key_cols in [(ANN_CSV, ("pdb_id", "chain")), (ERR_CSV, ("pdb_id", "chain"))]:
        if not path.is_file():
            continue
        try:
            df = pd.read_csv(path, dtype=str)
        except pd.errors.EmptyDataError:
            continue
        if df.empty:
            continue
        for _, row in df.iterrows():
            done.add((str(row[key_cols[0]]), str(row[key_cols[1]])))

    return done


def init_csv_with_header(path: Path, fieldnames: List[str]) -> None:
    """Cria o arquivo com header apenas se ele ainda nao existir (nao
    sobrescreve em caso de retomada)."""
    if path.is_file():
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def append_row(path: Path, fieldnames: List[str], row: Dict) -> None:
    """Escrita thread-safe de uma linha, com flush imediato -- garante que
    o dado esteja em disco assim que a linha eh processada, nao soh no
    fechamento do arquivo."""
    with write_lock:
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)
            f.flush()


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
    """"AF-P01911-F1-model_v4" -> "P01911". Works for any fragment number
    (F1, F2, ...) since it just reads the second "-"-separated field."""
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

def annotate_afdb_row(row: dict, index: int, total: int, logger: logging.Logger) -> None:
    model_id = str(row["pdb"]).strip()
    chain_id = str(row["chain"])
    tstart = int(row["tstart"])
    tend = int(row["tend"])
    target_length = tend - tstart + 1

    logger.info(f"[{index}/{total}] [AFDB] {model_id}")

    if target_length <= 0:
        append_row(ERR_CSV, ERR_FIELDNAMES, {
            "pdb_id": model_id, "chain": chain_id, "error": "invalid residue range",
        })
        return

    uniprot_id = extract_afdb_uniprot_id(model_id)
    meta = get_uniprot_metadata(uniprot_id, logger)

    if meta is None:
        append_row(ERR_CSV, ERR_FIELDNAMES, {
            "pdb_id": model_id, "chain": chain_id,
            "error": f"UniProt metadata failed: {uniprot_id}",
        })
        return

    append_row(ANN_CSV, ANN_FIELDNAMES, {
        "pdb_id": model_id,
        "chain": chain_id,
        "res_start": tstart,
        "res_end": tend,
        "uniprot_id": uniprot_id,
        "target_length": target_length,
        "mapped_length": meta["uniprot_length"],
        **meta,
    })


# =========================
# Main
# =========================

def main() -> None:
    make_output_dirs()
    validate_inputs()

    logger = setup_logger()

    init_csv_with_header(ANN_CSV, ANN_FIELDNAMES)
    init_csv_with_header(ERR_CSV, ERR_FIELDNAMES)

    input_df = load_primary_rows()
    total_all = len(input_df)

    # NOVO: pula linhas ja processadas em execucoes anteriores (resume).
    # Nota: linhas que ja falharam (estao em ERR_CSV) tambem sao puladas --
    # se voce quiser reprocessar especificamente os erros, apague as linhas
    # correspondentes de afdb_mhc_annotation_errors.csv antes de rodar de novo.
    done_keys = load_done_keys()
    if done_keys:
        input_df["_key"] = list(zip(input_df["pdb"].astype(str), input_df["chain"].astype(str)))
        input_df = input_df[~input_df["_key"].isin(done_keys)].drop(columns=["_key"])

    total = len(input_df)
    skipped = total_all - total

    logger.info(
        f"[START] AFDB annotations; total_rows={total_all}; "
        f"already_done={skipped}; remaining={total}"
    )

    if total == 0:
        logger.info("[DONE] Nada a processar -- todas as linhas ja foram feitas.")
        return

    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        futures = [
            executor.submit(annotate_afdb_row, row, index, total, logger)
            for index, row in enumerate(input_df.to_dict("records"), start=1)
        ]

        for future in as_completed(futures):
            future.result()  # propaga excecoes nao tratadas, se houver

    logger.info(f"[CSV] {ANN_CSV}")
    logger.info(f"[CSV] {ERR_CSV}")
    logger.info("[DONE] AFDB annotations")


if __name__ == "__main__":
    main()
