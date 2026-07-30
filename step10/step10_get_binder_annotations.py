#!/bin/python3
# ------------------------------------------------------------
# Robust binder UniProt + GO fallback annotation pipeline
# ------------------------------------------------------------

import re
import time
import logging
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed


# =========================
# CONFIGURATION
# =========================

BASE_DIR = "/mnt/c/Users/gio/Documents/foldseek_nefertari/filter/ligands_pipeline"

INPUT_CSV        = f"{BASE_DIR}/step9/pdb/modified_pdbs/step9_binders.csv"
CHAIN_MAP_CSV    = f"{BASE_DIR}/step5/pdb/chain_map.csv"
OUTPUT_CSV       = f"{BASE_DIR}/step10/binders_annotations_merged.csv"
ERROR_CSV        = f"{BASE_DIR}/step10/binders_annotations_errors.csv"

LOG_FILE         = f"{BASE_DIR}/step10/binders_annotations.log"

NUM_THREADS   = 32
REQUEST_SLEEP = 5

DATA_API = "https://data.rcsb.org/graphql"


# =========================
# LOGGING
# =========================

logging.basicConfig(
    filename=LOG_FILE,
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logging.getLogger().addHandler(console)


# =========================
# GLOBAL CONTAINERS
# =========================

ERROR_RECORDS = []

ALLOWED_ORGANISMS = {
    "Homo sapiens",
    "Human adenovirus E serotype 4",
    "Human adenovirus C serotype 6",
    "Human cytomegalovirus (strain AD169)",
    "Echovirus E6",
    "Echovirus E11",
    "Echovirus E30",
    "synthetic construct",
    "Plasmodium falciparum",
    "Plasmodium falciparum HB3",
    "Echovirus E18",
}

# ---------------------------------------------------------------
# MANUAL OVERRIDES
# ---------------------------------------------------------------

# PDB IDs to KEEP even though their organism is not in ALLOWED_ORGANISMS.
# Use the 4-character PDB ID (case-insensitive). This applies to every
# chain belonging to that entry, not just one — if you only want to keep
# one specific chain, filter the output csv manually afterwards instead.
ORGANISM_FILTER_EXCEPTIONS = {
    "6V7Y",
    "7BH8",
    "8SOS"
}

# Entries for which GO fallback should be used even when a usable
# SIFTS/UniProt mapping is found. Each item is either:
#   - a bare PDB ID, e.g. "6XYZ"   -> forces GO fallback for every chain of that entry
#   - "PDBID:CHAIN", e.g. "6XYZ:A" -> forces GO fallback only for that chain
#     (CHAIN here is the *new* chain id, i.e. the `chain_id` column from
#     step7_binders.csv / INPUT_CSV, not the original PDB author chain)
FORCE_GO_FALLBACK = {
    "4EN3:B",
    "4EN3:C",
    "4MJI:D",
    "5E6I:B",
    "6BJ1:E",
    "6BJ2:E",
    "6BJ3:E",
    "6BJ8:E",
    "6D78:D",
    "6D78:E",
    "6V80:D",
    "7Q9B:E",
}

# Normalized lookup sets used internally — built once from the lists above
# so the matching logic doesn't have to care about casing/whitespace.
ORGANISM_FILTER_EXCEPTIONS_NORM = {x.strip().upper() for x in ORGANISM_FILTER_EXCEPTIONS}


def _normalize_force_go_entry(entry):
    entry = entry.strip()
    if ":" in entry:
        pdb, chain = entry.split(":", 1)
        return f"{pdb.strip().upper()}:{chain.strip()}"
    return entry.upper()


FORCE_GO_FALLBACK_NORM = {_normalize_force_go_entry(e) for e in FORCE_GO_FALLBACK}


def log_error(pdb_id, old_chain, new_chain, message):
    ERROR_RECORDS.append({
        "pdb_id": pdb_id,
        "original_chain": old_chain,
        "new_chain": new_chain,
        "error": message,
    })
    logging.error(f"[{pdb_id}:{new_chain}] {message}")


def safe_nested_get(d, *keys, default=None):
    """
    Walk a nested dict via `keys`, treating BOTH a missing key and an
    explicit `None` value as "absent" and falling back to `default`.

    Plain chained `.get(key, {})` calls only fall back when a key is
    missing; if the JSON has the key present with an explicit `null`
    (which UniProt does return for some unreviewed/obsolete entries),
    `.get(key, {})` returns None instead of `{}`, and the next `.get()`
    call in the chain raises AttributeError. That exception used to
    propagate all the way out of map_entry() and get silently dropped
    by the thread pool (see main()). This helper closes that hole.
    """
    current = d
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


# =========================
# NETWORK
# =========================

def safe_get(url, label, timeout=10, retries=3):
    for attempt in range(1, retries + 1):
        try:
            time.sleep(REQUEST_SLEEP)
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            logging.warning(f"{label} attempt {attempt}/{retries} failed")
            time.sleep(2 ** attempt)
    logging.error(f"{label} failed after {retries} attempts")
    return None


# =========================
# GO FALLBACK
# =========================

QUERY = """
query($pdb: String!) {
  entry(entry_id: $pdb) {
    polymer_entities {
      rcsb_polymer_entity_container_identifiers {
        entity_id
        auth_asym_ids
      }
      rcsb_polymer_entity {
        pdbx_description
      }
      rcsb_entity_source_organism {
        ncbi_scientific_name
      }
    }
  }
}
"""

def go_fallback(pdb_id, old_chain):
    try:
        r = requests.post(
            DATA_API,
            json={"query": QUERY, "variables": {"pdb": pdb_id}},
            timeout=10
        )
        r.raise_for_status()

        entry = r.json().get("data", {}).get("entry")
        if not entry:
            return None, None

        for ent in entry.get("polymer_entities", []):
            auth_ids = (
                ent.get("rcsb_polymer_entity_container_identifiers", {})
                .get("auth_asym_ids") or []
            )

            if old_chain in auth_ids:
                polymer = (
                    ent.get("rcsb_polymer_entity", {})
                    .get("pdbx_description", "Missing entry")
                )

                organism_list = ent.get("rcsb_entity_source_organism") or []
                organism = (
                    organism_list[0].get("ncbi_scientific_name")
                    if organism_list else "Missing entry"
                )

                return polymer, organism

        return None, None

    except Exception as e:
        logging.warning(f"GO fallback failed for {pdb_id}:{old_chain} ({e})")
        return None, None


# =========================
# SIFTS + UNIPROT
# =========================

def get_sifts_mapping(pdb_id):
    url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id.lower()}"
    data = safe_get(url, f"SIFTS:{pdb_id}")
    if not data or pdb_id.lower() not in data:
        return {}
    return data[pdb_id.lower()].get("UniProt", {})


def get_uniprot_metadata(uniprot_id):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    data = safe_get(url, f"UniProt:{uniprot_id}")
    if not data:
        return None

    if not isinstance(data, dict):
        # Obsolete/merged/demerged accessions can come back as a list or
        # some other shape instead of a single-entry dict. Treat as a
        # missing record rather than letting AttributeError bubble up.
        logging.warning(
            f"UniProt:{uniprot_id} returned unexpected response shape "
            f"({type(data).__name__}); treating as missing"
        )
        return None

    name = safe_nested_get(
        data, "proteinDescription", "recommendedName", "fullName", "value",
        default="Missing entry"
    )

    gene_name = "Missing entry"
    try:
        gene_name = data["genes"][0]["geneName"]["value"]
    except Exception:
        pass

    organism = safe_nested_get(data, "organism", "scientificName", default="Missing entry")

    return {
        "uniprot_name": name,
        "gene_name": gene_name,
        "organism": organism,
    }


# =========================
# CORE LOGIC
# =========================

def map_entry(pdb_id, new_chain, chain_map):
    """
    Thin wrapper around _map_entry_core that guarantees no exception can
    cause a chain to vanish without a trace. Any unexpected error here
    (bad API response shape, parsing issue, etc.) is now recorded via
    log_error() -> ERROR_RECORDS -> binders_annotations_errors.csv,
    instead of only being caught by the bare `except Exception` in
    main()'s executor loop (which never wrote to ERROR_RECORDS at all).
    """
    pdb_clean = pdb_id.split("-")[0].upper()
    try:
        return _map_entry_core(pdb_clean, new_chain, chain_map)
    except Exception as exc:
        log_error(pdb_clean, None, new_chain, f"unhandled exception in map_entry: {exc}")
        logging.error(f"[{pdb_clean}:{new_chain}] unhandled exception in map_entry", exc_info=True)
        return []


def _map_entry_core(pdb_clean, new_chain, chain_map):

    old_chain_raw = chain_map.get(pdb_clean, {}).get(new_chain)
    if not old_chain_raw:
        log_error(pdb_clean, None, new_chain, "chain mapping missing")
        return []

    old_chain = str(old_chain_raw).split("-")[0]

    # 🚨 MANUAL OVERRIDE: force GO fallback even if SIFTS/UniProt works
    forced_go = (
        pdb_clean in FORCE_GO_FALLBACK_NORM
        or f"{pdb_clean}:{new_chain}" in FORCE_GO_FALLBACK_NORM
    )

    seen = []
    use_go = forced_go

    if forced_go:
        logging.info(
            f"[{pdb_clean}:{old_chain}] GO fallback forced via FORCE_GO_FALLBACK "
            f"override — skipping SIFTS/UniProt lookup"
        )
    else:
        mappings = get_sifts_mapping(pdb_clean)

        for uniprot_key, block in mappings.items():
            uniprot_id = uniprot_key.split("_")[0]
            for seg in block.get("mappings", []):
                if seg.get("chain_id") == old_chain:
                    meta = get_uniprot_metadata(uniprot_id)
                    if meta:
                        seen.append((uniprot_id, meta))
                    else:
                        logging.warning(
                            f"[{pdb_clean}:{old_chain}] UniProt metadata fetch returned None "
                            f"for {uniprot_id} — will fall through to GO"
                        )
                    break

        # 🚨 MULTIPLE MAPPINGS → FORCE GO
        if len(seen) > 1:
            logging.warning(
                f"[{pdb_clean}:{old_chain}] {len(seen)} UniProt mappings found "
                f"({', '.join(u for u, _ in seen)}) — forcing GO fallback"
            )
            seen = []

        if not seen:
            use_go = True
        else:
            meta = seen[0][1]
            gene = meta["gene_name"]
            name = meta["uniprot_name"]
            org  = meta["organism"]

            gene_missing = gene in [None, "", "Missing entry"]
            name_missing = name in [None, "", "Missing entry"]
            org_missing  = org in [None, "", "Missing entry"]

            if (gene_missing and name_missing) or org_missing or gene == "Genome polyprotein" or name == "Genome polyprotein":
                use_go = True

    if use_go:
        polymer, organism = go_fallback(pdb_clean, old_chain)
        if polymer:
            return [{
                "pdb_id": pdb_clean,
                "new_chain": new_chain,
                "original_chain": old_chain_raw,
                "uniprot_id": "GO_FALLBACK",
                "gene_name": "Missing entry",
                "uniprot_name": polymer,
                "organism": organism if organism else "Missing entry",
                "classification": polymer
            }]
        else:
            log_error(pdb_clean, old_chain_raw, new_chain, "GO fallback failed")
            return []

    # NORMAL CASE
    uniprot_id, meta = seen[0]
    gene = meta["gene_name"]
    name = meta["uniprot_name"]

    classification = gene if gene not in [None, "", "Missing entry"] else name

    return [{
        "pdb_id": pdb_clean,
        "new_chain": new_chain,
        "original_chain": old_chain_raw,
        "uniprot_id": uniprot_id,
        "gene_name": gene,
        "uniprot_name": name,
        "organism": meta["organism"],
        "classification": classification
    }]


# =========================
# MAIN
# =========================

def main():

    # ---------------------------------------
    # LOAD INPUT
    # ---------------------------------------
    df = pd.read_csv(INPUT_CSV)

    # ---------------------------------------
    # LOAD CHAIN MAP
    # ---------------------------------------
    df_chain_map = pd.read_csv(CHAIN_MAP_CSV)
    chain_map = {}
    for _, row in df_chain_map.iterrows():
        chain_map.setdefault(row["pdb"].upper(), {})[row["new_chain"]] = row["old_chain"]

    results = []

    # ---------------------------------------
    # THREAD EXECUTION
    # ---------------------------------------
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = {}
        for row in df.itertuples(index=False):
            fut = executor.submit(map_entry, row.pdb, row.chain_id, chain_map)
            futures[fut] = (row.pdb, row.chain_id)

        for f in as_completed(futures):
            pdb_id, chain_id = futures[f]
            try:
                results.extend(f.result())
            except Exception as exc:
                # Safety net: map_entry() already catches its own exceptions
                # and logs them via log_error(), but if something raises
                # outside that (e.g. in the executor machinery itself),
                # record it here too so the chain is never lost silently.
                log_error(pdb_id, None, chain_id, f"unhandled exception in worker thread: {exc}")
                logging.error(
                    f"[{pdb_id}:{chain_id}] Worker thread raised an unhandled exception: {exc}",
                    exc_info=True,
                )

    # ---------------------------------------
    # APPLY ORGANISM FILTER FIRST
    # ---------------------------------------
    final_kept_count = 0
    if results:
        results_df = pd.DataFrame(results)

        organism_ok = results_df["organism"].isin(ALLOWED_ORGANISMS)

        # 🚨 MANUAL OVERRIDE: keep these PDB IDs even if their organism
        # isn't in ALLOWED_ORGANISMS.
        pdb_exempt = results_df["pdb_id"].isin(ORGANISM_FILTER_EXCEPTIONS_NORM)
        keep_mask = organism_ok | pdb_exempt

        kept_via_exception = results_df[pdb_exempt & ~organism_ok]
        if not kept_via_exception.empty:
            logging.info(
                f"Kept {len(kept_via_exception)} row(s) that failed the organism "
                f"filter due to ORGANISM_FILTER_EXCEPTIONS "
                f"(pdb_ids: {sorted(kept_via_exception['pdb_id'].unique())})"
            )

        filtered_out = results_df[~keep_mask]
        if not filtered_out.empty:
            logging.warning(
                f"Organism filter dropped {len(filtered_out)} rows "
                f"(unique organisms: {sorted(filtered_out['organism'].unique())})"
            )
            for _, row in filtered_out.iterrows():
                log_error(
                    row["pdb_id"], row["original_chain"], row["new_chain"],
                    f"organism '{row['organism']}' not in ALLOWED_ORGANISMS"
                )

        results_df = results_df[keep_mask]
        final_kept_count = len(results_df)

        if not results_df.empty:
            results_df.to_csv(OUTPUT_CSV, index=False)

    # ---------------------------------------
    # ERRORS
    # ---------------------------------------
    if ERROR_RECORDS:
        pd.DataFrame(ERROR_RECORDS).to_csv(ERROR_CSV, index=False)

    # ---------------------------------------
    # RECONCILIATION CHECK
    # ---------------------------------------
    # Every input chain must end up either in the final kept output or in
    # ERROR_RECORDS. If this ever doesn't add up, something is silently
    # eating chains again (this is exactly the bug this patch fixed) -
    # fail loudly instead of requiring a manual row count.
    total_input = len(df)
    total_errors = len(ERROR_RECORDS)
    accounted = final_kept_count + total_errors
    if accounted != total_input:
        logging.error(
            f"RECONCILIATION MISMATCH: {total_input} input chains, but only "
            f"{accounted} accounted for ({final_kept_count} kept + {total_errors} errors). "
            f"{total_input - accounted} chain(s) are unaccounted for - investigate immediately."
        )
    else:
        logging.info(
            f"Reconciliation OK: {total_input} input chains = "
            f"{final_kept_count} kept + {total_errors} errors."
        )

if __name__ == "__main__":
    main()
