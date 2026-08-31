#!/usr/bin/env python3
"""
Build the primary/duplicate structure table from filtered Foldseek hits.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from afdb_dataset_config import STEP1_DIR as AFDB_STEP1_DIR

# =========================
# Configuration
# =========================

# Raw Foldseek alignment files live on a separate path from the
# afdb_pipeline/ tree used by step2 onward.
PIPELINE_DIR = Path(__file__).resolve().parents[2]

DATASET = "afdb"  # "pdb" or "afdb"

FOLDSEEK_ALN = {
    "pdb": PIPELINE_DIR / "foldseek_data" / "pdb" / "dbs_pdb_aln",
    "afdb": PIPELINE_DIR / "afdb_pipeline" / "foldseek_data" / "dbs_afdb_aln_amostra",
}

OUT_CSV = {
    "pdb": PIPELINE_DIR / "step1" / "pdb" / "pdb_assemblies.csv",
    "afdb": AFDB_STEP1_DIR / "afdb" / "afdb_models.csv",
}

EVALUE_CUTOFF = 0.01
MIN_ALIGNMENT_LENGTH = 90

FOLDSEEK_COLUMNS = [
    "query", "target", "fident", "alnlen", "mismatch", "gapopen",
    "qstart", "qend", "tstart", "tend", "evalue", "bits",
    "alntmscore", "qtmscore", "ttmscore", "lddt", "lddtfull", "prob",
]


# =========================
# Foldseek table
# =========================

def load_foldseek_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, delimiter=r"\s+")
    df.columns = FOLDSEEK_COLUMNS
    return df


def filter_hits(df: pd.DataFrame) -> pd.DataFrame:
    """Keep confident hits and one row per target (lowest e-value)."""
    df_filtered = df[
        (df["evalue"] <= EVALUE_CUTOFF)
        & (df["alnlen"] >= MIN_ALIGNMENT_LENGTH)
    ].copy()

    return (
        df_filtered
        .sort_values("evalue")
        .drop_duplicates(subset="target", keep="first")
    )


# =========================
# Target parsing
# =========================

def parse_targets(df_filtered: pd.DataFrame) -> pd.DataFrame:
    """Parse Foldseek target IDs into (pdb, chain, tstart, tend, status).

    PDB targets encode assembly/chain info (e.g. "1abc-assembly1_A") and
    can have duplicate/split entries to dedup. AFDB targets are plain
    model IDs (e.g. "AF-P01911-F1-model_v4") with no chain info, so they
    fall through to the "else" branch below with chain="NoChainInfo".
    """
    records = []
    seen_pdbs = set()
    all_targets = set(df_filtered["target"])

    for _, row in df_filtered.iterrows():
        entry = row["target"]
        has_assembly_info = "-assembly" in entry

        try:
            if has_assembly_info:
                pdb_id = entry.split("_")[0]
                chain_id = entry.split("_")[1]

                # Prefer assembly1 when an assembly1 version of this PDB exists.
                if not pdb_id.endswith("-assembly1"):
                    pdb_code = pdb_id.split("-")[0]
                    assembly1_prefix = f"{pdb_code}-assembly1"

                    if any(target.startswith(assembly1_prefix) for target in all_targets):
                        continue

            else:
                pdb_id = entry
                chain_id = "NoChainInfo"

                # Prefer the main model over split alternative entries.
                parts = entry.split("-")

                if len(parts) == 5:
                    main_entry = f"{parts[0]}-{parts[1]}-{parts[3]}-{parts[4]}"

                    if main_entry in all_targets:
                        continue

            status = "duplicate" if pdb_id in seen_pdbs else "primary"
            seen_pdbs.add(pdb_id)

            records.append({
                "pdb": pdb_id,
                "chain": chain_id,
                "tstart": row["tstart"],
                "tend": row["tend"],
                "status": status,
            })

        except IndexError:
            print(f"[SKIP] {entry}: incorrect target formatting")
            continue

    return pd.DataFrame.from_records(records)


# =========================
# Main
# =========================

def main() -> None:
    foldseek_path = FOLDSEEK_ALN[DATASET]
    out_csv = OUT_CSV[DATASET]
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df = load_foldseek_table(foldseek_path)
    df_filtered = filter_hits(df)
    structures = parse_targets(df_filtered)

    structures.to_csv(out_csv, index=False)

    print(f"[DONE] {DATASET} structures written: {out_csv}")
    print(f"[INFO] rows: {len(structures)}")


if __name__ == "__main__":
    main()
