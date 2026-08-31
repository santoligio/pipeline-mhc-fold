#!/usr/bin/env python3
"""
Check paired receptor/antibody binder annotations.

Input, in the current folder:
  binders_annotations_filtered.csv

Required columns:
  pdb_id,new_chain,original_chain,uniprot_id,organism,classification,mapped_class,mapped_superclass

Rules checked per pdb_id:
  TCR alpha-beta pair:
    if TCRa is present, TCRb must also be present
    if TCRb is present, TCRa must also be present

  TCR gamma-delta pair:
    if TCRg is present, TCRd must also be present
    if TCRd is present, TCRg must also be present

  Antibody heavy-light pair:
    if AbH is present, AbL must also be present
    if AbL is present, AbH must also be present

  NKG2 superclass:
    if any mapped_superclass is NKG2, the PDB should contain at least
    two NKG2-superclass binder rows. This uses mapped_superclass rather
    than mapped_class because NKG2 mapped_class can be uncertain.

Outputs, in the current folder:
  paired_annotation_missing_summary.csv
  paired_annotation_missing_details.csv
  paired_annotation_check_summary.txt
"""

from pathlib import Path
from typing import Dict, List
import pandas as pd


INPUT_CSV = Path("binders_annotations_filtered.csv")

OUTPUT_SUMMARY_CSV = Path("paired_annotation_missing_summary.csv")
OUTPUT_DETAILS_CSV = Path("paired_annotation_missing_details.csv")
OUTPUT_SUMMARY_TXT = Path("paired_annotation_check_summary.txt")

REQUIRED_COLUMNS = {
    "pdb_id",
    "new_chain",
    "original_chain",
    "uniprot_id",
    "organism",
    "classification",
    "mapped_class",
    "mapped_superclass",
}

PAIR_RULES = {
    "TCR_alpha_beta": {"TCRa", "TCRb"},
    "TCR_gamma_delta": {"TCRg", "TCRd"},
    "antibody_heavy_light": {"AbH", "AbL"},
}


def normalize_pdb_id(value) -> str:
    return str(value).strip().upper()


def normalize_mapped_class(value) -> str:
    """
    Normalize mapped_class labels while preserving the expected canonical names.

    Canonical labels:
      TCRa, TCRb, TCRg, TCRd, AbH, AbL
    """
    text = str(value).strip()

    if not text or text.lower() == "nan":
        return ""

    compact = (
        text
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
        .replace("/", "")
        .replace("\\", "")
        .lower()
    )

    aliases = {
        "tcra": "TCRa",
        "tcralpha": "TCRa",
        "tcrα": "TCRa",
        "tra": "TCRa",

        "tcrb": "TCRb",
        "tcrbeta": "TCRb",
        "tcrβ": "TCRb",
        "trb": "TCRb",

        "tcrg": "TCRg",
        "tcrgamma": "TCRg",
        "tcrγ": "TCRg",
        "trg": "TCRg",

        "tcrd": "TCRd",
        "tcrdelta": "TCRd",
        "tcrδ": "TCRd",
        "trd": "TCRd",

        "abh": "AbH",
        "abheavy": "AbH",
        "antibodyheavy": "AbH",
        "heavychain": "AbH",
        "igh": "AbH",

        "abl": "AbL",
        "ablight": "AbL",
        "antibodylight": "AbL",
        "lightchain": "AbL",
        "igk": "AbL",
        "igl": "AbL",
    }

    return aliases.get(compact, text)


def validate_input(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise SystemExit(
            f"ERROR: {INPUT_CSV} missing required columns: {sorted(missing)}"
        )


def class_chains_for_pdb(group: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Return canonical mapped_class -> list of new_chain IDs.
    """
    by_class: Dict[str, List[str]] = {}

    for _, row in group.iterrows():
        mapped_class = normalize_mapped_class(row["mapped_class"])

        if not mapped_class:
            continue

        new_chain = str(row["new_chain"]).strip()

        by_class.setdefault(mapped_class, [])

        if new_chain:
            by_class[mapped_class].append(new_chain)

    return by_class


def nkg2_chains_for_pdb(group: pd.DataFrame) -> List[str]:
    """
    Return unique chains whose mapped_superclass is NKG2.

    NKG2 is checked by mapped_superclass rather than mapped_class because
    chain-level class assignment can be uncertain.
    """
    chains = []

    for _, row in group.iterrows():
        superclass = str(row["mapped_superclass"]).strip().lower()

        if superclass != "nkg2":
            continue

        new_chain = str(row["new_chain"]).strip()

        if new_chain:
            chains.append(new_chain)

    return sorted(set(chains))


def check_pairs(df: pd.DataFrame):
    summary_rows = []
    detail_rows = []

    df = df.copy()
    df["pdb_id"] = df["pdb_id"].apply(normalize_pdb_id)
    df["mapped_class_normalized"] = df["mapped_class"].apply(normalize_mapped_class)

    for pdb_id, group in df.groupby("pdb_id", sort=True):
        by_class = class_chains_for_pdb(group)
        present_classes = set(by_class)

        for rule_name, required_classes in PAIR_RULES.items():
            present_in_rule = present_classes & required_classes

            if not present_in_rule:
                continue

            missing_classes = required_classes - present_classes

            if not missing_classes:
                continue

            present_text = ";".join(
                f"{cls}:{','.join(sorted(set(by_class.get(cls, []))))}"
                for cls in sorted(present_in_rule)
            )

            summary_rows.append({
                "pdb_id": pdb_id,
                "rule": rule_name,
                "present_classes": ";".join(sorted(present_in_rule)),
                "missing_classes": ";".join(sorted(missing_classes)),
                "present_chains": present_text,
            })

            for missing_class in sorted(missing_classes):
                detail_rows.append({
                    "pdb_id": pdb_id,
                    "rule": rule_name,
                    "missing_class": missing_class,
                    "present_classes": ";".join(sorted(present_in_rule)),
                    "present_chains": present_text,
                })

        nkg2_chains = nkg2_chains_for_pdb(group)

        if nkg2_chains and len(nkg2_chains) < 2:
            present_text = f"NKG2:{','.join(nkg2_chains)}"

            summary_rows.append({
                "pdb_id": pdb_id,
                "rule": "NKG2_superclass_pair",
                "present_classes": "NKG2",
                "missing_classes": "second_NKG2_superclass_binder",
                "present_chains": present_text,
            })

            detail_rows.append({
                "pdb_id": pdb_id,
                "rule": "NKG2_superclass_pair",
                "missing_class": "second_NKG2_superclass_binder",
                "present_classes": "NKG2",
                "present_chains": present_text,
            })

    return summary_rows, detail_rows


def write_text_summary(
    df: pd.DataFrame,
    summary_rows: List[dict],
) -> None:
    total_pdbs = df["pdb_id"].nunique()
    total_rows = len(df)
    problem_pdbs = sorted({row["pdb_id"] for row in summary_rows})

    with OUTPUT_SUMMARY_TXT.open("w", encoding="utf-8") as handle:
        handle.write("Paired binder annotation check\n")
        handle.write("==============================\n\n")

        handle.write(f"Input file: {INPUT_CSV}\n")
        handle.write(f"Total annotation rows: {total_rows}\n")
        handle.write(f"Total PDB IDs: {total_pdbs}\n")
        handle.write(f"PDB IDs with missing paired entries: {len(problem_pdbs)}\n\n")

        handle.write("Rules checked:\n")
        handle.write("  TCR_alpha_beta: TCRa + TCRb\n")
        handle.write("  TCR_gamma_delta: TCRg + TCRd\n")
        handle.write("  antibody_heavy_light: AbH + AbL\n")
        handle.write("  NKG2_superclass_pair: at least two mapped_superclass=NKG2 rows\n\n")

        if not summary_rows:
            handle.write("No missing paired entries found.\n")
            return

        handle.write("Missing paired entries:\n")

        for row in summary_rows:
            handle.write(
                f"  {row['pdb_id']} | {row['rule']} | "
                f"present={row['present_classes']} | "
                f"missing={row['missing_classes']} | "
                f"chains={row['present_chains']}\n"
            )


def main() -> None:
    if not INPUT_CSV.is_file():
        raise SystemExit(f"ERROR: input file not found in current folder: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV, keep_default_na=False)
    validate_input(df)

    summary_rows, detail_rows = check_pairs(df)

    pd.DataFrame(
        summary_rows,
        columns=[
            "pdb_id",
            "rule",
            "present_classes",
            "missing_classes",
            "present_chains",
        ],
    ).to_csv(OUTPUT_SUMMARY_CSV, index=False)

    pd.DataFrame(
        detail_rows,
        columns=[
            "pdb_id",
            "rule",
            "missing_class",
            "present_classes",
            "present_chains",
        ],
    ).to_csv(OUTPUT_DETAILS_CSV, index=False)

    write_text_summary(df, summary_rows)

    print(f"[INPUT] {INPUT_CSV}")
    print(f"[SUMMARY] problematic_pdbs={len(set(row['pdb_id'] for row in summary_rows))}")
    print(f"[CSV] {OUTPUT_SUMMARY_CSV}")
    print(f"[CSV] {OUTPUT_DETAILS_CSV}")
    print(f"[TXT] {OUTPUT_SUMMARY_TXT}")


if __name__ == "__main__":
    main()
