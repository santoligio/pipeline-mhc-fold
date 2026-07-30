#!/usr/bin/env python3
"""
Find PTM-like residues in chain A of current step7 filtered structures.

This version contains only the current step7 logic:
  - input is a folder of step7 PDB files
  - only chain A is analyzed
  - no tstart/tend CSV is required
  - no problematic_mhc_contacts.csv is required

Usage:
  python find_ptms.py /path/to/step7/pdb/1_filtered_structures [output_prefix]
"""

from pathlib import Path
import sys
import csv
from collections import defaultdict


# PTM-like residue codes found in the current residue review list plus common PTMs.
PTM_RESIDUES = {
    # Phosphorylation
    "SEP": "phosphorylation",
    "TPO": "phosphorylation",
    "PTR": "phosphorylation",

    # Glycosylation / common sugars
    "NAG": "glycosylation",
    "NDG": "glycosylation",
    "MAN": "glycosylation",
    "BMA": "glycosylation",
    "FUC": "glycosylation",
    "GAL": "glycosylation",
    "GLC": "glycosylation",
    "SIA": "glycosylation",
    "NAN": "glycosylation",

    # Acetylation
    "ALY": "acetylation",
    "ACE": "acetylation",

    # Methylation
    "MLY": "methylation",
    "M3L": "methylation",
    "MLZ": "methylation",
    "AGM": "methylation",
    "DM0": "methylation",

    # Hydroxylation
    "HYP": "hydroxylation",
    "HYL": "hydroxylation",

    # Sulfation
    "TYS": "sulfation",

    # Citrullination / deimination
    "CIR": "citrullination",

    # Carboxylation
    "KCX": "carboxylation",
    "CGU": "carboxylation",

    # Cysteine modifications / oxidation / redox-related PTMs
    "CSO": "cysteine modification",
    "CME": "cysteine modification",
    "CSD": "cysteine oxidation",
    "CSX": "cysteine oxidation",
    "OCS": "cysteine oxidation",
    "CSU": "cysteine oxidation",
    "SNC": "S-nitrosylation",

    # Other common modified amino acids
    "PCA": "pyroglutamate formation",
    "LLP": "cofactor-linked lysine modification",

    # Common engineered/substitutional modification
    "MSE": "selenomethionine substitution",

    # Amino-acid-like / peptide-like residues observed in current residue review list.
    # These are tracked because removing them may leave backbone gaps.
    "3AZ": "peptide-like residue",
    "3X9": "modified/unknown peptide residue",
    "4OG": "modified/unknown peptide residue",
    "ABA": "modified/unknown peptide residue",
    "BAL": "beta-amino acid / peptide-like residue",
    "CDE": "modified/unknown peptide residue",
    "F2F": "modified/unknown peptide residue",
    "GIC": "peptide-like residue",
    "KYN": "kynurenine substitution",
    "LPH": "modified/unknown peptide residue",
    "NVA": "modified/unknown peptide residue",
    "OSE": "modified/unknown peptide residue",
    "PFF": "modified/unknown peptide residue",
    "PRQ": "modified/unknown peptide residue",
    "PRV": "modified/unknown peptide residue",
    "QM8": "modified/unknown peptide residue",
    "QMB": "modified/unknown peptide residue",
    "TIG": "amino-acid analog",
    "XFW": "peptide-like residue",
}

PTM_TYPES = sorted(set(PTM_RESIDUES.values()))


def normalize_pdb_id(value: str) -> str:
    """
    Normalize a filename or PDB code to the first 4 characters, uppercase.
    Examples:
        1abc_mhc_complex.pdb -> 1ABC
        1abc                 -> 1ABC
    """
    value = str(value).strip()
    value = Path(value).name
    return value[:4].upper()


def read_unique_ptm_residues_from_chain_a(pdb_file: Path) -> list[dict]:
    """
    Read unique PTM residues from chain A.

    Each residue is represented once, even if it has many ATOM/HETATM lines.
    """
    pdb_id = normalize_pdb_id(pdb_file.name)
    seen_residues = set()
    residues = []

    with pdb_file.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue

            residue_name = line[17:20].strip().upper()
            chain_id = line[21].strip()

            if chain_id != "A":
                continue

            if residue_name not in PTM_RESIDUES:
                continue

            try:
                residue_number = int(line[22:26].strip())
            except ValueError:
                continue

            insertion_code = line[26].strip()
            hetflag = "HETATM" if line.startswith("HETATM") else "ATOM"
            residue_key = (chain_id, residue_number, insertion_code, residue_name, hetflag)

            if residue_key in seen_residues:
                continue

            seen_residues.add(residue_key)

            residues.append({
                "pdb_id": pdb_id,
                "chain_id": chain_id,
                "residue_number": residue_number,
                "insertion_code": insertion_code,
                "residue_name": residue_name,
                "hetflag": hetflag,
                "ptm_type": PTM_RESIDUES[residue_name],
            })

    return residues


def empty_counts() -> dict[str, int]:
    return {ptm_type: 0 for ptm_type in PTM_TYPES}


def analyze_step7_chain_a_folder(folder: Path):
    """
    Analyze PTM residues in chain A of step7 output structures.
    """
    all_pdb_files = sorted(folder.glob("*.pdb"))

    step7_results = {}
    step7_details = defaultdict(list)

    for pdb_file in all_pdb_files:
        pdb_id = normalize_pdb_id(pdb_file.name)
        step7_results[pdb_id] = empty_counts()

        residues = read_unique_ptm_residues_from_chain_a(pdb_file)

        for residue in residues:
            step7_results[pdb_id][residue["ptm_type"]] += 1
            step7_details[pdb_id].append(residue)

    return step7_results, step7_details


def has_any_ptm(counts: dict[str, int]) -> bool:
    return any(value > 0 for value in counts.values())


def write_ptm_counts_csv(results: dict[str, dict[str, int]], output_file: Path) -> None:
    fieldnames = ["pdb_id"] + PTM_TYPES

    with output_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for pdb_id, counts in sorted(results.items()):
            if not has_any_ptm(counts):
                continue

            row = {"pdb_id": pdb_id}

            for ptm_type in PTM_TYPES:
                row[ptm_type] = counts[ptm_type]

            writer.writerow(row)


def write_ptm_details_csv(details: dict[str, list[dict]], output_file: Path) -> None:
    fieldnames = [
        "pdb_id",
        "chain_id",
        "residue_name",
        "residue_number",
        "insertion_code",
        "hetflag",
        "ptm_type",
    ]

    rows = []

    for pdb_id, residues in details.items():
        rows.extend(residues)

    with output_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for row in sorted(
            rows,
            key=lambda x: (
                x["pdb_id"],
                x["chain_id"],
                x["residue_number"],
                x["insertion_code"],
                x["residue_name"],
            ),
        ):
            writer.writerow(row)


def summarize_step7_chain_a(
    results: dict[str, dict[str, int]],
    details: dict[str, list[dict]],
    output_file: Path,
) -> None:
    structures_per_ptm_type = defaultdict(set)
    structures_per_residue_code = defaultdict(set)

    for pdb_id, residues in details.items():
        for residue in residues:
            ptm_type = residue["ptm_type"]
            residue_code = residue["residue_name"]

            structures_per_ptm_type[ptm_type].add(pdb_id)
            structures_per_residue_code[residue_code].add(pdb_id)

    total_structures_with_ptm = sum(
        1 for counts in results.values()
        if has_any_ptm(counts)
    )

    with output_file.open("w", encoding="utf-8") as out:
        out.write("PTM summary for step7 chain A\n")
        out.write("==============================\n\n")

        out.write(f"Total step7 structures analyzed: {len(results)}\n")
        out.write(f"Total structures with at least one PTM in chain A: {total_structures_with_ptm}\n\n")

        out.write("Number of structures with each PTM type:\n")
        if structures_per_ptm_type:
            for ptm_type, pdb_ids in sorted(structures_per_ptm_type.items()):
                out.write(f"{ptm_type}\t{len(pdb_ids)}\n")
        else:
            out.write("None\n")

        out.write("\nNumber of structures with each PTM residue code:\n")
        if structures_per_residue_code:
            for residue_code, pdb_ids in sorted(structures_per_residue_code.items()):
                ptm_type = PTM_RESIDUES[residue_code]
                out.write(f"{residue_code}\t{ptm_type}\t{len(pdb_ids)}\n")
        else:
            out.write("None\n")

        out.write("\nFiles containing PTMs in chain A:\n")
        found_any = False

        for pdb_id, residues in sorted(details.items()):
            if not residues:
                continue

            found_any = True
            by_type = defaultdict(list)

            for residue in residues:
                by_type[residue["ptm_type"]].append(residue)

            out.write(f"\n{pdb_id}\n")

            for ptm_type, ptm_residues in sorted(by_type.items()):
                labels = [
                    f'{r["residue_name"]}:{r["chain_id"]}{r["residue_number"]}{r["insertion_code"]}'
                    for r in sorted(
                        ptm_residues,
                        key=lambda x: (
                            x["residue_number"],
                            x["insertion_code"],
                            x["residue_name"],
                        ),
                    )
                ]
                out.write(f"  - {ptm_type}: {len(ptm_residues)} residues ({', '.join(labels)})\n")

        if not found_any:
            out.write("None\n")


def main():
    if len(sys.argv) not in [2, 3]:
        print("Usage: python find_ptms.py /path/to/step7/pdb/1_filtered_structures [output_prefix]")
        sys.exit(1)

    folder = Path(sys.argv[1])

    if len(sys.argv) >= 3:
        output_prefix = sys.argv[2]
    else:
        output_prefix = "ptm_step7_chain_A"

    if not folder.is_dir():
        print(f"Error: {folder} is not a valid folder")
        sys.exit(1)

    counts_csv = Path(f"{output_prefix}.csv")
    details_csv = Path(f"{output_prefix}_details.csv")
    summary_txt = Path(f"{output_prefix}_summary.txt")

    step7_results, step7_details = analyze_step7_chain_a_folder(folder)

    write_ptm_counts_csv(step7_results, counts_csv)
    write_ptm_details_csv(step7_details, details_csv)
    summarize_step7_chain_a(step7_results, step7_details, summary_txt)

    print(f"Step7 chain-A counts CSV written to: {counts_csv}")
    print(f"Step7 chain-A details CSV written to: {details_csv}")
    print(f"Summary written to: {summary_txt}")


if __name__ == "__main__":
    main()
