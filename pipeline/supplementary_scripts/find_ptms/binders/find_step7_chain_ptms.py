#!/usr/bin/env python3
"""
Find dangerous PTM-like residues in step7 binder and/or ligand chains.

Goal:
  Count amino-acid-like PTM / modified peptide residues in chains already
  classified by step7.

Output files are written to the current working directory by default, not into
the step7 summaries folder.

Usage:
  python find_step7_chain_ptms.py PDB_FOLDER [output_prefix] [step7_binders.csv|none] [step7_ligands.csv|none]

Examples:
  python find_step7_chain_ptms.py ../step7/pdb/1_filtered_structures step7_ptms ../step7/pdb/summaries/step7_binders.csv ../step7/pdb/summaries/step7_ligands.csv
  python find_step7_chain_ptms.py ../step7/pdb/1_filtered_structures step7_ligand_ptms none ../step7/pdb/summaries/step7_ligands.csv

If neither step7_binders.csv nor step7_ligands.csv is supplied, the script
falls back to inferring binder-like chains as non-A chains with >=30 residues.
"""

from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict, Counter
import csv
import sys

MHC_CHAIN_ID = "A"
BINDER_MIN_RESIDUES_INCLUSIVE = 30

DANGEROUS_PTM_RESIDUES = {
    "3AZ": "peptide-like residue",
    "3X9": "modified/unknown peptide residue",
    "4OG": "modified/unknown peptide residue",
    "ABA": "modified/unknown peptide residue",
    "ALY": "acetylation",
    "BAL": "beta-amino acid / peptide-like residue",
    "CDE": "modified/unknown peptide residue",
    "CGU": "carboxylation",
    "CIR": "citrullination",
    "CME": "cysteine modification",
    "CSD": "cysteine oxidation",
    "CSO": "cysteine modification",
    "CSU": "cysteine oxidation",
    "CSX": "cysteine oxidation",
    "F2F": "modified/unknown peptide residue",
    "GIC": "peptide-like residue",
    "HYL": "hydroxylation",
    "HYP": "hydroxylation",
    "KCX": "carboxylation",
    "KYN": "kynurenine substitution",
    "LLP": "lysine cofactor-linked modification",
    "LPH": "modified/unknown peptide residue",
    "M3L": "methylation",
    "MLY": "methylation",
    "MLZ": "methylation",
    "MSE": "selenomethionine substitution",
    "NVA": "modified/unknown peptide residue",
    "OCS": "cysteine oxidation",
    "OSE": "modified/unknown peptide residue",
    "PCA": "pyroglutamate formation",
    "PFF": "modified/unknown peptide residue",
    "PRQ": "modified/unknown peptide residue",
    "PRV": "modified/unknown peptide residue",
    "PTR": "phosphorylation",
    "QM8": "modified/unknown peptide residue",
    "QMB": "modified/unknown peptide residue",
    "SEP": "phosphorylation",
    "SNC": "S-nitrosylation",
    "TIG": "amino-acid analog",
    "TPO": "phosphorylation",
    "TYS": "sulfation",
    "XFW": "peptide-like residue",
}


def normalize_pdb_id(value: str) -> str:
    value = str(value).strip()
    value = Path(value).name
    return value.split("_")[0][:4].upper()


def parse_optional_csv(value: str) -> Optional[Path]:
    text = str(value).strip()
    if not text or text.lower() in {"none", "na", "null", "-"}:
        return None
    return Path(text)


def parse_resseq(line: str) -> Optional[int]:
    try:
        return int(line[22:26].strip())
    except ValueError:
        return None


def read_unique_residues_by_chain(pdb_file: Path) -> Dict[str, List[dict]]:
    pdb_id = normalize_pdb_id(pdb_file.name)
    seen = set()
    chains = defaultdict(list)

    with pdb_file.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue

            resname = line[17:20].strip().upper()
            chain_id = line[21].strip()
            resseq = parse_resseq(line)

            if not chain_id or resseq is None:
                continue

            icode = line[26].strip()
            hetflag = "HETATM" if line.startswith("HETATM") else "ATOM"
            key = (chain_id, resseq, icode, resname, hetflag)

            if key in seen:
                continue

            seen.add(key)
            chains[chain_id].append({
                "pdb_id": pdb_id,
                "chain_id": chain_id,
                "resname": resname,
                "resseq": resseq,
                "icode": icode,
                "hetflag": hetflag,
            })

    return chains


def load_step7_chains(csv_path: Optional[Path], chain_type: str) -> Set[Tuple[str, str, str]]:
    if csv_path is None:
        return set()

    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing step7 {chain_type} CSV: {csv_path}")

    chains = set()

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)

        required = {"pdb", "chain_id"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{csv_path} missing columns {sorted(missing)}; "
                f"available columns: {reader.fieldnames}"
            )

        for row in reader:
            pdb_id = normalize_pdb_id(row["pdb"])
            chain_id = str(row["chain_id"]).strip()
            if chain_id:
                chains.add((pdb_id, chain_id, chain_type))

    return chains


def load_blacklist_resnames(folder: Path) -> Set[str]:
    candidates = [
        folder / "blacklist.csv",
        folder.parent / "blacklist.csv",
        folder.parent.parent / "blacklist.csv",
    ]

    for path in candidates:
        if path.is_file():
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)

                if not reader.fieldnames:
                    return set()

                column = "resname" if "resname" in reader.fieldnames else reader.fieldnames[0]
                return {
                    str(row[column]).strip().upper()
                    for row in reader
                    if str(row[column]).strip()
                }

    return set()


def infer_binder_chains(
    pdb_file: Path,
    residues_by_chain: Dict[str, List[dict]],
    blacklist_resnames: Set[str],
) -> Set[Tuple[str, str, str]]:
    pdb_id = normalize_pdb_id(pdb_file.name)
    binders = set()

    for chain_id, residues in residues_by_chain.items():
        if chain_id == MHC_CHAIN_ID:
            continue

        count = sum(
            1
            for residue in residues
            if residue["resname"].upper() not in blacklist_resnames
        )

        if count >= BINDER_MIN_RESIDUES_INCLUSIVE:
            binders.add((pdb_id, chain_id, "binder_inferred"))

    return binders


def analyze_folder(
    pdb_folder: Path,
    binders_csv: Optional[Path],
    ligands_csv: Optional[Path],
):
    requested_chains = set()
    requested_chains |= load_step7_chains(binders_csv, "binder")
    requested_chains |= load_step7_chains(ligands_csv, "ligand")

    use_fallback_inference = not requested_chains
    blacklist_resnames = load_blacklist_resnames(pdb_folder) if use_fallback_inference else set()

    details = []
    structures_with_ptm = defaultdict(set)
    chains_with_ptm = defaultdict(set)
    pdb_files = sorted(pdb_folder.glob("*.pdb"))

    for pdb_file in pdb_files:
        pdb_id = normalize_pdb_id(pdb_file.name)
        residues_by_chain = read_unique_residues_by_chain(pdb_file)

        if use_fallback_inference:
            chains_to_check = infer_binder_chains(
                pdb_file=pdb_file,
                residues_by_chain=residues_by_chain,
                blacklist_resnames=blacklist_resnames,
            )
        else:
            chains_to_check = {
                (csv_pdb, chain_id, chain_type)
                for (csv_pdb, chain_id, chain_type) in requested_chains
                if csv_pdb == pdb_id
            }

        for _, chain_id, chain_type in sorted(chains_to_check):
            residues = residues_by_chain.get(chain_id, [])

            for residue in residues:
                resname = residue["resname"].upper()

                if resname not in DANGEROUS_PTM_RESIDUES:
                    continue

                ptm_type = DANGEROUS_PTM_RESIDUES[resname]

                row = {
                    "pdb_id": pdb_id,
                    "chain_id": chain_id,
                    "chain_type": chain_type,
                    "resname": resname,
                    "resseq": residue["resseq"],
                    "icode": residue["icode"],
                    "hetflag": residue["hetflag"],
                    "ptm_type": ptm_type,
                    "removal_risk": "may_leave_backbone_gap",
                }

                details.append(row)
                structures_with_ptm[(chain_type, resname)].add(pdb_id)
                chains_with_ptm[(chain_type, resname)].add((pdb_id, chain_id))

    return {
        "details": details,
        "structures_with_ptm": structures_with_ptm,
        "chains_with_ptm": chains_with_ptm,
        "n_pdb_files": len(pdb_files),
        "used_binders_csv": binders_csv is not None,
        "used_ligands_csv": ligands_csv is not None,
        "used_fallback_inference": use_fallback_inference,
    }


def write_details_csv(details: List[dict], output_file: Path) -> None:
    fieldnames = [
        "pdb_id",
        "chain_id",
        "chain_type",
        "resname",
        "resseq",
        "icode",
        "hetflag",
        "ptm_type",
        "removal_risk",
    ]

    with output_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            sorted(
                details,
                key=lambda r: (
                    r["chain_type"],
                    r["pdb_id"],
                    r["chain_id"],
                    r["resseq"],
                    r["resname"],
                ),
            )
        )


def write_counts_csv(details: List[dict], output_file: Path) -> None:
    counts = Counter(
        (
            row["chain_type"],
            row["pdb_id"],
            row["chain_id"],
            row["resname"],
            row["ptm_type"],
        )
        for row in details
    )

    fieldnames = [
        "chain_type",
        "pdb_id",
        "chain_id",
        "resname",
        "ptm_type",
        "count",
    ]

    with output_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for (chain_type, pdb_id, chain_id, resname, ptm_type), count in sorted(counts.items()):
            writer.writerow({
                "chain_type": chain_type,
                "pdb_id": pdb_id,
                "chain_id": chain_id,
                "resname": resname,
                "ptm_type": ptm_type,
                "count": count,
            })


def write_summary_txt(results: dict, output_file: Path) -> None:
    details = results["details"]
    structures_with_ptm = results["structures_with_ptm"]
    chains_with_ptm = results["chains_with_ptm"]

    total_chains_with_any = {
        (row["chain_type"], row["pdb_id"], row["chain_id"])
        for row in details
    }
    total_structures_with_any = {
        (row["chain_type"], row["pdb_id"])
        for row in details
    }

    with output_file.open("w", encoding="utf-8") as out:
        out.write("Dangerous PTM-like residues in step7 chains\n")
        out.write("============================================\n\n")
        out.write(f"PDB files analyzed: {results['n_pdb_files']}\n")
        out.write(f"Used step7_binders.csv: {results['used_binders_csv']}\n")
        out.write(f"Used step7_ligands.csv: {results['used_ligands_csv']}\n")
        out.write(f"Used fallback inference: {results['used_fallback_inference']}\n")
        out.write(f"Chains with at least one dangerous PTM: {len(total_chains_with_any)}\n")
        out.write(f"Structures with at least one dangerous PTM: {len(total_structures_with_any)}\n")
        out.write(f"Total dangerous PTM residues: {len(details)}\n\n")

        out.write("By chain type and residue code:\n")
        if structures_with_ptm:
            for (chain_type, resname), pdb_ids in sorted(structures_with_ptm.items()):
                out.write(
                    f"{chain_type}\t"
                    f"{resname}\t"
                    f"type={DANGEROUS_PTM_RESIDUES[resname]}\t"
                    f"structures={len(pdb_ids)}\t"
                    f"chains={len(chains_with_ptm[(chain_type, resname)])}\n"
                )
        else:
            out.write("None\n")

        out.write("\nDetails by chain type and structure:\n")
        if not details:
            out.write("None\n")
            return

        grouped = defaultdict(list)
        for row in details:
            grouped[(row["chain_type"], row["pdb_id"], row["chain_id"])].append(row)

        for (chain_type, pdb_id, chain_id), rows in sorted(grouped.items()):
            labels = [
                f"{row['resname']}:{row['resseq']}{row['icode']}"
                for row in sorted(rows, key=lambda r: (r["resseq"], r["resname"]))
            ]
            out.write(f"{chain_type}\t{pdb_id} chain {chain_id}: {', '.join(labels)}\n")


def main():
    if len(sys.argv) not in [2, 3, 4, 5]:
        print("Usage:")
        print("  python find_step7_chain_ptms.py PDB_FOLDER [output_prefix] [step7_binders.csv|none] [step7_ligands.csv|none]")
        print()
        print("Examples:")
        print("  python find_step7_chain_ptms.py ../step7/pdb/1_filtered_structures step7_ptms ../step7/pdb/summaries/step7_binders.csv ../step7/pdb/summaries/step7_ligands.csv")
        print("  python find_step7_chain_ptms.py ../step7/pdb/1_filtered_structures step7_ligand_ptms none ../step7/pdb/summaries/step7_ligands.csv")
        sys.exit(1)

    pdb_folder = Path(sys.argv[1])

    if not pdb_folder.is_dir():
        print(f"Error: {pdb_folder} is not a valid folder")
        sys.exit(1)

    output_prefix = sys.argv[2] if len(sys.argv) >= 3 else "step7_chain_ptms"
    binders_csv = parse_optional_csv(sys.argv[3]) if len(sys.argv) >= 4 else None
    ligands_csv = parse_optional_csv(sys.argv[4]) if len(sys.argv) >= 5 else None

    results = analyze_folder(
        pdb_folder=pdb_folder,
        binders_csv=binders_csv,
        ligands_csv=ligands_csv,
    )

    details_csv = Path(f"{output_prefix}_details.csv")
    counts_csv = Path(f"{output_prefix}_counts.csv")
    summary_txt = Path(f"{output_prefix}_summary.txt")

    write_details_csv(results["details"], details_csv)
    write_counts_csv(results["details"], counts_csv)
    write_summary_txt(results, summary_txt)

    print(f"Details CSV written to: {details_csv}")
    print(f"Counts CSV written to: {counts_csv}")
    print(f"Summary written to: {summary_txt}")


if __name__ == "__main__":
    main()
