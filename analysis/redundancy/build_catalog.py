#!/usr/bin/env python3
"""Build the final-component sequence/signature catalog used for redundancy."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from redundancy_core import (
    AA3_TO_1, ChemComp, extract_pdb_residues, fetch_rcsb_metadata,
    is_peptide_residue, load_cached_metadata, log, normalize_pdb_id,
    peptide_token, read_chem_comps, read_csv, stable_id, tokens_text, write_csv,
)


SCRIPT_DIR = Path(__file__).resolve().parent
VERSION_DIR = SCRIPT_DIR.parents[1]
PIPELINE_DIR = VERSION_DIR / "pipeline"
DEFAULT_COMPLEX_DIR = PIPELINE_DIR / "step9" / "pdb" / "5_mhc_complex"
DEFAULT_BINDERS = PIPELINE_DIR / "step9" / "pdb" / "modified_pdbs" / "step9_binders.csv"
DEFAULT_LIGANDS = PIPELINE_DIR / "step9" / "pdb" / "modified_pdbs" / "step9_ligands.csv"
DEFAULT_TRIMS = PIPELINE_DIR / "step5" / "pdb" / "pdb_assemblies_remapped.csv"
DEFAULT_CIF_DIR = PIPELINE_DIR / "step2" / "pdb" / "1_assemblies"
DEFAULT_RESNAME_MAP = PIPELINE_DIR / "step6" / "pdb" / "summaries" / "resname_map.csv"
DEFAULT_REMODELED_DIR = PIPELINE_DIR / "step8" / "pdb" / "1_mhc_remodeled"
DEFAULT_MODIFIED_DIR = PIPELINE_DIR / "step9" / "pdb" / "modified_pdbs"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"
DEFAULT_OVERRIDES = SCRIPT_DIR / "manual_overrides.csv"
MHC_CHAIN = "A"


def portable_input_path(path: Path) -> str:
    """Record a reproducible input label without publishing a user's home path."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(VERSION_DIR).as_posix()
    except ValueError:
        return (Path(resolved.parent.name) / resolved.name).as_posix()


def load_chain_roles(path: Path) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = defaultdict(list)
    for row in read_csv(path):
        pdb_id = normalize_pdb_id(row.get("pdb", row.get("pdb_id", "")))
        chain = str(row.get("chain_id", "")).strip()
        if pdb_id and chain and chain not in result[pdb_id]:
            result[pdb_id].append(chain)
    return dict(result)


def load_role_counts(path: Path, component: str) -> Dict[Tuple[str, str, str], int]:
    result = {}
    for row in read_csv(path):
        pdb_id = normalize_pdb_id(row.get("pdb", row.get("pdb_id", "")))
        chain = str(row.get("chain_id", "")).strip()
        try:
            count = int(float(row.get("chain_residue_count", "")))
        except (TypeError, ValueError):
            continue
        result[(pdb_id, component, chain)] = count
    return result


def load_trim_ranges(path: Path) -> Dict[str, List[Tuple[int, int]]]:
    result: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for row in read_csv(path):
        pdb_id = normalize_pdb_id(row["pdb_id"])
        pair = (int(float(row["tstart"])), int(float(row["tend"])))
        if pair not in result[pdb_id]:
            result[pdb_id].append(pair)
    return dict(result)


def load_resname_aliases(path: Path) -> Tuple[Dict[Tuple[str, str], str], Dict[Tuple[str, str], str]]:
    """Restore 5-character CCD IDs shortened when structures were written as PDB."""
    if not path.is_file():
        return {}, {}
    candidates: Dict[Tuple[str, str], set] = defaultdict(set)
    position_candidates: Dict[Tuple[str, str], set] = defaultdict(set)
    for row in read_csv(path):
        pdb_id = normalize_pdb_id(row.get("pdb", row.get("pdb_id", "")))
        original = str(row.get("original_resname", row.get("old_resname", ""))).strip().upper()
        # PDB supports only three-character residue names. In altloc records,
        # the step6 repair table can retain the A/B altloc marker in front of a
        # five-character CCD ID (e.g. AA1B7N and BA1B7N -> A1B7N).
        if len(original) == 6 and original[0].isalpha():
            original = original[1:]
        shortened = str(row.get("short_resname", row.get("new_resname", ""))).strip().upper()
        if not shortened:
            # Current step6 table uses normalized_resname.
            shortened = str(row.get("normalized_resname", "")).strip().upper()
        if pdb_id and original and shortened:
            key = (pdb_id, shortened)
            candidates[key].add(original)
            resseq = str(row.get("resseq", "")).strip()
            if resseq:
                position_candidates[(pdb_id, resseq)].add(original)
    ambiguous = {key: values for key, values in candidates.items() if len(values) != 1}
    if ambiguous:
        example = next(iter(ambiguous.items()))
        raise ValueError(f"ambiguous residue alias {example[0]}: {sorted(example[1])}")
    by_name = {key: next(iter(values)) for key, values in candidates.items()}
    by_position = {
        key: next(iter(values)) for key, values in position_candidates.items()
        if len(values) == 1
    }
    return by_name, by_position


def load_overrides(path: Path) -> Dict[Tuple[str, str], dict]:
    if not path.is_file():
        return {}
    result = {}
    for row in read_csv(path):
        if not any(str(value).strip() for value in row.values()):
            continue
        pdb_id = normalize_pdb_id(row.get("pdb_id", ""))
        resname = str(row.get("residue_id", "")).strip().upper()
        classification = str(row.get("classification", "")).strip().lower()
        if classification not in {"peptide", "nonpeptide", "ignore"}:
            raise ValueError(f"invalid override classification for {pdb_id}/{resname}")
        if not pdb_id or not resname:
            raise ValueError("override rows require pdb_id and residue_id")
        ligand_descriptor = str(row.get("ligand_descriptor", "")).strip()
        if ligand_descriptor and not ligand_descriptor.startswith(("N:", "P:")):
            raise ValueError(
                f"invalid override ligand descriptor for {pdb_id}/{resname}: "
                f"{ligand_descriptor}"
            )
        result[(pdb_id, resname)] = row
    return result


def source_label(pdb_id: str, remodeled_dir: Path, modified_dir: Path) -> str:
    prefix = pdb_id.lower()
    if any(modified_dir.glob(f"{prefix}*.pdb")):
        return "modified"
    if any(remodeled_dir.glob(f"{prefix}*.pdb")):
        return "remodeled"
    return "step7"


def find_cif(pdb_id: str, cif_dir: Path) -> Optional[Path]:
    preferred = cif_dir / f"{pdb_id.lower()}-assembly1.cif"
    if preferred.is_file():
        return preferred
    matches = sorted(cif_dir.glob(f"{pdb_id.lower()}-assembly*.cif"))
    return matches[0] if matches else None


def classify_residue(
    pdb_id: str, resname: str, chem: Mapping[str, ChemComp], overrides: Mapping,
) -> Tuple[str, str, str]:
    """Return classification, matching token and provenance."""
    override = overrides.get((pdb_id, resname))
    if override:
        kind = str(override["classification"]).strip().lower()
        token = str(override.get("canonical_token", "")).strip()
        if kind == "peptide" and not token:
            token = peptide_token(resname, chem.get(resname))
        if kind == "nonpeptide":
            token = resname
        return kind, token, "manual_override"
    item = chem.get(resname)
    if is_peptide_residue(resname, item):
        return "peptide", peptide_token(resname, item), "ccd_or_standard"
    if item:
        return "nonpeptide", resname, "ccd"
    return "unknown", "", "unresolved"


def redundancy_peptide_token(resname: str, canonical_token: str) -> str:
    """Decorate modified amino acids while preserving their canonical parent."""
    if resname in AA3_TO_1:
        return canonical_token
    if canonical_token.startswith("["):
        return canonical_token
    return f"[{resname}>{canonical_token}]"


def add_issue(rows: List[dict], pdb_id: str, severity: str, code: str, component: str,
              chain_id: str, details: str, treatment: str) -> None:
    rows.append({
        "pdb_id": pdb_id, "severity": severity, "code": code,
        "component": component, "chain_id": chain_id,
        "details": details, "treatment": treatment,
    })


def effective_resname(
    pdb_id: str, residue, chem: Mapping[str, ChemComp],
    aliases_by_name: Mapping, aliases_by_position: Mapping,
) -> str:
    mapped = aliases_by_name.get((pdb_id, residue.resname))
    if mapped:
        return mapped
    # A manually rewritten final PDB may use generic LIG rather than the
    # shortened CCD ID. Position is safe only when the observed name itself
    # has no definition in the entry's _chem_comp table.
    if residue.resname not in chem:
        mapped = aliases_by_position.get((pdb_id, residue.resseq))
        if mapped:
            return mapped
    return residue.resname


def reconcile_final_roles(
    pdb_id: str, observed: Mapping, binder_chains: Sequence[str], ligand_chains: Sequence[str],
    role_counts: Mapping, chem: Mapping[str, ChemComp], overrides: Mapping,
    aliases_by_name: Mapping, aliases_by_position: Mapping, issue_rows: List[dict],
) -> List[Tuple[str, str]]:
    """Make final coordinates authoritative while retaining auditable CSV intent."""
    roles = [("binder", chain) for chain in binder_chains]
    roles += [("ligand", chain) for chain in ligand_chains]
    present = [(component, chain) for component, chain in roles if chain in observed]
    missing = [(component, chain) for component, chain in roles if chain not in observed]
    extras = set(observed) - {MHC_CHAIN} - {chain for _, chain in present}

    for component, old_chain in missing:
        expected_count = role_counts.get((pdb_id, component, old_chain))
        candidates = sorted(
            chain for chain in extras
            if expected_count is not None and len(observed[chain]) == expected_count
        )
        if len(candidates) == 1:
            new_chain = candidates[0]
            extras.remove(new_chain)
            present.append((component, new_chain))
            add_issue(issue_rows, pdb_id, "INFO", "classified_chain_reassigned", component,
                      new_chain, f"CSV chain={old_chain}; final chain={new_chain}; residues={expected_count}",
                      "final 5_mhc_complex chain used")
        else:
            add_issue(issue_rows, pdb_id, "INFO", "classified_chain_absent_in_final", component,
                      old_chain, f"expected_residues={expected_count}",
                      "CSV classification ignored because 5_mhc_complex prevails")

    for chain in sorted(extras):
        kinds = set()
        unknown = set()
        for residue in observed[chain]:
            resname = effective_resname(
                pdb_id, residue, chem, aliases_by_name, aliases_by_position
            )
            kind, _token, _provenance = classify_residue(pdb_id, resname, chem, overrides)
            if kind == "unknown":
                unknown.add(resname)
            elif kind != "ignore":
                kinds.add(kind)
        if kinds == {"nonpeptide"} and not unknown:
            present.append(("ligand", chain))
            add_issue(issue_rows, pdb_id, "INFO", "final_nonpeptide_chain_inferred_ligand",
                      "ligand", chain,
                      ";".join(sorted({res.resname for res in observed[chain]})),
                      "retained because final 5_mhc_complex prevails")
        else:
            add_issue(issue_rows, pdb_id, "ERROR", "unclassified_final_chain", "complex", chain,
                      f"residue_kinds={sorted(kinds)} unknown={sorted(unknown)}",
                      "manual review required")
    return sorted(present, key=lambda item: (item[0], item[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--complex-dir", type=Path, default=DEFAULT_COMPLEX_DIR)
    parser.add_argument("--binders", type=Path, default=DEFAULT_BINDERS)
    parser.add_argument("--ligands", type=Path, default=DEFAULT_LIGANDS)
    parser.add_argument("--trim-ranges", type=Path, default=DEFAULT_TRIMS)
    parser.add_argument("--cif-dir", type=Path, default=DEFAULT_CIF_DIR)
    parser.add_argument("--resname-map", type=Path, default=DEFAULT_RESNAME_MAP)
    parser.add_argument("--remodeled-dir", type=Path, default=DEFAULT_REMODELED_DIR)
    parser.add_argument("--modified-dir", type=Path, default=DEFAULT_MODIFIED_DIR)
    parser.add_argument(
        "--final-source-label", default="",
        help="Optional provenance label for every exported final complex",
    )
    parser.add_argument("--overrides", type=Path, default=DEFAULT_OVERRIDES)
    parser.add_argument(
        "--exclude-pdb", action="append", default=[], metavar="PDB_ID",
        help="Exclude a PDB ID from the analysis; may be repeated",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--offline", action="store_true", help="Use an existing RCSB cache only")
    parser.add_argument("--refresh-resolution", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    required = [args.complex_dir, args.binders, args.ligands, args.trim_ranges, args.cif_dir]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"missing inputs: {missing}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_pdb_paths = sorted(args.complex_dir.glob("*.pdb"))
    if not all_pdb_paths:
        raise SystemExit(f"no PDB files in {args.complex_dir}")
    all_pdb_ids = [normalize_pdb_id(path.name.split("_")[0]) for path in all_pdb_paths]
    if len(all_pdb_ids) != len(set(all_pdb_ids)):
        raise SystemExit("duplicate PDB IDs in final complex directory")
    excluded_pdb_ids = {normalize_pdb_id(pdb_id) for pdb_id in args.exclude_pdb}
    missing_exclusions = excluded_pdb_ids - set(all_pdb_ids)
    if missing_exclusions:
        raise SystemExit(f"excluded PDB IDs not found in final complex directory: {sorted(missing_exclusions)}")
    pdb_paths = [
        path for path, pdb_id in zip(all_pdb_paths, all_pdb_ids)
        if pdb_id not in excluded_pdb_ids
    ]
    pdb_ids = [
        pdb_id for pdb_id in all_pdb_ids
        if pdb_id not in excluded_pdb_ids
    ]
    if not pdb_paths:
        raise SystemExit("all final complexes were excluded")

    binders = load_chain_roles(args.binders)
    ligands = load_chain_roles(args.ligands)
    role_counts = load_role_counts(args.binders, "binder")
    role_counts.update(load_role_counts(args.ligands, "ligand"))
    trims = load_trim_ranges(args.trim_ranges)
    aliases_by_name, aliases_by_position = load_resname_aliases(args.resname_map)
    overrides = load_overrides(args.overrides)
    resolution_cache = args.output_dir / "rcsb_resolution_cache.csv"
    if args.offline:
        cached_metadata = load_cached_metadata(pdb_ids, resolution_cache)
    else:
        cached_metadata = fetch_rcsb_metadata(pdb_ids, resolution_cache, args.refresh_resolution)
    # Keep extra cache rows, including explicitly excluded structures, so a later
    # offline run can re-include them without requiring a new RCSB request.
    metadata = {pdb_id: cached_metadata[pdb_id] for pdb_id in pdb_ids}

    component_rows: List[dict] = []
    catalog_rows: List[dict] = []
    issue_rows: List[dict] = []
    used_overrides = set()

    for index, (pdb_id, pdb_path) in enumerate(zip(pdb_ids, pdb_paths), start=1):
        errors_before = sum(row["severity"] == "ERROR" for row in issue_rows if row["pdb_id"] == pdb_id)
        cif_path = find_cif(pdb_id, args.cif_dir)
        chem: Dict[str, ChemComp] = {}
        if cif_path:
            try:
                chem = read_chem_comps(cif_path)
            except Exception as exc:
                add_issue(issue_rows, pdb_id, "ERROR", "cif_chem_comp_parse_failed", "complex", "",
                          str(exc), "excluded from grouping until corrected")
        else:
            add_issue(issue_rows, pdb_id, "ERROR", "original_cif_missing", "complex", "", "",
                      "excluded from grouping until CIF is supplied")
        try:
            observed = extract_pdb_residues(pdb_path)
        except Exception as exc:
            add_issue(issue_rows, pdb_id, "ERROR", "final_pdb_parse_failed", "complex", "",
                      str(exc), "excluded from grouping")
            observed = {}

        reconciled_roles = reconcile_final_roles(
            pdb_id, observed, binders.get(pdb_id, []), ligands.get(pdb_id, []),
            role_counts, chem, overrides, aliases_by_name, aliases_by_position,
            issue_rows,
        )
        expected_roles = [("mhc", MHC_CHAIN)] + reconciled_roles
        role_owners: Dict[str, List[str]] = defaultdict(list)
        for component, chain_id in expected_roles:
            role_owners[chain_id].append(component)
        for chain_id, owners in role_owners.items():
            if len(owners) > 1:
                add_issue(issue_rows, pdb_id, "ERROR", "overlapping_chain_roles", "complex",
                          chain_id, ";".join(owners), "excluded until classifications are corrected")
        unexpected_chains = sorted(set(observed) - set(role_owners))
        if unexpected_chains:
            add_issue(issue_rows, pdb_id, "ERROR", "unclassified_final_chain", "complex", "",
                      ";".join(unexpected_chains),
                      "excluded until every final chain is classified")
        mhc_tokens: Tuple[str, ...] = tuple()
        mhc_redundancy_tokens: Tuple[str, ...] = tuple()
        binder_sequences: List[Tuple[str, ...]] = []
        binder_redundancy_sequences: List[Tuple[str, ...]] = []
        ligand_descriptors: List[str] = []
        total_peptide_length = 0

        for component, chain_id in expected_roles:
            residues = observed.get(chain_id, [])
            if not residues:
                add_issue(issue_rows, pdb_id, "ERROR", "classified_chain_missing", component, chain_id,
                          "chain absent or empty in final step9 complex", "excluded from grouping")
                continue
            peptide_tokens: List[str] = []
            redundancy_tokens: List[str] = []
            raw_peptide: List[str] = []
            nonpeptide: List[str] = []
            unknown: List[str] = []
            ignored: List[str] = []
            aliases_used: List[str] = []
            supplemental_ligands: List[str] = []
            override_positions: Dict[str, List[str]] = defaultdict(list)
            for residue in residues:
                resolved_resname = effective_resname(
                    pdb_id, residue, chem, aliases_by_name, aliases_by_position
                )
                if resolved_resname != residue.resname:
                    aliases_used.append(f"{residue.resname}->{resolved_resname}")
                    add_issue(issue_rows, pdb_id, "INFO", "pdb_resname_restored", component,
                              chain_id, f"{residue.resname}->{resolved_resname}",
                              "used full CCD ID recorded by step6 resname_map.csv")
                kind, token, provenance = classify_residue(
                    pdb_id, resolved_resname, chem, overrides
                )
                if provenance == "manual_override":
                    used_overrides.add((pdb_id, resolved_resname))
                    override = overrides[(pdb_id, resolved_resname)]
                    override_positions[resolved_resname].append(
                        f"{residue.resseq}{residue.insertion_code}"
                    )
                    supplemental = str(override.get("ligand_descriptor", "")).strip()
                    if supplemental:
                        supplemental_ligands.append(supplemental)
                        ligand_descriptors.append(supplemental)
                if kind == "peptide":
                    peptide_tokens.append(token)
                    redundancy_tokens.append(redundancy_peptide_token(resolved_resname, token))
                    raw_peptide.append(resolved_resname)
                    item = chem.get(resolved_resname)
                    if resolved_resname not in AA3_TO_1 and item and item.parent_id:
                        add_issue(issue_rows, pdb_id, "INFO", "modified_peptide_normalized", component,
                                  chain_id, f"{resolved_resname}->{token}",
                                  "matched through CCD parent amino acid")
                elif kind == "nonpeptide":
                    nonpeptide.append(token)
                elif kind == "ignore":
                    ignored.append(residue.resname)
                else:
                    unknown.append(residue.resname)
            for resname, positions in sorted(override_positions.items()):
                override = overrides[(pdb_id, resname)]
                details = (
                    f"residue_id={resname}; occurrences={len(positions)}; "
                    f"positions={';'.join(positions)}"
                )
                add_issue(
                    issue_rows, pdb_id, "INFO", "manual_override_applied", component,
                    chain_id, details, str(override.get("reason", "")),
                )
                supplemental = str(override.get("ligand_descriptor", "")).strip()
                if supplemental:
                    add_issue(
                        issue_rows, pdb_id, "INFO",
                        "supplemental_ligand_descriptor_added", component, chain_id,
                        f"{details}; descriptor={supplemental}",
                        "modified residue retained as its parent amino acid and chemical adduct retained in ligand signature",
                    )
            if unknown:
                add_issue(issue_rows, pdb_id, "ERROR", "unknown_residue_class", component, chain_id,
                          ";".join(sorted(set(unknown))),
                          "excluded; add a documented manual override")
            if component in {"mhc", "binder"} and nonpeptide:
                add_issue(issue_rows, pdb_id, "INFO", "nonpeptide_ignored_in_protein_chain", component,
                          chain_id, ";".join(nonpeptide), "not part of peptide-sequence matching")
            if component in {"mhc", "binder"} and not peptide_tokens:
                add_issue(issue_rows, pdb_id, "ERROR", "empty_peptide_sequence", component, chain_id,
                          "", "excluded from grouping")
            sequence = tuple(peptide_tokens)
            redundancy_sequence = tuple(redundancy_tokens)
            if component == "mhc":
                mhc_tokens = sequence
                mhc_redundancy_tokens = redundancy_sequence
            elif component == "binder":
                binder_sequences.append(sequence)
                binder_redundancy_sequences.append(redundancy_sequence)
            else:
                if sequence:
                    ligand_descriptors.append("P:" + tokens_text(sequence))
                ligand_descriptors.extend("N:" + comp_id for comp_id in nonpeptide)
                if not sequence and not nonpeptide and not unknown:
                    add_issue(issue_rows, pdb_id, "ERROR", "empty_ligand_descriptor", component,
                              chain_id, ";".join(ignored), "excluded from grouping")
            total_peptide_length += len(sequence)
            component_rows.append({
                "pdb_id": pdb_id, "component": component, "chain_id": chain_id,
                "observed_residue_count": len(residues), "peptide_length": len(sequence),
                "canonical_sequence": tokens_text(sequence),
                "matching_sequence": tokens_text(redundancy_sequence),
                "raw_peptide_residue_ids": "-".join(raw_peptide),
                "nonpeptide_residue_ids": ";".join(nonpeptide),
                "unknown_residue_ids": ";".join(unknown),
                "residue_id_aliases": ";".join(aliases_used),
                "supplemental_ligand_descriptors_json": json.dumps(
                    supplemental_ligands, separators=(",", ":")
                ),
                "sequence_sha256_16": stable_id(sequence),
                "redundancy_sequence_sha256_16": stable_id(redundancy_sequence),
                "source_pdb": pdb_path.name,
            })

        trim_ranges = trims.get(pdb_id, [])
        expected_lengths = sorted({end - start + 1 for start, end in trim_ranges})
        source = args.final_source_label or source_label(pdb_id, args.remodeled_dir, args.modified_dir)
        if not trim_ranges:
            add_issue(issue_rows, pdb_id, "ERROR", "trim_metadata_missing", "mhc", MHC_CHAIN, "",
                      "excluded from grouping")
        elif len(mhc_tokens) not in expected_lengths:
            severity = "INFO" if source in {"remodeled", "exported_final"} else "WARNING"
            treatment = (
                "final MHC retained as authoritative trimmed sequence"
                if source in {"remodeled", "exported_final"}
                else "final step9 MHC retained; review before downstream removal"
            )
            add_issue(issue_rows, pdb_id, severity, "mhc_length_differs_from_trim_span", "mhc",
                      MHC_CHAIN, f"observed={len(mhc_tokens)} expected={expected_lengths}", treatment)

        errors_after = sum(row["severity"] == "ERROR" for row in issue_rows if row["pdb_id"] == pdb_id)
        status = "valid" if errors_after == errors_before else "invalid"
        binder_signature = tuple(sorted(tuple(seq) for seq in binder_sequences))
        binder_redundancy_signature = tuple(
            sorted(tuple(seq) for seq in binder_redundancy_sequences)
        )
        ligand_signature = tuple(sorted(ligand_descriptors))
        meta = metadata.get(pdb_id, {})
        catalog_rows.append({
            "pdb_id": pdb_id, "status": status, "final_source": source,
            "observed_chain_count": len(observed), "classified_chain_count": len(role_owners),
            "mhc_length": len(mhc_tokens), "mhc_sequence": tokens_text(mhc_tokens),
            "mhc_sequence_json": json.dumps(mhc_tokens, separators=(",", ":")),
            "mhc_signature_sha256_16": stable_id(mhc_tokens),
            "mhc_redundancy_sequence_json": json.dumps(
                mhc_redundancy_tokens, separators=(",", ":")
            ),
            "mhc_redundancy_signature_sha256_16": stable_id(mhc_redundancy_tokens),
            "binder_count": len(binder_signature),
            "binder_lengths": ";".join(map(lambda seq: str(len(seq)), binder_signature)),
            "binder_signature_json": json.dumps(binder_signature, separators=(",", ":")),
            "binder_signature_sha256_16": stable_id(binder_signature),
            "binder_redundancy_signature_json": json.dumps(
                binder_redundancy_signature, separators=(",", ":")
            ),
            "binder_redundancy_signature_sha256_16": stable_id(
                binder_redundancy_signature
            ),
            "ligand_chain_count": sum(component == "ligand" for component, _chain in reconciled_roles),
            "ligand_descriptor_count": len(ligand_signature),
            "ligand_signature_json": json.dumps(ligand_signature, separators=(",", ":")),
            "ligand_signature_sha256_16": stable_id(ligand_signature),
            "total_peptide_length": total_peptide_length,
            "trim_ranges": ";".join(f"{start}-{end}" for start, end in trim_ranges),
            "expected_trim_lengths": ";".join(map(str, expected_lengths)),
            "resolution_angstrom": meta.get("resolution_angstrom", ""),
            "experimental_method": meta.get("experimental_method", ""),
            "resolution_api_status": meta.get("api_status", ""),
            "source_pdb": pdb_path.name,
        })
        if index % 100 == 0 or index == len(pdb_paths):
            log(f"[CATALOG] {index}/{len(pdb_paths)}")

    for key, row in overrides.items():
        if key[0] in excluded_pdb_ids:
            continue
        if key not in used_overrides:
            add_issue(issue_rows, key[0], "WARNING", "unused_manual_override", "", "",
                      f"residue_id={key[1]} reason={row.get('reason', '')}",
                      "review or remove stale override")
    final_ids = set(pdb_ids)
    for label, role_map in (("binder", binders), ("ligand", ligands)):
        for pdb_id in sorted(set(role_map) - final_ids - excluded_pdb_ids):
            add_issue(issue_rows, pdb_id, "WARNING", "classification_without_final_complex", label, "",
                      ";".join(role_map[pdb_id]), "not included")

    write_csv(args.output_dir / "component_sequences.csv", [
        "pdb_id", "component", "chain_id", "observed_residue_count", "peptide_length",
        "canonical_sequence", "matching_sequence", "raw_peptide_residue_ids", "nonpeptide_residue_ids",
        "unknown_residue_ids", "residue_id_aliases", "supplemental_ligand_descriptors_json",
        "sequence_sha256_16", "redundancy_sequence_sha256_16", "source_pdb",
    ], component_rows)
    write_csv(args.output_dir / "complex_catalog.csv", [
        "pdb_id", "status", "final_source", "observed_chain_count", "classified_chain_count",
        "mhc_length", "mhc_sequence",
        "mhc_sequence_json", "mhc_signature_sha256_16",
        "mhc_redundancy_sequence_json", "mhc_redundancy_signature_sha256_16",
        "binder_count", "binder_lengths", "binder_signature_json", "binder_signature_sha256_16",
        "binder_redundancy_signature_json", "binder_redundancy_signature_sha256_16",
        "ligand_chain_count",
        "ligand_descriptor_count", "ligand_signature_json", "ligand_signature_sha256_16",
        "total_peptide_length", "trim_ranges", "expected_trim_lengths",
        "resolution_angstrom", "experimental_method", "resolution_api_status", "source_pdb",
    ], catalog_rows)
    write_csv(args.output_dir / "validation_issues.csv", [
        "pdb_id", "severity", "code", "component", "chain_id", "details", "treatment",
    ], sorted(issue_rows, key=lambda row: (row["pdb_id"], row["severity"], row["code"])))
    all_cif_ids = {
        path.name.split("-assembly")[0].upper()
        for path in args.cif_dir.glob("*.cif")
        if "-assembly" in path.name.lower()
    }
    summary = {
        "final_complexes": len(pdb_paths),
        "available_assembly_cifs": len(all_cif_ids),
        "assembly_cifs_ignored_without_final_complex": len(all_cif_ids - set(pdb_ids)),
        "analysis_universe": "PDB IDs present in final 5_mhc_complex directory after explicit exclusions",
        "excluded_pdb_ids": sorted(excluded_pdb_ids),
        "excluded_final_complexes": len(excluded_pdb_ids),
        "valid_complexes": sum(row["status"] == "valid" for row in catalog_rows),
        "invalid_complexes": sum(row["status"] != "valid" for row in catalog_rows),
        "binder_chains": sum(row["component"] == "binder" for row in component_rows),
        "ligand_chains": sum(row["component"] == "ligand" for row in component_rows),
        "issues_by_severity": {
            level: sum(row["severity"] == level for row in issue_rows)
            for level in ("ERROR", "WARNING", "INFO")
        },
        "matching_peptide_basis": "canonical residue sequence; CCD parent used for modified amino acids",
        "ligand_basis": "exact multiset of peptide sequences and nonpeptide CCD residue IDs",
        "input_paths": {
            "complex_dir": portable_input_path(args.complex_dir),
            "cif_dir": portable_input_path(args.cif_dir),
            "binders": portable_input_path(args.binders),
            "ligands": portable_input_path(args.ligands),
            "trim_ranges": portable_input_path(args.trim_ranges),
            "resname_map": portable_input_path(args.resname_map),
        },
    }
    (args.output_dir / "catalog_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    log(f"[DONE] catalog valid={summary['valid_complexes']} invalid={summary['invalid_complexes']}")


if __name__ == "__main__":
    main()
