#!/usr/bin/env python3
"""Map final MHC+binder+ligand redundancy and choose group representatives."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from redundancy_core import (
    contiguous_relation, find_subsequence, log, match_sequence_multisets,
    read_csv, stable_id, write_csv,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"
DEFAULT_CLASS_REPRESENTATIVES = SCRIPT_DIR / "top1_representatives_tm_score.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_OUTPUT_DIR / "complex_catalog.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--class-representatives", type=Path, default=DEFAULT_CLASS_REPRESENTATIVES,
        help="CSV containing one class representative per row in a 'pdb' column",
    )
    return parser.parse_args()


def load_catalog(path: Path) -> Dict[str, dict]:
    records = {}
    for row in read_csv(path):
        if row["status"] != "valid":
            continue
        pdb_id = row["pdb_id"]
        row["mhc_tokens"] = tuple(json.loads(
            row.get("mhc_redundancy_sequence_json") or row["mhc_sequence_json"]
        ))
        row["binder_sequences"] = tuple(
            tuple(sequence) for sequence in json.loads(
                row.get("binder_redundancy_signature_json") or row["binder_signature_json"]
            )
        )
        row["ligand_signature"] = tuple(json.loads(row["ligand_signature_json"]))
        records[pdb_id] = row
    return records


def load_class_representatives(path: Path) -> Dict[str, dict]:
    """Load and normalize the curated class-representative table."""
    if not path.is_file():
        raise SystemExit(f"class representative file not found: {path}")
    rows = read_csv(path)
    if not rows or "pdb" not in rows[0]:
        raise SystemExit(f"class representative CSV must contain a 'pdb' column: {path}")
    representatives: Dict[str, dict] = {}
    for line_number, source_row in enumerate(rows, start=2):
        pdb_id = source_row.get("pdb", "").strip().upper()
        if not pdb_id:
            raise SystemExit(f"blank class representative PDB at {path}:{line_number}")
        if pdb_id in representatives:
            raise SystemExit(f"duplicate class representative PDB {pdb_id} in {path}")
        row = dict(source_row)
        row["pdb"] = pdb_id
        representatives[pdb_id] = row
    return representatives


def numeric_resolution(row: dict) -> Tuple[int, float]:
    try:
        return (0, float(row["resolution_angstrom"]))
    except (TypeError, ValueError):
        return (1, float("inf"))


def total_binder_length(row: dict) -> int:
    """Return the combined peptide length of all binder chains."""
    return sum(len(sequence) for sequence in row["binder_sequences"])


def representative_key(row: dict) -> tuple:
    """Longer MHC, lower resolution, longer binders, then PDB ID."""
    return (
        -int(row["mhc_length"]),
        *numeric_resolution(row),
        -total_binder_length(row),
        row["pdb_id"],
    )


def prioritized_representative_key(row: dict, class_representatives: set[str]) -> tuple:
    """Place curated class representatives before the ordinary ranking criteria."""
    return (0 if row["pdb_id"] in class_representatives else 1, *representative_key(row))


def selection_reason(
    representative: str,
    members: Sequence[str],
    records: Dict[str, dict],
    class_representatives: set[str] | None = None,
) -> str:
    if len(members) == 1:
        return "singleton"
    if representative in (class_representatives or set()):
        standard_representative = min(members, key=lambda pdb_id: representative_key(records[pdb_id]))
        if representative == standard_representative:
            return "class_representative_priority_confirmed_standard_selection"
        return "class_representative_priority_changed_selection"
    longest = max(int(records[pdb_id]["mhc_length"]) for pdb_id in members)
    candidates = [pdb_id for pdb_id in members if int(records[pdb_id]["mhc_length"]) == longest]
    if len(candidates) == 1:
        return "greatest_mhc_length"
    available = [pdb_id for pdb_id in candidates if numeric_resolution(records[pdb_id])[0] == 0]
    if available:
        best = min(numeric_resolution(records[pdb_id])[1] for pdb_id in available)
        candidates = [pdb_id for pdb_id in available if numeric_resolution(records[pdb_id])[1] == best]
        if len(candidates) == 1:
            return "best_resolution"
        resolution_state = "same_resolution"
    else:
        resolution_state = "resolution_unavailable"
    longest_binders = max(total_binder_length(records[pdb_id]) for pdb_id in candidates)
    candidates = [
        pdb_id for pdb_id in candidates
        if total_binder_length(records[pdb_id]) == longest_binders
    ]
    if len(candidates) == 1:
        return "greatest_total_binder_length"
    return f"pdb_id_tiebreak_same_mhc_{resolution_state}_and_binder_length"


def build_groups(
    records: Dict[str, dict],
    adjacency: Dict[str, set[str]],
    class_representatives: set[str],
) -> List[dict]:
    """Build a stable representative-anchored partition with protected class representatives."""
    remaining = set(records)
    groups = []
    while remaining:
        representative = min(
            remaining,
            key=lambda pdb_id: prioritized_representative_key(
                records[pdb_id], class_representatives
            ),
        )
        # Two curated class representatives can be redundant to each other, but
        # neither is allowed to remove the other. Each therefore anchors its own
        # group; any cross-group edge remains visible in redundant_pairs.csv.
        protected = (adjacency[representative] & remaining) & class_representatives
        direct_members = {representative} | (
            (adjacency[representative] & remaining) - protected
        )
        ordered_members = [representative] + sorted(
            direct_members - {representative},
            key=lambda pdb_id: representative_key(records[pdb_id]),
        )
        groups.append({"representative": representative, "members": ordered_members})
        remaining -= direct_members

    # Group IDs remain as stable as possible when the priority list changes.
    # This reproduces the former ordering by the best ordinary-rank member.
    return sorted(
        groups,
        key=lambda group: representative_key(records[min(
            group["members"], key=lambda pdb_id: representative_key(records[pdb_id])
        )]),
    )


def relation_details(left: dict, right: dict):
    if left["ligand_signature"] != right["ligand_signature"]:
        return None
    mhc_relation = contiguous_relation(left["mhc_tokens"], right["mhc_tokens"])
    if not mhc_relation:
        return None
    binder_matches = match_sequence_multisets(
        left["binder_sequences"], right["binder_sequences"]
    )
    if binder_matches is None:
        return None
    match_rows = []
    directions = set()
    for left_index, right_index, relation in binder_matches:
        left_seq = left["binder_sequences"][left_index]
        right_seq = right["binder_sequences"][right_index]
        if relation == "left_in_right":
            directions.add("left_in_right")
            start = find_subsequence(right_seq, left_seq)
        elif relation == "right_in_left":
            directions.add("right_in_left")
            start = find_subsequence(left_seq, right_seq)
        else:
            start = 0
        match_rows.append({
            "left_index": left_index, "right_index": right_index,
            "left_length": len(left_seq), "right_length": len(right_seq),
            "relation": relation, "start_in_longer_0based": start,
        })
    if mhc_relation != "exact":
        directions.add(mhc_relation)
    if not directions:
        direction = "exact"
    elif len(directions) == 1:
        direction = next(iter(directions))
    else:
        direction = "mixed_component_directions"
    return mhc_relation, match_rows, direction


def main() -> None:
    args = parse_args()
    if not args.catalog.is_file():
        raise SystemExit(f"catalog not found: {args.catalog}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_catalog(args.catalog)
    if not records:
        raise SystemExit("catalog has no valid complexes")
    class_representative_rows = load_class_representatives(args.class_representatives)
    class_representatives = set(class_representative_rows)

    # Ligands must be identical; these buckets avoid comparisons that cannot pass.
    buckets: Dict[Tuple[str, ...], List[str]] = defaultdict(list)
    for pdb_id, row in records.items():
        buckets[row["ligand_signature"]].append(pdb_id)

    pair_rows: List[dict] = []
    adjacency = {pdb_id: set() for pdb_id in records}
    mixed_pairs = []
    for members in buckets.values():
        members = sorted(members)
        for left_index, left_id in enumerate(members):
            for right_id in members[left_index + 1 :]:
                left = records[left_id]
                right = records[right_id]
                if len(left["binder_sequences"]) != len(right["binder_sequences"]):
                    continue
                details = relation_details(left, right)
                if details is None:
                    continue
                mhc_relation, binder_matches, direction = details
                adjacency[left_id].add(right_id)
                adjacency[right_id].add(left_id)
                if direction == "mixed_component_directions":
                    mixed_pairs.append((left_id, right_id))
                pair_rows.append({
                    "pdb_id_1": left_id, "pdb_id_2": right_id,
                    "mhc_relation": mhc_relation,
                    "binder_matches_json": json.dumps(binder_matches, separators=(",", ":")),
                    "overall_containment_direction": direction,
                    "binder_count": len(left["binder_sequences"]),
                    "ligand_descriptor_count": len(left["ligand_signature"]),
                    "ligand_signature_sha256_16": left["ligand_signature_sha256_16"],
                })

    # Containment is not transitive. Groups are therefore representative-anchored:
    # every removed member has a direct verified pair with its representative.
    membership_rows: List[dict] = []
    group_by_pdb: Dict[str, str] = {}
    groups = build_groups(records, adjacency, class_representatives)
    for group_number, group in enumerate(groups, start=1):
        representative = group["representative"]
        ordered_members = group["members"]
        standard_representative = min(
            ordered_members, key=lambda pdb_id: representative_key(records[pdb_id])
        )
        reason = selection_reason(
            representative, ordered_members, records, class_representatives
        )
        group_id = f"RED{group_number:05d}"
        for rank, pdb_id in enumerate(ordered_members, start=1):
            row = records[pdb_id]
            group_by_pdb[pdb_id] = group_id
            membership_rows.append({
                "redundancy_group": group_id,
                "group_size": len(ordered_members),
                "pdb_id": pdb_id,
                "representative_pdb_id": representative,
                "is_representative": "yes" if pdb_id == representative else "no",
                "is_class_representative": "yes" if pdb_id in class_representatives else "no",
                "standard_rank_representative": standard_representative,
                "class_priority_changed_selection": (
                    "yes" if representative != standard_representative else "no"
                ),
                "representative_selection_reason": reason,
                "rank": rank,
                "total_peptide_length": row["total_peptide_length"],
                "mhc_length": row["mhc_length"],
                "binder_lengths": row["binder_lengths"],
                "resolution_angstrom": row["resolution_angstrom"],
                "experimental_method": row["experimental_method"],
            })

    for row in pair_rows:
        row["group_1"] = group_by_pdb[row["pdb_id_1"]]
        row["group_2"] = group_by_pdb[row["pdb_id_2"]]
        row["same_assigned_group"] = "yes" if row["group_1"] == row["group_2"] else "no"

    multi_members = [row for row in membership_rows if int(row["group_size"]) > 1]
    removals = [row for row in multi_members if row["is_representative"] == "no"]
    representatives = [row for row in multi_members if row["is_representative"] == "yes"]

    membership_by_id = {row["pdb_id"]: row for row in membership_rows}
    class_audit_rows = []
    for pdb_id, source_row in class_representative_rows.items():
        member = membership_by_id.get(pdb_id)
        if member is None:
            audit_status = "not_in_valid_catalog"
            group_id = group_size = is_representative = standard_representative = ""
            priority_changed = reason = ""
        else:
            group_id = member["redundancy_group"]
            group_size = member["group_size"]
            is_representative = member["is_representative"]
            standard_representative = member["standard_rank_representative"]
            priority_changed = member["class_priority_changed_selection"]
            reason = member["representative_selection_reason"]
            audit_status = (
                "representative_redundant_group"
                if int(group_size) > 1 else "representative_singleton"
            )
        class_audit_rows.append({
            "analysis_group": source_row.get("analysis_group", ""),
            "mapped_class": source_row.get("mapped_class", ""),
            "gene_name": source_row.get("gene_name", ""),
            "pdb_id": pdb_id,
            "mean_tm_score_avg": source_row.get("mean_tm_score_avg", ""),
            "n_pairs": source_row.get("n_pairs", ""),
            "n_structures": source_row.get("n_structures", ""),
            "audit_status": audit_status,
            "redundancy_group": group_id,
            "group_size": group_size,
            "is_redundancy_representative": is_representative,
            "standard_rank_representative": standard_representative,
            "class_priority_changed_selection": priority_changed,
            "representative_selection_reason": reason,
        })

    write_csv(args.output_dir / "redundant_pairs.csv", [
        "pdb_id_1", "pdb_id_2", "mhc_relation", "binder_matches_json",
        "overall_containment_direction", "binder_count", "ligand_descriptor_count",
        "ligand_signature_sha256_16", "group_1", "group_2", "same_assigned_group",
    ], pair_rows)
    write_csv(args.output_dir / "group_membership.csv", [
        "redundancy_group", "group_size", "pdb_id", "representative_pdb_id",
        "is_representative", "is_class_representative",
        "standard_rank_representative", "class_priority_changed_selection",
        "representative_selection_reason", "rank",
        "total_peptide_length", "mhc_length",
        "binder_lengths", "resolution_angstrom", "experimental_method",
    ], membership_rows)
    write_csv(args.output_dir / "representatives.csv", [
        "redundancy_group", "group_size", "pdb_id", "total_peptide_length",
        "mhc_length", "binder_lengths", "resolution_angstrom", "experimental_method",
        "is_class_representative", "standard_rank_representative",
        "class_priority_changed_selection", "representative_selection_reason",
    ], representatives)
    write_csv(args.output_dir / "class_representative_audit.csv", [
        "analysis_group", "mapped_class", "gene_name", "pdb_id",
        "mean_tm_score_avg", "n_pairs", "n_structures", "audit_status",
        "redundancy_group", "group_size", "is_redundancy_representative",
        "standard_rank_representative", "class_priority_changed_selection",
        "representative_selection_reason",
    ], class_audit_rows)
    # Deliberately minimal: this is the only file intended as a downstream removal index.
    write_csv(args.output_dir / "removal_index.csv", ["PDB_ID", "redundancy_group"], (
        {"PDB_ID": row["pdb_id"], "redundancy_group": row["redundancy_group"]}
        for row in removals
    ))

    summary = {
        "valid_complexes": len(records),
        "redundant_pairs": len(pair_rows),
        "assigned_groups_including_singletons": len(groups),
        "redundant_groups": len(representatives),
        "representatives": len(representatives),
        "structures_marked_for_removal": len(removals),
        "mixed_component_direction_pairs": len(mixed_pairs),
        "grouping_rule": "representative-anchored partition; every removed PDB directly matches its representative; curated class representatives cannot remove each other",
        "representative_rule": "curated class representative, then greatest trimmed MHC length, then lowest numeric resolution, then greatest total binder peptide length, then PDB ID",
        "class_representative_source": args.class_representatives.name,
        "class_representatives_listed": len(class_representatives),
        "class_representatives_in_valid_catalog": sum(
            pdb_id in records for pdb_id in class_representatives
        ),
        "class_representatives_not_in_valid_catalog": sum(
            pdb_id not in records for pdb_id in class_representatives
        ),
        "class_representatives_in_redundant_groups": sum(
            row["audit_status"] == "representative_redundant_group"
            for row in class_audit_rows
        ),
        "class_priority_changed_representative": sum(
            row["class_priority_changed_selection"] == "yes"
            for row in class_audit_rows
        ),
        "representative_selection_reasons": {
            reason: sum(row["representative_selection_reason"] == reason for row in representatives)
            for reason in sorted({row["representative_selection_reason"] for row in representatives})
        },
        "run_signature_sha256_16": stable_id({
            "pairs": pair_rows,
            "groups": [
                {"representative": group["representative"], "members": group["members"]}
                for group in groups
            ],
            "class_representatives": sorted(class_representatives),
        }),
    }
    (args.output_dir / "redundancy_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    log(
        f"[DONE] pairs={len(pair_rows)} redundant_groups={len(representatives)} "
        f"remove={len(removals)}"
    )


if __name__ == "__main__":
    main()
