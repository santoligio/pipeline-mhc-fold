#!/usr/bin/env python3
"""Fail-fast consistency checks for all definitive redundancy outputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from redundancy_core import contiguous_relation, match_sequence_multisets, read_csv


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"
DEFAULT_CLASS_REPRESENTATIVES = SCRIPT_DIR / "top1_representatives_tm_score.csv"


def load_class_representatives(path: Path) -> dict[str, dict]:
    rows = read_csv(path)
    if not rows or "pdb" not in rows[0]:
        raise AssertionError("class representative CSV must contain a 'pdb' column")
    result = {}
    for row in rows:
        pdb_id = row["pdb"].strip().upper()
        assert pdb_id, "blank class representative PDB ID"
        assert pdb_id not in result, f"duplicate class representative PDB ID: {pdb_id}"
        result[pdb_id] = row
    return result


def validate(
    output_dir: Path,
    class_representatives_path: Path = DEFAULT_CLASS_REPRESENTATIVES,
) -> dict:
    catalog = read_csv(output_dir / "complex_catalog.csv")
    components = read_csv(output_dir / "component_sequences.csv")
    issues = read_csv(output_dir / "validation_issues.csv")
    pairs = read_csv(output_dir / "redundant_pairs.csv")
    membership = read_csv(output_dir / "group_membership.csv")
    removals = read_csv(output_dir / "removal_index.csv")
    representatives = read_csv(output_dir / "representatives.csv")
    class_audit = read_csv(output_dir / "class_representative_audit.csv")
    class_rows = load_class_representatives(class_representatives_path)
    class_ids = set(class_rows)

    catalog_ids = [row["pdb_id"] for row in catalog]
    assert len(catalog_ids) == len(set(catalog_ids)), "duplicate catalog PDB ID"
    valid_ids = {row["pdb_id"] for row in catalog if row["status"] == "valid"}
    invalid_ids = set(catalog_ids) - valid_ids
    error_ids = {row["pdb_id"] for row in issues if row["severity"] == "ERROR"}
    assert invalid_ids == (error_ids & set(catalog_ids)), "catalog status/error mismatch"
    assert not invalid_ids, (
        f"{len(invalid_ids)} invalid PDBs remain; inspect validation_issues.csv before removal"
    )

    component_keys = [(row["pdb_id"], row["component"], row["chain_id"]) for row in components]
    assert len(component_keys) == len(set(component_keys)), "duplicate component chain row"
    for row in catalog:
        pdb_id = row["pdb_id"]
        assert int(row["observed_chain_count"]) == int(row["classified_chain_count"]), (
            f"{pdb_id}: final/classified chain count mismatch"
        )
        mhc = [item for item in components if item["pdb_id"] == pdb_id and item["component"] == "mhc"]
        binders = [item for item in components if item["pdb_id"] == pdb_id and item["component"] == "binder"]
        ligands = [item for item in components if item["pdb_id"] == pdb_id and item["component"] == "ligand"]
        assert len(mhc) == 1, f"{pdb_id}: expected one MHC component"
        assert len(binders) == int(row["binder_count"]), f"{pdb_id}: binder count mismatch"
        assert len(ligands) == int(row["ligand_chain_count"]), f"{pdb_id}: ligand count mismatch"
        supplemental = Counter(
            descriptor
            for item in components if item["pdb_id"] == pdb_id
            for descriptor in json.loads(item.get("supplemental_ligand_descriptors_json", "[]"))
        )
        ligand_signature = Counter(json.loads(row["ligand_signature_json"]))
        assert all(ligand_signature[key] >= count for key, count in supplemental.items()), (
            f"{pdb_id}: supplemental chemical descriptor missing from ligand signature"
        )
        canonical_mhc = json.loads(row["mhc_sequence_json"])
        redundancy_mhc = json.loads(row["mhc_redundancy_sequence_json"])
        assert len(canonical_mhc) == len(redundancy_mhc) == int(row["mhc_length"]), (
            f"{pdb_id}: canonical/redundancy MHC length mismatch"
        )
        canonical_binders = json.loads(row["binder_signature_json"])
        redundancy_binders = json.loads(row["binder_redundancy_signature_json"])
        assert sorted(map(len, canonical_binders)) == sorted(map(len, redundancy_binders)), (
            f"{pdb_id}: canonical/redundancy binder length mismatch"
        )

    membership_ids = [row["pdb_id"] for row in membership]
    assert set(membership_ids) == valid_ids, "membership must contain each valid PDB"
    assert len(membership_ids) == len(set(membership_ids)), "PDB assigned to multiple groups"
    group_counts = Counter(row["redundancy_group"] for row in membership)
    for row in membership:
        assert int(row["group_size"]) == group_counts[row["redundancy_group"]], "group size mismatch"
    reps_by_group = {
        group: [row for row in membership if row["redundancy_group"] == group and row["is_representative"] == "yes"]
        for group in group_counts
    }
    assert all(len(rows) == 1 for rows in reps_by_group.values()), "group without exactly one representative"

    membership_by_id = {row["pdb_id"]: row for row in membership}
    valid_class_ids = class_ids & valid_ids
    for pdb_id in valid_class_ids:
        member = membership_by_id[pdb_id]
        assert member["is_representative"] == "yes", (
            f"{pdb_id}: curated class representative is not its redundancy-group representative"
        )
        assert member["is_class_representative"] == "yes"
    for pdb_id in valid_ids - class_ids:
        assert membership_by_id[pdb_id]["is_class_representative"] == "no"

    pair_set = {frozenset((row["pdb_id_1"], row["pdb_id_2"])) for row in pairs}
    removal_ids = [row["PDB_ID"] for row in removals]
    assert len(removal_ids) == len(set(removal_ids)), "duplicate removal PDB ID"
    assert set(removal_ids) <= valid_ids, "invalid PDB in removal index"
    for row in removals:
        member = membership_by_id[row["PDB_ID"]]
        assert member["redundancy_group"] == row["redundancy_group"], "removal group mismatch"
        representative = member["representative_pdb_id"]
        assert frozenset((row["PDB_ID"], representative)) in pair_set, (
            f"{row['PDB_ID']}: no direct redundancy pair with representative {representative}"
        )
    expected_reps = {
        row["pdb_id"] for row in membership
        if int(row["group_size"]) > 1 and row["is_representative"] == "yes"
    }
    assert {row["pdb_id"] for row in representatives} == expected_reps, "representative table mismatch"
    assert not (set(removal_ids) & expected_reps), "representative marked for removal"
    for row in pairs:
        assert row["pdb_id_1"] in valid_ids and row["pdb_id_2"] in valid_ids

    catalog_by_id = {row["pdb_id"]: row for row in catalog if row["status"] == "valid"}
    for pair in pairs:
        left = catalog_by_id[pair["pdb_id_1"]]
        right = catalog_by_id[pair["pdb_id_2"]]
        left_mhc = tuple(json.loads(left["mhc_redundancy_sequence_json"]))
        right_mhc = tuple(json.loads(right["mhc_redundancy_sequence_json"]))
        assert contiguous_relation(left_mhc, right_mhc) == pair["mhc_relation"]
        left_binders = [tuple(seq) for seq in json.loads(left["binder_redundancy_signature_json"])]
        right_binders = [tuple(seq) for seq in json.loads(right["binder_redundancy_signature_json"])]
        assert match_sequence_multisets(left_binders, right_binders) is not None
        assert left["ligand_signature_json"] == right["ligand_signature_json"]

    def rep_key(row):
        try:
            resolution_key = (0, float(row["resolution_angstrom"]))
        except (TypeError, ValueError):
            resolution_key = (1, float("inf"))
        binder_length = sum(len(seq) for seq in json.loads(row["binder_signature_json"]))
        return (-int(row["mhc_length"]), *resolution_key, -binder_length, row["pdb_id"])

    for group, rows in reps_by_group.items():
        representative = rows[0]["pdb_id"]
        members = [row["pdb_id"] for row in membership if row["redundancy_group"] == group]
        standard_representative = min(
            members, key=lambda pdb_id: rep_key(catalog_by_id[pdb_id])
        )
        curated_members = set(members) & class_ids
        assert len(curated_members) <= 1, (
            f"{group}: multiple curated class representatives were assigned together"
        )
        expected_representative = (
            next(iter(curated_members)) if curated_members else standard_representative
        )
        assert representative == expected_representative, (
            f"{group}: representative violates class/MHC/resolution/binder/PDB ordering"
        )
        changed = "yes" if representative != standard_representative else "no"
        for member in (row for row in membership if row["redundancy_group"] == group):
            assert member["standard_rank_representative"] == standard_representative
            assert member["class_priority_changed_selection"] == changed

    audit_ids = [row["pdb_id"] for row in class_audit]
    assert len(audit_ids) == len(set(audit_ids)), "duplicate class representative audit row"
    assert set(audit_ids) == class_ids, "class representative audit/source mismatch"
    audit_by_id = {row["pdb_id"]: row for row in class_audit}
    for pdb_id in class_ids:
        audit = audit_by_id[pdb_id]
        source = class_rows[pdb_id]
        assert audit["analysis_group"] == source.get("analysis_group", "")
        assert audit["mapped_class"] == source.get("mapped_class", "")
        assert audit["gene_name"] == source.get("gene_name", "")
        if pdb_id not in valid_ids:
            assert audit["audit_status"] == "not_in_valid_catalog"
            assert not audit["redundancy_group"]
            continue
        member = membership_by_id[pdb_id]
        expected_status = (
            "representative_redundant_group"
            if int(member["group_size"]) > 1 else "representative_singleton"
        )
        assert audit["audit_status"] == expected_status
        assert audit["redundancy_group"] == member["redundancy_group"]
        assert audit["group_size"] == member["group_size"]
        assert audit["is_redundancy_representative"] == "yes"
        assert audit["standard_rank_representative"] == member["standard_rank_representative"]
        assert (
            audit["class_priority_changed_selection"]
            == member["class_priority_changed_selection"]
        )

    result = {
        "catalog_rows": len(catalog), "valid_pdbs": len(valid_ids),
        "invalid_pdbs": len(invalid_ids), "component_rows": len(components),
        "pair_rows": len(pairs), "groups": len(group_counts),
        "redundant_groups": len(expected_reps), "removal_rows": len(removals),
        "class_representatives_listed": len(class_ids),
        "class_representatives_in_valid_catalog": len(valid_class_ids),
        "class_representatives_in_redundant_groups": sum(
            row["audit_status"] == "representative_redundant_group"
            for row in class_audit
        ),
        "class_priority_changed_representative": sum(
            row["class_priority_changed_selection"] == "yes"
            for row in class_audit
        ),
        "status": "passed",
    }
    (output_dir / "validation_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--class-representatives", type=Path, default=DEFAULT_CLASS_REPRESENTATIVES,
    )
    args = parser.parse_args()
    result = validate(args.output_dir, args.class_representatives)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
