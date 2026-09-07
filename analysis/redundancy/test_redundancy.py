from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from redundancy_core import (
    ChemComp, contiguous_relation, extract_pdb_residues, find_subsequence,
    match_sequence_multisets, peptide_token, read_chem_comps,
)
from build_catalog import (
    classify_residue, load_overrides, load_resname_aliases,
    redundancy_peptide_token,
)
from map_redundancy import representative_key, selection_reason


class RedundancyCoreTests(unittest.TestCase):
    @staticmethod
    def ranking_row(pdb_id, mhc_length, resolution, binder_lengths):
        return {
            "pdb_id": pdb_id,
            "mhc_length": str(mhc_length),
            "resolution_angstrom": resolution,
            "binder_sequences": tuple(tuple("A" * length) for length in binder_lengths),
        }

    def test_contiguous_only(self):
        self.assertEqual(contiguous_relation(tuple("ABC"), tuple("XABCY")), "left_in_right")
        self.assertEqual(contiguous_relation(tuple("XABCY"), tuple("ABC")), "right_in_left")
        self.assertEqual(contiguous_relation(tuple("ABC"), tuple("ABC")), "exact")
        self.assertIsNone(contiguous_relation(tuple("ABC"), tuple("ABXC")))
        self.assertEqual(find_subsequence(tuple("XXABCY"), tuple("ABC")), 2)

    def test_multiset_matching_preserves_count(self):
        left = [tuple("ABC"), tuple("QQ")]
        right = [tuple("XABCY"), tuple("QQ")]
        self.assertIsNotNone(match_sequence_multisets(left, right))
        self.assertIsNone(match_sequence_multisets([tuple("ABC")], right))
        self.assertIsNone(match_sequence_multisets([tuple("ABC"), tuple("ABC")], [tuple("ABC"), tuple("QQ")]))

    def test_modified_residue_parent(self):
        comp = ChemComp("MSE", "L-peptide linking", "MET")
        self.assertTrue(comp.is_peptide)
        self.assertEqual(peptide_token("MSE", comp), "M")
        self.assertEqual(redundancy_peptide_token("MSE", "M"), "[MSE>M]")
        self.assertEqual(redundancy_peptide_token("MET", "M"), "M")

    def test_manual_modified_residue_keeps_parent_and_ligand_descriptor(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "manual_overrides.csv"
            path.write_text(
                "pdb_id,residue_id,classification,canonical_token,ligand_descriptor,reason\n"
                "1KCG,CGL,peptide,C,N:GSH,test\n"
                "6BJ2,PCA,peptide,Q,,test\n"
                "9TJX,CGU,peptide,E,,test\n",
                encoding="utf-8",
            )
            overrides = load_overrides(path)
            kind, token, provenance = classify_residue("1KCG", "CGL", {}, overrides)
            self.assertEqual((kind, token, provenance), ("peptide", "C", "manual_override"))
            self.assertEqual(overrides[("1KCG", "CGL")]["ligand_descriptor"], "N:GSH")
            self.assertEqual(redundancy_peptide_token("CGL", token), "[CGL>C]")
            self.assertEqual(classify_residue("6BJ2", "PCA", {}, overrides)[:2], ("peptide", "Q"))
            self.assertEqual(redundancy_peptide_token("PCA", "Q"), "[PCA>Q]")
            self.assertEqual(classify_residue("9TJX", "CGU", {}, overrides)[:2], ("peptide", "E"))
            self.assertEqual(redundancy_peptide_token("CGU", "E"), "[CGU>E]")

    def test_cif_and_pdb_parsers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            cif = temp / "x.cif"
            cif.write_text(
                "data_x\nloop_\n_chem_comp.id\n_chem_comp.type\n"
                "_chem_comp.mon_nstd_parent_comp_id\nALA 'L-peptide linking' ?\n"
                "MSE 'L-peptide linking' MET\nLIG non-polymer ?\n#\n",
                encoding="utf-8",
            )
            comps = read_chem_comps(cif)
            self.assertTrue(comps["ALA"].is_peptide)
            self.assertEqual(comps["MSE"].parent_id, "MET")
            self.assertFalse(comps["LIG"].is_peptide)
            pdb = temp / "x.pdb"
            pdb.write_text(
                "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N  \n"
                "ATOM      2  CA AALA A   1       0.000   0.000   0.000  1.00  0.00           C  \n"
                "HETATM    3  C1  LIG C   2       0.000   0.000   0.000  1.00  0.00           C  \n",
                encoding="utf-8",
            )
            chains = extract_pdb_residues(pdb)
            self.assertEqual([res.resname for res in chains["A"]], ["ALA"])
            self.assertEqual([res.resname for res in chains["C"]], ["LIG"])

    def test_altloc_prefix_is_removed_from_long_ccd_alias(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "resname_map.csv"
            path.write_text(
                "pdb,old_resname,new_resname,chain_id,resseq\n"
                "9O06,AA1B7N,B7N,A,268\n"
                "9O06,BA1B7N,B7N,A,268\n",
                encoding="utf-8",
            )
            by_name, by_position = load_resname_aliases(path)
            self.assertEqual(by_name[("9O06", "B7N")], "A1B7N")
            self.assertEqual(by_position[("9O06", "268")], "A1B7N")

    def test_representative_order_is_mhc_resolution_binders_then_pdb_id(self):
        rows = [
            self.ranking_row("BBBB", 180, "1.8", [120]),
            self.ranking_row("AAAA", 181, "3.0", [1]),
        ]
        self.assertEqual(min(rows, key=representative_key)["pdb_id"], "AAAA")

        rows = [
            self.ranking_row("BBBB", 180, "1.8", [1]),
            self.ranking_row("AAAA", 180, "2.0", [200]),
        ]
        self.assertEqual(min(rows, key=representative_key)["pdb_id"], "BBBB")

        rows = [
            self.ranking_row("BBBB", 180, "1.8", [100, 20]),
            self.ranking_row("AAAA", 180, "1.8", [121]),
        ]
        self.assertEqual(min(rows, key=representative_key)["pdb_id"], "AAAA")
        records = {row["pdb_id"]: row for row in rows}
        self.assertEqual(
            selection_reason("AAAA", ["AAAA", "BBBB"], records),
            "greatest_total_binder_length",
        )

        rows = [
            self.ranking_row("BBBB", 180, "1.8", [121]),
            self.ranking_row("AAAA", 180, "1.8", [121]),
        ]
        self.assertEqual(min(rows, key=representative_key)["pdb_id"], "AAAA")

        rows = [
            self.ranking_row("BBBB", 180, "", [200]),
            self.ranking_row("AAAA", 180, "2.5", [1]),
        ]
        self.assertEqual(min(rows, key=representative_key)["pdb_id"], "AAAA")


if __name__ == "__main__":
    unittest.main()
