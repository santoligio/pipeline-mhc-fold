#!/usr/bin/env python3
"""
Calculate the minimum distance from DSSP-defined helix atoms of chain A
to every other chain in one PDB structure.

This reproduces the Step 7 distance criterion:
- Run DSSP.
- Select chain A residues with DSSP codes H, G, or I.
- Collect all atoms from those residues.
- For each other chain, report the minimum Euclidean atom-to-atom distance.
- Mark whether the chain is within the default 4.0 Å cutoff.

Requirements:
    Biopython
    NumPy
    pandas
    DSSP executable available as "mkdssp" or "dssp"

Usage:
    python3 min_distance_chain_a_dssp.py structure.pdb

Optional:
    python3 min_distance_chain_a_dssp.py structure.pdb --dssp dssp
    python3 min_distance_chain_a_dssp.py structure.pdb --cutoff 4.0
    python3 min_distance_chain_a_dssp.py structure.pdb --csv distances.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Set, Tuple

import numpy as np
import pandas as pd
from Bio.PDB import NeighborSearch, PDBParser
from Bio.PDB.DSSP import DSSP


HELIX_CODES = {"H", "G", "I"}
WATER_RESNAMES = {"HOH", "WAT", "H2O"}


def identify_helix_residues_with_dssp(
    model,
    pdb_path: Path,
    chain_id: str,
    dssp_executable: str,
) -> Set[Tuple[str, int]]:
    """
    Identify DSSP helix residues on the selected chain.

    Helix codes:
        H = alpha helix
        G = 3-10 helix
        I = pi helix
    """
    dssp = DSSP(
        model,
        str(pdb_path),
        dssp=dssp_executable,
    )

    helix_residues: Set[Tuple[str, int]] = set()

    for key in dssp.keys():
        current_chain_id, residue_id = key
        secondary_structure = dssp[key][2]

        if current_chain_id != chain_id:
            continue

        if secondary_structure in HELIX_CODES:
            helix_residues.add(
                (current_chain_id, residue_id[1])
            )

    return helix_residues


def collect_helix_atoms(
    model,
    helix_residues: Set[Tuple[str, int]],
) -> List:
    """Collect all atoms from the DSSP-defined helix residues."""
    helix_atoms = []

    for chain_id, residue_number in helix_residues:
        if chain_id not in model:
            continue

        chain = model[chain_id]

        for residue in chain:
            if residue.id[1] != residue_number:
                continue

            helix_atoms.extend(list(residue.get_atoms()))

    return helix_atoms


def format_atom(atom) -> str:
    """Return a compact chain/residue/atom label."""
    residue = atom.get_parent()
    chain = residue.get_parent()

    resname = residue.get_resname().strip()
    residue_number = residue.id[1]
    insertion_code = str(residue.id[2]).strip()
    atom_name = atom.get_name().strip()

    return (
        f"{chain.id}:{resname}"
        f"{residue_number}{insertion_code}:{atom_name}"
    )


def get_nonwater_atoms(chain) -> List:
    """Return all atoms from a chain, excluding water residues."""
    atoms = []
    for residue in chain.get_residues():
        resname = residue.get_resname().strip().upper()
        if resname in WATER_RESNAMES:
            continue
        atoms.extend(residue.get_atoms())
    return atoms


def calculate_minimum_distances(
    pdb_path: Path,
    reference_chain_id: str,
    cutoff: float,
    dssp_executable: str,
) -> tuple[pd.DataFrame, int]:
    """
    Calculate minimum distances from DSSP helix atoms of the reference chain
    to every other chain.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(
        pdb_path.stem,
        str(pdb_path),
    )
    model = structure[0]

    if reference_chain_id not in model:
        available = ", ".join(
            chain.id for chain in model.get_chains()
        )
        raise ValueError(
            f"Reference chain {reference_chain_id!r} not found. "
            f"Available chains: {available or 'none'}"
        )

    helix_residues = identify_helix_residues_with_dssp(
        model=model,
        pdb_path=pdb_path,
        chain_id=reference_chain_id,
        dssp_executable=dssp_executable,
    )

    if not helix_residues:
        raise ValueError(
            f"DSSP found no H/G/I helix residues in "
            f"chain {reference_chain_id!r}."
        )

    helix_atoms = collect_helix_atoms(
        model=model,
        helix_residues=helix_residues,
    )

    if not helix_atoms:
        raise ValueError(
            "No atoms were collected from the DSSP helix residues."
        )

    all_atoms = [
        atom
        for chain in model.get_chains()
        for atom in get_nonwater_atoms(chain)
    ]
    neighbor_search = NeighborSearch(all_atoms)

    # Same cutoff-based contact logic used in Step 7.
    chains_within_cutoff: set[str] = set()

    for helix_atom in helix_atoms:
        neighbors = neighbor_search.search(
            helix_atom.coord,
            cutoff,
            level="A",
        )

        for neighbor in neighbors:
            chain_id = neighbor.get_parent().get_parent().id

            if chain_id != reference_chain_id:
                chains_within_cutoff.add(chain_id)

    helix_coords = np.array(
        [atom.coord for atom in helix_atoms],
        dtype=float,
    )

    rows: list[dict[str, object]] = []

    for chain in model.get_chains():
        if chain.id == reference_chain_id:
            continue

        chain_atoms = get_nonwater_atoms(chain)

        # Skip chains that contain only water after filtering.
        if not chain_atoms:
            continue

        other_coords = np.array(
            [atom.coord for atom in chain_atoms],
            dtype=float,
        )

        differences = (
            helix_coords[:, None, :]
            - other_coords[None, :, :]
        )
        squared_distances = np.sum(
            differences * differences,
            axis=2,
        )

        flat_index = int(np.argmin(squared_distances))
        helix_index, other_index = np.unravel_index(
            flat_index,
            squared_distances.shape,
        )

        minimum_distance = float(
            np.sqrt(
                squared_distances[
                    helix_index,
                    other_index,
                ]
            )
        )

        rows.append(
            {
                "reference_chain": reference_chain_id,
                "other_chain": chain.id,
                "minimum_distance_to_mhc_helix": round(
                    minimum_distance,
                    3,
                ),
                "within_cutoff": (
                    "yes"
                    if chain.id in chains_within_cutoff
                    else "no"
                ),
                "cutoff_angstrom": cutoff,
                "closest_helix_atom": format_atom(
                    helix_atoms[helix_index]
                ),
                "closest_other_atom": format_atom(
                    chain_atoms[other_index]
                ),
            }
        )

    result = pd.DataFrame(rows)

    if not result.empty:
        result = result.sort_values(
            "minimum_distance_to_mhc_helix",
            na_position="last",
        ).reset_index(drop=True)

    return result, len(helix_residues)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate minimum distances from DSSP-defined "
            "chain-A helix atoms to every other chain."
        )
    )
    parser.add_argument(
        "pdb_file",
        help="Input PDB file.",
    )
    parser.add_argument(
        "--reference-chain",
        default="A",
        help="Reference MHC chain. Default: A",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=4.0,
        help="Contact cutoff in Å. Default: 4.0",
    )
    parser.add_argument(
        "--dssp",
        default="mkdssp",
        help=(
            "DSSP executable. Default: mkdssp. "
            "Use --dssp dssp if required."
        ),
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional output CSV path.",
    )

    args = parser.parse_args()

    pdb_path = Path(args.pdb_file)

    if not pdb_path.is_file():
        raise SystemExit(
            f"ERROR: PDB file not found: {pdb_path}"
        )

    try:
        result, helix_residue_count = (
            calculate_minimum_distances(
                pdb_path=pdb_path,
                reference_chain_id=args.reference_chain,
                cutoff=args.cutoff,
                dssp_executable=args.dssp,
            )
        )
    except Exception as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    print()
    print(
        f"DSSP helix residues in chain "
        f"{args.reference_chain}: "
        f"{helix_residue_count}"
    )
    print()

    if result.empty:
        print(
            f"No chains other than "
            f"{args.reference_chain!r} were found."
        )
        return

    print(result.to_string(index=False))
    print()

    if args.csv:
        output_csv = Path(args.csv)
        output_csv.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        result.to_csv(output_csv, index=False)
        print(f"CSV written to: {output_csv}")


if __name__ == "__main__":
    main()
