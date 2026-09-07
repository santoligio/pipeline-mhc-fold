#!/usr/bin/env python3
"""Run catalog extraction, redundancy mapping, and consistency validation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--complex-dir", type=Path)
    parser.add_argument("--binders", type=Path)
    parser.add_argument("--ligands", type=Path)
    parser.add_argument("--trim-ranges", type=Path)
    parser.add_argument("--cif-dir", type=Path)
    parser.add_argument("--resname-map", type=Path)
    parser.add_argument("--overrides", type=Path)
    parser.add_argument("--final-source-label")
    parser.add_argument(
        "--exclude-pdb", action="append", default=[], metavar="PDB_ID",
        help="Exclude a PDB ID from the analysis; may be repeated",
    )
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "outputs")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--refresh-resolution", action="store_true")
    args = parser.parse_args()

    catalog_command = [sys.executable, str(SCRIPT_DIR / "build_catalog.py"),
                       "--output-dir", str(args.output_dir)]
    for option in ("complex_dir", "binders", "ligands", "trim_ranges", "cif_dir", "resname_map", "overrides"):
        value = getattr(args, option)
        if value is not None:
            catalog_command.extend(("--" + option.replace("_", "-"), str(value)))
    if args.offline:
        catalog_command.append("--offline")
    if args.refresh_resolution:
        catalog_command.append("--refresh-resolution")
    if args.final_source_label:
        catalog_command.extend(("--final-source-label", args.final_source_label))
    for pdb_id in args.exclude_pdb:
        catalog_command.extend(("--exclude-pdb", pdb_id))
    subprocess.run(catalog_command, check=True)
    subprocess.run([
        sys.executable, str(SCRIPT_DIR / "map_redundancy.py"),
        "--catalog", str(args.output_dir / "complex_catalog.csv"),
        "--output-dir", str(args.output_dir),
    ], check=True)
    subprocess.run([
        sys.executable, str(SCRIPT_DIR / "validate_outputs.py"),
        "--output-dir", str(args.output_dir),
    ], check=True)


if __name__ == "__main__":
    main()
