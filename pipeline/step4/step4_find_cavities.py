#!/usr/bin/env python3
"""
Align MHC-only PDBs, find cavities, and keep reference-groove cavities.
"""

import csv
import os
import subprocess
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ["KMP_WARNINGS"] = "0"

import numpy as np
import pyKVFinder
from Bio.PDB import PDBParser, PDBIO


# =========================
# Configuration
# =========================

PIPELINE_DIR = Path("/mnt/c/Users/gio/Documents/foldseek_nefertari/filter/ligands_pipeline")

INPUT_PDB_DIR = PIPELINE_DIR / "step3" / "pdb" / "1_mhc_only"
STEP4_DIR = PIPELINE_DIR / "step4"

ALIGN_REF_PDB = STEP4_DIR / "1ao7_pep_ref.pdb"
CAVITY_REF_PDB = STEP4_DIR / "1ao7_pep_only_CA.pdb"

ALIGNED_DIR = STEP4_DIR / "pdb" / "1_aligned"
MATRIX_DIR = ALIGNED_DIR / "matrices"
KVFINDER_DIR = STEP4_DIR / "pdb" / "2_kvfinder"
CAVITY_DIR = STEP4_DIR / "pdb" / "3_filtered_cavities"

SUMMARY_CSV = CAVITY_DIR / "cavities.csv"
ERROR_LOG = STEP4_DIR / "step4_errors.log"

TMALIGN_EXE = "TMalign"
INPUT_PATTERN = "*.pdb"

PROBE_IN = 1.4
PROBE_OUT = 6.0
VOLUME_CUTOFF = 30.0
REMOVAL_DISTANCE = 1.8
INCLUDE_DEPTH = True
INCLUDE_HYDROPATHY = True
HYDROPHOBICITY_SCALE = "EisenbergWeiss"

CAVITY_CUTOFF = 1.0

OVERWRITE_KVFINDER = True
SKIP_EXISTING_ALIGNED = False
DEBUG_ALIGNMENT = False


# =========================
# Setup
# =========================

def log(message: str) -> None:
    print(message, flush=True)


def make_output_dirs() -> None:
    for path in [ALIGNED_DIR, MATRIX_DIR, KVFINDER_DIR, CAVITY_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def validate_inputs() -> None:
    if not INPUT_PDB_DIR.is_dir():
        raise SystemExit(f"ERROR: INPUT_PDB_DIR not found: {INPUT_PDB_DIR}")

    if not ALIGN_REF_PDB.is_file():
        raise SystemExit(f"ERROR: ALIGN_REF_PDB not found: {ALIGN_REF_PDB}")

    if not CAVITY_REF_PDB.is_file():
        raise SystemExit(f"ERROR: CAVITY_REF_PDB not found: {CAVITY_REF_PDB}")


def collect_input_pdbs() -> List[Path]:
    pdbs = sorted(INPUT_PDB_DIR.glob(INPUT_PATTERN))

    if not pdbs:
        raise SystemExit(f"ERROR: no PDB files found in {INPUT_PDB_DIR}")

    return pdbs


def get_pdb_id(path: Path) -> str:
    return path.name.split("_")[0].lower()


# =========================
# TM-align
# =========================

def run_tmalign(mobile_pdb: Path, ref_pdb: Path, matrix_path: Path) -> None:
    cmd = [TMALIGN_EXE, str(mobile_pdb), str(ref_pdb), "-m", str(matrix_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.returncode != 0:
        raise RuntimeError(
            "TM-align failed\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDERR:\n{proc.stderr}\n"
            f"STDOUT:\n{proc.stdout}"
        )


def read_tmalign_matrix(matrix_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    lines = matrix_path.read_text().splitlines()

    start = None
    for idx, line in enumerate(lines):
        if "rotation matrix" in line.lower():
            start = idx + 1
            break

    if start is None:
        raise ValueError(f"Rotation matrix header not found: {matrix_path}")

    rows = []

    for line in lines[start:]:
        try:
            nums = [float(x) for x in line.strip().split()]
        except ValueError:
            continue

        if len(nums) == 5:
            rows.append(nums[1:5])
        elif len(nums) == 4:
            rows.append(nums)

        if len(rows) == 3:
            break

    if len(rows) != 3:
        raise ValueError(f"Could not parse matrix rows: {matrix_path}")

    matrix = np.array(rows, dtype=float)
    t_vec = matrix[:, 0]
    u_mat = matrix[:, 1:4]

    return u_mat, t_vec


def apply_transform(in_pdb: Path, out_pdb: Path, u_mat: np.ndarray, t_vec: np.ndarray) -> None:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("mobile", str(in_pdb))

    # TM-align: X = t + U*x. Biopython uses X = x*R + t.
    rotation = u_mat.T

    for atom in structure.get_atoms():
        atom.transform(rotation, t_vec)

    io = PDBIO()
    io.set_structure(structure)
    io.save(str(out_pdb))


def check_rotation(u_mat: np.ndarray) -> Tuple[float, float]:
    det = float(np.linalg.det(u_mat))
    ortho_err = float(np.linalg.norm(u_mat @ u_mat.T - np.eye(3)))
    return det, ortho_err


def align_one(pdb_path: Path) -> Path:
    pdb_id = get_pdb_id(pdb_path)

    out_pdb = ALIGNED_DIR / f"{pdb_id}_aligned.pdb"
    matrix_path = MATRIX_DIR / f"{pdb_id}_matrix.txt"

    if SKIP_EXISTING_ALIGNED and out_pdb.exists():
        log(f"[SKIPPED] {pdb_id}: aligned PDB exists")
        return out_pdb

    run_tmalign(pdb_path, ALIGN_REF_PDB, matrix_path)

    u_mat, t_vec = read_tmalign_matrix(matrix_path)
    det, ortho_err = check_rotation(u_mat)

    if DEBUG_ALIGNMENT:
        log(f"[MATRIX] {pdb_id}: det={det:.6f} ortho_err={ortho_err:.3e}")

    if ortho_err > 1e-2:
        log(f"[WARNING] {pdb_id}: rotation may not be orthonormal")

    apply_transform(pdb_path, out_pdb, u_mat, t_vec)

    log(f"[ALIGNED] {pdb_id}: {out_pdb.name}")
    return out_pdb


def align_all(pdbs: List[Path]) -> List[Path]:
    aligned = []

    for pdb_path in pdbs:
        try:
            aligned.append(align_one(pdb_path))
        except Exception as exc:
            log_error(f"[ALIGN_FAILED] {pdb_path.name}: {exc}\n{traceback.format_exc()}")

    return aligned


# =========================
# PDB cleanup for pyKVFinder
# =========================

def clean_pdb_for_kvfinder(in_pdb: Path, out_pdb: Path) -> None:
    """
    Apply only the known MSE fix needed by pyKVFinder.

    In the temporary KVFinder input only:
      MSE SE -> MET SD with sulfur element.
    All other lines are copied unchanged.
    """
    with in_pdb.open("r") as fin, out_pdb.open("w") as fout:
        for line in fin:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                fout.write(line)
                continue

            atom_name = line[12:16].strip()
            resname = line[17:20].strip()

            if resname == "MSE" and atom_name == "SE":
                line = f"{line[:12]} SD {line[16:17]}MET{line[20:76]} S{line[78:]}"
            elif resname == "MSE":
                line = f"{line[:17]}MET{line[20:]}"

            fout.write(line)


# =========================
# pyKVFinder
# =========================

def run_kvfinder_one(pdb_path: Path) -> Optional[Tuple[Path, Path, Path]]:
    pdb_id = get_pdb_id(pdb_path)

    out_dir = KVFINDER_DIR / pdb_id
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_pdb = out_dir / f"{pdb_id}_kvfinder_input.pdb"
    results_toml = out_dir / "results.toml"
    cavity_pdb = out_dir / "cavity.pdb"

    if not OVERWRITE_KVFINDER and results_toml.exists() and cavity_pdb.exists():
        log(f"[SKIPPED] {pdb_id}: KVFinder output exists")
        return out_dir, results_toml, cavity_pdb

    clean_pdb_for_kvfinder(pdb_path, clean_pdb)

    results = pyKVFinder.run_workflow(
        str(clean_pdb),
        include_depth=INCLUDE_DEPTH,
        include_hydropathy=INCLUDE_HYDROPATHY,
        hydrophobicity_scale=HYDROPHOBICITY_SCALE,
        probe_in=PROBE_IN,
        probe_out=PROBE_OUT,
        volume_cutoff=VOLUME_CUTOFF,
        removal_distance=REMOVAL_DISTANCE,
    )

    results.export_all(
        fn=str(results_toml),
        output=str(cavity_pdb),
        include_frequencies_pdf=False,
    )

    log(f"[KVFINDER] {pdb_id}")
    return out_dir, results_toml, cavity_pdb


def run_kvfinder_all(aligned_pdbs: List[Path]) -> List[Tuple[Path, Path, Path]]:
    outputs = []

    for pdb_path in aligned_pdbs:
        try:
            result = run_kvfinder_one(pdb_path)
            if result is not None:
                outputs.append(result)
        except Exception as exc:
            log_error(f"[KVFINDER_FAILED] {pdb_path.name}: {exc}\n{traceback.format_exc()}")

    return outputs


# =========================
# Cavity filtering
# =========================

try:
    import tomllib  # type: ignore
except Exception:
    tomllib = None
    import tomli  # type: ignore


def load_toml(path: Path) -> dict:
    data = path.read_bytes()
    if tomllib is not None:
        return tomllib.loads(data.decode("utf-8"))
    return tomli.loads(data.decode("utf-8"))


def read_reference_ca_coords(reference_pdb: Path) -> np.ndarray:
    coords = []

    with reference_pdb.open("r") as handle:
        for line in handle:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue

            if line[12:16].strip() != "CA":
                continue

            coords.append((
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            ))

    if not coords:
        raise ValueError(f"No CA atoms found in reference: {reference_pdb}")

    return np.array(coords, dtype=float)


def parse_cavity_points(cavity_pdb: Path) -> Dict[str, np.ndarray]:
    points: Dict[str, List[Tuple[float, float, float]]] = {}

    with cavity_pdb.open("r") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue

            cavity_id = line[17:20].strip()
            if not cavity_id:
                continue

            points.setdefault(cavity_id, []).append((
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            ))

    return {key: np.array(value, dtype=float) for key, value in points.items()}


def min_distance(points_a: np.ndarray, points_b: np.ndarray) -> float:
    try:
        from scipy.spatial import cKDTree  # type: ignore

        tree = cKDTree(points_b)
        dists, _ = tree.query(points_a, k=1)
        return float(np.min(dists))

    except Exception:
        diff = points_a[:, None, :] - points_b[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        return float(np.sqrt(np.min(dist2)))


def filter_cavities(
    cavity_points: Dict[str, np.ndarray],
    reference_ca: np.ndarray,
) -> Tuple[List[str], Dict[str, float]]:
    kept = []
    distances = {}

    for cavity_id, points in cavity_points.items():
        if points.size == 0:
            continue

        dmin = min_distance(points, reference_ca)
        distances[cavity_id] = dmin

        if dmin < CAVITY_CUTOFF:
            kept.append(cavity_id)

    kept.sort()
    return kept, distances


def write_filtered_cavity(in_pdb: Path, out_pdb: Path, keep_ids: set) -> None:
    with in_pdb.open("r") as fin, out_pdb.open("w") as fout:
        for line in fin:
            if line.startswith("ATOM"):
                cavity_id = line[17:20].strip()
                if cavity_id in keep_ids:
                    fout.write(line)
            else:
                fout.write(line)


def get_metrics(results: dict, cavity_id: str) -> dict:
    block = results.get("RESULTS", {})

    def get_value(section: str) -> Optional[float]:
        value = block.get(section, {}).get(cavity_id, None)
        return round(float(value), 3) if value is not None else None

    return {
        "volume": get_value("VOLUME"),
        "area": get_value("AREA"),
        "max_depth": get_value("MAX_DEPTH"),
        "avg_depth": get_value("AVG_DEPTH"),
        "avg_hydropathy": get_value("AVG_HYDROPATHY"),
    }


def filter_all_cavities(kv_outputs: List[Tuple[Path, Path, Path]]) -> None:
    reference_ca = read_reference_ca_coords(CAVITY_REF_PDB)

    rows = []
    n_with_kept = 0

    for out_dir, results_path, cavity_path in kv_outputs:
        pdb_id = out_dir.name

        try:
            results = load_toml(results_path)
            cavity_points = parse_cavity_points(cavity_path)
            kept_ids, distances = filter_cavities(cavity_points, reference_ca)

            if kept_ids:
                n_with_kept += 1

            filtered_pdb = CAVITY_DIR / f"{pdb_id}_filtered_cavity.pdb"
            write_filtered_cavity(cavity_path, filtered_pdb, set(kept_ids))

            if kept_ids:
                for cavity_id in kept_ids:
                    rows.append({
                        "pdb": pdb_id,
                        "cavity_id": cavity_id,
                        "min_dist_to_reference_ca": round(float(distances[cavity_id]), 3),
                        **get_metrics(results, cavity_id),
                    })
            else:
                rows.append({
                    "pdb": pdb_id,
                    "cavity_id": "NA",
                    "min_dist_to_reference_ca": "NA",
                    "volume": "NA",
                    "area": "NA",
                    "max_depth": "NA",
                    "avg_depth": "NA",
                    "avg_hydropathy": "NA",
                })

            log(f"[FILTERED] {pdb_id}: kept={len(kept_ids)}")

        except Exception as exc:
            log_error(f"[FILTER_FAILED] {pdb_id}: {exc}\n{traceback.format_exc()}")

    fieldnames = [
        "pdb",
        "cavity_id",
        "min_dist_to_reference_ca",
        "volume",
        "area",
        "max_depth",
        "avg_depth",
        "avg_hydropathy",
    ]

    with SUMMARY_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    log(f"[SUMMARY] structures={len(kv_outputs)} with_cavity={n_with_kept}")
    log(f"[CSV] {SUMMARY_CSV}")


# =========================
# Logging
# =========================

def log_error(message: str) -> None:
    log(message)
    with ERROR_LOG.open("a") as handle:
        handle.write(message)
        handle.write("\n")
        handle.write("-" * 80)
        handle.write("\n")


# =========================
# Main
# =========================

def main() -> None:
    make_output_dirs()
    validate_inputs()

    if ERROR_LOG.exists():
        ERROR_LOG.unlink()

    log("[START] STEP4")

    input_pdbs = collect_input_pdbs()
    log(f"[INPUT] PDBs={len(input_pdbs)}")

    aligned_pdbs = align_all(input_pdbs)
    log(f"[ALIGNED] total={len(aligned_pdbs)}")

    kv_outputs = run_kvfinder_all(aligned_pdbs)
    log(f"[KVFINDER] total={len(kv_outputs)}")

    filter_all_cavities(kv_outputs)

    log("[DONE] STEP4")


if __name__ == "__main__":
    main()
