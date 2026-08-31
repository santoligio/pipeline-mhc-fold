#!/usr/bin/env bash
#
# Compare paired PDB files between two 2_trimmed_mhc directories
# (e.g. step6 vs step6_fix) and report which ones changed.
#
# Usage:
#   ./compare_trimmed_mhc.sh <old_dir> <new_dir> [report.csv]
#
# Example:
#   ./compare_trimmed_mhc.sh \
#       /mnt/c/.../ligands_pipeline/step6/pdb/2_trimmed_mhc \
#       /mnt/c/.../ligands_pipeline/step6_fix/pdb/2_trimmed_mhc \
#       trimmed_mhc_diff_report.csv

set -euo pipefail

OLD_DIR="${1:?Usage: $0 <old_dir> <new_dir> [report.csv]}"
NEW_DIR="${2:?Usage: $0 <old_dir> <new_dir> [report.csv]}"
REPORT="${3:-trimmed_mhc_diff_report.csv}"

if [[ ! -d "$OLD_DIR" ]]; then
    echo "ERROR: old dir does not exist: $OLD_DIR" >&2
    exit 1
fi

if [[ ! -d "$NEW_DIR" ]]; then
    echo "ERROR: new dir does not exist: $NEW_DIR" >&2
    exit 1
fi

echo "pdb,status,old_path,new_path" > "$REPORT"

# Collect the union of filenames from both directories (basenames only).
mapfile -t all_files < <(
    { find "$OLD_DIR" -maxdepth 1 -name '*.pdb' -printf '%f\n'; \
      find "$NEW_DIR" -maxdepth 1 -name '*.pdb' -printf '%f\n'; } \
    | sort -u
)

n_same=0
n_changed=0
n_only_old=0
n_only_new=0

for fname in "${all_files[@]}"; do
    old_file="$OLD_DIR/$fname"
    new_file="$NEW_DIR/$fname"
    pdb_id="${fname%%_*}"   # e.g. 9btz_trimmed_mhc.pdb -> 9btz

    if [[ -f "$old_file" && -f "$new_file" ]]; then
        old_hash=$(md5sum "$old_file" | cut -d' ' -f1)
        new_hash=$(md5sum "$new_file" | cut -d' ' -f1)

        if [[ "$old_hash" == "$new_hash" ]]; then
            status="unchanged"
            n_same=$((n_same + 1))
        else
            status="changed"
            n_changed=$((n_changed + 1))
        fi

    elif [[ -f "$old_file" ]]; then
        status="only_in_old"
        n_only_old=$((n_only_old + 1))

    else
        status="only_in_new"
        n_only_new=$((n_only_new + 1))
    fi

    echo "$pdb_id,$status,$old_file,$new_file" >> "$REPORT"

    if [[ "$status" != "unchanged" ]]; then
        echo "[$status] $pdb_id"
    fi
done

echo ""
echo "[SUMMARY]"
echo "unchanged:   $n_same"
echo "changed:     $n_changed"
echo "only_in_old: $n_only_old"
echo "only_in_new: $n_only_new"
echo ""
echo "Full report written to: $REPORT"
