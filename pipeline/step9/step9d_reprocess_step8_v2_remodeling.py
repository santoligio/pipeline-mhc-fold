#!/usr/bin/env python3
"""
NOVO: reprocessa cirurgicamente as estruturas afetadas pela SEGUNDA rodada de
remodelagem do step8 (2026-09-18, `compressed_files/MHCs-set1.tar.gz` +
`MHCs-set2.tar.gz`, pasta interna `backmapping_noH_OK_TERfixed/`), que
substituiu 25 dos 85 arquivos antigos de `step8/pdb/1_mhc_remodeled/` por
versoes novas (mais 1 estrutura nova, 8EC5, que nunca tinha sido remodelada
antes) -- os outros 60 arquivos remodelados sao byte-identicos aos antigos,
nao precisam de reprocessamento.

Confirmado antes de rodar:
  - Todos os TER espurios no meio de uma cadeia (24 casos flagados
    anteriormente) sumiram nos arquivos novos -- todo TER agora cai numa
    fronteira real entre cadeias.
  - 7Q15 agora tem a cadeia C (binder) que estava faltando na rodada
    anterior do step8 -- bate com step7_binders.csv (chain C, resseq 209).
  - 7OKV, 9OHZ e 9OHX NAO mudaram nessa rodada (arquivos identicos aos
    antigos) -- continuam com os problemas ja documentados/resolvidos por
    outra via (overrides em modified_pdbs/ pros dois primeiros; 9OHX
    permanece propositalmente sem fix, instrucao explicita do usuario
    2026-09-10 de nao mexer no EKG).

IDS processados aqui (24): os 24 TER-fragmentados antigos + 7Q15, MENOS
9C96 (que foi removido fisicamente do dataset em 2026-09-04 por motivo nao
relacionado -- binder com muitos gaps -- e nao deve ser reintroduzido so
porque o step8 forneceu um arquivo remodelado novo para ele).

8EC5 (a outra estrutura "nova" nessa rodada de remodelagem) foi EXCLUIDA
daqui -- nao existe em NENHUM outro lugar do pipeline (step2/step7/step9),
ou seja, nunca fez parte do dataset da v6. Nao da pra so "reprocessar o
step9" pra ela; precisaria entrar pelo pipeline inteiro (step1+) ou ser
investigada primeiro por que a pessoa que remodelou incluiu ela. Pendente
de decisao do usuario, nao tratado aqui.

Nenhuma dessas 25 tem entrada em modified_pdbs/ (prioridade 1 em
resolve_pdb_path) -- confirmado antes de rodar -- entao process_one() vai
pegar o arquivo novo de step8/pdb/1_mhc_remodeled/ automaticamente
(prioridade 2).

Backups dos arquivos de saida sobrescritos em .backup_step8_v2_remodeling/.
"""

import os
import shutil
from pathlib import Path

import pandas as pd

# NOVO: FILTER_DIR configurável via env var, com fallback relativo ao script (portável entre máquinas).
FILTER_DIR = Path(os.environ.get("PIPELINE_DIR", str(Path(__file__).resolve().parent.parent)))

PDB_IDS = [
    "13BR", "1KCG", "3VWJ", "3VWK", "4EN3", "4L4V", "4PJE", "4PJH", "4PJI",
    "4PJX", "4V3E", "5WHK", "6ZKX", "7Q15", "7RYM", "7ZT9", "8C44",
    "8SGM", "9K2R", "9K2S", "9L4I", "9RU5", "9TJX", "9WBD",
]

MODIFIED_PDB_DIR = FILTER_DIR / "step9" / "pdb" / "modified_pdbs"


def load_module(name: str, path: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def backup(path: Path) -> None:
    if not path.is_file():
        return
    backup_dir = path.parent / ".backup_step8_v2_remodeling"
    backup_dir.mkdir(exist_ok=True)
    dest = backup_dir / path.name
    if not dest.is_file():
        shutil.copy2(path, dest)
        print(f"[BACKUP] {path} -> {dest}")


def main():
    step9 = load_module(
        "step9_separate_mhc_complexes",
        FILTER_DIR / "step9" / "step9_separate_mhc_complexes.py",
    )

    # Safety check: none of these 25 should have a modified_pdbs override
    # (that would silently take priority over the new step8 file).
    for pdb_id in PDB_IDS:
        prefix = pdb_id.lower()
        clashes = sorted(
            p for p in MODIFIED_PDB_DIR.glob("*.pdb")
            if p.stem.lower().startswith(prefix)
        )
        assert not clashes, (
            f"{pdb_id}: found modified_pdbs override {clashes} -- "
            f"would shadow the new step8 file, aborting"
        )

    binders = step9.load_chain_map(step9.BINDERS_CSV, "binders")
    ligands = step9.load_chain_map(step9.LIGANDS_CSV, "ligands")

    # Backup any existing output files for these IDs before overwriting.
    for out_dir in [
        step9.OUTPUT_MHC_COMPLEX,
        step9.OUTPUT_MHC_BINDER,
        step9.OUTPUT_MHC_ONLY,
        step9.OUTPUT_MHC_ALL,
        step9.OUTPUT_MHC_LIGANDS,
        step9.OUTPUT_BINDERS,
    ]:
        for pdb_id in PDB_IDS:
            for f in out_dir.glob(f"{pdb_id.lower()}*"):
                backup(f)

    all_rows = []
    processed = []
    skipped = []
    for pdb_id in PDB_IDS:
        step7_pdb_path = (
            FILTER_DIR / "step7" / "pdb" / "1_filtered_structures"
            / f"{pdb_id.lower()}_mhc_complex.pdb"
        )
        if not step7_pdb_path.is_file():
            print(f"[SKIP] {pdb_id}: no step7 source at {step7_pdb_path}")
            skipped.append(pdb_id)
            continue
        rows = step9.process_one(pdb_path=step7_pdb_path, binders=binders, ligands=ligands)
        if rows is None:
            print(f"[SKIP] {pdb_id}: process_one returned None")
            skipped.append(pdb_id)
            continue
        all_rows.extend(rows)
        processed.append(pdb_id)
        print(f"[OK] {pdb_id}: {len(rows)} flagged residue rows")

    # Merge into step9_residues.csv: drop old rows for processed IDs, add new ones.
    backup(step9.RESIDUES_CSV)
    cols = ["pdb_id", "resname", "resid", "type", "nonstandard", "in_ptm_dict"]
    old_df = pd.read_csv(step9.RESIDUES_CSV, keep_default_na=False, na_filter=False)
    affected_upper = set(processed)
    kept_df = old_df[~old_df["pdb_id"].astype(str).str.upper().isin(affected_upper)]
    new_df = pd.DataFrame(all_rows)[cols] if all_rows else pd.DataFrame(columns=cols)
    merged = (
        pd.concat([kept_df, new_df], ignore_index=True)
        .drop_duplicates()
        .sort_values(["pdb_id", "type", "resid", "resname"])
    )
    merged.to_csv(step9.RESIDUES_CSV, index=False)
    print(
        f"[PATCHED] step9_residues.csv: {len(old_df)} -> {len(merged)} rows "
        f"(removed {len(old_df) - len(kept_df)} old, added {len(new_df)} new)"
    )

    # Recopy to 6_mhc_binder for any of these that are binder=yes and already
    # have a complex file there (annotations/binder classification unchanged
    # by this reprocessing, only physical chain content changed).
    ann_df = pd.read_csv(step9.ANNOTATIONS_OUT_CSV)
    binder_yes = set(
        ann_df[ann_df["binder"].astype(str).str.lower() == "yes"]["pdb_id"]
        .astype(str).str.upper()
    )
    for pdb_id in processed:
        if pdb_id not in binder_yes:
            continue
        src = step9.OUTPUT_MHC_COMPLEX / f"{pdb_id.lower()}_mhc_complex.pdb"
        dst = step9.OUTPUT_MHC_BINDER / f"{pdb_id.lower()}_mhc_complex.pdb"
        if src.is_file():
            shutil.copy2(src, dst)
            print(f"[RECOPIED to 6_mhc_binder] {dst}")

    print(f"\n[DONE] Reprocessed {len(processed)}/{len(PDB_IDS)} PDBs.")
    if skipped:
        print(f"[WARNING] Skipped: {skipped}")


if __name__ == "__main__":
    main()
