#!/usr/bin/env python3
"""
NOVO: reconcilia cirurgicamente 3 estruturas onde o arquivo remodelado do
step8 (ferramenta externa, sem script no repo) diverge do conjunto de
cadeias binder/ligante ja classificado pelo step6/step7 -- reportado pela
Giovanna (7Q15, 7OKV) e achado numa varredura sistematica das 85 estruturas
remodeladas comparando cadeias step7 vs step8 (9OHX).

Casos (nenhum tem entrada em binder_removals.csv, ou seja, nenhuma das 3
divergencias e uma remocao manual intencional):

  - 7Q15: step7 tem cadeia C (binder, step7_binders.csv). O arquivo do
    step8 (7q15_modeled_loop.pdb) so tem cadeia A -- a cadeia C sumiu.
  - 9OHX: step7 tem cadeia C (ligante EKG, step7_ligands.csv). O arquivo
    do step8 nao tem essa cadeia C, mas tem uma cadeia E nova com um
    ligante DIFERENTE (R16) que nunca passou pela classificacao do
    step6/step7.
  - 7OKV: do step5 ao step7 so existe cadeia A (MHC), nunca houve
    ligante/binder registrado. O arquivo do step8 introduziu uma cadeia B
    nova com ligante R16.

R16 esta na blacklist do step6 (step6/residue_removal_list.csv) -- aditivo
de cristalizacao, nao ligante real. Como o step8 roda fora do pipeline
(depois do ponto onde o filtro de blacklist age), essas atomos de R16
nunca foram filtrados. EKG e um ligante real, corretamente classificado
pelo step7 -- so foi perdido porque o step8 nao preservou essa cadeia ao
remodelar.

Fix: para cada uma das 3, reconstroi um "modified_pdbs/<pdb>_mhc_complex.pdb"
= cadeia A do step8 (MHC remodelada) + cadeias nao-A do step7 que estavam
registradas em step7_binders.csv/step7_ligands.csv (nunca as cadeias
novas/nao registradas do step8, ex: R16). Isso restaura o binder/ligante
perdido e garante que nenhum residuo de blacklist introduzido pelo step8
sobreviva no complexo final. step7_binders.csv/step7_ligands.csv/
step9_annotations.csv ja estavam corretos (baseados no step7); nenhuma
dessas CSVs precisa de patch, so o conteudo fisico do PDB final.

Depois disso, reprocessa step9_separate_mhc_complexes.process_one() para as
3 (resolve_pdb_path vai pegar o modified_pdbs regenerado com prioridade),
regenerando 5_mhc_complex/6_mhc_binder/1_mhc_only/2_binders/3_mhc_all/
4_mhc_ligands + merge em step9_residues.csv.

Backups dos arquivos sobrescritos em .backup_step8_mismatch_fix/.
"""

import importlib.util
import os
import shutil
from pathlib import Path

from Bio.PDB import PDBIO, PDBParser

# NOVO: FILTER_DIR configurável via env var, com fallback relativo ao script (portável entre máquinas).
FILTER_DIR = Path(os.environ.get("PIPELINE_DIR", str(Path(__file__).resolve().parent.parent)))

# pdb_id -> (chains to keep from step7 filtered structure, in addition to
# step8's chain A)
# NOVO: 2026-09-03 - user reverted the 7Q15/9OHX fixes (those splice a real
# binder/ligand chain that step8 lost entirely from step7 into the step8
# file - a merge of biological content across pipeline stages, which the
# user does not want done automatically). 7OKV (pure removal of an
# unregistered blacklisted chain) and 9OHZ (same chain/coords, only
# resname+occupancy corrected) were confirmed OK to keep applied. Do not
# reintroduce 7Q15/9OHX here without explicit user confirmation.
CASES = {
    "7OKV": {
        "keep_from_step7": [],
        "reason": "step8_modeled_loop introduced an unregistered chain B with blacklisted residue R16 (never present in step5-step7); dropped, matches step7 (chain A only)",
    },
    "9OHZ": {
        "keep_from_step7": ["C"],
        "reason": "step8_modeled_loop kept chain C (same coords/resseq) but genericized the ligand resname CB9 (A1CB9 CCD) to a placeholder 'LIG' and zeroed occupancy; restored the correctly-named/occupancy chain C from step7 (chain set was already correct, only resname/occupancy were corrupted)",
    },
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def backup(path: Path) -> None:
    if not path.is_file():
        return
    backup_dir = path.parent / ".backup_step8_mismatch_fix"
    backup_dir.mkdir(exist_ok=True)
    dest = backup_dir / path.name
    if not dest.is_file():
        shutil.copy2(path, dest)
        print(f"[BACKUP] {path} -> {dest}")


def main():
    import pandas as pd

    step9 = load_module("step9_separate_mhc_complexes", FILTER_DIR / "step9" / "step9_separate_mhc_complexes.py")

    modified_pdb_dir = FILTER_DIR / "step9" / "pdb" / "modified_pdbs"
    modification_log = modified_pdb_dir / "modified_pdbs_log.csv"

    parser = PDBParser(QUIET=True)
    log_rows = []

    for pdb_id, case in CASES.items():
        pdb_lower = pdb_id.lower()
        print(f"\n=== {pdb_id} ===")

        step8_path = FILTER_DIR / "step8" / "pdb" / "1_mhc_remodeled" / f"{pdb_lower}_modeled_loop.pdb"
        step7_path = FILTER_DIR / "step7" / "pdb" / "1_filtered_structures" / f"{pdb_lower}_mhc_complex.pdb"
        assert step8_path.is_file(), f"nao encontrado: {step8_path}"
        assert step7_path.is_file(), f"nao encontrado: {step7_path}"

        step8_structure = parser.get_structure(pdb_id, str(step8_path))
        step8_model = step8_structure[0]

        step8_chains_before = sorted(c.id for c in step8_model)
        for chain in list(step8_model):
            if chain.id != "A":
                step8_model.detach_child(chain.id)
        print(f"[STEP8] chains before={step8_chains_before}, kept=['A']")

        if case["keep_from_step7"]:
            step7_structure = parser.get_structure(pdb_id, str(step7_path))
            step7_model = step7_structure[0]
            for chain_id in case["keep_from_step7"]:
                assert chain_id in step7_model, f"{pdb_id}: cadeia {chain_id} nao encontrada no step7"
                chain = step7_model[chain_id]
                step7_model.detach_child(chain_id)
                step8_model.add(chain)
                print(f"[RESTORED] {pdb_id}: chain {chain_id} from step7")

        final_chains = sorted(c.id for c in step8_model)
        print(f"[FINAL] chains={final_chains}")

        out_path = modified_pdb_dir / f"{pdb_lower}_mhc_complex.pdb"
        backup(out_path)

        io = PDBIO()
        io.set_structure(step8_structure)
        io.save(str(out_path))
        print(f"[SAVED] {out_path}")

        log_rows.append({
            "timestamp": pd.Timestamp.now().isoformat(timespec="seconds"),
            "pdb_id": pdb_id,
            "removed_chain_id": "",
            "reason": case["reason"],
            "source": "step8+step7_reconciled",
            "status": "reconciled",
        })

    modified_pdb_dir.mkdir(parents=True, exist_ok=True)
    write_header = not modification_log.is_file()
    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(modification_log, mode="a", header=write_header, index=False)
    print(f"\n[LOGGED] {modification_log}: +{len(log_df)} linhas")

    # ------------------------------------------------------------
    # Reprocess step9 for the 3 affected PDBs
    # ------------------------------------------------------------
    print("\n=== step9 reprocess ===")
    binders = step9.load_chain_map(step9.BINDERS_CSV, "binders")
    ligands = step9.load_chain_map(step9.LIGANDS_CSV, "ligands")

    for out_dir in [
        step9.OUTPUT_MHC_COMPLEX,
        step9.OUTPUT_MHC_BINDER,
        step9.OUTPUT_MHC_ONLY,
        step9.OUTPUT_MHC_ALL,
        step9.OUTPUT_MHC_LIGANDS,
        step9.OUTPUT_BINDERS,
    ]:
        for pdb_id in CASES:
            for f in out_dir.glob(f"{pdb_id.lower()}*"):
                backup(f)

    all_rows = []
    for pdb_id in CASES:
        step7_pdb_path = FILTER_DIR / "step7" / "pdb" / "1_filtered_structures" / f"{pdb_id.lower()}_mhc_complex.pdb"
        rows = step9.process_one(pdb_path=step7_pdb_path, binders=binders, ligands=ligands)
        assert rows is not None, f"{pdb_id}: process_one retornou None"
        all_rows.extend(rows)
        print(f"[OK] {pdb_id}: {len(rows)} linhas de residuo flagado")

    backup(step9.RESIDUES_CSV)
    cols = ["pdb_id", "resname", "resid", "type", "nonstandard", "in_ptm_dict"]
    old_df = pd.read_csv(step9.RESIDUES_CSV, keep_default_na=False, na_filter=False)
    affected_upper = set(CASES)
    kept_df = old_df[~old_df["pdb_id"].astype(str).str.upper().isin(affected_upper)]
    new_df = pd.DataFrame(all_rows)[cols] if all_rows else pd.DataFrame(columns=cols)
    merged = (
        pd.concat([kept_df, new_df], ignore_index=True)
        .drop_duplicates()
        .sort_values(["pdb_id", "type", "resid", "resname"])
    )
    merged.to_csv(step9.RESIDUES_CSV, index=False)
    print(f"[PATCHED] step9_residues.csv: {len(old_df)} -> {len(merged)} linhas "
          f"(removidas {len(old_df) - len(kept_df)} antigas, adicionadas {len(new_df)} novas)")

    for pdb_id in CASES:
        binder_dst = step9.OUTPUT_MHC_BINDER / f"{pdb_id.lower()}_mhc_complex.pdb"
        binder_src = step9.OUTPUT_MHC_COMPLEX / f"{pdb_id.lower()}_mhc_complex.pdb"
        if binder_dst.is_file():
            shutil.copy2(binder_src, binder_dst)
            print(f"[RECOPIED] {binder_dst}")

    print(f"\n[DONE] Reconciliacao step8/step7 concluida para {', '.join(CASES)}.")


if __name__ == "__main__":
    main()
