# Pipeline de heatmaps MHC

Os pipelines `rmsd` e `tm-score` usam os mesmos dados TM-align, mas selecionam os representantes com criterios opostos:

- RMSD: menor media.
- TM-score: maior media.

As classes `HLA Class Ia`, `HLA Class Ib`, `CD1` e `MIC` sao subdivididas por `gene_name`. Grupos com uma unica estrutura PDB recebem essa propria estrutura como representante (RMSD 0 ou TM-score 1, sem pares).

Estruturas AFDB participam do UMAP e dos calculos medios, mas nao podem ser
selecionadas como representantes. Estruturas anotadas como `SC` ou `chimera`
tambem sao inelegiveis. Grupos sem candidato valido sao registrados em
`groups_without_representative_rmsd.csv` ou
`groups_without_representative_tm_score.csv`.

## Ordem de execucao

Para cada metrica:

1. `python separate_in_mapped_class.py`
2. `python <classes>/mean_rmsd_by_class.py`
3. `python <classes>/correlation_pca_rmsd.py`
4. `python <classes>/heatmap_w_clustering.py`
5. `python <classes>/plot_pca_w_representatives.py`

Os scripts podem ser chamados de qualquer diretorio. Novos arquivos sao organizados em:

- `<classes>/pairs/`: pares intragrupo e `class_members.csv`.
- `<classes>/results/`: representantes, tabelas, correlacoes e figuras.

Os arquivos consolidados de representantes sao distintos por metrica:

- RMSD: `top1_representatives_rmsd.csv`.
- TM-score: `top1_representatives_tm_score.csv`.

Figuras e tabelas destinadas a compartilhamento tambem incluem a metrica no
nome. Por exemplo: `umap_representatives_rmsd.png` e
`umap_representatives_tm_score.png`.

`pipeline_common.py` concentra validacao, nomes de grupos e logica compartilhada pelas duas metricas.

Os resultados da versao anterior foram preservados em
`<classes>/legacy_outputs_pre_gene_split/`. Os scripts duplicados antigos foram
preservados em `legacy_scripts/`.
