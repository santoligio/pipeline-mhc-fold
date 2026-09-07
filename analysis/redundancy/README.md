# MHC-complex redundancy analysis

`analysis/redundancy` is the canonical directory for this analysis. It maps
redundancy only among components retained in the final
step9 complex: trimmed MHC, curated binders, and curated ligands. It never
aligns sequences and never inserts gaps. The scripts and configuration live
directly in this directory, while generated CSVs and JSON summaries are kept
separately in `outputs/`.

## Current default inputs

- `pipeline/step9/pdb/5_mhc_complex/*.pdb`: authoritative final structures.
- `pipeline/step9/pdb/modified_pdbs/step9_binders.csv`: final binder chains.
- `pipeline/step9/pdb/modified_pdbs/step9_ligands.csv`: final ligand chains,
  after manual chain separation when applicable.
- `pipeline/step5/pdb/pdb_assemblies_remapped.csv`: `tstart`/`tend` audit
  metadata. These coordinates are not applied again to step9 structures.
- `pipeline/step2/pdb/1_assemblies/*.cif`: `_chem_comp` definitions used to
  distinguish peptide and nonpeptide residues.
- `pipeline/step6/pdb/summaries/resname_map.csv`: restoration of long CCD IDs
  shortened to three characters by PDB-format output.
- `manual_overrides.csv`: small, documented exception table. An override is
  never silent; its use is written to `validation_issues.csv`.
- RCSB Data API: `rcsb_entry_info.resolution_combined` and experimental method.
  Results are persisted in `outputs/rcsb_resolution_cache.csv`.

All input paths can be replaced on the command line for the future assembly
dataset. The final-complex directory defines the universe of PDBs analyzed.
Assemblies without a corresponding final complex are counted in the summary
but never enter the catalog or redundancy grouping.

## Run

From `analysis/redundancy`:

```powershell
python run_pipeline.py
```

After the first online run, a reproducible run without RCSB access is:

```powershell
python run_pipeline.py --offline
```

For a new dataset, pass any changed paths explicitly, for example:

```powershell
python run_pipeline.py --complex-dir PATH --binders FILE.csv --ligands FILE.csv `
  --trim-ranges FILE.csv --cif-dir PATH --resname-map FILE.csv --output-dir PATH `
  --final-source-label exported_final
```

Specific structures can be excluded explicitly and reproducibly without
altering the input files. Repeat the option for multiple PDB IDs:

```powershell
python run_pipeline.py --exclude-pdb 7Q15 --exclude-pdb 9OHX
```

Requested exclusions are normalized, checked against the final-complex
directory, and recorded in `catalog_summary.json`.

Use `--refresh-resolution` to replace cached RCSB metadata.

## Matching rules

### Peptide components

Residues classified by `_chem_comp.type` as peptide-linking are converted into
one-residue sequence tokens. Standard amino acids use one-letter codes.
Modified peptide residues use their `_chem_comp.mon_nstd_parent_comp_id` when
that parent is a standard amino acid; otherwise a lossless bracketed token such
as `[XYZ]` is retained.

Canonical sequence fields contain the parent amino acid, but redundancy
sequence fields decorate every modified residue (for example `[MSE>M]`). Thus
a modified and an unmodified chain are not redundant even when their canonical
parent sequences are identical. Modification position is preserved naturally
inside the decorated sequence, including under the contiguous-containment rule.

For a documented covalent modification that contains both a polymer residue
and a chemical adduct, `manual_overrides.csv` may provide both a canonical
amino-acid token and a supplemental ligand descriptor. This preserves the
protein sequence and independently makes the chemical state part of the exact
ligand signature.

Nonpeptide molecules that merely share a final chain ID with an MHC or binder
are excluded from both the protein sequence and ligand signature. They remain
listed in `validation_issues.csv` for audit and do not differentiate complexes.

Two MHC sequences match only if they are exactly equal or if one token sequence
is a contiguous substring of the other. Binder chains use the same rule. A
complex with multiple binders is compared as a multiset: binder-chain order and
chain IDs are ignored, multiplicity is preserved, and a one-to-one perfect
matching between chains is required.

This is not local/global alignment: internal substitutions, deletions, or
insertions fail. Only terminal overhangs around an exact contiguous occurrence
are accepted.

### Ligands

The ligand signature is an exact multiset. Each peptide ligand chain contributes
its full canonical sequence. Each nonpeptide residue contributes its full CCD
ID. Multiplicity is preserved and chain IDs are ignored. Ligand containment is
not accepted: the complete ligand signature must be identical.

### Whole-complex redundancy

A pair is redundant only when all three conditions hold:

1. trimmed final MHC sequences match by exact/contiguous containment;
2. binder multisets admit a complete one-to-one exact/containment matching;
3. ligand signatures are exactly identical.

Because containment is not transitive, graph connected components are unsafe.
Groups here are a deterministic representative-anchored partition: the best
remaining candidate becomes representative and only structures with a direct,
verified redundancy edge to it join that group. `redundant_pairs.csv` retains
all edges, including valid edges that cross two assigned groups.

Representatives are ordered by:

1. greatest trimmed MHC peptide length;
2. lowest numeric RCSB resolution in ångström;
3. greatest combined peptide length across all binder chains;
4. lexicographically smallest PDB ID as deterministic final tie-breaker.

Missing resolution (for example NMR) ranks after available numeric resolution
when sequence length ties.

`total_peptide_length` remains in the catalog only as descriptive audit data.
The binder criterion is calculated separately from the binder sequences and
does not include the MHC or peptide ligands.

## Scripts and principal functions

- `redundancy_core.py` contains parsers and comparison primitives.
  `extract_pdb_residues()` reads the first PDB model and removes altloc atom
  duplication at residue level. `read_cif_loop()` and `read_chem_comps()` read
  CCD typing from mmCIF without Biopython. `contiguous_relation()` implements
  the gapless substring rule. `match_sequence_multisets()` performs the
  one-to-one binder-chain matching. `fetch_rcsb_metadata()` queries GraphQL in
  batches with retry and writes the local cache.
- `build_catalog.py` resolves final chain roles, restores long CCD IDs,
  classifies residues, builds component and whole-complex signatures, and
  records every warning/error/special treatment. `classify_residue()` applies
  CCD logic and explicit overrides. The final step9 chain A is authoritative;
  trim spans are checked rather than applied a second time.
- `map_redundancy.py` uses `relation_details()` to require MHC, binder, and
  ligand agreement, writes every redundant pair, builds representative-anchored
  groups, and produces the minimal removal index.
- `validate_outputs.py` independently rechecks sequences, binder matching,
  ligand equality, group sizes, direct member-to-representative edges,
  representative ranking, and disjointness between representatives/removals.
- `run_pipeline.py` executes the three stages in order and stops on any failure.
- `test_redundancy.py` tests parsing, modified-residue parent normalization,
  contiguous-only matching, and multiset matching.

## Outputs

All outputs below are written to `analysis/redundancy/outputs` by default.
The compact results, validation audit, and summaries are versioned. The two
large intermediate catalogs and the local RCSB cache are reproducible and are
therefore excluded by `.gitignore`.

- `component_sequences.csv`: auditable sequence/residue inventory by component
  and final chain.
- `complex_catalog.csv`: one signature and RCSB metadata row per PDB.
- `rcsb_resolution_cache.csv`: local API cache used to support offline reruns.
- `validation_issues.csv`: all automatic special-case records. `ERROR` excludes
  a PDB; `WARNING` requires review; `INFO` documents an accepted treatment.
- `redundant_pairs.csv`: all directly verified redundant pairs.
- `group_membership.csv`: all valid PDBs, including singleton groups, with the
  selected representative and ranking fields.
- `representatives.csv`: representatives only for groups with more than one PDB.
- `removal_index.csv`: downstream deletion index. It intentionally contains
  exactly `PDB_ID,redundancy_group` and only non-representatives.
- `*_summary.json`: machine-readable run and validation summaries.

## Current documented special handling

- `1KCG/CGL`: treated as a glutathionylated cysteine. It contributes canonical
  `C` to the MHC peptide sequence, remains `[CGL>C]` in the redundancy sequence,
  while `N:GSH` is added independently to the exact ligand signature. These
  actions are documented in
  `validation_issues.csv`.
- `6BJ2/PCA` and `9TJX/CGU`: canonicalized respectively to `Q` and `E`, while
  their modified identities and positions remain in the decorated binder
  redundancy sequences.
- `8SGM/AHF`, `8Y6X/LXU`, and `8ZOX/L2B`: restored respectively to the full CCD
  IDs `A1AHF`, `A1LXU`, and `A1L2B` using the existing step6 residue-name map.
- The 81 analyzed MHCs whose observed final length differs from the original
  `tstart`/`tend` span retain their final chain A. Every case is listed
  in `validation_issues.csv`.
- `7Q15` and `9OHX` are explicitly excluded from the committed analysis
  snapshot. The exclusions are recorded in `catalog_summary.json`.
- `9OHZ`: shortened final residue `CB9` is restored to `A1CB9` using the
  existing step6 residue-name map.
- `9O06`: altloc-prefixed map values `AA1B7N`/`BA1B7N` are normalized to the
  single five-character CCD ID `A1B7N`.

## Pending ligand-audit decisions

The committed snapshot preserves the ligand policy used in the validated run;
the following observations are documented but have not been silently applied:

- `1T7V/P6G` is crystallization PEG according to the associated publication.
- `6VQY` contains a separate free `ARG` attributed by the authors to the
  refolding buffer. `6VQZ` also contains a separate `ARG`, but the same origin
  is not stated explicitly for that entry.
- `1HSB` models only the resolved termini of a heterogeneous endogenous peptide
  population, rather than one defined complete ligand sequence.
- `7KGT` contains three observed peptide residues in the final coordinates,
  whereas the deposited ligand sequence is the nine-residue `RLNQLESKM`.

Any change to these signatures should be made as an explicit, auditable policy
and followed by a complete rerun.
