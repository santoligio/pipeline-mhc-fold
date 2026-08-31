#!/usr/bin/env python3

from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr, spearmanr


PAIR_FILE = "tmalign_result_db_mhc_only_afdb_reviewed_new.csv"
UMAP_FILE = "umap_coordinates_precomputed_1_minus_tm.csv"
GENE_SPLIT_CLASSES = {"HLA Class Ia", "HLA Class Ib", "CD1", "MIC"}


def metric_output_suffix(metric):
    suffixes = {"rmsd": "rmsd", "tm_score_avg": "tm_score"}
    try:
        return suffixes[metric]
    except KeyError as exc:
        raise ValueError(f"Metrica sem sufixo de saida: {metric}") from exc


def representatives_filename(metric):
    return f"top1_representatives_{metric_output_suffix(metric)}.csv"


def require_columns(df, columns, source):
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"{source}: colunas ausentes: {', '.join(missing)}")


def normalize_ids(series):
    return series.astype(str).str.strip().str.upper()


def safe_name(value):
    name = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")
    if not name:
        raise ValueError(f"Nome de grupo invalido: {value!r}")
    return name


def add_analysis_group(metadata):
    metadata = metadata.copy()
    require_columns(metadata, {"pdb_id", "mapped_class", "gene_name"}, "metadados UMAP")

    metadata["pdb_id"] = normalize_ids(metadata["pdb_id"])
    metadata["mapped_class"] = metadata["mapped_class"].astype("string").str.strip()
    metadata["gene_name"] = metadata["gene_name"].astype("string").str.strip()

    if metadata["pdb_id"].duplicated().any():
        duplicated = metadata.loc[metadata["pdb_id"].duplicated(), "pdb_id"].tolist()
        raise ValueError(f"PDBs duplicados nos metadados: {duplicated[:5]}")
    if metadata["mapped_class"].isna().any():
        raise ValueError("Existem estruturas sem mapped_class nos metadados")

    split_mask = metadata["mapped_class"].isin(GENE_SPLIT_CLASSES)
    if metadata.loc[split_mask, "gene_name"].isna().any():
        missing = metadata.loc[split_mask & metadata["gene_name"].isna(), "pdb_id"].tolist()
        raise ValueError(f"Estruturas de classes subdivididas sem gene_name: {missing[:5]}")

    metadata["analysis_group"] = metadata["mapped_class"]
    metadata.loc[split_mask, "analysis_group"] = (
        metadata.loc[split_mask, "mapped_class"]
        + " / "
        + metadata.loc[split_mask, "gene_name"]
    )
    return metadata


def read_metadata(metric_dir):
    return add_analysis_group(pd.read_csv(metric_dir / UMAP_FILE))


def structure_id_from_filename(value):
    stem = Path(str(value)).stem
    for suffix in ("_mhc_groove", "_mhc"):
        if stem.lower().endswith(suffix):
            return stem[: -len(suffix)].upper()
    return stem.upper()


def add_pair_ids(pairs):
    pairs = pairs.copy()
    require_columns(pairs, {"file1", "file2"}, "resultado TM-align")
    pairs["pdb1"] = normalize_ids(pairs["file1"].map(structure_id_from_filename))
    pairs["pdb2"] = normalize_ids(pairs["file2"].map(structure_id_from_filename))
    return pairs


def separate_by_analysis_group(metric_dir, classes_dir):
    metadata = read_metadata(metric_dir)
    pairs = add_pair_ids(pd.read_csv(metric_dir / PAIR_FILE))

    group_map = metadata.set_index("pdb_id")["analysis_group"]
    pairs["analysis_group1"] = pairs["pdb1"].map(group_map)
    pairs["analysis_group2"] = pairs["pdb2"].map(group_map)

    unmapped = pairs["analysis_group1"].isna() | pairs["analysis_group2"].isna()
    if unmapped.any():
        raise ValueError(f"{int(unmapped.sum())} pares possuem PDB sem metadados")

    same_group = pairs[pairs["analysis_group1"] == pairs["analysis_group2"]].copy()
    pairs_dir = classes_dir / "pairs"
    results_dir = classes_dir / "results"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    filenames = {}
    manifest_parts = []
    for group, members in metadata.groupby("analysis_group", sort=True):
        filename = f"{safe_name(group)}.csv"
        if filename in filenames and filenames[filename] != group:
            raise ValueError(f"Colisao de nomes entre {filenames[filename]!r} e {group!r}")
        filenames[filename] = group

        group_pairs = same_group[same_group["analysis_group1"] == group]
        group_pairs.to_csv(pairs_dir / filename, index=False)

        member_rows = members[
            ["pdb_id", "source", "mapped_class", "gene_name", "analysis_group"]
        ].copy()
        member_rows["class_file"] = filename
        member_rows["n_structures"] = len(member_rows)
        manifest_parts.append(member_rows)
        print(f"{group}: {len(member_rows)} estruturas, {len(group_pairs)} pares")

    manifest = pd.concat(manifest_parts, ignore_index=True)
    manifest.to_csv(pairs_dir / "class_members.csv", index=False)
    print(f"\nManifesto salvo: {pairs_dir / 'class_members.csv'}")


def select_representatives(classes_dir, metric, higher_is_better, singleton_value):
    pairs_dir = classes_dir / "pairs"
    results_dir = classes_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(pairs_dir / "class_members.csv")
    require_columns(
        manifest,
        {
            "pdb_id",
            "source",
            "mapped_class",
            "gene_name",
            "analysis_group",
            "class_file",
            "n_structures",
        },
        "class_members.csv",
    )
    manifest["pdb_id"] = normalize_ids(manifest["pdb_id"])

    mean_column = f"mean_{metric}"
    output_suffix = metric_output_suffix(metric)
    representatives = []
    skipped_groups = []

    for group, members in manifest.groupby("analysis_group", sort=True):
        members = members.copy()
        member_ids = set(members["pdb_id"])
        class_file = members["class_file"].iloc[0]
        n_structures = len(member_ids)

        if n_structures == 1:
            pdb = next(iter(member_ids))
            summary = pd.DataFrame({"pdb": [pdb], "n_pairs": [0], mean_column: [singleton_value]})
        else:
            class_pairs = add_pair_ids(pd.read_csv(pairs_dir / class_file))
            require_columns(class_pairs, {metric}, class_file)
            if class_pairs[metric].isna().any():
                raise ValueError(f"{class_file}: valores ausentes em {metric}")

            observed_pairs = {
                tuple(sorted(pair))
                for pair in class_pairs[["pdb1", "pdb2"]].itertuples(index=False, name=None)
            }
            expected_pairs = n_structures * (n_structures - 1) // 2
            if len(observed_pairs) != expected_pairs:
                raise ValueError(
                    f"{class_file}: esperados {expected_pairs} pares unicos, encontrados {len(observed_pairs)}"
                )
            if len(class_pairs) != expected_pairs:
                raise ValueError(f"{class_file}: existem pares duplicados")

            endpoints = pd.concat(
                [
                    class_pairs[["pdb1", metric]].rename(columns={"pdb1": "pdb"}),
                    class_pairs[["pdb2", metric]].rename(columns={"pdb2": "pdb"}),
                ],
                ignore_index=True,
            )
            summary = (
                endpoints.groupby("pdb", as_index=False)[metric]
                .agg(["count", "mean"])
                .reset_index()
                .rename(columns={"count": "n_pairs", "mean": mean_column})
            )
            if set(summary["pdb"]) != member_ids:
                raise ValueError(f"{class_file}: membros do manifesto e dos pares nao coincidem")

        source_map = members.set_index("pdb_id")["source"].astype(str).str.lower()
        class_map = members.set_index("pdb_id")["mapped_class"].astype(str)
        summary["source"] = summary["pdb"].map(source_map)
        summary["mapped_class"] = summary["pdb"].map(class_map)
        disallowed_annotation = summary["mapped_class"].str.contains(
            r"(?:^|[ _])SC(?:$|[ _])|chimera",
            case=False,
            regex=True,
            na=False,
        )
        summary["eligible_representative"] = ~(
            summary["source"].isin({"af", "afdb", "alphafold"})
            | summary["pdb"].str.startswith("AF-")
            | disallowed_annotation
        )
        summary = summary.sort_values(mean_column, ascending=not higher_is_better)
        eligible = summary[summary["eligible_representative"]].copy()
        stem = Path(class_file).stem
        summary.to_csv(results_dir / f"{stem}_{mean_column}_per_pdb.csv", index=False)

        unit = " A" if metric == "rmsd" else ""
        direction = "menor" if not higher_is_better else "maior"
        with (
            results_dir / f"{stem}_top10_representatives_{output_suffix}.txt"
        ).open("w") as handle:
            handle.write(f"Top 10 estruturas com {direction} {metric} medio\n")
            handle.write("=" * 48 + "\n\n")
            for rank, row in enumerate(eligible.head(10).itertuples(), start=1):
                handle.write(f"{rank}. {row.pdb} {mean_column}={getattr(row, mean_column):.4f}{unit}\n")

        if eligible.empty:
            skipped_groups.append(
                {
                    "analysis_group": group,
                    "reason": "grupo sem estrutura elegivel; AFDB, SC e chimera nao podem representar",
                    "n_structures": n_structures,
                }
            )
            print(f"{group}: sem representante PDB elegivel")
            continue

        top = eligible.iloc[0]
        source_class = members["mapped_class"].iloc[0]
        genes = sorted(members["gene_name"].dropna().astype(str).unique())
        representatives.append(
            {
                "analysis_group": group,
                "mapped_class": source_class,
                "gene_name": genes[0] if source_class in GENE_SPLIT_CLASSES else "",
                "pdb": top["pdb"],
                mean_column: top[mean_column],
                "n_pairs": int(top["n_pairs"]),
                "n_structures": n_structures,
            }
        )
        print(f"{group}: {top['pdb']} ({mean_column}={top[mean_column]:.4f})")

    top1 = pd.DataFrame(representatives).sort_values(mean_column, ascending=not higher_is_better)
    top1_filename = representatives_filename(metric)
    top1.to_csv(results_dir / top1_filename, index=False)
    pd.DataFrame(
        skipped_groups,
        columns=["analysis_group", "reason", "n_structures"],
    ).to_csv(
        results_dir / f"groups_without_representative_{output_suffix}.csv",
        index=False,
    )
    print(f"\nRepresentantes salvos: {results_dir / top1_filename}")


def representative_pairs(classes_dir, metric_dir, metric, output_name, plot_name, metric_label):
    results_dir = classes_dir / "results"
    top1_filename = representatives_filename(metric)
    top1 = pd.read_csv(results_dir / top1_filename)
    require_columns(top1, {"pdb", "analysis_group"}, top1_filename)
    top1["pdb"] = normalize_ids(top1["pdb"])
    representative_ids = set(top1["pdb"])

    umap = pd.read_csv(metric_dir / UMAP_FILE)
    require_columns(umap, {"pdb_id", "UMAP1", "UMAP2"}, UMAP_FILE)
    umap["pdb_id"] = normalize_ids(umap["pdb_id"])
    coords = umap.set_index("pdb_id")[["UMAP1", "UMAP2"]]
    missing_coords = sorted(representative_ids - set(coords.index))
    if missing_coords:
        raise ValueError(f"Representantes sem coordenadas UMAP: {missing_coords}")

    pairs = add_pair_ids(pd.read_csv(metric_dir / PAIR_FILE))
    require_columns(pairs, {metric}, PAIR_FILE)
    pairs = pairs[pairs["pdb1"].isin(representative_ids) & pairs["pdb2"].isin(representative_ids)].copy()
    expected = len(representative_ids) * (len(representative_ids) - 1) // 2
    if len(pairs) != expected:
        raise ValueError(f"Esperados {expected} pares entre representantes, encontrados {len(pairs)}")

    xy1 = coords.loc[pairs["pdb1"]].to_numpy()
    xy2 = coords.loc[pairs["pdb2"]].to_numpy()
    pairs["dist_umap"] = np.sqrt(np.square(xy1 - xy2).sum(axis=1))
    pairs.to_csv(results_dir / output_name, index=False)

    pearson_r, pearson_p = pearsonr(pairs["dist_umap"], pairs[metric])
    spearman_rho, spearman_p = spearmanr(pairs["dist_umap"], pairs[metric])
    print(f"Pares entre representantes: {len(pairs)}")
    print(f"Pearson r={pearson_r:.4f}, p={pearson_p:.4e}")
    print(f"Spearman rho={spearman_rho:.4f}, p={spearman_p:.4e}")

    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.regplot(
        data=pairs,
        x="dist_umap",
        y=metric,
        scatter_kws={"s": 50, "alpha": 0.7},
        line_kws={"color": "red"},
        ax=ax,
    )
    ax.set_xlabel("Distancia UMAP")
    ax.set_ylabel(metric_label)
    ax.set_title(f"Representantes dos grupos\n{metric_label} vs distancia UMAP")
    ax.text(
        0.05,
        0.95,
        f"Pearson r = {pearson_r:.3f}\nSpearman rho = {spearman_rho:.3f}",
        transform=ax.transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.8},
    )
    fig.tight_layout()
    fig.savefig(results_dir / plot_name, dpi=300)
    plt.close(fig)


def representative_heatmap(
    classes_dir,
    pair_filename,
    metric,
    matrix_filename,
    plot_filename,
    metric_label,
    similarity=False,
    exclude=None,
):
    results_dir = classes_dir / "results"
    pairs = add_pair_ids(pd.read_csv(results_dir / pair_filename))
    top1_filename = representatives_filename(metric)
    top1 = pd.read_csv(results_dir / top1_filename)
    require_columns(top1, {"pdb", "analysis_group"}, top1_filename)
    top1["pdb"] = normalize_ids(top1["pdb"])
    group_map = top1.set_index("pdb")["analysis_group"]
    pairs["analysis_group1"] = pairs["pdb1"].map(group_map)
    pairs["analysis_group2"] = pairs["pdb2"].map(group_map)

    excluded = set(exclude or [])
    groups = sorted(set(top1.loc[~top1["analysis_group"].isin(excluded), "analysis_group"]))
    pairs = pairs[
        pairs["analysis_group1"].isin(groups) & pairs["analysis_group2"].isin(groups)
    ].copy()
    matrix = pd.DataFrame(np.nan, index=groups, columns=groups)
    np.fill_diagonal(matrix.values, 1.0 if similarity else 0.0)

    for row in pairs.itertuples():
        value = getattr(row, metric)
        matrix.loc[row.analysis_group1, row.analysis_group2] = value
        matrix.loc[row.analysis_group2, row.analysis_group1] = value
    if matrix.isna().any().any():
        missing = int(matrix.isna().sum().sum() / 2)
        raise ValueError(f"Matriz incompleta: {missing} pares ausentes")
    matrix.to_csv(results_dir / matrix_filename)

    distance_matrix = 1.0 - matrix if similarity else matrix
    condensed = squareform(distance_matrix.to_numpy())
    cluster_linkage = linkage(condensed, method="average")

    import matplotlib.pyplot as plt
    import seaborn as sns

    kwargs = {"vmin": 0.5, "vmax": 1.0} if similarity else {}
    grid = sns.clustermap(
        matrix,
        row_linkage=cluster_linkage,
        col_linkage=cluster_linkage,
        cmap="mako_r",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        figsize=(14, 14),
        cbar_kws={"label": metric_label},
        **kwargs,
    )
    grid.fig.suptitle("Similaridade estrutural entre grupos representativos", y=1.02, fontsize=16)
    grid.ax_heatmap.set_xlabel("")
    grid.ax_heatmap.set_ylabel("")
    grid.savefig(results_dir / plot_filename, dpi=300, bbox_inches="tight")
    plt.close(grid.fig)


def plot_umap_representatives(classes_dir, metric_dir):
    results_dir = classes_dir / "results"
    umap = read_metadata(metric_dir)
    metric = "tm_score_avg" if metric_dir.name == "tm-score" else "rmsd"
    output_suffix = metric_output_suffix(metric)
    metric_title = "TM-score" if metric == "tm_score_avg" else "RMSD"
    top1_filename = representatives_filename(metric)
    top1 = pd.read_csv(results_dir / top1_filename)
    require_columns(top1, {"pdb", "analysis_group"}, top1_filename)
    top1["pdb"] = normalize_ids(top1["pdb"])

    representative_map = top1.set_index("pdb")["analysis_group"]
    umap["representative_group"] = umap["pdb_id"].map(representative_map)
    representatives = umap[umap["representative_group"].notna()].copy()
    missing = sorted(set(top1["pdb"]) - set(representatives["pdb_id"]))
    if missing:
        raise ValueError(f"Representantes ausentes no UMAP: {missing}")

    import matplotlib.pyplot as plt
    import plotly.express as px
    import plotly.graph_objects as go
    import seaborn as sns
    from matplotlib.lines import Line2D

    groups = sorted(umap["analysis_group"].dropna().unique())
    palette = sns.color_palette("tab20", len(groups))
    color_map = dict(zip(groups, palette))
    group_sizes = umap.groupby("analysis_group").size().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(24, 12))
    for group in group_sizes.index:
        subset = umap[umap["analysis_group"] == group]
        ax.scatter(subset["UMAP1"], subset["UMAP2"], s=40, alpha=0.5, color=color_map[group], zorder=1)
    ax.scatter(
        representatives["UMAP1"],
        representatives["UMAP2"],
        marker="*",
        s=120,
        color="gold",
        edgecolors="black",
        linewidths=0.8,
        zorder=1000,
    )
    for row in representatives.itertuples():
        ax.annotate(row.representative_group, (row.UMAP1, row.UMAP2), xytext=(5, 5), textcoords="offset points", fontsize=8)

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map[group], markersize=7, label=group)
        for group in groups
    ]
    legend = ax.legend(handles=handles, title="analysis_group", bbox_to_anchor=(1.0, 1), loc="upper left")
    ax.add_artist(legend)
    ax.legend(
        handles=[Line2D([0], [0], marker="*", color="gold", markeredgecolor="black", linewidth=0, markersize=10, label="Representative")],
        loc="lower right",
    )
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title(
        f"UMAP de estruturas MHC-like\nRepresentantes selecionados por {metric_title}"
    )
    fig.savefig(
        results_dir / f"umap_representatives_{output_suffix}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    plotly_colors = px.colors.qualitative.Dark24
    plotly_color_map = {group: plotly_colors[i % len(plotly_colors)] for i, group in enumerate(groups)}
    interactive = go.Figure()
    for group in group_sizes.index:
        subset = umap[umap["analysis_group"] == group]
        interactive.add_trace(
            go.Scatter(
                x=subset["UMAP1"],
                y=subset["UMAP2"],
                mode="markers",
                name=group,
                marker={"size": 6, "color": plotly_color_map[group], "opacity": 0.6},
                customdata=subset[["pdb_id", "gene_name", "mapped_class", "mapped_superclass"]].values,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>Gene: %{customdata[1]}<br>"
                    "Class: %{customdata[2]}<br>Superclass: %{customdata[3]}<br>"
                    "UMAP1=%{x:.3f}<br>UMAP2=%{y:.3f}<extra></extra>"
                ),
            )
        )
    interactive.add_trace(
        go.Scatter(
            x=representatives["UMAP1"],
            y=representatives["UMAP2"],
            mode="markers+text",
            text=representatives["representative_group"],
            textposition="top center",
            name="Representative",
            marker={"symbol": "star", "size": 10, "color": "gold", "line": {"color": "black", "width": 1}},
            customdata=representatives[["pdb_id", "representative_group"]].values,
            hovertemplate="<b>%{customdata[0]}</b><br>Representative of %{customdata[1]}<extra></extra>",
        )
    )
    interactive.update_layout(
        width=1400,
        height=1000,
        legend_title="analysis_group",
        title=(
            "UMAP de estruturas MHC-like<br>"
            f"Representantes selecionados por {metric_title}"
        ),
    )
    interactive.write_html(
        results_dir / f"umap_representatives_{output_suffix}.html"
    )
