#!/usr/bin/env python3

from pathlib import Path
import sys

CLASSES_DIR = Path(__file__).resolve().parent
HEATMAPS_DIR = CLASSES_DIR.parents[1]
sys.path.insert(0, str(HEATMAPS_DIR))

from pipeline_common import representative_heatmap


if __name__ == "__main__":
    representative_heatmap(
        CLASSES_DIR,
        pair_filename="top1_tm_score_avg_vs_umap.csv",
        metric="tm_score_avg",
        matrix_filename="representative_tm_score_avg_matrix.csv",
        plot_filename="representative_tm_score_avg_clustermap.png",
        metric_label="TM-score",
        similarity=True,
    )
