#!/usr/bin/env python3

from pathlib import Path
import sys

CLASSES_DIR = Path(__file__).resolve().parent
METRIC_DIR = CLASSES_DIR.parent
HEATMAPS_DIR = METRIC_DIR.parent
sys.path.insert(0, str(HEATMAPS_DIR))

from pipeline_common import representative_pairs


if __name__ == "__main__":
    representative_pairs(
        CLASSES_DIR,
        METRIC_DIR,
        metric="tm_score_avg",
        output_name="top1_tm_score_avg_vs_umap.csv",
        plot_name="top1_tm_score_avg_vs_umap.png",
        metric_label="TM-score",
    )

