#!/usr/bin/env python3

from pathlib import Path
import sys

CLASSES_DIR = Path(__file__).resolve().parent
METRIC_DIR = CLASSES_DIR.parent
HEATMAPS_DIR = METRIC_DIR.parent
sys.path.insert(0, str(HEATMAPS_DIR))

from pipeline_common import plot_umap_representatives


if __name__ == "__main__":
    plot_umap_representatives(CLASSES_DIR, METRIC_DIR)

