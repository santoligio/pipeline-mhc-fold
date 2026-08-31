#!/usr/bin/env python3

from pathlib import Path
import sys

METRIC_DIR = Path(__file__).resolve().parent
HEATMAPS_DIR = METRIC_DIR.parent
sys.path.insert(0, str(HEATMAPS_DIR))

from pipeline_common import separate_by_analysis_group


if __name__ == "__main__":
    separate_by_analysis_group(METRIC_DIR, METRIC_DIR / "rmsd_classes")

