#!/usr/bin/env python3

from pathlib import Path
import sys

CLASSES_DIR = Path(__file__).resolve().parent
HEATMAPS_DIR = CLASSES_DIR.parents[1]
sys.path.insert(0, str(HEATMAPS_DIR))

from pipeline_common import select_representatives


if __name__ == "__main__":
    select_representatives(
        CLASSES_DIR,
        metric="tm_score_avg",
        higher_is_better=True,
        singleton_value=1.0,
    )

