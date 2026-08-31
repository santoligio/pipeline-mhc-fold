#!/usr/bin/env python3
"""
Plot Foldseek score distributions for PDB or AFDB alignments.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Configuration
# =========================

PIPELINE_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = PIPELINE_DIR.parent.parent
VERSION_02_DIR = WORKSPACE_DIR / "version_02"
DATASET = "pdb"  # "pdb" or "afdb"

FOLDSEEK_ALN = {
    "pdb": VERSION_02_DIR / "pdb" / "dbs_pdb_aln",
    "afdb": VERSION_02_DIR / "alphafold" / "3mre_afdb_aln",
}

OUT_DIR = PIPELINE_DIR / "step0" / DATASET
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIGSIZE = (7, 5)

FOLDSEEK_COLUMNS = [
    "query", "target", "fident", "alnlen", "mismatch", "gapopen",
    "qstart", "qend", "tstart", "tend", "evalue", "bits",
    "alntmscore", "qtmscore", "ttmscore", "lddt", "lddtfull", "prob",
]


# =========================
# Plot helpers
# =========================

def load_foldseek_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, delimiter=r"\s+")
    df.columns = FOLDSEEK_COLUMNS
    df = df[df["evalue"] > 0].copy()
    df["log_evalue"] = np.log(df["evalue"])
    return df


def plot_tm_score(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)

    scatter = ax.scatter(
        df["qtmscore"], df["log_evalue"],
        c=df["prob"], cmap="viridis",
        s=1.0, alpha=0.3,
    )

    ax.set_xlabel("Query TM-score")
    ax.set_ylabel("log E-value")

    ax.axhline(y=np.log(0.1), color="gray", linestyle="--", alpha=0.5)
    ax.axhline(y=np.log(0.01), color="blue", linestyle="--")
    ax.axhline(y=np.log(0.001), color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Probability")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "tm_score.png", dpi=300)
    plt.close(fig)


def plot_alignment_length(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)

    scatter = ax.scatter(
        df["alnlen"], df["log_evalue"],
        c=df["prob"], cmap="viridis",
        s=1.0, alpha=0.3,
    )

    ax.set_xlabel("Alignment length")
    ax.set_ylabel("log E-value")

    ax.axhline(y=np.log(0.01), color="blue", linestyle="--")
    ax.axvline(x=90, color="green", linestyle="--")

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Probability")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "alignment_length.png", dpi=300)
    plt.close(fig)


# =========================
# Main
# =========================

def main() -> None:
    aln_path = FOLDSEEK_ALN[DATASET]
    df = load_foldseek_table(aln_path)

    plot_tm_score(df)
    plot_alignment_length(df)

    print(f"[DONE] {DATASET} plots written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
