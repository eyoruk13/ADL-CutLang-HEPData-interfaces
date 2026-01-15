"""
python3 event_correlation_heatmap.py SR1.csv SR2.csv SR3.csv SR4.csv SR5.csv etc. --output heatmap.png --matrix-out matrix.csv
"""

import argparse
from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def build_event_matrix(csv_files, has_header=False):
    """
    Build a binary event matrix from a list of CSV files.

    Rows    -> file names
    Columns -> event1 ... eventN, where N is the maximum event id seen
    Cell    -> 1 if the event appears in that file, else 0
    """
    file_to_events = {}

    for file in csv_files:
        try:
            if has_header:
                s = pd.read_csv(file).iloc[:, 0]
            else:
                s = pd.read_csv(file, header=None).iloc[:, 0]
        except pd.errors.EmptyDataError:
            # Completely empty CSV -> no events
            file_to_events[file] = set()
            continue

        nums = (
            pd.to_numeric(s, errors="coerce")
            .dropna()
            .astype(int)
        )
        file_to_events[file] = set(nums.tolist())

    if not file_to_events:
        raise ValueError("No CSV files were provided.")

    max_event = max(
        (max(events) for events in file_to_events.values() if events),
        default=0
    )

    if max_event == 0:
        # All files empty / no valid numbers -> all-zero matrix with one column
        columns = ["event1"]
        index = [Path(f).name for f in csv_files]
        return pd.DataFrame(0, index=index, columns=columns, dtype=int)

    columns = [f"event{i}" for i in range(1, max_event + 1)]
    index = [Path(f).name for f in csv_files]
    matrix = pd.DataFrame(0, index=index, columns=columns, dtype=int)

    for file, events in file_to_events.items():
        row_name = Path(file).name
        for ev in events:
            if 1 <= ev <= max_event:
                col = f"event{ev}"
                if col in matrix.columns:
                    matrix.at[row_name, col] = 1

    return matrix


def compute_row_correlation_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Pearson correlation between rows of a binary event matrix.

    Correlations between all-zero rows (or other degenerate cases) will be NaN.
    """
    corr_matrix = matrix.T.corr(method="pearson")
    return corr_matrix


def make_mid_bwr_cmap():
    """
    Create a blue - white - red colormap,
    """
    cmap = LinearSegmentedColormap.from_list(
        "mid_bwr",
        [
            (0.0, (0.40, 0.60, 1.0)),   # medium blue
            (0.5, (1.0, 1.0, 1.0)),     # white
            (1.0, (1.0, 0.40, 0.40)),   # medium red
        ]
    )
    cmap.set_bad(color="lightgray")
    return cmap


def plot_correlation_heatmap(corr_matrix: pd.DataFrame, figsize=None, output_file=None):
    """
    Plot a 2D heatmap of the correlation matrix using matplotlib.

    """
    labels = [Path(name).stem for name in corr_matrix.index]
    data = corr_matrix.values
    n = len(labels)

    # Figure size: scale with number of regions
    if figsize is None:
        side = min(max(10.0, n * 0.35), 32.0)
        figsize = (side, side)

    # Font sizes for labels
    if n <= 20:
        label_fontsize = 12
    elif n <= 35:
        label_fontsize = 9
    elif n <= 60:
        label_fontsize = 7
    else:
        label_fontsize = 5

    annot_fontsize = max(3, int(label_fontsize * 0.7))

    fig, ax = plt.subplots(figsize=figsize)

    cmap = make_mid_bwr_cmap()

    im = ax.imshow(data, interpolation="nearest", vmin=-1.0, vmax=1.0, cmap=cmap)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=label_fontsize)
    ax.set_yticklabels(labels, fontsize=label_fontsize)

    # Annotate all non-NaN cells
    n_rows, n_cols = data.shape
    for i in range(n_rows):
        for j in range(n_cols):
            value = data[i, j]
            if np.isnan(value):
                continue
            ax.text(
                j, i, f"{value:.2f}",
                ha="center", va="center",
                fontsize=annot_fontsize,
            )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Pearson correlation")

    # Space for rotated x labels
    plt.subplots_adjust(bottom=0.42)
    plt.tight_layout()

    if output_file is not None:
        plt.savefig(output_file, bbox_inches="tight", dpi=300)

    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a binary event matrix from CSV files and plot a Pearson correlation heatmap."
    )
    parser.add_argument(
        "csv_files", nargs="*",
        help="List of CSV files. If empty and --glob is given, the glob pattern will be used."
    )
    parser.add_argument(
        "--glob", type=str, default=None,
        help="Glob pattern to collect CSV files, e.g. '*.csv'"
    )
    parser.add_argument(
        "--has-header", action="store_true",
        help="Use this flag if the CSV files have a header row."
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Optional path to save the heatmap image."
    )
    parser.add_argument(
        "--matrix-out", type=str, default=None,
        help="Optional path to save the event matrix as CSV."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    csv_files = list(args.csv_files)

    if not csv_files and args.glob is not None:
        csv_files = glob(args.glob)

    if not csv_files:
        raise SystemExit("No CSV files provided. Use positional arguments or --glob pattern.")

    event_matrix = build_event_matrix(csv_files, has_header=args.has_header)

    if args.matrix_out is not None:
        event_matrix.to_csv(args.matrix_out, index=True)

    corr_matrix = compute_row_correlation_matrix(event_matrix)

    plot_correlation_heatmap(corr_matrix, figsize=None, output_file=args.output)


if __name__ == "__main__":
    main()
