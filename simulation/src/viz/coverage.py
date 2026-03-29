"""Data quality & coverage visualizations.

Functions take a LoadedDataset and/or CoverageReport and return a matplotlib Figure.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ..core_types import LoadedDataset
from ..report_types import CoverageReport
from .style import DEMAND_COLOR, SUPPLY_COLOR, NEUTRAL_COLOR, apply_style


def plot_coverage_heatmap(
    dataset: LoadedDataset,
    expected_index: pd.DatetimeIndex,
) -> Figure:
    """Heatmap: meters (y) x time (x), colored present (green) / missing (red).

    Args:
        dataset: LoadedDataset with prosumers and production assets.
        expected_index: The canonical simulation clock to check against.
    """
    apply_style()

    all_series = dataset.prosumers + dataset.production_assets
    ids = [s.meter_id for s in all_series]
    n_meters = len(ids)
    n_timesteps = len(expected_index)

    matrix = np.zeros((n_meters, n_timesteps), dtype=np.float32)
    for i, s in enumerate(all_series):
        vals = pd.Series(s.value, index=s.timestamp).reindex(expected_index)
        matrix[i] = vals.notna().astype(np.float32)

    fig, ax = plt.subplots(figsize=(14, max(3, n_meters * 0.35)))
    ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1, origin="upper", interpolation="nearest")

    ax.set_yticks(range(n_meters))
    ax.set_yticklabels(ids, fontsize=8)

    # Sparse x ticks
    n_labels = min(10, n_timesteps)
    step = max(1, n_timesteps // n_labels)
    ax.set_xticks(range(0, n_timesteps, step))
    ax.set_xticklabels(
        [str(expected_index[i])[:16] for i in range(0, n_timesteps, step)],
        rotation=45, ha="right", fontsize=7,
    )

    ax.set_title("Data Coverage (green = present, red = missing)")
    fig.tight_layout()
    return fig


def plot_missing_fraction_bars(report: CoverageReport) -> Figure:
    """Horizontal bar chart of missing data fraction per meter."""
    apply_style()

    ids = sorted(report.per_meter_missing_fraction.keys())
    fractions = [report.per_meter_missing_fraction[m] for m in ids]

    fig, ax = plt.subplots(figsize=(8, max(3, len(ids) * 0.35)))
    colors = ["#ef4444" if f > 0.1 else "#f59e0b" if f > 0 else "#22c55e" for f in fractions]
    ax.barh(range(len(ids)), fractions, color=colors, alpha=0.7)
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels(ids, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Missing fraction")
    ax.set_xlim(0, max(max(fractions) * 1.1, 0.05) if fractions else 1.0)
    ax.set_title("Missing Data Fraction per Meter")

    for i, f in enumerate(fractions):
        ax.text(f + 0.005, i, f"{f:.1%}", va="center", fontsize=8)

    fig.tight_layout()
    return fig


def plot_coverage_timeline(
    dataset: LoadedDataset,
    expected_index: pd.DatetimeIndex,
) -> Figure:
    """Stacked area: number of meters with data vs. missing over time."""
    apply_style()

    all_series = dataset.prosumers + dataset.production_assets
    n_meters = len(all_series)
    n_timesteps = len(expected_index)

    present_count = np.zeros(n_timesteps, dtype=np.int32)
    for s in all_series:
        vals = pd.Series(s.value, index=s.timestamp).reindex(expected_index)
        present_count += vals.notna().astype(np.int32)

    missing_count = n_meters - present_count

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(expected_index, present_count, alpha=0.6, color=SUPPLY_COLOR, label="Present")
    ax.fill_between(expected_index, present_count, present_count + missing_count, alpha=0.6, color="#ef4444", label="Missing")

    ax.set_ylabel("Number of meters")
    ax.set_title("Data Coverage Over Time")
    ax.legend(loc="upper right")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig
