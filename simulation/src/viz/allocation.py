"""Allocation analysis visualizations.

All functions take an AllocationResult (and optionally AggregatedStep)
and return a matplotlib Figure.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ..core_types import AggregatedStep, AllocationResult
from .style import (
    DEMAND_COLOR,
    SUPPLY_COLOR,
    LOCAL_ALLOC_COLOR,
    GRID_IMPORT_COLOR,
    GRID_EXPORT_COLOR,
    apply_style,
)


def plot_energy_flow(allocation: AllocationResult, step: AggregatedStep) -> Figure:
    """Stacked area showing how demand is met: local allocation + grid import."""
    apply_style()
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ts = allocation.timestamp
    total_alloc = sum(allocation.allocations[m] for m in allocation.prosumer_ids)

    # Top: demand side
    ax = axes[0]
    ax.fill_between(ts, total_alloc, alpha=0.6, color=LOCAL_ALLOC_COLOR, label="Local allocation")
    ax.fill_between(
        ts, total_alloc, total_alloc + allocation.grid_import,
        alpha=0.6, color=GRID_IMPORT_COLOR, label="Grid import",
    )
    ax.plot(ts, step.demand_total, color=DEMAND_COLOR, linewidth=0.8, linestyle="--", label="Total demand")
    ax.set_ylabel("kWh")
    ax.set_title("How Demand Is Met")
    ax.legend(loc="upper right")

    # Bottom: supply side
    ax = axes[1]
    ax.fill_between(ts, total_alloc, alpha=0.6, color=LOCAL_ALLOC_COLOR, label="Locally allocated")
    ax.fill_between(
        ts, total_alloc, total_alloc + allocation.grid_export,
        alpha=0.6, color=GRID_EXPORT_COLOR, label="Grid export (unused)",
    )
    ax.plot(ts, step.supply_total, color=SUPPLY_COLOR, linewidth=0.8, linestyle="--", label="Total supply")
    ax.set_ylabel("kWh")
    ax.set_title("How Supply Is Used")
    ax.legend(loc="upper right")

    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_self_sufficiency(allocation: AllocationResult, step: AggregatedStep) -> Figure:
    """Self-sufficiency rate: fraction of demand met locally over time."""
    apply_style()
    fig, ax = plt.subplots(figsize=(14, 4))

    total_alloc = sum(allocation.allocations[m] for m in allocation.prosumer_ids)
    rate = np.where(step.demand_total > 0, total_alloc / step.demand_total, 0.0)

    ax.fill_between(allocation.timestamp, rate, alpha=0.3, color=LOCAL_ALLOC_COLOR)
    ax.plot(allocation.timestamp, rate, color=LOCAL_ALLOC_COLOR, linewidth=0.8)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.5)

    avg_rate = float(total_alloc.sum() / step.demand_total.sum()) if step.demand_total.sum() > 0 else 0
    ax.axhline(avg_rate, color=LOCAL_ALLOC_COLOR, linestyle=":", linewidth=1, label=f"Period avg: {avg_rate:.1%}")

    ax.set_ylabel("Self-sufficiency rate")
    ax.set_ylim(0, min(1.05, rate.max() * 1.1) if rate.max() > 0 else 1.05)
    ax.set_title("Local Self-Sufficiency Rate")
    ax.legend(loc="upper right")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_prosumer_allocation_heatmap(allocation: AllocationResult) -> Figure:
    """Heatmap: prosumers (y) x time (x), color = allocated kWh."""
    apply_style()

    ids = allocation.prosumer_ids
    matrix = np.stack([allocation.allocations[m] for m in ids], axis=0)

    fig, ax = plt.subplots(figsize=(14, max(3, len(ids) * 0.4)))
    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap="YlOrRd",
        origin="upper",
        interpolation="nearest",
    )
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels(ids, fontsize=8)

    # Sparse x-axis labels
    n_labels = min(10, len(allocation.timestamp))
    step = max(1, len(allocation.timestamp) // n_labels)
    ax.set_xticks(range(0, len(allocation.timestamp), step))
    ax.set_xticklabels(
        [str(allocation.timestamp[i])[:16] for i in range(0, len(allocation.timestamp), step)],
        rotation=45, ha="right", fontsize=7,
    )

    ax.set_title("Per-Prosumer Allocation (kWh)")
    fig.colorbar(im, ax=ax, label="kWh", shrink=0.8)
    fig.tight_layout()
    return fig


def plot_allocation_fairness(allocation: AllocationResult) -> Figure:
    """Boxplot: distribution of allocation per timestep for each prosumer."""
    apply_style()
    fig, ax = plt.subplots(figsize=(max(6, len(allocation.prosumer_ids) * 0.8), 5))

    data = [allocation.allocations[m] for m in allocation.prosumer_ids]
    bp = ax.boxplot(data, labels=allocation.prosumer_ids, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor(LOCAL_ALLOC_COLOR)
        patch.set_alpha(0.6)

    ax.set_ylabel("kWh per interval")
    ax.set_title("Allocation Fairness (distribution per prosumer)")
    if len(allocation.prosumer_ids) > 10:
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def plot_grid_dependency(allocation: AllocationResult, step: AggregatedStep) -> Figure:
    """Daily stacked bar: local allocation vs. grid import."""
    apply_style()
    fig, ax = plt.subplots(figsize=(12, 5))

    total_alloc = sum(allocation.allocations[m] for m in allocation.prosumer_ids)
    df = pd.DataFrame({
        "local": total_alloc,
        "grid_import": allocation.grid_import,
        "date": allocation.timestamp.date,
    })
    daily = df.groupby("date").sum()

    ax.bar(range(len(daily)), daily["local"], color=LOCAL_ALLOC_COLOR, label="Local allocation")
    ax.bar(
        range(len(daily)), daily["grid_import"],
        bottom=daily["local"], color=GRID_IMPORT_COLOR, label="Grid import",
    )

    ax.set_xticks(range(0, len(daily), max(1, len(daily) // 10)))
    ax.set_xticklabels(
        [str(d) for d in daily.index[::max(1, len(daily) // 10)]],
        rotation=45, ha="right", fontsize=8,
    )
    ax.set_ylabel("kWh")
    ax.set_title("Daily Grid Dependency")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_curtailment(allocation: AllocationResult) -> Figure:
    """Area chart of grid_export (unallocated local supply) over time."""
    apply_style()
    fig, ax = plt.subplots(figsize=(14, 4))

    ax.fill_between(allocation.timestamp, allocation.grid_export, alpha=0.5, color=GRID_EXPORT_COLOR)
    ax.plot(allocation.timestamp, allocation.grid_export, color=GRID_EXPORT_COLOR, linewidth=0.6)

    total = float(allocation.grid_export.sum())
    ax.set_ylabel("kWh")
    ax.set_title(f"Unallocated Local Supply (Curtailment) — Total: {total:,.1f} kWh")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig
