"""Single-page dashboard combining key plots from all pipeline stages.

The main entry point is `plot_dashboard()`, which produces a 2x2 grid
with a KPI text strip at the bottom.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from ..core_types import AggregatedStep, AllocationResult, PricingResult
from .style import (
    DEMAND_COLOR,
    SUPPLY_COLOR,
    LOCAL_ALLOC_COLOR,
    GRID_IMPORT_COLOR,
    GRID_EXPORT_COLOR,
    COST_COLOR,
    apply_style,
)


def plot_dashboard(
    step: AggregatedStep,
    allocation: AllocationResult,
    pricing: PricingResult,
    grid_tariff_eur_per_kwh: float = 0.25,
) -> Figure:
    """Single-page executive summary of a simulation run.

    Layout:
        Top-left:     Supply vs. Demand time series
        Top-right:    Self-sufficiency rate
        Bottom-left:  Energy flow (demand side)
        Bottom-right: Per-prosumer cost bars
        Footer:       KPI text box
    """
    apply_style()
    fig = plt.figure(figsize=(18, 13))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 0.2], hspace=0.35, wspace=0.25)

    ts = step.timestamp
    total_alloc = sum(allocation.allocations[m] for m in allocation.prosumer_ids)

    # --- Top-left: Supply vs Demand ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(ts, step.demand_total, color=DEMAND_COLOR, linewidth=0.8, label="Demand")
    ax1.plot(ts, step.supply_total, color=SUPPLY_COLOR, linewidth=0.8, label="Supply")
    ax1.fill_between(ts, step.supply_total, step.demand_total,
                      where=step.supply_total >= step.demand_total, alpha=0.12, color=SUPPLY_COLOR)
    ax1.fill_between(ts, step.supply_total, step.demand_total,
                      where=step.supply_total < step.demand_total, alpha=0.12, color=DEMAND_COLOR)
    ax1.set_ylabel("kWh")
    ax1.set_title("Supply vs. Demand")
    ax1.legend(loc="upper right", fontsize=8)
    fig.autofmt_xdate()

    # --- Top-right: Self-sufficiency ---
    ax2 = fig.add_subplot(gs[0, 1])
    rate = np.where(step.demand_total > 0, total_alloc / step.demand_total, 0.0)
    ax2.fill_between(ts, rate, alpha=0.25, color=LOCAL_ALLOC_COLOR)
    ax2.plot(ts, rate, color=LOCAL_ALLOC_COLOR, linewidth=0.8)
    avg_rate = float(total_alloc.sum() / step.demand_total.sum()) if step.demand_total.sum() > 0 else 0
    ax2.axhline(avg_rate, color=LOCAL_ALLOC_COLOR, linestyle=":", linewidth=1, label=f"Avg: {avg_rate:.1%}")
    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=0.5)
    ax2.set_ylabel("Rate")
    ax2.set_ylim(0, min(1.05, max(rate.max() * 1.1, 0.1)))
    ax2.set_title("Self-Sufficiency Rate")
    ax2.legend(loc="upper right", fontsize=8)

    # --- Bottom-left: Energy flow (demand side) ---
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.fill_between(ts, total_alloc, alpha=0.6, color=LOCAL_ALLOC_COLOR, label="Local")
    ax3.fill_between(ts, total_alloc, total_alloc + allocation.grid_import,
                      alpha=0.6, color=GRID_IMPORT_COLOR, label="Grid import")
    ax3.plot(ts, step.demand_total, color=DEMAND_COLOR, linewidth=0.8, linestyle="--", label="Demand")
    ax3.set_ylabel("kWh")
    ax3.set_title("How Demand Is Met")
    ax3.legend(loc="upper right", fontsize=8)

    # --- Bottom-right: Per-prosumer cost ---
    ax4 = fig.add_subplot(gs[1, 1])
    ids = sorted(
        pricing.prosumer_ids,
        key=lambda m: pricing.total_local_cost_eur_by_prosumer[m],
        reverse=True,
    )
    costs = [pricing.total_local_cost_eur_by_prosumer[m] for m in ids]
    display_ids = ids[:15]  # cap at 15 for readability
    display_costs = costs[:15]
    ax4.barh(range(len(display_ids)), display_costs, color=COST_COLOR, alpha=0.7)
    ax4.set_yticks(range(len(display_ids)))
    ax4.set_yticklabels(display_ids, fontsize=8)
    ax4.invert_yaxis()
    ax4.set_xlabel("EUR")
    ax4.set_title("Per-Prosumer Local Cost")
    if len(ids) > 15:
        ax4.text(0.95, 0.95, f"(+{len(ids)-15} more)", transform=ax4.transAxes,
                 ha="right", va="top", fontsize=8, color="gray")

    # --- Footer: KPI text ---
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis("off")

    demand_total = float(step.demand_total.sum())
    supply_total = float(step.supply_total.sum())
    alloc_total = float(total_alloc.sum())
    grid_imp = float(allocation.grid_import.sum())
    grid_exp = float(allocation.grid_export.sum())
    community_cost = float(pricing.total_local_cost_eur.sum())
    grid_cost = alloc_total * grid_tariff_eur_per_kwh
    total_savings = grid_cost - community_cost

    kpi_text = (
        f"Period: {ts[0].strftime('%d-%m-%Y')} to {ts[-1].strftime('%d-%m-%Y')}   |   "
        f"Demand: {demand_total:,.0f} kWh   |   "
        f"Supply: {supply_total:,.0f} kWh   |   "
        f"Locally allocated: {alloc_total:,.0f} kWh ({avg_rate:.1%})   |   "
        f"Grid import: {grid_imp:,.0f} kWh   |   "
        f"Grid export: {grid_exp:,.0f} kWh\n"
        f"Local cost: {community_cost:,.2f} EUR   |   "
        f"Grid equivalent: {grid_cost:,.2f} EUR (@ {grid_tariff_eur_per_kwh:.2f} EUR/kWh)   |   "
        f"Community savings: {total_savings:,.2f} EUR"
    )
    ax5.text(0.5, 0.5, kpi_text, transform=ax5.transAxes,
             ha="center", va="center", fontsize=10, family="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", edgecolor="#e2e8f0"))

    fig.suptitle("Energy Sharing Simulation — Summary Dashboard", fontsize=14, fontweight="bold", y=0.98)
    return fig
