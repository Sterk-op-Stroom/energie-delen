"""Build per-prosumer summary table from a PipelineResult."""

from __future__ import annotations

import numpy as np
import pandas as pd

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cli import PipelineResult


def build_prosumer_table(pipeline: PipelineResult) -> pd.DataFrame:
    """Return a summary DataFrame with one row per prosumer.

    Columns:
        meter_id            — prosumer identifier
        total_demand_kWh    — sum of positive meter values over the period
        total_allocated_kWh — sum of locally allocated energy
        self_sufficiency_pct — fraction of demand met locally [0–1]
        grid_import_kWh     — unmet demand (from grid) over the period
        total_cost_eur      — total local energy charge
        avg_eur_per_kwh     — average price paid per kWh allocated
    """
    dataset = pipeline.dataset
    allocation = pipeline.allocation
    pricing = pipeline.pricing

    # Build a demand lookup for all meters (prosumers + production assets)
    demand_by_meter: dict[str, float] = {}
    for m in dataset.prosumers + dataset.production_assets:
        demand_by_meter[m.meter_id] = float(np.nansum(np.clip(m.value, 0, None)))

    rows = []
    for pid in allocation.prosumer_ids:
        demand = demand_by_meter.get(pid, 0.0)
        allocated = float(allocation.allocations[pid].sum())
        cost = float(pricing.total_local_cost_eur_by_prosumer.get(pid, 0.0))
        self_suff = allocated / demand if demand > 0 else 0.0
        grid_imp = max(demand - allocated, 0.0)
        avg_price = cost / allocated if allocated > 0 else 0.0
        rows.append(
            {
                "meter_id": pid,
                "total_demand_kWh": round(demand, 3),
                "total_allocated_kWh": round(allocated, 3),
                "self_sufficiency_pct": round(self_suff, 4),
                "grid_import_kWh": round(grid_imp, 3),
                "total_cost_eur": round(cost, 4),
                "avg_eur_per_kwh": round(avg_price, 5),
            }
        )

    return pd.DataFrame(rows)
