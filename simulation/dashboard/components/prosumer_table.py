"""Build per-prosumer summary table from a PipelineResult."""

from __future__ import annotations

import numpy as np
import pandas as pd

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cli import PipelineResult


def build_prosumer_table(pipeline: PipelineResult) -> pd.DataFrame:
    """Return a summary DataFrame with one row per prosumer.

    Columns (always present):
        meter_id                — prosumer identifier
        total_cost_eur          — total cost in sharing scenario
                                  (local sharing + market import − market export)
        avg_eur_per_kwh         — total_cost_eur / total demand
        total_demand_kWh        — sum of positive meter values over the period
        total_allocated_kWh     — sum of locally allocated energy
        self_sufficiency_pct    — fraction of demand met locally [0–1]
        local_sharing_cost_eur  — charge for locally allocated energy only

    Columns present when market pricing is configured:
        market_import_cost_eur  — cost for grid import after local sharing
        market_export_revenue_eur — revenue from grid export after local sharing
        net_market_cost_eur     — market_import_cost_eur − market_export_revenue_eur

    Columns present when counterfactual pricing is configured:
        total_cf_cost_eur       — total cost without local sharing
                                  (cf import − cf export)
        cf_import_cost_eur      — import cost without local sharing
        cf_export_revenue_eur   — export revenue without local sharing
    """
    dataset = pipeline.dataset
    allocation = pipeline.allocation
    pricing = pipeline.pricing
    market_import = pipeline.pricing_market_import
    market_export = pipeline.pricing_market_export
    cf_import = pipeline.pricing_counterfactual_import
    cf_export = pipeline.pricing_counterfactual_export

    has_market = market_import is not None or market_export is not None
    has_cf = cf_import is not None or cf_export is not None

    demand_by_meter: dict[str, float] = {}
    for m in dataset.prosumers + dataset.production_assets:
        demand_by_meter[m.meter_id] = float(np.nansum(np.clip(m.value, 0, None)))

    rows = []
    for pid in allocation.prosumer_ids:
        demand = demand_by_meter.get(pid, 0.0)
        allocated = float(allocation.allocations[pid].sum())
        local_cost = float(pricing.total_cost_eur_by_prosumer.get(pid, 0.0))
        self_suff = allocated / demand if demand > 0 else 0.0

        mkt_imp = float(market_import.total_cost_eur_by_prosumer.get(pid, 0.0)) if market_import is not None else 0.0
        mkt_exp = float(market_export.total_cost_eur_by_prosumer.get(pid, 0.0)) if market_export is not None else 0.0
        total_sharing_cost = local_cost + mkt_imp - mkt_exp

        avg_price = total_sharing_cost / demand if demand > 0 else 0.0

        row: dict = {
            "meter_id": pid,
        }

        row["total_cost_eur"] = round(total_sharing_cost, 2) if has_market else None

        if has_cf:
            cf_imp = float(cf_import.total_cost_eur_by_prosumer.get(pid, 0.0)) if cf_import is not None else 0.0
            cf_exp = float(cf_export.total_cost_eur_by_prosumer.get(pid, 0.0)) if cf_export is not None else 0.0
            row["total_cf_cost_eur"] = round(cf_imp - cf_exp, 2)

        row["local_sharing_cost_eur"] = round(local_cost, 2)
        row["avg_eur_per_kwh"] = round(avg_price, 5) if has_market else None
        row["total_demand_kWh"] = round(demand, 3)
        row["total_allocated_kWh"] = round(allocated, 3)
        row["self_sufficiency_pct"] = round(self_suff, 4)

        if has_market:
            row["market_import_cost_eur"] = round(mkt_imp, 2)
            row["market_export_revenue_eur"] = round(mkt_exp, 2)
            if market_import is not None and market_export is not None:
                row["net_market_cost_eur"] = round(mkt_imp - mkt_exp, 2)

        if has_cf:
            row["cf_import_cost_eur"] = round(cf_imp, 2)
            row["cf_export_revenue_eur"] = round(cf_exp, 2)

        rows.append(row)

    return pd.DataFrame(rows)
