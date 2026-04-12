"""KPI summary rows built from a PipelineResult."""

from __future__ import annotations

from typing import TYPE_CHECKING

import panel as pn

if TYPE_CHECKING:
    from cli import PipelineResult


def build_kpi_row(pipeline: PipelineResult) -> pn.Column:
    """Return three rows of KPI cards grouped by unit (kWh / % / EUR)."""
    step = pipeline.step
    alloc = pipeline.allocation
    pricing = pipeline.pricing

    total_demand = float(step.demand_total.sum())
    total_supply = float(step.supply_total.sum())
    total_allocated = sum(alloc.allocations[m].sum() for m in alloc.prosumer_ids)
    total_cost = float(pricing.total_local_cost_eur.sum())
    self_suff = total_allocated / total_demand if total_demand > 0 else 0.0
    self_cons = total_allocated / total_supply if total_supply > 0 else 0.0
    grid_import = float(alloc.grid_import.sum())
    grid_export = float(alloc.grid_export.sum())

    def _card(title: str, value: str, unit: str = "", description: str = "") -> pn.Column:
        children = [
            pn.pane.Markdown(f"**{title}**", margin=(0, 0, 2, 0)),
            pn.pane.Markdown(
                f"<span style='font-size:1.6em;font-weight:bold'>{value}</span>"
                f"<span style='color:#6b7280'> {unit}</span>",
                margin=(0, 0, 0, 0),
            ),
        ]
        if description:
            children.append(
                pn.pane.Markdown(
                    f"<div style='font-size:0.78em;color:#9ca3af;line-height:1.35;"
                    f"white-space:normal;word-wrap:break-word;max-width:180px'>"
                    f"{description}</div>",
                    margin=(4, 0, 0, 0),
                )
            )
        return pn.Column(
            *children,
            styles={
                "background": "#f8fafc",
                "border": "1px solid #e2e8f0",
                "border-radius": "8px",
                "padding": "12px 16px",
            },
            min_width=160,
        )

    row_label_style = {"font-size": "0.75em", "color": "#6b7280", "font-weight": "bold",
                       "text-transform": "uppercase", "letter-spacing": "0.05em"}

    kwh_row = pn.Column(
        pn.pane.Markdown("Energy (kWh)", styles=row_label_style, margin=(8, 0, 4, 0)),
        pn.Row(
            _card("Total Demand", f"{total_demand:,.0f}", "kWh",
                  "Total energy consumed over the period"),
            _card("Total Supply", f"{total_supply:,.0f}", "kWh",
                  "Total locally generated energy available for sharing"),
            sizing_mode="stretch_width",
        ),
        pn.Row(
            _card("Locally Allocated", f"{total_allocated:,.0f}", "kWh",
                  "Local supply actually distributed to the community"),
            _card("Grid Import", f"{grid_import:,.0f}", "kWh",
                  "Demand not met locally; drawn from the public grid"),
            _card("Grid Export", f"{grid_export:,.0f}", "kWh",
                  "Local supply not consumed locally; fed back to the grid"),
            sizing_mode="stretch_width",
        ),
        sizing_mode="stretch_width",
    )

    pct_row = pn.Column(
        pn.pane.Markdown("Efficiency (%)", styles=row_label_style, margin=(8, 0, 4, 0)),
        pn.Row(
            _card("Self-Sufficiency", f"{self_suff:.1%}", "",
                  "Share of total demand covered by local supply"),
            _card("Self-Consumption", f"{self_cons:.1%}", "",
                  "Share of local supply actually consumed within the community"),
            sizing_mode="stretch_width",
        ),
        sizing_mode="stretch_width",
    )

    eur_row = pn.Column(
        pn.pane.Markdown("Cost (EUR)", styles=row_label_style, margin=(8, 0, 4, 0)),
        pn.Row(
            _card("Community Cost", f"{total_cost:,.2f}", "EUR",
                  "Total charge for locally allocated energy across all prosumers"),
            sizing_mode="stretch_width",
        ),
        sizing_mode="stretch_width",
    )

    return pn.Column(kwh_row, pct_row, eur_row, sizing_mode="stretch_width")
