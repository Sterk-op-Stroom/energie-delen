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
    total_cost = float(pricing.total_cost_eur.sum())
    self_suff = total_allocated / total_demand if total_demand > 0 else 0.0
    self_cons = total_allocated / total_supply if total_supply > 0 else 0.0
    grid_import = float(alloc.grid_import.sum())
    grid_export = float(alloc.grid_export.sum())
    market_import = pipeline.pricing_market_import
    market_export = pipeline.pricing_market_export
    cf_import = pipeline.pricing_counterfactual_import
    cf_export = pipeline.pricing_counterfactual_export

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
        pn.pane.Markdown("Energie (kWh)", styles=row_label_style, margin=(8, 0, 4, 0)),
        pn.Row(
            _card("Totale Vraag", f"{total_demand:,.0f}", "kWh",
                  "Totaal verbruikte energie over de periode"),
            _card("Totaal Aanbod", f"{total_supply:,.0f}", "kWh",
                  "Totaal lokaal opgewekte energie beschikbaar voor delen"),
            sizing_mode="stretch_width",
        ),
        pn.Row(
            _card("Lokaal Toegewezen", f"{total_allocated:,.0f}", "kWh",
                  "Lokaal aanbod daadwerkelijk verdeeld aan de gemeenschap"),
            _card("Netimport", f"{grid_import:,.0f}", "kWh",
                  "Vraag niet lokaal gedekt; afgenomen van het openbare net"),
            _card("Netexport", f"{grid_export:,.0f}", "kWh",
                  "Lokaal aanbod niet lokaal verbruikt; teruggeleverd aan het net"),
            sizing_mode="stretch_width",
        ),
        sizing_mode="stretch_width",
    )

    pct_row = pn.Column(
        pn.pane.Markdown("Efficiëntie (%)", styles=row_label_style, margin=(8, 0, 4, 0)),
        pn.Row(
            _card("Zelfvoorzienendheid", f"{self_suff:.1%}", "",
                  "Aandeel van de totale vraag gedekt door lokaal aanbod"),
            _card("Zelfconsumptie", f"{self_cons:.1%}", "",
                  "Aandeel van het lokale aanbod gebruikt binnen de gemeenschap"),
            sizing_mode="stretch_width",
        ),
        sizing_mode="stretch_width",
    )

    eur_cards = [
        _card("Bedrag gedeelde energie", f"{total_cost:,.2f}", "EUR",
              "Bedrag aan lokaal gedeelde energie"),
    ]
    if market_import is not None:
        import_cost = float(market_import.total_cost_eur.sum())
        eur_cards.append(_card(
            "Marktkosten import", f"{import_cost:,.2f}", "EUR",
            "Totale kosten voor netimport na lokaal delen",
        ))
    if market_export is not None:
        export_rev = float(market_export.total_cost_eur.sum())
        eur_cards.append(_card(
            "Marktopbrengsten export", f"{export_rev:,.2f}", "EUR",
            "Totale opbrengsten voor netexport na lokaal delen",
        ))
    if market_import is not None and market_export is not None:
        net = float(market_import.total_cost_eur.sum()) - float(market_export.total_cost_eur.sum())
        eur_cards.append(_card(
            "Netto marktkosten", f"{net:,.2f}", "EUR",
            "Marktkosten import minus marktopbrengsten export",
        ))

    eur_row = pn.Column(
        pn.pane.Markdown("Kosten (EUR)", styles=row_label_style, margin=(8, 0, 4, 0)),
        pn.Row(*eur_cards, sizing_mode="stretch_width"),
        sizing_mode="stretch_width",
    )

    rows = [kwh_row, pct_row, eur_row]

    if cf_import is not None or cf_export is not None:
        cf_import_cost = float(cf_import.total_cost_eur.sum()) if cf_import is not None else 0.0
        cf_export_rev = float(cf_export.total_cost_eur.sum()) if cf_export is not None else 0.0
        cf_cards = []
        if cf_import is not None:
            cf_cards.append(_card(
                "Marktkosten import", f"{cf_import_cost:,.2f}", "EUR",
                "Wat prosumers zouden betalen zonder lokaal energiedelen",
            ))
        if cf_export is not None:
            cf_cards.append(_card(
                "Marktopbrengsten export", f"{cf_export_rev:,.2f}", "EUR",
                "Wat producenten zouden ontvangen zonder lokaal energiedelen",
            ))
        if cf_import is not None and cf_export is not None:
            cf_net = cf_import_cost - cf_export_rev
            cf_cards.append(_card(
                "Netto zonder deling", f"{cf_net:,.2f}", "EUR",
                "Netto energiekosten zonder lokaal energiedelen",
            ))
        rows.append(pn.Column(
            pn.pane.Markdown(
                "Vergelijking met kosten bij normale energieleverancier (EUR)",
                styles=row_label_style, margin=(8, 0, 4, 0),
            ),
            pn.Row(*cf_cards, sizing_mode="stretch_width"),
            sizing_mode="stretch_width",
        ))

    return pn.Column(*rows, sizing_mode="stretch_width")
