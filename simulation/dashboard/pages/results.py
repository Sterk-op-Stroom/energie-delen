"""Results page — two tabs: Explore (hvplot), Table + Export."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import panel as pn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dashboard.state import AppState
from dashboard.transforms import (
    available_profiles,
    make_allocation_df,
    make_avg_profile_df,
    make_community_cost_df,
    make_efficiency_df,
    make_market_cost_df,
    make_prosumer_timeseries_df,
    make_supply_demand_df,
    resample_df,
)
from dashboard.components.prosumer_table import build_prosumer_table
from dashboard.components.export import (
    prosumer_csv_bytes,
    timeseries_csv_bytes,
)

# Colour palette (mirrors src/viz/style.py)
_DEMAND_COLOR = "#3b82f6"
_SUPPLY_COLOR = "#22c55e"
_ALLOC_COLOR = "#f59e0b"
_IMPORT_COLOR = "#ef4444"
_EXPORT_COLOR = "#8b5cf6"
_COST_COLOR = "#0ea5e9"

_AGG_OPTIONS = {
    "Onbewerkt (15 min)": None,
    "Per uur": "1h",
    "Dagelijks": "1D",
    "Wekelijks": "1W",
}


class ResultsPage:
    def __init__(self, state: AppState) -> None:
        self._state = state

    # ------------------------------------------------------------------
    # Tab 1: Explore
    # ------------------------------------------------------------------

    def _explore_tab(self, pipeline) -> pn.viewable.Viewable:
        import hvplot.pandas  # noqa: F401

        ts = pipeline.step.timestamp
        start_dt = ts[0].to_pydatetime()
        end_dt = ts[-1].to_pydatetime()

        date_slider = pn.widgets.DateRangeSlider(
            name="Datumbereik",
            start=start_dt,
            end=end_dt,
            value=(start_dt, end_dt),
            sizing_mode="stretch_width",
        )
        agg_select = pn.widgets.Select(
            name="Aggregatie",
            options=list(_AGG_OPTIONS),
            value="Onbewerkt (15 min)",
            width=200,
        )

        # Pre-compute base DataFrames
        sd_df = make_supply_demand_df(pipeline.step)
        alloc_df = make_allocation_df(pipeline.allocation)
        eff_df = make_efficiency_df(pipeline.step, pipeline.allocation)
        cost_df = make_community_cost_df(pipeline.pricing)
        market_cost_df = make_market_cost_df(
            pipeline.pricing_market_import,
            pipeline.pricing_market_export,
        )
        prosumer_df = make_prosumer_timeseries_df(pipeline.allocation, pipeline.pricing)
        n_prosumers = len(pipeline.allocation.prosumer_ids)

        def _to_utc(dt):
            t = pd.Timestamp(dt)
            return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")

        def _filter_resample(df, date_range, freq, agg="sum"):
            start, end = _to_utc(date_range[0]), _to_utc(date_range[1])
            mask = (df["timestamp"] >= start) & (df["timestamp"] <= end)
            filtered = df[mask].copy()
            if freq:
                filtered = resample_df(filtered, freq, agg=agg)
            return filtered

        # -- Sub-tab 1: Supply/Demand + Energy Flows --
        @pn.depends(date_slider.param.value, agg_select.param.value)
        def _energy_charts(date_range, agg_label):
            freq = _AGG_OPTIONS[agg_label]
            sd = _filter_resample(sd_df, date_range, freq)
            al = _filter_resample(alloc_df, date_range, freq)
            charts = []

            if not sd.empty:
                chart = sd.hvplot.area(
                    x="timestamp",
                    y=["demand_kWh", "supply_kWh"],
                    alpha=0.6,
                    color=[_DEMAND_COLOR, _SUPPLY_COLOR],
                    title="Aanbod vs Vraag (kWh)",
                    responsive=True,
                    min_height=280,
                    legend="top_left",
                )
                charts.append(pn.pane.HoloViews(chart, sizing_mode="stretch_width"))

            if not al.empty:
                chart2 = al.hvplot.area(
                    x="timestamp",
                    y=["local_allocation_kWh", "grid_import_kWh"],
                    alpha=0.6,
                    color=[_ALLOC_COLOR, _IMPORT_COLOR],
                    title="Energiestromen (kWh)",
                    responsive=True,
                    min_height=280,
                    legend="top_left",
                ) * al.hvplot.line(
                    x="timestamp",
                    y="grid_export_kWh",
                    color=_EXPORT_COLOR,
                    label="grid_export_kWh",
                )
                charts.append(pn.pane.HoloViews(chart2, sizing_mode="stretch_width"))

            return pn.Column(*charts, sizing_mode="stretch_width") if charts else pn.pane.Markdown("Geen gegevens in dit bereik.")

        # -- Sub-tab 2: Self-Sufficiency + Self-Consumption --
        @pn.depends(date_slider.param.value, agg_select.param.value)
        def _efficiency_charts(date_range, agg_label):
            freq = _AGG_OPTIONS[agg_label]
            eff = _filter_resample(eff_df, date_range, freq, agg="mean")
            charts = []

            if not eff.empty:
                chart3 = eff.hvplot.line(
                    x="timestamp",
                    y="self_sufficiency_pct",
                    ylim=(0, 1.05),
                    color=_ALLOC_COLOR,
                    title="Zelfvoorzieningspercentage  (lokale toewijzing ÷ vraag)",
                    responsive=True,
                    min_height=240,
                )
                charts.append(pn.pane.HoloViews(chart3, sizing_mode="stretch_width"))

            if not eff.empty:
                chart3b = eff.hvplot.line(
                    x="timestamp",
                    y="self_consumption_pct",
                    ylim=(0, 1.05),
                    color=_SUPPLY_COLOR,
                    title="Zelfconsumptiepercentage  (lokale toewijzing ÷ aanbod)",
                    responsive=True,
                    min_height=240,
                )
                charts.append(pn.pane.HoloViews(chart3b, sizing_mode="stretch_width"))

            # Per-prosumer allocation (only if ≤20 prosumers)
            if n_prosumers <= 20 and not prosumer_df.empty:
                start, end = _to_utc(date_range[0]), _to_utc(date_range[1])
                mask_p = (prosumer_df["timestamp"] >= start) & (prosumer_df["timestamp"] <= end)
                pdf = prosumer_df[mask_p].copy()
                if freq:
                    parts = []
                    for pid, grp in pdf.groupby("meter_id"):
                        r = resample_df(grp.drop(columns="meter_id"), freq)
                        r["meter_id"] = pid
                        parts.append(r)
                    pdf = pd.concat(parts, ignore_index=True) if parts else pdf

                if not pdf.empty:
                    chart5 = pdf.hvplot.line(
                        x="timestamp",
                        y="allocated_kWh",
                        by="meter_id",
                        title=f"Toewijzing per prosumer ({n_prosumers} meters)",
                        responsive=True,
                        min_height=280,
                        legend="right",
                    )
                    charts.append(pn.pane.HoloViews(chart5, sizing_mode="stretch_width"))

            return pn.Column(*charts, sizing_mode="stretch_width") if charts else pn.pane.Markdown("Geen gegevens in dit bereik.")

        # -- Sub-tab 3: Cost --
        @pn.depends(date_slider.param.value, agg_select.param.value)
        def _cost_charts(date_range, agg_label):
            freq = _AGG_OPTIONS[agg_label]
            ct = _filter_resample(cost_df, date_range, freq)
            charts = []

            if not ct.empty:
                chart4 = ct.hvplot.area(
                    x="timestamp",
                    y="cost_eur",
                    alpha=0.6,
                    color=_COST_COLOR,
                    title="Gemeenschapskosten lokaal delen (EUR)",
                    responsive=True,
                    min_height=280,
                )
                charts.append(pn.pane.HoloViews(chart4, sizing_mode="stretch_width"))

            # Per-prosumer cost (only if ≤20 prosumers)
            if n_prosumers <= 20 and not prosumer_df.empty:
                start, end = _to_utc(date_range[0]), _to_utc(date_range[1])
                mask_p = (prosumer_df["timestamp"] >= start) & (prosumer_df["timestamp"] <= end)
                pdf = prosumer_df[mask_p].copy()
                if freq:
                    parts = []
                    for pid, grp in pdf.groupby("meter_id"):
                        r = resample_df(grp.drop(columns="meter_id"), freq)
                        r["meter_id"] = pid
                        parts.append(r)
                    pdf = pd.concat(parts, ignore_index=True) if parts else pdf

                if not pdf.empty and "cost_eur" in pdf.columns:
                    chart6 = pdf.hvplot.line(
                        x="timestamp",
                        y="cost_eur",
                        by="meter_id",
                        title=f"Kosten lokaal delen per prosumer ({n_prosumers} meters)",
                        responsive=True,
                        min_height=280,
                        legend="right",
                    )
                    charts.append(pn.pane.HoloViews(chart6, sizing_mode="stretch_width"))

            # Market cost charts (only when market pricing was run)
            if not market_cost_df.empty:
                mc = _filter_resample(market_cost_df, date_range, freq)
                if not mc.empty:
                    market_ys = [c for c in ["import_cost_eur", "export_revenue_eur"] if c in mc.columns]
                    if market_ys:
                        _MARKET_COLORS = {
                            "import_cost_eur": _IMPORT_COLOR,
                            "export_revenue_eur": _SUPPLY_COLOR,
                        }
                        colors = [_MARKET_COLORS[y] for y in market_ys]
                        chart_market = mc.hvplot.area(
                            x="timestamp",
                            y=market_ys,
                            alpha=0.5,
                            color=colors,
                            title="Marktkosten: import & export (EUR)",
                            responsive=True,
                            min_height=280,
                            legend="top_left",
                        )
                        charts.append(pn.pane.HoloViews(chart_market, sizing_mode="stretch_width"))
                    if "net_cost_eur" in mc.columns:
                        chart_net = mc.hvplot.line(
                            x="timestamp",
                            y="net_cost_eur",
                            color=_ALLOC_COLOR,
                            title="Netto marktkosten (EUR)",
                            responsive=True,
                            min_height=240,
                        )
                        charts.append(pn.pane.HoloViews(chart_net, sizing_mode="stretch_width"))

            return pn.Column(*charts, sizing_mode="stretch_width") if charts else pn.pane.Markdown("Geen gegevens in dit bereik.")

        inner_tabs = pn.Tabs(
            ("Energiestromen", _energy_charts),
            ("Zelfvoorzienendheid & consumptie", _efficiency_charts),
            ("Kosten", _cost_charts),
            ("Gemiddeld profiel", self._profile_tab(pipeline)),
            dynamic=True,
            sizing_mode="stretch_width",
        )

        return pn.Column(
            pn.Row(date_slider, agg_select),
            inner_tabs,
            sizing_mode="stretch_width",
        )

    # ------------------------------------------------------------------
    # Average Profile tab (inside Explore inner tabs)
    # ------------------------------------------------------------------

    def _profile_tab(self, pipeline) -> pn.viewable.Viewable:
        import hvplot.pandas  # noqa: F401

        profiles = available_profiles(pipeline.step)
        if not profiles:
            return pn.pane.Alert(
                "Onvoldoende gegevens voor een gemiddeld profiel (minimaal 2 dagen vereist).",
                alert_type="warning",
            )

        _PROFILE_NL = {
            "Daily": "Dagelijks",
            "Weekly": "Wekelijks",
            "Yearly": "Jaarlijks",
        }
        _PROFILE_FROM_NL = {v: k for k, v in _PROFILE_NL.items()}
        profiles_nl = [_PROFILE_NL[p] for p in profiles]

        toggle = pn.widgets.RadioButtonGroup(
            options=profiles_nl,
            value=profiles_nl[0],
            button_type="default",
            button_style="outline",
        )

        _X_LABELS = {
            "Daily": "hour_of_day",
            "Weekly": "week_position",
            "Yearly": "day_of_year",
        }
        _TITLES = {
            "Daily": "Gemiddeld dagprofiel (gemiddeld kWh per kwartier)",
            "Weekly": "Gemiddeld weekprofiel (gemiddeld kWh per kwartier)",
            "Yearly": "Gemiddeld jaarprofiel (gemiddelde dagelijkse kWh per dag van het jaar)",
        }
        _XLABELS = {
            "Daily": "Uur van de dag",
            "Weekly": "Dag van de week",
            "Yearly": "Dag van het jaar",
        }
        _YLABELS = {
            "Daily": "kWh (gemiddeld per tijdvak)",
            "Weekly": "kWh (gemiddeld per tijdvak)",
            "Yearly": "kWh (gemiddeld dagelijks totaal)",
        }
        # week_position tick positions and labels (at start of each day)
        _WEEK_XTICKS = [(i, d) for i, d in enumerate(["Ma", "Di", "Wo", "Do", "Vr", "Za", "Zo"])]

        # Pre-compute all needed profile DataFrames
        _dfs = {p: make_avg_profile_df(pipeline.step, p) for p in profiles}

        @pn.depends(toggle.param.value)
        def _chart(profile_nl):
            profile = _PROFILE_FROM_NL[profile_nl]
            df = _dfs[profile]
            x = _X_LABELS[profile]

            extra_opts = {}
            if profile == "Weekly":
                extra_opts = {"xticks": _WEEK_XTICKS, "xlim": (0, 7)}

            demand_chart = df.hvplot.area(
                x=x,
                y="demand_kWh",
                alpha=0.4,
                color=_DEMAND_COLOR,
                label="Vraag",
                responsive=True,
                min_height=320,
            ) * df.hvplot.line(
                x=x,
                y="demand_kWh",
                color=_DEMAND_COLOR,
                line_width=1.5,
                label="",
            )
            supply_chart = df.hvplot.area(
                x=x,
                y="supply_kWh",
                alpha=0.4,
                color=_SUPPLY_COLOR,
                label="Aanbod",
            ) * df.hvplot.line(
                x=x,
                y="supply_kWh",
                color=_SUPPLY_COLOR,
                line_width=1.5,
                label="",
            )
            chart = (demand_chart * supply_chart).opts(
                title=_TITLES[profile],
                xlabel=_XLABELS[profile],
                ylabel=_YLABELS[profile],
                legend_position="top_right",
                **extra_opts,
            )

            return pn.pane.HoloViews(chart, sizing_mode="stretch_width")

        return pn.Column(
            pn.Row(toggle),
            _chart,
            sizing_mode="stretch_width",
        )

    # ------------------------------------------------------------------
    # Tab 2: Prosumer Table + Export
    # ------------------------------------------------------------------

    def _table_tab(self, pipeline) -> pn.viewable.Viewable:
        df = build_prosumer_table(pipeline)

        table = pn.widgets.Tabulator(
            df,
            pagination="remote",
            page_size=20,
            show_index=False,
            sizing_mode="stretch_width",
        )

        prosumer_dl = pn.widgets.FileDownload(
            callback=lambda: io.BytesIO(prosumer_csv_bytes(pipeline)),
            filename="prosumer_summary.csv",
            label="Download prosumer-CSV",
            button_type="primary",
        )
        timeseries_dl = pn.widgets.FileDownload(
            callback=lambda: io.BytesIO(timeseries_csv_bytes(pipeline)),
            filename="timeseries.csv",
            label="Download tijdreeks-CSV",
            button_type="light",
        )

        return pn.Column(
            pn.Row(prosumer_dl, timeseries_dl),
            table,
            sizing_mode="stretch_width",
        )

    # ------------------------------------------------------------------
    # Public panel method
    # ------------------------------------------------------------------

    def _content(self, pipeline) -> pn.viewable.Viewable:
        if pipeline is None:
            return pn.pane.Alert(
                "Nog geen simulatieresultaten. Voer eerst de simulatie uit.",
                alert_type="warning",
            )
        return pn.Tabs(
            ("Verkennen", self._explore_tab(pipeline)),
            ("Prosumertabel", self._table_tab(pipeline)),
            dynamic=True,
            sizing_mode="stretch_width",
        )

    def panel(self) -> pn.viewable.Viewable:
        return pn.Column(
            pn.pane.Markdown("# Resultaten"),
            pn.bind(self._content, self._state.param.pipeline),
            sizing_mode="stretch_width",
        )
