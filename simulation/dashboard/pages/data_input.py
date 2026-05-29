"""Data Input page — 3-step progressive flow.

Step 1: Select prosumer and production paths (file upload or text path).
Step 2: Show meter lists (after inspect).
Step 3: Show inspect results and coverage charts.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import panel as pn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cli import inspect_dataset
from src.sample_data import SampleDataGenerator
from dashboard.components.file_upload import bytes_to_tempfile
from dashboard.state import AppState


class DataInputPage:
    def __init__(self, state: AppState) -> None:
        self._state = state
        self._build_widgets()

    # ------------------------------------------------------------------
    # Widget construction
    # ------------------------------------------------------------------

    def _build_widgets(self) -> None:
        s = self._state

        # --- Prosumer input ---
        self._prosumer_path_input = pn.widgets.TextInput(
            name="Prosumer pad (bestand of map)",
            placeholder="/pad/naar/prosumers.parquet  of  /pad/naar/map/",
            sizing_mode="stretch_width",
        )
        self._prosumer_upload = pn.widgets.FileInput(
            name="of upload een enkel bestand",
            accept=".parquet",
            sizing_mode="stretch_width",
        )

        # --- Production input ---
        self._production_path_input = pn.widgets.TextInput(
            name="Productie-assets pad (bestand of map)",
            placeholder="/pad/naar/production.parquet  of  /pad/naar/map/",
            sizing_mode="stretch_width",
        )
        self._production_upload = pn.widgets.FileInput(
            name="of upload een enkel bestand",
            accept=".parquet",
            sizing_mode="stretch_width",
        )

        # --- Action buttons ---
        self._sample_btn = pn.widgets.Button(
            name="Laad voorbeelddata",
            button_type="light",
            icon="database",
            sizing_mode="stretch_width",
        )
        self._inspect_btn = pn.widgets.Button(
            name="Laad & inspecteer",
            button_type="primary",
            icon="search",
            sizing_mode="stretch_width",
        )

        # --- Status ---
        self._status_pane = pn.pane.Markdown("", sizing_mode="stretch_width")

        # Wire callbacks
        self._prosumer_upload.param.watch(self._on_prosumer_upload, "value")
        self._production_upload.param.watch(self._on_production_upload, "value")
        self._sample_btn.on_click(self._on_sample)
        self._inspect_btn.on_click(self._on_inspect)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_prosumer_upload(self, event) -> None:
        if event.new is None:
            return
        p = bytes_to_tempfile(event.new, role="prosumer")
        self._prosumer_path_input.value = str(p)
        self._state.prosumer_path = p

    def _on_production_upload(self, event) -> None:
        if event.new is None:
            return
        p = bytes_to_tempfile(event.new, role="production")
        self._production_path_input.value = str(p)
        self._state.production_path = p

    def _on_sample(self, _event) -> None:
        import tempfile
        self._status_pane.object = "Voorbeelddata genereren…"
        tmpdir = Path(tempfile.mkdtemp(prefix="energie_demo_"))
        prosumer_path, production_path = SampleDataGenerator.generate_sample_dataset(
            output_dir=tmpdir, num_prosumers=5, num_assets=2, num_days=7
        )
        self._state.prosumer_path = prosumer_path
        self._state.production_path = production_path
        self._prosumer_path_input.value = str(prosumer_path)
        self._production_path_input.value = str(production_path)
        self._status_pane.object = f"Voorbeelddata klaar in `{tmpdir}`"

    def _on_inspect(self, _event) -> None:
        # Resolve paths from text inputs (may differ from uploaded temp files)
        p_str = self._prosumer_path_input.value.strip()
        a_str = self._production_path_input.value.strip()

        prosumer_path = Path(p_str) if p_str else self._state.prosumer_path
        production_path = Path(a_str) if a_str else self._state.production_path

        if not prosumer_path and not production_path:
            self._status_pane.object = "⚠ Geef ten minste één datapad op."
            return

        self._state.prosumer_path = prosumer_path
        self._state.production_path = production_path
        self._state.inspect_status = "loading"
        self._state.inspect_result = None
        self._status_pane.object = "Data laden en inspecteren…"
        self._inspect_btn.disabled = True

        def _worker() -> None:
            try:
                result = inspect_dataset(
                    prosumer_path=prosumer_path,
                    production_path=production_path,
                )
                self._state.inspect_result = result
                self._state.inspect_status = "done"
                self._status_pane.object = "Inspectie voltooid."
            except Exception as exc:  # noqa: BLE001
                self._state.inspect_status = f"error: {exc}"
                self._status_pane.object = f"⚠ {exc}"
            finally:
                self._inspect_btn.disabled = False

        threading.Thread(target=_worker, daemon=True).start()

    # ------------------------------------------------------------------
    # Reactive panels
    # ------------------------------------------------------------------

    @pn.depends("_state.inspect_status")
    def _meter_lists(self) -> pn.viewable.Viewable:
        result = self._state.inspect_result
        if result is None:
            return pn.pane.Markdown("")

        prosumer_rows = [
            {
                "meter_id": m.meter_id,
                "start": str(m.start)[:19],
                "end": str(m.end)[:19],
                "n_points": m.n_points,
                "freq": m.freq or "?",
                "NaN %": f"{m.nan_fraction * 100:.1f}%",
            }
            for m in result.meters
            if m.role == "prosumer"
        ]
        asset_rows = [
            {
                "meter_id": m.meter_id,
                "start": str(m.start)[:19],
                "end": str(m.end)[:19],
                "n_points": m.n_points,
                "freq": m.freq or "?",
                "NaN %": f"{m.nan_fraction * 100:.1f}%",
            }
            for m in result.meters
            if m.role == "asset"
        ]

        p_tab = pn.widgets.Tabulator(
            pd.DataFrame(prosumer_rows) if prosumer_rows else pd.DataFrame(),
            name=f"Prosumers ({len(prosumer_rows)})",
            show_index=False,
            pagination="remote",
            page_size=15,
            sizing_mode="stretch_width",
        )
        a_tab = pn.widgets.Tabulator(
            pd.DataFrame(asset_rows) if asset_rows else pd.DataFrame(),
            name=f"Productie-assets ({len(asset_rows)})",
            show_index=False,
            pagination="remote",
            page_size=15,
            sizing_mode="stretch_width",
        )

        return pn.Column(
            pn.pane.Markdown("## Meters gevonden"),
            pn.Row(
                pn.Column(pn.pane.Markdown(f"**Prosumers — {len(prosumer_rows)}**"), p_tab),
                pn.Column(pn.pane.Markdown(f"**Productie-assets — {len(asset_rows)}**"), a_tab),
            ),
            sizing_mode="stretch_width",
        )

    @pn.depends("_state.inspect_status")
    def _inspect_results(self) -> pn.viewable.Viewable:
        result = self._state.inspect_result
        if result is None:
            return pn.pane.Markdown("")

        # Summary card
        has_complete = result.suggested_start is not None
        if has_complete:
            sug_start = pd.to_datetime(result.suggested_start, dayfirst=True)
            sug_end = pd.to_datetime(result.suggested_end, dayfirst=True)
            complete_days = (sug_end - sug_start).days
        no_overlap_warn = "" if has_complete else "\n\n⚠ **Geen volledige overlap tussen prosumer- en assetdata.**"
        summary_md = f"""
## Inspectierapport

| | |
|---|---|
| **Voorgestelde start** | `{result.suggested_start}` |
| **Voorgesteld einde** | `{result.suggested_end}` |
| **Voorgestelde frequentie** | `{result.suggested_freq}` |
| **Volledige periode** | {f"{complete_days} dagen" if has_complete else "n/a"} |
| **Frequentie consistent** | {"ja" if result.freq_consistent else "nee ⚠"} |
{no_overlap_warn}
"""
        summary = pn.pane.Markdown(summary_md)

        # Interactive coverage charts in tabs
        coverage_tabs = self._build_coverage_panes(result)

        # Pre-fill simulation settings and navigate
        next_btn = pn.widgets.Button(
            name="Volgende: Simulatie-instellingen →",
            button_type="success",
            sizing_mode="stretch_width",
        )
        next_btn.on_click(self._on_next)

        return pn.Column(
            summary,
            coverage_tabs,
            next_btn,
            sizing_mode="stretch_width",
        )

    def _build_coverage_panes(self, result) -> pn.viewable.Viewable:
        """Build interactive hvplot coverage charts with a RadioButtonGroup selector."""
        import hvplot.pandas  # noqa: F401

        if not result.raw_meters:
            return pn.pane.Markdown("_Geen meterdata beschikbaar._")

        # Build expected index over the full global extent so all data is visible.
        # Must be tz-aware (UTC) to match meter timestamps.
        expected_index = pd.date_range(
            start=min(m.start for m in result.meters),
            end=max(m.end for m in result.meters),
            freq=result.suggested_freq,
            tz="UTC",
        )

        n_meters = len(result.raw_meters)
        chart_height = max(280, n_meters * 40)

        # Reindex each meter once — reused by all three charts
        reindexed = {
            m.meter_id: pd.Series(m.value, index=m.timestamp).reindex(expected_index)
            for m in result.raw_meters
        }

        # --- Chart 1: Coverage Heatmap ---
        # Build one heatmap per resolution; auto-select a sensible default (~12 columns)
        n_total_days = max(1, (expected_index[-1] - expected_index[0]).days + 1)
        _HEATMAP_RESOLUTIONS = {}  # label -> pandas resample freq
        if n_total_days <= 366:
            _HEATMAP_RESOLUTIONS["Dagelijks"] = "1D"
        if n_total_days >= 14:
            _HEATMAP_RESOLUTIONS["Wekelijks"] = "1W"
        if n_total_days >= 28:
            _HEATMAP_RESOLUTIONS["Maandelijks"] = "MS"
        if n_total_days >= 90:
            _HEATMAP_RESOLUTIONS["Kwartaal"] = "QS"
        if not _HEATMAP_RESOLUTIONS:
            _HEATMAP_RESOLUTIONS["Dagelijks"] = "1D"

        # Default resolution: ~12 columns
        if n_total_days <= 14:
            _heatmap_default = "Dagelijks"
        elif n_total_days <= 90:
            _heatmap_default = "Wekelijks"
        elif n_total_days <= 365:
            _heatmap_default = "Maandelijks"
        else:
            _heatmap_default = "Kwartaal"

        def _build_heatmap_df(freq: str) -> pd.DataFrame:
            rows = []
            for m in result.raw_meters:
                for ts, frac in reindexed[m.meter_id].notna().resample(freq).mean().items():
                    rows.append({
                        "date": str(ts.date()),
                        "meter_id": m.meter_id,
                        "coverage": float(frac) if not pd.isna(frac) else 0.0,
                    })
            return pd.DataFrame(rows)

        _heatmap_panes = {
            label: pn.pane.HoloViews(
                _build_heatmap_df(freq).hvplot.heatmap(
                    x="date", y="meter_id", C="coverage",
                    cmap="RdYlGn", clim=(0, 1),
                    title=f"Dekkingsheatmap ({label})",
                    responsive=True, min_height=chart_height,
                    rot=45,
                ),
                sizing_mode="stretch_width",
            )
            for label, freq in _HEATMAP_RESOLUTIONS.items()
        }
        _res_selector = pn.widgets.RadioButtonGroup(
            options=list(_HEATMAP_RESOLUTIONS.keys()),
            value=_heatmap_default,
            button_type="light",
            sizing_mode="stretch_width",
        )
        heatmap_pane = pn.Column(
            pn.Row(
                pn.pane.Markdown("**Resolutie:**", margin=(8, 6, 0, 0)),
                _res_selector,
            ),
            pn.bind(lambda res: _heatmap_panes[res], _res_selector),
            sizing_mode="stretch_width",
        )

        # --- Chart 2: Missing % bars ---
        bars_rows = []
        for m in result.raw_meters:
            vals = reindexed[m.meter_id]
            missing_pct = float(vals.isna().sum() / len(vals) * 100) if len(vals) > 0 else 100.0
            bars_rows.append({"meter_id": m.meter_id, "missing_pct": missing_pct})
        bars_df = pd.DataFrame(bars_rows).sort_values("missing_pct")
        bars_pane = pn.pane.HoloViews(
            bars_df.hvplot.barh(
                x="meter_id", y="missing_pct",
                color="#ef4444", alpha=0.7,
                xlabel="Ontbrekend (%)", ylabel="",
                title="Ontbrekende data % per meter",
                responsive=True, min_height=chart_height,
            ),
            sizing_mode="stretch_width",
        )

        # --- Chart 3: Coverage Timeline ---
        # Use DatetimeIndex as the DataFrame index (more reliable with hvplot.area)
        present_counts = np.zeros(len(expected_index), dtype=int)
        for m in result.raw_meters:
            present_counts += reindexed[m.meter_id].notna().astype(int).values
        timeline_df = pd.DataFrame(
            {"present": present_counts, "missing": n_meters - present_counts},
            index=expected_index,
        )
        timeline_df.index.name = "timestamp"
        # Downsample to hourly so Bokeh doesn't choke on 15-min points
        timeline_df = timeline_df.resample("1h").mean()
        timeline_pane = pn.pane.HoloViews(
            timeline_df.hvplot.area(
                y=["present", "missing"],
                stacked=True,
                color=["#22c55e", "#ef4444"], alpha=0.6,
                ylabel="Meters", title="Dekking over tijd",
                responsive=True, min_height=300,
                hover=False,
            ),
            sizing_mode="stretch_width",
        )

        # RadioButtonGroup acts as the tab bar — more reliable than pn.Tabs
        # inside a reactive pn.bind context; Timeline shown first
        _CHART_OPTIONS = ["Tijdlijn", "Dekkingsheatmap", "Ontbrekend %"]
        _CHART_MAP = {
            "Dekkingsheatmap": heatmap_pane,
            "Ontbrekend %": bars_pane,
            "Tijdlijn": timeline_pane,
        }
        selector = pn.widgets.RadioButtonGroup(
            options=_CHART_OPTIONS,
            value="Tijdlijn",
            button_type="light",
            sizing_mode="stretch_width",
        )
        chart_area = pn.bind(lambda sel: _CHART_MAP[sel], selector)

        return pn.Column(selector, chart_area, sizing_mode="stretch_width")

    def _on_next(self, _event) -> None:
        result = self._state.inspect_result
        if result:
            if result.suggested_start:
                self._state.start_date = result.suggested_start
            if result.suggested_end:
                self._state.end_date = result.suggested_end
            if result.suggested_freq:
                freq = result.suggested_freq
                if freq in ["15min", "30min", "1H"]:
                    self._state.freq = freq
        self._state.active_page = "simulation"

    # ------------------------------------------------------------------
    # Public panel method
    # ------------------------------------------------------------------

    def panel(self) -> pn.viewable.Viewable:
        return pn.Column(
            pn.pane.Markdown("# Gegevensinvoer"),
            # Step 1: path selection
            pn.pane.Markdown("## Stap 1: Selecteer uw gegevens"),
            pn.Row(
                pn.Column(
                    self._prosumer_path_input,
                    self._prosumer_upload,
                    sizing_mode="stretch_width",
                ),
                pn.Column(
                    self._production_path_input,
                    self._production_upload,
                    sizing_mode="stretch_width",
                ),
            ),
            pn.Row(self._sample_btn, self._inspect_btn),
            self._status_pane,
            pn.layout.Divider(),
            # Step 2: meter lists (reactive)
            pn.bind(lambda _: self._meter_lists(), self._state.param.inspect_status),
            pn.layout.Divider(),
            # Step 3: inspect results (reactive)
            pn.bind(lambda _: self._inspect_results(), self._state.param.inspect_status),
            sizing_mode="stretch_width",
        )
