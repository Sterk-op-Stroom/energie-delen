"""Data Input page — 3-step progressive flow.

Step 1: Select prosumer and production paths (file upload or text path).
        Each upload is appended to a per-role list. 'Geselecteerde bestanden'
        opens two independent checkbox columns — check any combination of
        prosumer and production files to use them all together.
Step 2: Show meter lists (after inspect).
Step 3: Show inspect results and coverage charts.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import panel as pn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cli import inspect_dataset
from src.sample_data import SampleDataGenerator
from dashboard.components.file_upload import bytes_to_tempfile, register_cleanup, resolve_role_path
from dashboard.state import AppState


class DataInputPage:
    def __init__(self, state: AppState) -> None:
        self._state = state
        self._build_widgets()

    # ------------------------------------------------------------------
    # Widget construction
    # ------------------------------------------------------------------

    def _build_widgets(self) -> None:
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

        # --- File set list toggle ---
        self._file_list_toggle = pn.widgets.Toggle(
            name=self._file_list_label(),
            button_type="light",
            icon="list",
            value=False,
            sizing_mode="stretch_width",
        )

        # --- Status ---
        self._status_pane = pn.pane.Markdown("", sizing_mode="stretch_width")

        # Wire callbacks
        self._prosumer_upload.param.watch(self._on_prosumer_upload, "value")
        self._production_upload.param.watch(self._on_production_upload, "value")
        self._sample_btn.on_click(self._on_sample)
        self._inspect_btn.on_click(self._on_inspect)
        self._state.param.watch(self._sync_file_list_label, "file_sets_version")

    # ------------------------------------------------------------------
    # File list helpers
    # ------------------------------------------------------------------

    def _file_list_label(self) -> str:
        n_p = len(self._state.prosumer_files)
        n_a = len(self._state.production_files)
        sel_p = len(self._state.selected_prosumer_indices)
        sel_a = len(self._state.selected_production_indices)
        total = n_p + n_a
        selected = sel_p + sel_a
        if total == 0:
            return "Geselecteerde bestanden (0)"
        return f"Geselecteerde bestanden ({selected}/{total} geselecteerd)"

    def _sync_file_list_label(self, _=None) -> None:
        self._file_list_toggle.name = self._file_list_label()

    def _build_role_path(self, role: str) -> Path | None:
        """Resolve the effective path for a role from the checkbox selection.

        If the text input holds a valid absolute or existing path it is used as
        a manual override. Otherwise delegates to resolve_role_path().
        """
        if role == "prosumer":
            files = self._state.prosumer_files
            indices = self._state.selected_prosumer_indices
            text = self._prosumer_path_input.value.strip()
        else:
            files = self._state.production_files
            indices = self._state.selected_production_indices
            text = self._production_path_input.value.strip()

        # Manual text input takes precedence when it looks like a real path
        fallback: Path | None = None
        if text:
            candidate = Path(text)
            if candidate.is_absolute() or candidate.exists():
                fallback = candidate

        return resolve_role_path(files, indices, role, fallback=fallback)

    def _on_toggle_selection(self, role: str, idx: int, checked: bool) -> None:
        if role == "prosumer":
            sel = list(self._state.selected_prosumer_indices)
            if checked and idx not in sel:
                sel.append(idx)
            elif not checked and idx in sel:
                sel.remove(idx)
            self._state.selected_prosumer_indices = sel
        else:
            sel = list(self._state.selected_production_indices)
            if checked and idx not in sel:
                sel.append(idx)
            elif not checked and idx in sel:
                sel.remove(idx)
            self._state.selected_production_indices = sel
        self._state.file_sets_version += 1

    def _on_delete_file(self, role: str, idx: int) -> None:
        if role == "prosumer":
            files = list(self._state.prosumer_files)
            sel = list(self._state.selected_prosumer_indices)
        else:
            files = list(self._state.production_files)
            sel = list(self._state.selected_production_indices)

        if idx < len(files):
            try:
                files[idx][0].unlink(missing_ok=True)
            except OSError:
                pass
            files.pop(idx)

        # Remove the deleted index and shift down any higher indices
        sel = [s for s in sel if s != idx]
        sel = [s - 1 if s > idx else s for s in sel]

        if role == "prosumer":
            self._state.prosumer_files = files
            self._state.selected_prosumer_indices = sel
        else:
            self._state.production_files = files
            self._state.selected_production_indices = sel

        self._state.file_sets_version += 1

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_prosumer_upload(self, event) -> None:
        if event.new is None:
            return
        p = bytes_to_tempfile(event.new, role="prosumer")
        filename = self._prosumer_upload.filename or ""
        pf = list(self._state.prosumer_files)
        pf.append((p, filename))
        self._state.prosumer_files = pf
        new_idx = len(pf) - 1
        sel = list(self._state.selected_prosumer_indices)
        if new_idx not in sel:
            sel.append(new_idx)
        self._state.selected_prosumer_indices = sel
        self._prosumer_path_input.value = filename
        self._state.file_sets_version += 1

    def _on_production_upload(self, event) -> None:
        if event.new is None:
            return
        p = bytes_to_tempfile(event.new, role="production")
        filename = self._production_upload.filename or ""
        af = list(self._state.production_files)
        af.append((p, filename))
        self._state.production_files = af
        new_idx = len(af) - 1
        sel = list(self._state.selected_production_indices)
        if new_idx not in sel:
            sel.append(new_idx)
        self._state.selected_production_indices = sel
        self._production_path_input.value = filename
        self._state.file_sets_version += 1

    def _on_sample(self, _event) -> None:
        self._status_pane.object = "Voorbeelddata genereren…"
        tmpdir = Path(tempfile.mkdtemp(prefix="energie_demo_"))
        register_cleanup(tmpdir)
        prosumer_path, production_path = SampleDataGenerator.generate_sample_dataset(
            output_dir=tmpdir, num_prosumers=5, num_assets=2, num_days=7
        )
        pf = list(self._state.prosumer_files)
        af = list(self._state.production_files)
        pf.append((prosumer_path, prosumer_path.name))
        af.append((production_path, production_path.name))
        self._state.prosumer_files = pf
        self._state.production_files = af
        sel_p = list(self._state.selected_prosumer_indices)
        sel_a = list(self._state.selected_production_indices)
        new_p = len(pf) - 1
        new_a = len(af) - 1
        if new_p not in sel_p:
            sel_p.append(new_p)
        if new_a not in sel_a:
            sel_a.append(new_a)
        self._state.selected_prosumer_indices = sel_p
        self._state.selected_production_indices = sel_a
        self._prosumer_path_input.value = prosumer_path.name
        self._production_path_input.value = production_path.name
        self._state.file_sets_version += 1
        self._status_pane.object = f"Voorbeelddata klaar in `{tmpdir}`"

    def _on_inspect(self, _event) -> None:
        prosumer_path = self._build_role_path("prosumer")
        production_path = self._build_role_path("production")

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
    # File set list panel
    # ------------------------------------------------------------------

    def _render_file_list(self) -> pn.viewable.Viewable:
        pf = self._state.prosumer_files
        af = self._state.production_files
        sel_p = set(self._state.selected_prosumer_indices)
        sel_a = set(self._state.selected_production_indices)

        def _make_column(files, selected_set, role, header):
            rows = [pn.pane.Markdown(f"**{header}**", margin=(0, 0, 4, 0))]
            if not files:
                rows.append(pn.pane.Markdown("_Geen bestanden._", styles={"color": "#9ca3af"}))
            else:
                for i, (_path, filename) in enumerate(files):
                    cb = pn.widgets.Checkbox(
                        value=(i in selected_set),
                        width=20,
                        margin=(6, 6, 0, 2),
                    )
                    cb.param.watch(
                        lambda e, idx=i, r=role: self._on_toggle_selection(r, idx, e.new),
                        "value",
                    )
                    trash_btn = pn.widgets.Button(
                        name="",
                        icon="trash",
                        button_type="light",
                        width=36,
                        height=32,
                        margin=(2, 0, 2, 4),
                    )
                    trash_btn.on_click(lambda _, idx=i, r=role: self._on_delete_file(r, idx))
                    rows.append(pn.Row(
                        cb,
                        pn.pane.Markdown(filename, sizing_mode="stretch_width", margin=(6, 4)),
                        trash_btn,
                    ))
            return pn.Column(*rows, sizing_mode="stretch_width")

        return pn.Row(
            _make_column(pf, sel_p, "prosumer", "Prosumers"),
            pn.Spacer(width=16),
            _make_column(af, sel_a, "production", "Productie-assets"),
            sizing_mode="stretch_width",
        )

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
        coverage_tabs = self._build_coverage_panes(result)

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

        expected_index = pd.date_range(
            start=min(m.start for m in result.meters),
            end=max(m.end for m in result.meters),
            freq=result.suggested_freq,
            tz="UTC",
        )

        n_meters = len(result.raw_meters)
        chart_height = max(280, n_meters * 40)

        reindexed = {
            m.meter_id: pd.Series(m.value, index=m.timestamp).reindex(expected_index)
            for m in result.raw_meters
        }

        n_total_days = max(1, (expected_index[-1] - expected_index[0]).days + 1)
        _HEATMAP_RESOLUTIONS = {}
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

        present_counts = np.zeros(len(expected_index), dtype=int)
        for m in result.raw_meters:
            present_counts += reindexed[m.meter_id].notna().astype(int).values
        timeline_df = pd.DataFrame(
            {"present": present_counts, "missing": n_meters - present_counts},
            index=expected_index,
        )
        timeline_df.index.name = "timestamp"
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
                self._state.freq = result.suggested_freq
        self._state.active_page = "simulation"

    # ------------------------------------------------------------------
    # Public panel method
    # ------------------------------------------------------------------

    def panel(self) -> pn.viewable.Viewable:
        file_list_area = pn.bind(
            lambda toggle_val, _ver: self._render_file_list() if toggle_val else pn.pane.Markdown(""),
            self._file_list_toggle,
            self._state.param.file_sets_version,
        )

        return pn.Column(
            pn.pane.Markdown("# Gegevensinvoer"),
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
            self._file_list_toggle,
            file_list_area,
            self._status_pane,
            pn.layout.Divider(),
            pn.bind(lambda _: self._meter_lists(), self._state.param.inspect_status),
            pn.layout.Divider(),
            pn.bind(lambda _: self._inspect_results(), self._state.param.inspect_status),
            sizing_mode="stretch_width",
        )
