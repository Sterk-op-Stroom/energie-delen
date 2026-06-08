"""Simulation Settings page.

Lets the user configure pipeline parameters, run the simulation,
and see a KPI summary on success.
"""

from __future__ import annotations

import html as _html
import io
import sys
import threading
import contextlib
from pathlib import Path

import panel as pn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cli import run_pipeline
from dashboard.state import AppState
from dashboard.components.kpi_cards import build_kpi_row

_MISSING_DATA_HELP = {
    "fill_zero": "Vul gaten op met 0 (standaard — ontbrekende meters behandeld als afwezig)",
    "fill_forward": "Vul gaten op vanuit de laatste bekende waarde",
    "keep_nan": "Laat gaten als NaN (doorgegeven door de aggregator)",
    "error": "Geef een fout als een meter ontbrekende gegevens heeft",
}
_NAN_POLICY_HELP = {
    "treat_as_zero": "NaN-waarden dragen 0 bij aan vraag/aanbod-totalen",
    "propagate": "Een NaN bij een tijdstap maakt het totaal van die tijdstap NaN",
}


class SimulationPage:
    def __init__(self, state: AppState) -> None:
        self._state = state
        self._log_buffer = io.StringIO()
        self._build_widgets()

    def _build_widgets(self) -> None:
        s = self._state

        self._start_input = pn.widgets.TextInput(
            name="Startdatum (DD-MM-JJJJ)",
            value=s.start_date or "01-01-2025",
            sizing_mode="stretch_width",
        )
        self._end_input = pn.widgets.TextInput(
            name="Einddatum (DD-MM-JJJJ)",
            value=s.end_date or "07-01-2025",
            sizing_mode="stretch_width",
        )
        freq_number, freq_unit = self._parse_freq_to_widgets(s.freq)
        self._freq_infer_toggle = pn.widgets.Toggle(
            name="Automatisch detecteren",
            value=s.freq_infer,
            button_type="primary",
            width=180,
        )
        self._freq_number = pn.widgets.IntInput(
            name="Interval",
            value=freq_number,
            start=1,
            end=999,
            width=90,
            disabled=s.freq_infer,
        )
        self._freq_unit = pn.widgets.Select(
            name="Eenheid",
            options=["min", "sec"],
            value=freq_unit,
            width=90,
            disabled=s.freq_infer,
        )
        self._freq_hint = pn.pane.Markdown(
            self._freq_hint_text(s.freq, s.freq_infer),
            styles={"font-size": "0.82em", "color": "#6b7280"},
            margin=(0, 0, 0, 0),
        )
        self._missing_select = pn.widgets.Select(
            name="Beleid voor ontbrekende gegevens",
            options=list(_MISSING_DATA_HELP),
            value=s.missing_data,
            sizing_mode="stretch_width",
        )
        self._nan_select = pn.widgets.RadioButtonGroup(
            name="NaN-beleid",
            options=list(_NAN_POLICY_HELP),
            value=s.nan_policy,
            sizing_mode="stretch_width",
        )
        self._price_input = pn.widgets.FloatInput(
            name="Lokale prijs (EUR / kWh)",
            value=s.price_eur_per_kwh,
            step=0.005,
            sizing_mode="stretch_width",
        )
        self._run_btn = pn.widgets.Button(
            name="Simulatie starten",
            button_type="success",
            icon="play",
            sizing_mode="stretch_width",
        )
        self._log_pane = pn.pane.HTML(
            self._log_html(""),
            sizing_mode="stretch_width",
        )

        # Wire run button — always connected regardless of inspect flow
        self._run_btn.on_click(self._on_run)

        # Watch start/end input changes to update state
        self._start_input.param.watch(lambda e: setattr(s, "start_date", e.new), "value")
        self._end_input.param.watch(lambda e: setattr(s, "end_date", e.new), "value")
        self._freq_infer_toggle.param.watch(self._on_freq_infer_toggle, "value")
        self._freq_number.param.watch(self._on_freq_widget_change, "value")
        self._freq_unit.param.watch(self._on_freq_widget_change, "value")
        self._missing_select.param.watch(lambda e: setattr(s, "missing_data", e.new), "value")
        self._nan_select.param.watch(lambda e: setattr(s, "nan_policy", e.new), "value")
        self._price_input.param.watch(lambda e: setattr(s, "price_eur_per_kwh", e.new), "value")

        # Pre-fill date/freq widgets whenever a new inspect result arrives
        s.param.watch(self._on_inspect_result, "inspect_result")

    @staticmethod
    def _log_html(text: str) -> str:
        return (
            "<div style='height:200px;overflow-y:auto;white-space:pre-wrap;"
            "font-family:monospace;font-size:0.82em;background:#f8fafc;"
            "border:1px solid #e2e8f0;border-radius:6px;padding:8px;"
            "color:#374151'>"
            + (_html.escape(text) if text else "<span style='color:#9ca3af'>Nog geen logboek.</span>")
            + "</div>"
        )

    def _on_inspect_result(self, event) -> None:
        result = event.new
        if result is None or result.suggested_start is None:
            return
        self._start_input.value = result.suggested_start
        self._end_input.value = result.suggested_end
        suggested = result.suggested_freq or ""
        if self._freq_infer_toggle.value:
            self._state.freq = suggested
            self._freq_hint.object = self._freq_hint_text(suggested, infer=True)
        else:
            n, u = self._parse_freq_to_widgets(suggested)
            self._freq_number.value = n
            self._freq_unit.value = u

    def _on_freq_infer_toggle(self, event) -> None:
        infer = event.new
        self._freq_number.disabled = infer
        self._freq_unit.disabled = infer
        self._state.freq_infer = infer
        if infer:
            result = self._state.inspect_result
            suggested = (result.suggested_freq or "") if result else ""
            self._state.freq = suggested or self._state.freq
            self._freq_hint.object = self._freq_hint_text(self._state.freq, infer=True)
        else:
            self._on_freq_widget_change(None)

    def _on_freq_widget_change(self, _event) -> None:
        if self._freq_infer_toggle.value:
            return
        n = self._freq_number.value
        u = self._freq_unit.value
        if n is None or n < 1:
            self._freq_hint.object = "⚠ Voer een getal in van 1 tot 999."
            return
        unit_str = "min" if u == "min" else "S"
        freq = f"{n}{unit_str}"
        self._state.freq = freq
        self._freq_hint.object = self._freq_hint_text(freq, infer=False)

    @staticmethod
    def _freq_hint_text(freq: str, infer: bool) -> str:
        if not freq:
            return ""
        if infer:
            return f"Gedetecteerde frequentie: **{freq}**"
        return f"Pandas-frequentie: **{freq}**"

    @staticmethod
    def _parse_freq_to_widgets(freq_str: str) -> tuple[int, str]:
        """Parse a pandas freq string into (number, unit) for the widgets.

        Converts hours to minutes. Falls back to (15, "min") if unparseable.
        """
        import re
        if freq_str:
            m = re.match(r'^(\d+)(min|T|H|S)$', freq_str, re.IGNORECASE)
            if m:
                n, unit = int(m.group(1)), m.group(2).upper()
                if unit in ('MIN', 'T'):
                    return min(n, 999), 'min'
                if unit == 'H':
                    return min(n * 60, 999), 'min'
                if unit == 'S':
                    return min(n, 999), 'sec'
        return 15, 'min'

    def _on_run(self, _event) -> None:
        s = self._state
        if not s.prosumer_path and not s.production_path:
            s.run_status = "error: Geen datapaden ingesteld. Ga eerst naar Data Input."
            return

        s.run_status = "running"
        s.pipeline = None
        self._run_btn.disabled = True
        self._log_pane.object = self._log_html("")

        def _worker() -> None:
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    result = run_pipeline(
                        start=s.start_date,
                        end=s.end_date,
                        prosumer_path=s.prosumer_path,
                        production_path=s.production_path,
                        freq=s.freq,
                        missing_data=s.missing_data,
                        nan_policy=s.nan_policy,
                        price_eur_per_kwh=s.price_eur_per_kwh,
                    )
                s.pipeline = result
                s.run_status = "done"
            except Exception as exc:  # noqa: BLE001
                s.run_status = f"error: {exc}"
            finally:
                self._log_pane.object = self._log_html(buf.getvalue())
                self._run_btn.disabled = False

        threading.Thread(target=_worker, daemon=True).start()

    @pn.depends("_state.run_status")
    def _status_panel(self) -> pn.viewable.Viewable:
        status = self._state.run_status
        if status == "idle":
            return pn.pane.Markdown("")
        if status == "running":
            return pn.Row(
                pn.indicators.Progress(active=True, sizing_mode="fixed", width=200),
                pn.pane.Markdown("Simulatie uitvoeren…"),
            )
        if status == "done":
            pipeline = self._state.pipeline
            kpi = build_kpi_row(pipeline) if pipeline is not None else pn.pane.Markdown("")
            next_btn = pn.widgets.Button(
                name="Bekijk resultaten →",
                button_type="primary",
                sizing_mode="stretch_width",
            )
            next_btn.on_click(lambda _: setattr(self._state, "active_page", "results"))
            return pn.Column(
                pn.pane.Alert("Simulatie voltooid!", alert_type="success"),
                kpi,
                next_btn,
            )
        if status.startswith("error:"):
            return pn.pane.Alert(status[len("error:"):].strip(), alert_type="danger")
        return pn.pane.Markdown("")

    def panel(self) -> pn.viewable.Viewable:
        return pn.Column(
            pn.pane.Markdown("# Simulatie-instellingen"),
            pn.Row(
                pn.Column(
                    self._start_input,
                    self._end_input,
                    pn.pane.Markdown("**Frequentie**", margin=(8, 0, 2, 0)),
                    pn.Row(
                        self._freq_number,
                        self._freq_unit,
                        pn.Column(
                            pn.Spacer(height=22),
                            self._freq_infer_toggle,
                            margin=(0, 0, 0, 0),
                        ),
                        align="end",
                    ),
                    self._freq_hint,
                    sizing_mode="stretch_width",
                ),
                pn.Column(
                    self._missing_select,
                    pn.pane.Markdown(
                        "\n".join(f"- **{k}**: {v}" for k, v in _MISSING_DATA_HELP.items()),
                        styles={"font-size": "0.85em", "color": "#6b7280"},
                    ),
                    sizing_mode="stretch_width",
                ),
            ),
            pn.pane.Markdown("**NaN-beleid** (aggregatiefase)"),
            self._nan_select,
            pn.pane.Markdown(
                "\n".join(f"- **{k}**: {v}" for k, v in _NAN_POLICY_HELP.items()),
                styles={"font-size": "0.85em", "color": "#6b7280"},
            ),
            self._price_input,
            pn.layout.Divider(),
            self._run_btn,
            pn.pane.Markdown("**Pipeline-logboek**", margin=(8, 0, 2, 0)),
            self._log_pane,
            pn.bind(lambda _: self._status_panel(), self._state.param.run_status),
            sizing_mode="stretch_width",
        )
