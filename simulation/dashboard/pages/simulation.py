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
    "fill_zero": "Fill gaps with 0 (default — missing meters treated as absent)",
    "fill_forward": "Forward-fill gaps from the last known value",
    "keep_nan": "Leave gaps as NaN (propagated by the aggregator)",
    "error": "Raise an error if any meter has missing data",
}
_NAN_POLICY_HELP = {
    "treat_as_zero": "NaN values contribute 0 to supply/demand totals",
    "propagate": "Any NaN at a timestep makes that timestep's total NaN",
}


class SimulationPage:
    def __init__(self, state: AppState) -> None:
        self._state = state
        self._log_buffer = io.StringIO()
        self._build_widgets()

    def _build_widgets(self) -> None:
        s = self._state

        self._start_input = pn.widgets.TextInput(
            name="Start date (DD-MM-YYYY)",
            value=s.start_date or "01-01-2025",
            sizing_mode="stretch_width",
        )
        self._end_input = pn.widgets.TextInput(
            name="End date (DD-MM-YYYY)",
            value=s.end_date or "07-01-2025",
            sizing_mode="stretch_width",
        )
        self._freq_select = pn.widgets.Select(
            name="Frequency",
            options=["15min", "30min", "1H"],
            value=s.freq,
            sizing_mode="stretch_width",
        )
        self._missing_select = pn.widgets.Select(
            name="Missing data policy",
            options=list(_MISSING_DATA_HELP),
            value=s.missing_data,
            sizing_mode="stretch_width",
        )
        self._nan_select = pn.widgets.RadioButtonGroup(
            name="NaN policy",
            options=list(_NAN_POLICY_HELP),
            value=s.nan_policy,
            sizing_mode="stretch_width",
        )
        self._price_input = pn.widgets.FloatInput(
            name="Local price (EUR / kWh)",
            value=s.price_eur_per_kwh,
            step=0.005,
            sizing_mode="stretch_width",
        )
        self._run_btn = pn.widgets.Button(
            name="Run Simulation",
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
        self._freq_select.param.watch(lambda e: setattr(s, "freq", e.new), "value")
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
            + (_html.escape(text) if text else "<span style='color:#9ca3af'>No log yet.</span>")
            + "</div>"
        )

    def _on_inspect_result(self, event) -> None:
        result = event.new
        if result is None or result.suggested_start is None:
            return
        self._start_input.value = result.suggested_start
        self._end_input.value = result.suggested_end
        self._freq_select.value = result.suggested_freq

    def _on_run(self, _event) -> None:
        s = self._state
        if not s.prosumer_path and not s.production_path:
            s.run_status = "error: No data paths set. Go to Data Input first."
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
                pn.pane.Markdown("Running simulation…"),
            )
        if status == "done":
            pipeline = self._state.pipeline
            kpi = build_kpi_row(pipeline) if pipeline is not None else pn.pane.Markdown("")
            next_btn = pn.widgets.Button(
                name="View Results →",
                button_type="primary",
                sizing_mode="stretch_width",
            )
            next_btn.on_click(lambda _: setattr(self._state, "active_page", "results"))
            return pn.Column(
                pn.pane.Alert("Simulation complete!", alert_type="success"),
                kpi,
                next_btn,
            )
        if status.startswith("error:"):
            return pn.pane.Alert(status[len("error:"):].strip(), alert_type="danger")
        return pn.pane.Markdown("")

    def panel(self) -> pn.viewable.Viewable:
        return pn.Column(
            pn.pane.Markdown("# Simulation Settings"),
            pn.Row(
                pn.Column(
                    self._start_input,
                    self._end_input,
                    self._freq_select,
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
            pn.pane.Markdown("**NaN policy** (aggregation stage)"),
            self._nan_select,
            pn.pane.Markdown(
                "\n".join(f"- **{k}**: {v}" for k, v in _NAN_POLICY_HELP.items()),
                styles={"font-size": "0.85em", "color": "#6b7280"},
            ),
            self._price_input,
            pn.layout.Divider(),
            self._run_btn,
            pn.pane.Markdown("**Pipeline log**", margin=(8, 0, 2, 0)),
            self._log_pane,
            pn.bind(lambda _: self._status_panel(), self._state.param.run_status),
            sizing_mode="stretch_width",
        )
