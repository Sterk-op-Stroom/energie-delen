"""Dashboard entry point.

Launch with:
    uv run --group dashboard panel serve dashboard/app.py --show --autoreload

Or from Python:
    from dashboard.app import create_app
    app = create_app()
    app.show()
"""

from __future__ import annotations

import sys
from pathlib import Path

import os

os.environ.setdefault("BOKEH_RESOURCES", "server")

import threading
import panel as pn

sys.path.insert(0, str(Path(__file__).parent.parent))

from dashboard.state import AppState
from dashboard.pages.data_input import DataInputPage
from dashboard.pages.simulation import SimulationPage
from dashboard.pages.results import ResultsPage

pn.extension(
    "tabulator",
    "filedropper",
    sizing_mode="stretch_width",
    notifications=True,
)

def _on_session_destroyed(session_context, _pn=pn, _os=os, _threading=threading) -> None:
    existing = _pn.state.cache.get("_shutdown_timer")
    if existing is not None:
        existing.cancel()

    def _check_and_exit() -> None:
        if _pn.state.session_info.get("live", 0) == 0:
            _os._exit(0)

    timer = _threading.Timer(3.0, _check_and_exit)
    timer.daemon = True
    timer.start()
    _pn.state.cache["_shutdown_timer"] = timer


if not pn.state.cache.get("_shutdown_registered"):
    pn.state.cache["_shutdown_registered"] = True
    pn.state.on_session_destroyed(_on_session_destroyed)


def create_app() -> pn.Template:
    """Build and return the Panel FastListTemplate app."""
    state = AppState()

    data_page = DataInputPage(state)
    sim_page = SimulationPage(state)
    results_page = ResultsPage(state)

    # --- Sidebar navigation ---
    nav_data = pn.widgets.Button(
        name="1 · Data Input",
        button_type="light",
        sizing_mode="stretch_width",
    )
    nav_sim = pn.widgets.Button(
        name="2 · Simulation",
        button_type="light",
        sizing_mode="stretch_width",
    )
    nav_results = pn.widgets.Button(
        name="3 · Results",
        button_type="light",
        sizing_mode="stretch_width",
    )

    def _go_data(_):
        state.active_page = "data_input"

    def _go_sim(_):
        state.active_page = "simulation"

    def _go_results(_):
        state.active_page = "results"

    nav_data.on_click(_go_data)
    nav_sim.on_click(_go_sim)
    nav_results.on_click(_go_results)

    # Disable Results until a pipeline result exists
    @pn.depends(state.param.pipeline, watch=True)
    def _toggle_results_btn(pipeline):
        nav_results.disabled = pipeline is None

    nav_results.disabled = True  # initial state

    sidebar_content = pn.Column(
        pn.pane.Markdown("### Navigation", margin=(10, 5)),
        nav_data,
        nav_sim,
        nav_results,
        pn.layout.Divider(),
        pn.pane.Markdown(
            "Energie Delen\n\n"
            "[Data formats](docs/data_formats.md) · "
            "[Privacy](docs/PRIVACY.md)",
            styles={"font-size": "0.8em", "color": "#9ca3af"},
        ),
    )

    # --- Main area: swap pages reactively ---
    _page_instances = {
        "data_input": data_page,
        "simulation": sim_page,
        "results": results_page,
    }

    def _main_content(active_page: str) -> pn.viewable.Viewable:
        return _page_instances[active_page].panel()

    main_pane = pn.bind(_main_content, state.param.active_page)

    template = pn.template.FastListTemplate(
        title="Energie Delen Simulator",
        sidebar=[sidebar_content],
        main=[main_pane],
        accent_base_color="#22c55e",
        header_background="#1e3a5f",
        theme="default",
    )
    return template


# Allow `panel serve dashboard/app.py`
app = create_app()
app.servable()
