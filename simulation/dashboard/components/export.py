"""Export helpers: pipeline results → downloadable bytes."""

from __future__ import annotations

import io
import zipfile
import matplotlib
from typing import TYPE_CHECKING

import pandas as pd

from dashboard.components.prosumer_table import build_prosumer_table
from dashboard.transforms import make_supply_demand_df, make_allocation_df, make_community_cost_df

if TYPE_CHECKING:
    from cli import PipelineResult


def prosumer_csv_bytes(pipeline: PipelineResult) -> bytes:
    """Return UTF-8 CSV bytes of the per-prosumer summary table."""
    df = build_prosumer_table(pipeline)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()


def timeseries_csv_bytes(pipeline: PipelineResult) -> bytes:
    """Return UTF-8 CSV bytes of the main community time series.

    Columns: timestamp, demand_kWh, supply_kWh, net_kWh,
             local_allocation_kWh, grid_import_kWh, grid_export_kWh, cost_eur
    """
    sd = make_supply_demand_df(pipeline.step)
    alloc = make_allocation_df(pipeline.allocation)
    cost = make_community_cost_df(pipeline.pricing)[["timestamp", "cost_eur"]]

    merged = sd.merge(alloc, on="timestamp").merge(cost, on="timestamp")
    buf = io.StringIO()
    merged.to_csv(buf, index=False)
    return buf.getvalue().encode()


def figure_png_bytes(fig: matplotlib.figure.Figure, dpi: int = 150) -> bytes:
    """Return PNG bytes for a single matplotlib figure."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


def report_zip_bytes(
    figures: dict[str, matplotlib.figure.Figure], dpi: int = 150
) -> bytes:
    """Pack a dict of {filename: Figure} into a zip archive and return bytes.

    Args:
        figures: Mapping of filename (without extension) to matplotlib Figure.
        dpi: Resolution for PNG export.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, fig in figures.items():
            png = figure_png_bytes(fig, dpi=dpi)
            zf.writestr(f"{name}.png", png)
    buf.seek(0)
    return buf.read()
