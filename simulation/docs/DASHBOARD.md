# Dashboard — User Guide

The dashboard is the primary way to use Energie Delen. It is a browser-based interface that walks you through the full simulation workflow without needing to use the command line.

---

## Starting the dashboard

### Windows

Double-click `launch_dashboard.bat`, or run it from a terminal:

```
launch_dashboard.bat
```

### macOS / Linux

```bash
./launch_dashboard.sh
```

The dashboard is served at `http://localhost:5006`. If the browser does not open automatically, navigate there manually.

### Manual start from the command line (advanced)

```bash
uv run --group dashboard panel serve dashboard/app.py --show
```

Add `--autoreload` during development to reload on code changes.

---

## Workflow overview

The dashboard is split into three pages, navigated with the sidebar buttons on the left.

```
1 · Data Input  →  2 · Simulation  →  3 · Results
```

You must complete each step in order. The **Results** button stays disabled until a simulation has been run successfully.

---

## Page 1 — Data Input

This page loads your data and gives you a first look at its quality before you configure the simulation.

### Step 1: Select your data

You have three options:

| Option | When to use |
|--------|-------------|
| **Load Sample Data** | First-time users; generates 5 prosumers + 2 production assets for 7 days |
| **Type or paste a file/folder path** | You have Parquet files on disk |
| **Upload a file** | You want to drag-and-drop a single `.parquet` file from your computer |

Two types of data can be supplied:
- **Prosumer data** — smart meter readings for each household (positive = consumption, negative = net injection)
- **Production assets** — output of shared generation assets (solar panels, etc.)

See `docs/data_formats.md` for the required Parquet schema to prepare your data.

### Step 2: Load & Inspect

Click **Load & Inspect**. The dashboard will:

1. Scan all meters.
2. Display a table listing every meter: its ID, time range, number of data points, inferred frequency, and percentage of missing values.
3. Show three interactive coverage charts:

| Chart | What it shows |
|-------|---------------|
| **Timeline** | How many meters have data at every point in time (stacked area) |
| **Coverage Heatmap** | Per-meter coverage fraction across time periods (green = full, red = gaps). Switch between Daily / Weekly / Monthly / Quarterly resolution. |
| **Missing %** | Horizontal bar chart ranking meters by their missing-data fraction |

Above the charts you will see the **Inspect Results** summary:

| Field | Meaning |
|-------|---------|
| Suggested start / end | The longest period where all meters overlap |
| Suggested frequency | The most common interval detected across meters |
| Complete period | Number of days in the overlap window |
| Freq consistent | Whether all meters share the same sampling frequency |

### Step 3: Proceed

Click **Next: Simulation Settings →** to move to the next page. The suggested start, end, and frequency are pre-filled in the simulation form automatically. Adjust the settings as desired.

---

## Page 2 — Simulation Settings

Configure and run the pipeline.

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| **Start date** | Simulation window start (DD-MM-YYYY) | From inspect |
| **End date** | Simulation window end (DD-MM-YYYY) | From inspect |
| **Frequency** | Timestep resolution: `15min`, `30min`, or `1H` | From inspect |
| **Missing data policy** | How to handle gaps in meter data | `fill_zero` |
| **NaN policy** | How NaN values behave during aggregation | `treat_as_zero` |
| **Local price (EUR/kWh)** | Fixed price charged for locally shared energy | `0.075` |

#### Missing data policy options

| Option | Behaviour |
|--------|-----------|
| `fill_zero` | Missing timesteps are treated as zero consumption / production |
| `fill_forward` | Gaps are filled by propagating the last known value |
| `keep_nan` | Gaps remain as NaN and propagate through the pipeline |
| `error` | The run fails immediately if any meter has missing data |

#### NaN policy options

| Option | Behaviour |
|--------|-----------|
| `treat_as_zero` | NaN values contribute 0 to supply/demand totals |
| `propagate` | Any NaN at a timestep makes that timestep's total NaN |

### Running the simulation

Click **Run Simulation**. The **Pipeline log** area shows the output from each pipeline stage (loader → aggregator → allocator → pricing).

On success, a row of **KPI cards** appears immediately:

**Energy (kWh)**
- Total Demand — total consumption over the period
- Total Supply — total locally generated energy
- Locally Allocated — local supply actually distributed to prosumers
- Grid Import — demand not met locally; drawn from the public grid
- Grid Export — local supply not consumed locally; fed back to the grid

**Efficiency (%)**
- Self-Sufficiency — share of total demand covered by local supply
- Self-Consumption — share of local supply consumed within the community

**Cost (EUR)**
- Community Cost — total charge for locally allocated energy across all prosumers

Click **View Results →** to open the full results explorer.

---

## Page 3 — Results

The results page has two tabs.

### Tab 1 — Explore

Interactive time-series charts with the option to download them. Two controls apply to all charts:

- **Date range slider** — zoom into any sub-period of the simulation window
- **Aggregation selector** — view raw 15-min data, or resample to Hourly / Daily / Weekly

Four sub-tabs:

#### Energy Flows
- **Supply vs Demand** — stacked area chart comparing total community demand (blue) and available local supply (green)
- **Energy Flows** — stacked area showing locally allocated energy (amber) and grid import (red), with grid export (purple) as a line overlay

#### Self-Sufficiency & Consumption
- **Self-Sufficiency Rate** — ratio of local allocation to total demand over time
- **Self-Consumption Rate** — ratio of local allocation to total supply over time
- **Per-Prosumer Allocation** — individual meter allocation lines (shown when ≤ 20 prosumers)

#### Cost
- **Community Cost** — total EUR cost of locally shared energy over time
- **Per-Prosumer Cost** — individual meter cost lines (shown when ≤ 20 prosumers)

#### Average Profile
Aggregates all timesteps by time-of-day / week / year to reveal structural patterns:

| Profile | X-axis | Use |
|---------|--------|-----|
| Daily | Hour of day (0–24) | Morning/evening peaks |
| Weekly | Day of week (Mon–Sun) | Weekday vs weekend patterns |
| Yearly | Day of year | Seasonal variation |

Each profile shows mean supply (green) and demand (blue) as overlaid area charts.

### Tab 2 — Prosumer Table & Export

A paginated table with one row per prosumer, summarising their totals for the simulation period: allocated kWh, grid import, grid export, self-sufficiency rate, and cost in EUR.

Two download buttons:

| Button | File | Contents |
|--------|------|----------|
| **Download Prosumer CSV** | `prosumer_summary.csv` | One row per prosumer, aggregated totals |
| **Download Time Series CSV** | `timeseries.csv` | Full timestep-level data for every prosumer |

---

## Data flow summary

```
Data Input page
  └─ prosumer_path + production_path
        ↓  inspect_dataset()
  └─ inspect_result (meter list, coverage, suggested config)

Simulation page
  └─ SimulationConfig (start, end, freq, missing_data, nan_policy, price)
        ↓  run_pipeline()
  └─ PipelineResult (dataset → step → allocation → pricing)

Results page
  └─ reads PipelineResult (read-only)
  └─ interactive charts + CSV export
```

---

## Tips

- **No data yet?** Use **Load Sample Data** on the Data Input page to generate a working dataset instantly.
- **Frequency mismatch warning?** The inspect step will flag it. Check that all meter files use the same sampling interval.
- **No overlap warning?** Your prosumer and production files cover different time windows. Check that their date ranges intersect.
- **Results button is greyed out?** A simulation has not been run yet on this session. Complete page 2 first.
- **Re-running with different settings:** Go back to page 2 at any time and click **Run Simulation** again. The results page updates automatically.
