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
| **Laad voorbeelddata** | First-time users; generates 5 prosumers + 2 production assets for 7 days |
| **Type or paste a file/folder path** | You have Parquet files on disk |
| **Upload a file** | You want to drag-and-drop a single `.parquet` file from your computer |

Two types of data can be supplied:
- **Prosumer data** — smart meter readings for each household (positive = consumption, negative = net injection)
- **Production assets** — output of shared generation assets (solar panels, etc.)

See `docs/data_formats.md` for the required Parquet schema to prepare your data.

#### Stacking and combining files

Every upload is remembered — files are not replaced when you upload a new one. Instead each upload is appended to its own per-role list (prosumer files, production files separately).

Click **Geselecteerde bestanden (x/n geselecteerd)** (below the action buttons) to expand the file list. It shows two independent columns:

| Column | Contents |
|--------|----------|
| **Prosumers** | All uploaded prosumer files |
| **Productie-assets** | All uploaded production-asset files |

Each entry has:

| Control | Action |
|---------|--------|
| **☑ checkbox** (left) | Include this file in the active selection. Any combination across both columns can be checked simultaneously. |
| **🗑** (right) | Remove this entry from the list and delete its temp file. |

When **Laad & inspecteer** is clicked (see Step 2), all checked files for each role are merged and loaded together. If only one file is checked per role, it is used directly; if multiple are checked they are combined into a temporary directory that the loader auto-discovers. You can change the selection at any time and re-inspect without losing other uploaded files.

### Step 2: Load & Inspect

Click **Laad & inspecteer**. The dashboard will:

1. Scan all meters.
2. Display a table listing every meter: its ID, time range, number of data points, inferred frequency, and percentage of missing values.
3. Show three interactive coverage charts:

| Chart | What it shows |
|-------|---------------|
| **Tijdlijn** | How many meters have data at every point in time (stacked area) |
| **Dekkingsheatmap** | Per-meter coverage fraction across time periods (green = full, red = gaps). Switch between **Dagelijks / Wekelijks / Maandelijks / Kwartaal** resolution. |
| **Ontbrekend %** | Horizontal bar chart ranking meters by their missing-data fraction |

Above the charts you will see the **Inspectierapport** summary:

| Field | Meaning |
|-------|---------|
| Voorgestelde start / Voorgesteld einde | The longest period where all meters overlap |
| Voorgestelde frequentie | The most common interval detected across meters |
| Volledige periode | Number of days in the overlap window |
| Frequentie consistent | Whether all meters share the same sampling frequency (`ja` / `nee ⚠`) |

### Step 3: Proceed

Click **Volgende: Simulatie-instellingen →** to move to the next page. The suggested start, end, and frequency are pre-filled in the simulation form automatically. Adjust the settings as desired.

---

## Page 2 — Simulation Settings

Configure and run the pipeline.

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| **Start date** | Simulation window start (DD-MM-YYYY) | From inspect |
| **End date** | Simulation window end (DD-MM-YYYY) | From inspect |
| **Pricing model (local sharing)** | Strategy for pricing locally shared energy. Currently only **Vaste prijs** (fixed price) is available. | `Vaste prijs` |
| **Local price (EUR/kWh)** | Fixed price charged per kWh of locally allocated energy | `0.075` |
| **Market pricing model** | Optional pricing for residual grid flows. Select **Geen** to skip, **Vaste prijs** to set fixed import/export prices. | `Geen` |
| **Market import price (EUR/kWh)** | Price paid per kWh drawn from the grid after local sharing (visible when market pricing is active) | `0.25` |
| **Export compensation price (EUR/kWh)** | Revenue received per kWh exported to the grid after local sharing (visible when market pricing is active) | `0.09` |
| **Frequency** | Timestep resolution — enter a number and choose `min` or `sec` (e.g. 15 min, 30 sec). Toggle **Automatisch detecteren** to use the frequency inferred from the data. | From inspect |
| **Missing data policy** | How to handle gaps in meter data | `fill_zero` |
| **NaN policy** | How NaN values behave during aggregation | `treat_as_zero` |

The **Frequency**, **Missing data policy**, and **NaN policy** fields are hidden inside the collapsed **Geavanceerde instellingen** card. Click it to expand.

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

Click **Simulatie starten**. The **Pipeline-logboek** area shows the output from each pipeline stage (loader → aggregator → allocator → pricing).

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

**Market Cost (EUR)** — shown only when **Prijsmodel markt** is configured
- Market Import Cost — total grid import charges after local sharing
- Market Export Revenue — total grid export revenues after local sharing
- Net Market Cost — import cost minus export revenue

Click **Bekijk resultaten →** to open the full results explorer.

---

## Page 3 — Results

The results page has two tabs.

### Tab 1 — Verkennen

Interactive time-series charts with the option to download them. Two controls apply to all charts:

- **Datumbereik** slider — zoom into any sub-period of the simulation window
- **Aggregatie** selector — view raw data (**Onbewerkt (15 min)**), or resample to **Per uur / Dagelijks / Wekelijks**

Four sub-tabs:

#### Energiestromen
- **Aanbod vs Vraag (kWh)** — stacked area chart comparing total community demand (blue) and available local supply (green)
- **Energiestromen (kWh)** — stacked area showing locally allocated energy (amber) and grid import (red), with grid export (purple) as a line overlay

#### Zelfvoorzienendheid & consumptie
- **Zelfvoorzieningspercentage** — ratio of local allocation to total demand over time
- **Zelfconsumptiepercentage** — ratio of local allocation to total supply over time
- **Toewijzing per prosumer** — individual meter allocation lines (shown when ≤ 20 prosumers)

#### Kosten
- **Gemeenschapskosten lokaal delen (EUR)** — total EUR cost of locally shared energy over time
- **Kosten lokaal delen per prosumer** — individual meter cost lines (shown when ≤ 20 prosumers)
- **Marktkosten: import & export (EUR)** — area chart of grid import costs (red) and export revenues (green) per timestep (shown when market pricing is configured)
- **Netto marktkosten (EUR)** — net market cost per timestep: import cost minus export revenue (shown when both market import and export prices are configured)

#### Gemiddeld profiel
Aggregates all timesteps by time-of-day / week / year to reveal structural patterns:

| Profile | X-axis | Use |
|---------|--------|-----|
| **Dagelijks** | Hour of day (0–24) | Morning/evening peaks |
| **Wekelijks** | Day of week (Ma–Zo) | Weekday vs weekend patterns |
| **Jaarlijks** | Day of year | Seasonal variation |

Each profile shows mean supply (green) and demand (blue) as overlaid area charts.

### Tab 2 — Prosumertabel

A paginated table with one row per prosumer, summarising their totals for the simulation period: allocated kWh, grid import, self-sufficiency rate, and local sharing cost in EUR. When **Prijsmodel markt** is configured, three additional columns appear: market import cost, market export revenue, and net market cost per prosumer.

Two download buttons:

| Button | File | Contents |
|--------|------|----------|
| **Download prosumer-CSV** | `prosumer_summary.csv` | One row per prosumer, aggregated totals |
| **Download tijdreeks-CSV** | `timeseries.csv` | Full timestep-level data for every prosumer |

---

## Data flow summary

```
Data Input page
  └─ prosumer_files[] + production_files[]  (stacked upload history)
  └─ selected_prosumer_indices + selected_production_indices → prosumer_path + production_path
        ↓  inspect_dataset()
  └─ inspect_result (meter list, coverage, suggested config)

Simulation page
  └─ pricing settings (local price, optional market import/export prices)
  └─ SimulationConfig (start, end, freq, missing_data, nan_policy)
        ↓  run_pipeline()
  └─ PipelineResult
       ├─ dataset → step → allocation → pricing  (local sharing)
       ├─ pricing_market_import  (optional — grid import costs)
       └─ pricing_market_export  (optional — grid export revenues)

Results page
  └─ reads PipelineResult (read-only)
  └─ interactive charts + CSV export
```

---

## Tips

- **No data yet?** Use **Laad voorbeelddata** on the Data Input page to generate a working dataset instantly.
- **Frequency mismatch warning?** The inspect step will flag it. Check that all meter files use the same sampling interval.
- **No overlap warning?** Your prosumer and production files cover different time windows. Check that their date ranges intersect.
- **Results button is greyed out?** A simulation has not been run yet on this session. Complete page 2 first.
- **Re-running with different settings:** Go back to page 2 at any time and click **Simulatie starten** again. The results page updates automatically.
- **Testing different data combinations:** Upload multiple prosumer or production files, then use the **Geselecteerde bestanden** panel to check any combination, re-inspect, and run — without re-uploading.
