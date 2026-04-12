# Documentation Index - Energie Delen Simulation

## Where to start

| Goal | File |
|------|------|
| Preparing my data | `docs/data_formats.md` and `docs/PRIVACY.md`(NOT YET) |
| **Use the dashboard** | `docs/DASHBOARD.md` |
| Understand the project | `README.md` |
| Find your path | `GETTING_STARTED.md` |
| All commands and architecture summary | `CLAUDE.md` (project root) |

---

## Documentation map

### User docs (`simulation/`)

| File | What it covers |
|------|---------------|
| `README.md` | Architecture overview, CLI reference, Python API, module reference |
| `GETTING_STARTED.md` | Role-based entry points: run data, understand code, extend, privacy |
| `docs/DASHBOARD.md` | How to start the dashboard, page-by-page workflow, charts, export |

### Technical reference (`simulation/docs/`)

| File | What it covers |
|------|---------------|
| `data_formats.md` | Parquet schema, sign convention, validation rules |
| `PRIVACY.md` | GDPR, pseudonymization, Dutch regulatory context |

---

## Module overview

| Module | Input | Output |
|--------|-------|--------|
| Loader (`src/loader.py`) | Parquet files + `SimulationConfig` | `LoadedDataset` + `CoverageReport` |
| Aggregator (`src/aggregator.py`) | `LoadedDataset` | `AggregatedStep` |
| Allocation (`src/allocation.py`) | `LoadedDataset` + `AggregatedStep` | `AllocationResult` |
| Pricing (`src/pricing.py`) | `AllocationResult` | `PricingResult` |
| Visualization (`src/viz/`) | Any result dataclass | `matplotlib.Figure` |
| Dashboard (`dashboard/`) | `PipelineResult` (via Panel/hvplot) | Interactive browser UI |