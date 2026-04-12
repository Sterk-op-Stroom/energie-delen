# Data Format Specification

This document describes the expected format for input data files used in the energy sharing simulation.

## Overview

The simulation accepts time series data for two types of entities:

1. **Prosumers**: Households/buildings that consume electricity
2. **Production Assets**: Generation systems (e.g., solar PV, wind turbines)

All data is expected in **Parquet format** for efficient storage and loading.

## Value Sign Convention

Internally, the loader enforces a uniform sign convention for energy values:
- **Positive values** represent **consumption** (energy flowing into the system)
- **Negative values** represent **production** (energy flowing out of the system)

In input data:
- Prosumer data should have positive values for consumption
- Production asset data should have positive values for production (these are negated internally to become negative)

## Unified Data Schema

Both prosumers and production assets use the same tidy data format:

### File Structure
- **Format**: Parquet
- **Expected columns**:
  - `timestamp` (datetime, required): UTC timestamp of the measurement
  - `meter_id` (string, required): Unique identifier for the meter
  - `value` (float, required): Energy value in kWh

### Example Schema

```
timestamp: timestamp[ns, tz=UTC]
meter_id: string
value: double
```

### Example Data

**Prosumer data** (consumption, positive values):
```
timestamp                | meter_id | value
2025-01-01 00:00:00 UTC | p001     | 0.75
2025-01-01 00:15:00 UTC | p001     | 0.78
2025-01-01 00:30:00 UTC | p001     | 0.81
2025-01-01 00:45:00 UTC | p001     | 0.79
```

**Production asset data** (production, positive values in input):
```
timestamp                | meter_id | value
2025-01-01 00:00:00 UTC | a001     | 0.0
2025-01-01 00:15:00 UTC | a001     | 0.0
2025-01-01 06:00:00 UTC | a001     | 0.5
2025-01-01 12:00:00 UTC | a001     | 4.8
```

### Validation Rules

1. **Timestamps**:
   - Must be in UTC timezone (timestamps without timezone info are assumed UTC with a warning)
   - No duplicate timestamps per meter_id
   - Gaps are allowed; they are logged as warnings and handled according to the `missing_data` policy (`fill_zero` by default). Use `missing_data="error"` to reject any gaps.

2. **Meter IDs**:
   - Must be unique per meter
   - String format (e.g., "p001" for prosumers, "a001" for assets)
   - Should be pseudonymized (see Privacy Recommendations below)

3. **Values**:
   - In kWh
   - Typical ranges:
     - Prosumers: 0.5–3.5 kWh (consumption)
     - Production assets: 0–10 kWh (production)

### Loading Options

**Single File**:
All data can be in one Parquet file for prosumers and one for production assets. The loader will split by `meter_id`.

**Folder Loading**:
Multiple Parquet files can be stored in a folder. One folder for prosumers and a seperate folder for production assets. All matching files will be loaded.


## Missing Value Handling

Supported values for `missing_data`:

1. **`"fill_zero"`** (default): Fill gaps with 0.0 — suitable for energy data where missing means no consumption/production.
2. **`"fill_forward"`**: Forward-fill with the last known value — suitable for slowly-varying data.
3. **`"keep_nan"`**: Leave gaps as NaN — useful if you want to track and handle missing values downstream.
4. **`"error"`**: Raise a `ValueError` on any gap — strict mode for data quality checks.

## Timezone Handling

- **Requirement**: All timestamps must be in UTC
- **Automatic conversion**: If timestamps lack timezone info, UTC will be assumed and logged
- **DST handling**: No DST transitions assumed (use UTC exclusively)


## File Organization Best Practices

### Naming Conventions

- **Prosumer files**: `prosumers.parquet`, `prosumers_*.parquet`
- **Production files**: `production.parquet`, `production_*.parquet`
- **Date ranges**: Include in file names if needed: `prosumers_2025-01.parquet`

## Format Validation

All data is validated upon loading. Validation errors will raise `ValueError` with descriptive messages.


