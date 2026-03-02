"""Data loader module for prosumer and production asset time series.

This module handles:
1. Loading Parquet files from single files or directories
2. Schema validation (timestamps, columns, data types)
3. Timezone handling
4. Missing value handling policies
"""

import logging
from enum import Enum
from pathlib import Path
from typing import Optional, Union, Tuple

import numpy as np
import pandas as pd

from .core_types import LoadedDataset, MeterTimeSeries, SimulationConfig, CoverageReport

logger = logging.getLogger(__name__)


class MissingValuePolicy(str, Enum):
    """Policy for handling missing timestamps in the time series."""

    ERROR = "error"  # Raise an error on missing timestamps
    FILL_FORWARD = "fill_forward"  # Forward-fill missing values
    FILL_ZERO = "fill_zero"  # Fill missing values with zero


class SchemaValidator:
    """Validates Parquet file schemas for prosumers and production assets."""

    # Both prosumers and production assets use the same tidy schema
    TIDY_REQUIRED_COLUMNS = {"timestamp", "value", "meter_id"}

    @classmethod
    def validate_tidy_schema(cls, df: pd.DataFrame, file_path: Optional[str] = None) -> None:
        """Validate tidy meter-style data schema (timestamp, meter_id, value).

        Args:
            df: DataFrame loaded from Parquet.
            file_path: Path to the file (for error messages).

        Raises:
            ValueError: If schema is invalid.
        """
        missing_cols = cls.TIDY_REQUIRED_COLUMNS - set(df.columns)
        if missing_cols:
            msg = f"Tidy meter data missing columns: {missing_cols}"
            if file_path:
                msg += f" (file: {file_path})"
            raise ValueError(msg)

        # Check timestamp column
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            raise ValueError(
                f"Column 'timestamp' must be datetime type, got {df['timestamp'].dtype}"
            )

        # Check value column is numeric
        if not pd.api.types.is_numeric_dtype(df["value"]):
            raise ValueError(
                f"Column 'value' must be numeric, got {df['value'].dtype}"
            )


class TimestampValidator:
    """Validates timestamp consistency across time series."""

    @classmethod
    def validate_timestamps(
        cls,
        timestamps: pd.DatetimeIndex,
        policy: MissingValuePolicy = MissingValuePolicy.ERROR,
        series_id: Optional[str] = None,
        freq: Optional[str] = None,
    ) -> Tuple[pd.DatetimeIndex, Optional[str]]:
        """Validate and potentially fix timestamps.

        Args:
            timestamps: The timestamp index to validate.
            policy: How to handle missing timestamps.
            series_id: Identifier of the series (for logging).
            freq: Expected frequency (e.g., "15min"). If None, will infer from data.

        Returns:
            Tuple of (validated_timestamps, inferred_or_provided_freq).
            If freq was provided, inferred freq will match. If freq was None,
            returns the inferred frequency string.

        Raises:
            ValueError: If timestamps are invalid and policy is ERROR.
        """
        if len(timestamps) == 0:
            raise ValueError("Timestamp index is empty")

        # Check for duplicates
        if timestamps.duplicated().any():
            dup_count = timestamps.duplicated().sum()
            msg = f"Found {dup_count} duplicate timestamps"
            if series_id:
                msg += f" in {series_id}"
            raise ValueError(msg)

        # Check for timezone
        if timestamps.tz is None:
            logger.warning("Timestamps have no timezone info; assuming UTC")
            timestamps = timestamps.tz_localize("UTC")

        # Infer frequency if not provided
        inferred_freq = freq
        if freq is None:
            inferred_freq = cls._infer_frequency(timestamps, series_id)
            logger.info(f"Inferred frequency {inferred_freq} from timestamps")

        # Check for gaps
        gaps = cls._find_gaps(timestamps, freq=inferred_freq)

        if gaps:
            gap_msg = f"Found {len(gaps)} gaps in timestamps"
            if series_id:
                gap_msg += f" in {series_id}"

            if policy == MissingValuePolicy.ERROR:
                raise ValueError(gap_msg)
            elif policy == MissingValuePolicy.FILL_FORWARD:
                logger.warning(f"{gap_msg}; filling with forward fill")
            elif policy == MissingValuePolicy.FILL_ZERO:
                logger.warning(f"{gap_msg}; filling with zero")

        return timestamps, inferred_freq

    @classmethod
    def _infer_frequency(cls, timestamps: pd.DatetimeIndex, series_id: Optional[str] = None) -> str:
        """Infer frequency from timestamp deltas with safeguards.

        Args:
            timestamps: The timestamp index.
            series_id: Identifier of the series (for logging).

        Returns:
            Inferred frequency string (e.g., "15min", "1H").

        Raises:
            ValueError: If frequency cannot be inferred reliably.
        """
        if len(timestamps) < 2:
            raise ValueError(f"Cannot infer frequency from single timestamp (series_id={series_id})")

        sorted_ts = timestamps.sort_values()
        deltas = sorted_ts.to_series().diff()[1:]  # Skip first NaT

        # Check if all deltas are the same (constant frequency)
        unique_deltas = deltas.unique()
        if len(unique_deltas) == 1:
            delta = unique_deltas[0]
        else:
            # Multiple different deltas: infer from most common
            delta_counts = deltas.value_counts()
            most_common_delta = delta_counts.index[0]
            common_count = delta_counts.iloc[0]
            total_count = len(deltas)
            coverage = common_count / total_count

            if coverage < 0.9:  # Less than 90% regular spacing
                msg = f"Frequency inconsistent: only {coverage*100:.1f}% of intervals match most common delta"
                if series_id:
                    msg += f" (series_id={series_id})"
                raise ValueError(msg)

            logger.warning(f"Frequency inferred from {coverage*100:.1f}% of intervals (has gaps)")
            delta = most_common_delta

        # Convert Timedelta to frequency string
        # Common intervals: 15min, 30min, 1H, 1D, etc.
        total_seconds = int(delta.total_seconds())

        if total_seconds % 3600 == 0:  # Hourly or longer
            hours = total_seconds // 3600
            if hours == 1:
                return "1H"
            else:
                return f"{hours}H"
        elif total_seconds % 60 == 0:  # Minutes
            minutes = total_seconds // 60
            if minutes == 15:
                return "15min"
            elif minutes == 30:
                return "30min"
            else:
                return f"{minutes}min"
        else:
            # Seconds or sub-second
            return f"{total_seconds}S"

    @classmethod
    def _find_gaps(cls, timestamps: pd.DatetimeIndex, freq: str) -> list[tuple]:
        """Find gaps in timestamp sequence.

        Args:
            timestamps: The timestamp index.
            freq: Expected frequency (e.g., "15min").

        Returns:
            List of (gap_start, gap_end, gap_size_minutes) tuples.
        """
        if len(timestamps) < 2:
            return []

        gaps = []
        sorted_ts = timestamps.sort_values()
        diffs = sorted_ts.to_series().diff()

        # Parse frequency string to Timedelta
        # Create a dummy DatetimeIndex and check the inferred frequency
        try:
            offset = pd.tseries.frequencies.to_offset(freq)
            expected_delta = pd.Timedelta(offset)
        except Exception:
            # Fallback: try to parse directly
            expected_delta = pd.Timedelta(freq)

        gap_mask = diffs > expected_delta

        for idx in diffs[gap_mask].index:
            gap_start = sorted_ts[sorted_ts.get_loc(idx) - 1]
            gap_end = idx
            gap_minutes = int((idx - gap_start).total_seconds() / 60)
            gaps.append((gap_start, gap_end, gap_minutes))

        return gaps

class MeterLoader:
    """Loader for time series data for meter id's."""

    def __init__(
        self,
        missing_value_policy: MissingValuePolicy = MissingValuePolicy.ERROR,
    ):
        """Initialize loader.

        Args:
            missing_value_policy: How to handle missing timestamps.
        """
        self.policy = missing_value_policy
        self.validator = SchemaValidator()
        self.ts_validator = TimestampValidator()

    def load(self, path: Path, unit: str, pattern: str = "*.parquet", freq: Optional[str] = None) -> Tuple[list[MeterTimeSeries], dict]:
        """Load from a file or folder, auto-detecting which.

        If path is a file: loads that single Parquet file.
        If path is a folder: loads all matching Parquet files from the folder.

        Args:
            path: Path to file or folder.
            unit: Unit of the values (e.g., "kWh").
            pattern: Glob pattern for folder matching (default: *.parquet).
                     Ignored if path is a file.
            freq: Expected frequency (e.g., "15min"). If None, will infer from data.

        Returns:
            Tuple of (list of MeterTimeSeries, metadata dict with inferred frequencies).

        Raises:
            ValueError: If path doesn't exist or is invalid.
        """
        path = Path(path)

        if path.is_file():
            logger.info(f"Path is file: {path}")
            return self._load_file(path, unit=unit, freq=freq)
        elif path.is_dir():
            logger.info(f"Path is folder: {path}")
            return self._load_folder(path, unit=unit, pattern=pattern, freq=freq)
        else:
            raise ValueError(f"Path does not exist: {path}")

    def _load_file(self, file_path: Path, unit: str, freq: Optional[str] = None) -> Tuple[list[MeterTimeSeries], dict]:
        """Load from a single Parquet file.

        Expected format: tidy structure with columns {timestamp, meter_id, value}.
        Multiple meter ids can be in the same file (grouped by meter_id).

        Returns raw series with native timestamps (one meter may have different
        timestamp coverage than another). No canonical reindexing here.

        Args:
            file_path: Path to Parquet file.
            unit: Unit of the values (e.g., "kWh").
            freq: Expected frequency. If None, will infer from data.

        Returns:
            Tuple of (list of MeterTimeSeries, metadata dict).
        """
        logger.info(f"Loading meters from {file_path}")

        df = pd.read_parquet(file_path)
        self.validator.validate_tidy_schema(df, str(file_path))

        meters = []
        metadata = {"inferred_frequencies": {}}
        inferred_freq = freq

        for meter_id, group in df.groupby("meter_id"):
            # Sort by timestamp
            group = group.sort_values("timestamp").reset_index(drop=True)

            # Validate timestamps and infer freq if needed
            timestamps = pd.DatetimeIndex(group["timestamp"])
            timestamps, inferred_freq_for_meter = self.ts_validator.validate_timestamps(
                timestamps, self.policy, f"meter {meter_id}", freq=freq or inferred_freq
            )

            # Store inferred frequency for this meter
            metadata["inferred_frequencies"][str(meter_id)] = inferred_freq_for_meter

            # Update global inferred_freq from first meter if not provided
            if inferred_freq is None and freq is None:
                inferred_freq = inferred_freq_for_meter

            # No reindexing here — return each meter on its native timestamps
            meter = MeterTimeSeries(
                meter_id=str(meter_id),
                timestamp=timestamps,
                value=group["value"].to_numpy(dtype="float32"),
                unit=unit,
            )
            meters.append(meter)
            logger.debug(f"Loaded meter {meter_id}: {len(timestamps)} timesteps at freq {inferred_freq_for_meter}")

        return meters, metadata

    def _load_folder(self, folder_path: Path, unit: str, pattern: str = "*.parquet", freq: Optional[str] = None) -> Tuple[list[MeterTimeSeries], dict]:
        """Load from all Parquet files in a folder.

        Each meter retains its native timestamps. No alignment across files.
        Frequency consistency is still validated (all meters must have same freq).

        Args:
            folder_path: Path to folder containing Parquet files.
            unit: Unit of the values (e.g., "kWh").
            pattern: Glob pattern for file matching (default: *.parquet).
            freq: Expected frequency. If None, will infer from data.

        Returns:
            Tuple of (list of MeterTimeSeries, metadata dict).
        """
        folder_path = Path(folder_path)
        files = sorted(folder_path.glob(pattern))

        if not files:
            raise ValueError(f"No files matching {pattern} found in {folder_path}")

        logger.info(f"Found {len(files)} Parquet files in {folder_path}")

        all_meters = []
        all_metadata = {"inferred_frequencies": {}, "files_processed": []}
        inferred_freq = freq

        for file_path in files:
            meters, file_metadata = self._load_file(file_path, unit=unit, freq=freq or inferred_freq)
            all_meters.extend(meters)
            all_metadata["inferred_frequencies"].update(file_metadata.get("inferred_frequencies", {}))
            all_metadata["files_processed"].append(str(file_path))
            # Update global inferred_freq from first file if not provided
            if inferred_freq is None and freq is None and meters:
                inferred_freq = file_metadata["inferred_frequencies"][meters[0].meter_id]

        # Check for duplicate meter IDs across files
        meter_ids = [m.meter_id for m in all_meters]
        if len(meter_ids) != len(set(meter_ids)):
            duplicates = [mid for mid in set(meter_ids) if meter_ids.count(mid) > 1]
            logger.warning(f"Duplicate meter IDs across files: {duplicates}")


        return all_meters, all_metadata



class DatasetLoader:
    """High-level loader orchestrating prosumer and production asset loading."""

    def __init__(
        self,
        missing_value_policy: MissingValuePolicy = MissingValuePolicy.ERROR,
    ):
        """Initialize loader.

        Args:
            missing_value_policy: How to handle missing timestamps.
        """
        self.prosumer_loader = MeterLoader(missing_value_policy)
        self.asset_loader = MeterLoader(missing_value_policy)

    def load(
        self,
        prosumer_data_path: Optional[Path] = None,
        production_data_path: Optional[Path] = None,
        simulation_config: Optional["SimulationConfig"] = None,
        return_coverage_report: bool = False,
    ) -> Union[LoadedDataset, Tuple[LoadedDataset, CoverageReport]]:
        """Load a complete dataset.

        MeterLoader automatically detects if paths are files or folders.
        Frequency can be provided in SimulationConfig, or will be inferred from data.

        Args:
            prosumer_data_path: Path to prosumer data (file or folder).
            production_data_path: Path to production data (file or folder).
            simulation_config: Optional SimulationConfig with freq and other settings.
            return_coverage_report: If True, also return CoverageReport.

        Returns:
            LoadedDataset with validated data, and optionally CoverageReport.

        Raises:
            ValueError: If no data paths provided, or if frequency validation fails.
        """
        if not prosumer_data_path and not production_data_path:
            raise ValueError("At least one of prosumer_data_path or production_data_path must be provided")

        # Extract freq from simulation_config if provided
        freq = simulation_config.freq if simulation_config else None

        prosumers = []
        production_assets = []
        all_metadata = {"inferred_frequencies": {}, "data_sources": {}}

        if prosumer_data_path:
            prosumers, prosumer_metadata = self.prosumer_loader.load(prosumer_data_path, unit="kWh", freq=freq)
            all_metadata["inferred_frequencies"].update(prosumer_metadata.get("inferred_frequencies", {}))
            all_metadata["data_sources"]["prosumers"] = prosumer_metadata

        if production_data_path:
            production_assets, asset_metadata = self.asset_loader.load(production_data_path, unit="kWh", freq=freq)
            all_metadata["inferred_frequencies"].update(asset_metadata.get("inferred_frequencies", {}))
            all_metadata["data_sources"]["production_assets"] = asset_metadata

        # Validate frequency consistency across all meters
        self._validate_frequency_consistency(all_metadata["inferred_frequencies"], freq)

        # If simulation_config provided, enforce coverage checks and optionally return a CoverageReport
        coverage_report = None
        if simulation_config is not None:
            expected_index = simulation_config.to_index(tz="UTC")
            missing_prosumers = []
            missing_production_assets = []
            per_meter_full_span = {}
            per_meter_missing_count = {}
            per_meter_missing_fraction = {}
            per_meter_first_missing = {}
            per_meter_last_missing = {}

            def check_coverage_per_series(series_list, missing_list):
                """Check coverage: timestamp exists AND value is not NaN.

                Correctly checks: for each timestep in expected_index, does the meter have
                a non-NaN value? Computes detailed metrics for reporting.
                """
                longest_span = pd.Timedelta(0)
                for s in series_list:
                    ts = pd.DatetimeIndex(s.timestamp)
                    # Reindex the VALUES (not just check timestamp presence)
                    # For each expected timestamp, get the meter's value
                    values = pd.Series(s.value, index=ts).reindex(expected_index)
                    # Coverage = has non-NaN value
                    present = values.notna().to_numpy()

                    # Count and fraction of missing
                    missing_count = (~present).sum()
                    missing_fraction = missing_count / len(present) if len(present) > 0 else 0.0

                    # First and last missing timestamp
                    missing_indices = np.where(~present)[0]
                    first_missing = expected_index[missing_indices[0]] if len(missing_indices) > 0 else None
                    last_missing = expected_index[missing_indices[-1]] if len(missing_indices) > 0 else None

                    per_meter_missing_count[s.meter_id] = int(missing_count)
                    per_meter_missing_fraction[s.meter_id] = float(missing_fraction)
                    per_meter_first_missing[s.meter_id] = first_missing
                    per_meter_last_missing[s.meter_id] = last_missing

                    if not present.all():
                        missing_list.append(s.meter_id)

                    # Compute longest contiguous True run in present
                    if present.any():
                        arr = present.astype(int)
                        padded = np.concatenate([[0], arr, [0]])
                        diff = np.diff(padded)
                        starts = np.where(diff == 1)[0]
                        ends = np.where(diff == -1)[0]
                        counts = (ends - starts)
                        if len(counts) > 0:
                            lengths = counts * pd.Timedelta(simulation_config.freq)
                            max_len = lengths.max()
                        else:
                            max_len = pd.Timedelta(0)
                    else:
                        max_len = pd.Timedelta(0)

                    per_meter_full_span[s.meter_id] = max_len
                    if max_len > longest_span:
                        longest_span = max_len
                return longest_span

            prosumer_span = check_coverage_per_series(prosumers, missing_prosumers)
            asset_span = check_coverage_per_series(production_assets, missing_production_assets)
            overall_longest = max(prosumer_span, asset_span)

            coverage_report = CoverageReport(
                missing_prosumers=missing_prosumers,
                missing_production_assets=missing_production_assets,
                per_meter_full_span=per_meter_full_span,
                overall_longest_full_span=overall_longest,
                per_meter_missing_count=per_meter_missing_count,
                per_meter_missing_fraction=per_meter_missing_fraction,
                per_meter_first_missing=per_meter_first_missing,
                per_meter_last_missing=per_meter_last_missing,
            )

            if (missing_prosumers or missing_production_assets) and not simulation_config.allow_partial:
                logger.warning(
                    "Missing data detected for the requested simulation period. "
                    f"Missing prosumers: {missing_prosumers}; Missing assets: {missing_production_assets}"
                )
                logger.info(
                    f"Longest full-coverage contiguous subperiod within requested period: {overall_longest}"
                )
                raise ValueError(
                    "Simulation period contains missing data for one or more meters. "
                    "Set SimulationConfig.allow_partial=True to override and continue."
                )

        dataset = LoadedDataset(
            prosumers=prosumers,
            production_assets=production_assets,
            timestamp_index=simulation_config.to_index(tz="UTC") if simulation_config else None,
            timezone="UTC",
            metadata=all_metadata,
        )

        # If align_to_clock=True, reindex all series to the simulation clock
        if simulation_config is not None and simulation_config.align_to_clock:
            expected_index = simulation_config.to_index(tz="UTC")

            def align_and_fill(series_list, fill_method):
                """Reindex series to canonical clock and apply fill method."""
                aligned = []
                for s in series_list:
                    # Create a DataFrame with the original data
                    df = pd.DataFrame({"value": s.value}, index=s.timestamp)
                    # Reindex to canonical clock
                    df_reindexed = df.reindex(expected_index)
                    # Apply fill method
                    if fill_method == "zero":
                        df_reindexed["value"] = df_reindexed["value"].fillna(0.0)
                    elif fill_method == "ffill":
                        df_reindexed["value"] = df_reindexed["value"].ffill()
                    # else: fill_method is None, leave as NaN
                    # Create new MeterTimeSeries with aligned data
                    aligned_series = MeterTimeSeries(
                        meter_id=s.meter_id,
                        timestamp=expected_index,
                        value=df_reindexed["value"].to_numpy(dtype="float32"),
                        unit=s.unit,
                    )
                    aligned.append(aligned_series)
                return aligned

            prosumers = align_and_fill(prosumers, simulation_config.fill_method)
            production_assets = align_and_fill(production_assets, simulation_config.fill_method)
            dataset = LoadedDataset(
                prosumers=prosumers,
                production_assets=production_assets,
                timestamp_index=expected_index,
                timezone="UTC",
                metadata=all_metadata,
            )

        if return_coverage_report:
            return dataset, coverage_report
        return dataset

    @staticmethod
    def _validate_frequency_consistency(inferred_frequencies: dict, provided_freq: Optional[str]) -> None:
        """Validate that all meters have consistent frequency.

        Args:
            inferred_frequencies: dict mapping meter_id -> inferred frequency string
            provided_freq: frequency from SimulationConfig (if any)

        Raises:
            ValueError: If frequencies are inconsistent or don't match provided freq.
        """
        if not inferred_frequencies:
            return  # No meters loaded, nothing to validate

        unique_freqs = set(inferred_frequencies.values())

        if len(unique_freqs) > 1:
            # Multiple different frequencies across meters
            freq_breakdown = {}
            for meter_id, freq in inferred_frequencies.items():
                if freq not in freq_breakdown:
                    freq_breakdown[freq] = []
                freq_breakdown[freq].append(meter_id)

            msg = (
                f"Frequency mismatch across meters. Cannot create single simulation clock.\n"
                f"Found {len(unique_freqs)} different frequencies:\n"
            )
            for freq, meter_list in sorted(freq_breakdown.items()):
                msg += f"  {freq}: {len(meter_list)} meters (e.g., {meter_list[:3]})\n"
            msg += (
                "\nEnsure all data files have consistent frequency or provide an explicit freq in SimulationConfig."
            )
            raise ValueError(msg)

        inferred_freq = unique_freqs.pop()

        if provided_freq is not None:
            # User provided a frequency; verify it matches
            if inferred_freq != provided_freq:
                msg = (
                    f"Frequency mismatch: SimulationConfig.freq='{provided_freq}' "
                    f"but data infers '{inferred_freq}'.\n"
                    f"Check your data or update SimulationConfig.freq to match."
                )
                raise ValueError(msg)
            logger.info(f"Frequency validation passed: all meters match provided freq '{provided_freq}'")
        else:
            # Frequency was inferred; log the result
            logger.info(f"Frequency validation passed: all meters have consistent inferred freq '{inferred_freq}'")

