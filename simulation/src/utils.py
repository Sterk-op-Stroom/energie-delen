"""Shared utilities for the energy sharing simulation."""

import numpy as np
import pandas as pd


def ensure_array(value: object, dtype: np.dtype) -> np.ndarray:
    """Coerce a value to a numpy array with the given dtype.

    If value is already an ndarray with the correct dtype, it is returned as-is.
    Otherwise a new array is created or cast.
    """
    if not isinstance(value, np.ndarray):
        return np.asarray(value, dtype=dtype)
    if value.dtype != dtype:
        return value.astype(dtype)
    return value


def infer_freq(timestamp: pd.DatetimeIndex) -> str | None:
    """Infer a frequency string from a DatetimeIndex (best-effort).

    Returns a human-readable frequency string (e.g. "15min", "1H") based on
    the delta between the first two timestamps, or None if the index has
    fewer than 2 entries or the delta is non-positive.
    """
    if len(timestamp) < 2:
        return None
    delta = timestamp[1] - timestamp[0]
    total_seconds = int(delta.total_seconds())
    if total_seconds <= 0:
        return None
    if total_seconds % 86400 == 0:
        days = total_seconds // 86400
        return f"{days}D"
    if total_seconds % 3600 == 0:
        hours = total_seconds // 3600
        return f"{hours}h"
    if total_seconds % 60 == 0:
        return f"{total_seconds // 60}min"
    return f"{total_seconds}s"
