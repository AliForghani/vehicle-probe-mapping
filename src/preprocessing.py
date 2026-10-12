"""
GPS probe preprocessing: cleaning, validation, and derived fields.

Each step is explained with WHY it matters for vehicle probe data.
"""

import logging

import numpy as np
import pandas as pd
from pyproj import Transformer

logger = logging.getLogger(__name__)

_TO_UTM = Transformer.from_crs("EPSG:4326", "EPSG:32618", always_xy=True)

# Hard limits for sanity checks
LAT_MIN, LAT_MAX = -90.0, 90.0
LON_MIN, LON_MAX = -180.0, 180.0
MAX_SPEED_KMH = 250.0          # nothing faster than a sports car on a road
MAX_JUMP_M = 500.0             # impossible gap between consecutive points (metres)


def preprocess_probes(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and enrich raw probe observations.

    Steps (in order):
    1. Drop rows with missing lat/lon  — sensors occasionally drop fields
    2. Reject impossible coordinates   — data pipeline bugs produce 0.0, 999, etc.
    3. Remove exact duplicates         — duplicate rows inflate counts and bias aggregations
    4. Sort by vehicle + timestamp     — required before any per-trajectory calculations
    5. Flag duplicate timestamps       — same vehicle, same second = sensor glitch
    6. Calculate time delta            — needed to derive speed and detect unrealistic jumps
    7. Calculate distance delta        — metric distance between consecutive obs (UTM)
    8. Calculate derived speed         — cross-check against reported speed

    Returns a cleaned DataFrame with additional columns:
        dt_seconds, dist_m, derived_speed_kmh
    """
    n_start = len(df)

    # Step 1: Drop missing coordinates
    # A GPS fix with no coordinates is useless and will crash any spatial operation.
    df = df.dropna(subset=["latitude", "longitude"])
    logger.debug("After drop NaN coords: %d rows (removed %d)", len(df), n_start - len(df))

    # Step 2: Reject impossible coordinates
    # Corrupt telemetry pipelines often emit default values like (0.0, 0.0) or (-999, -999).
    valid_coords = (
        (df["latitude"] >= LAT_MIN) & (df["latitude"] <= LAT_MAX)
        & (df["longitude"] >= LON_MIN) & (df["longitude"] <= LON_MAX)
        # Reject exact (0,0) — almost certainly a missing-value sentinel, not the Gulf of Guinea
        & ~((df["latitude"] == 0.0) & (df["longitude"] == 0.0))
    )
    df = df[valid_coords].copy()
    logger.debug("After coord range filter: %d rows", len(df))

    # Step 3: Remove exact duplicate rows
    # The same fix can arrive twice via redundant telemetry channels.
    dup_cols = ["vehicle_id", "timestamp", "latitude", "longitude"]
    df = df.drop_duplicates(subset=dup_cols)
    logger.debug("After dedup: %d rows", len(df))

    # Step 4: Sort by vehicle and time
    # All subsequent per-trajectory computations assume chronological order.
    df = df.sort_values(["vehicle_id", "timestamp"]).reset_index(drop=True)

    # Step 5: Flag duplicate timestamps within a vehicle
    # Two observations from the same vehicle at the exact same second indicates a sensor glitch.
    df["dup_timestamp"] = df.duplicated(subset=["vehicle_id", "timestamp"], keep=False)

    # Step 6: Time delta between consecutive observations (seconds)
    df["dt_seconds"] = (
        df.groupby("vehicle_id")["timestamp"]
        .diff()
        .dt.total_seconds()
        .fillna(0.0)
    )

    # Step 7: Distance delta in metres (UTM projection for metric accuracy)
    utm_x, utm_y = _TO_UTM.transform(df["longitude"].values, df["latitude"].values)
    df["_utm_x"] = utm_x
    df["_utm_y"] = utm_y

    prev_x = df.groupby("vehicle_id")["_utm_x"].shift(1)
    prev_y = df.groupby("vehicle_id")["_utm_y"].shift(1)
    df["dist_m"] = np.sqrt((df["_utm_x"] - prev_x) ** 2 + (df["_utm_y"] - prev_y) ** 2).fillna(0.0)
    df = df.drop(columns=["_utm_x", "_utm_y"])

    # Step 8: Derived speed from position deltas
    # If reported speed is wildly inconsistent with derived speed, the fix is suspect.
    with np.errstate(divide="ignore", invalid="ignore"):
        df["derived_speed_kmh"] = np.where(
            df["dt_seconds"] > 0,
            (df["dist_m"] / df["dt_seconds"]) * 3.6,
            np.nan,
        )

    n_end = len(df)
    logger.info("Preprocessing: %d → %d rows (removed %d)", n_start, n_end, n_start - n_end)
    return df


def filter_gps_noise(df: pd.DataFrame) -> pd.DataFrame:
    """Remove obvious GPS noise observations.

    Strategy: remove *isolated* impossible observations while preserving
    persistent deviations (which may be real road changes, not noise).

    Filters applied:
    - Reject observations with derived speed > MAX_SPEED_KMH
      (the vehicle cannot have moved that fast between two fixes)
    - Reject isolated jumps > MAX_JUMP_M from the previous fix
      (a single massive jump with quick return = multipath reflection)

    NOT applied here:
    - Distance from road — that is the signal we WANT to study
    - Smoothing of the candidate new-road trajectory

    Returns (cleaned_df, removed_df) so callers can inspect what was dropped.
    """
    # Flag impossible speeds
    impossible_speed = df["derived_speed_kmh"] > MAX_SPEED_KMH

    # Flag isolated jumps: large dist_m followed by small dist_m on same vehicle
    # "isolated" means the NEXT point is close to where we were before the jump
    next_dist = df.groupby("vehicle_id")["dist_m"].shift(-1).fillna(0.0)
    isolated_jump = (df["dist_m"] > MAX_JUMP_M) & (next_dist < MAX_JUMP_M / 2)

    noise_mask = impossible_speed | isolated_jump
    removed = df[noise_mask].copy()
    cleaned = df[~noise_mask].copy()

    logger.info(
        "Noise filter: removed %d observations (speed=%d, jump=%d)",
        noise_mask.sum(),
        impossible_speed.sum(),
        isolated_jump.sum(),
    )
    return cleaned, removed
