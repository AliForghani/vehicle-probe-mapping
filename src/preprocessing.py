"""
GPS probe preprocessing: cleaning, validation, and derived fields.

Two distinct concepts kept deliberately separate:

  filter_gps_noise()
      Asks: "Is this observation inconsistent with the vehicle's OWN trajectory?"
      Looks only at the vehicle's internal sequence — speeds, local geometry.
      Does NOT use distance from the road map.

  (downstream) detect_candidate_map_changes()
      Asks: "Are many reliable vehicle trajectories consistently inconsistent
             with the EXISTING MAP?"
      Uses road distance as the signal.

Keeping these separate prevents the noise filter from erasing the very evidence
the change detector needs.
"""

import logging
import math
from typing import Optional

import numpy as np
import pandas as pd
from pyproj import Transformer

logger = logging.getLogger(__name__)

_TO_UTM = Transformer.from_crs("EPSG:4326", "EPSG:32618", always_xy=True)

# Defaults — all configurable via filter_gps_noise() parameters
LAT_MIN, LAT_MAX   = -90.0, 90.0
LON_MIN, LON_MAX   = -180.0, 180.0
MAX_SPEED_KMH      = 250.0   # conservative configurable threshold for detecting implausible jumps in this demo
MAX_TIME_GAP_S     = 30.0    # seconds; don't apply local-geometry test across gaps
LOCAL_OUTLIER_M    = 25.0    # metres; perpendicular deviation from prev→next segment


# ── Structural cleaning ────────────────────────────────────────────────────────

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
    8. Calculate derived speed         — dist_m / dt_seconds; accounts for sampling rate

    Returns a cleaned DataFrame with added columns:
        dt_seconds, dist_m, derived_speed_kmh
    """
    n_start = len(df)

    # Step 1: Drop missing coordinates
    df = df.dropna(subset=["latitude", "longitude"])

    # Step 2: Reject impossible / sentinel coordinates
    valid_coords = (
        (df["latitude"]  >= LAT_MIN) & (df["latitude"]  <= LAT_MAX)
        & (df["longitude"] >= LON_MIN) & (df["longitude"] <= LON_MAX)
        & ~((df["latitude"] == 0.0) & (df["longitude"] == 0.0))
    )
    df = df[valid_coords].copy()

    # Step 3: Remove exact duplicate rows
    df = df.drop_duplicates(subset=["vehicle_id", "timestamp", "latitude", "longitude"])

    # Step 4: Sort chronologically per vehicle
    df = df.sort_values(["vehicle_id", "timestamp"]).reset_index(drop=True)

    # Step 5: Flag duplicate timestamps within a vehicle
    df["dup_timestamp"] = df.duplicated(subset=["vehicle_id", "timestamp"], keep=False)

    # Step 6: Time delta (seconds) between consecutive fixes — per vehicle
    df["dt_seconds"] = (
        df.groupby("vehicle_id")["timestamp"]
        .diff()
        .dt.total_seconds()
        .fillna(0.0)
    )

    # Step 7: Distance delta in metres (UTM for metric accuracy)
    utm_x, utm_y = _TO_UTM.transform(df["longitude"].values, df["latitude"].values)
    df["_utm_x"] = utm_x
    df["_utm_y"] = utm_y
    prev_x = df.groupby("vehicle_id")["_utm_x"].shift(1)
    prev_y = df.groupby("vehicle_id")["_utm_y"].shift(1)
    df["dist_m"] = np.sqrt(
        (df["_utm_x"] - prev_x) ** 2 + (df["_utm_y"] - prev_y) ** 2
    ).fillna(0.0)
    df = df.drop(columns=["_utm_x", "_utm_y"])

    # Step 8: Derived speed — bakes in the time gap so a slow vehicle with a
    # long sampling interval is not confused with a fast-moving one
    with np.errstate(divide="ignore", invalid="ignore"):
        df["derived_speed_kmh"] = np.where(
            df["dt_seconds"] > 0,
            (df["dist_m"] / df["dt_seconds"]) * 3.6,
            np.nan,
        )

    logger.info("Preprocessing: %d → %d rows (removed %d)", n_start, len(df), n_start - len(df))
    return df


# ── GPS noise filtering ────────────────────────────────────────────────────────

def _perpendicular_distance_m(
    px: float, py: float,   # current point (UTM)
    ax: float, ay: float,   # previous point (UTM)
    bx: float, by: float,   # next point (UTM)
) -> float:
    """Distance (metres) from point P to segment A→B.

    Uses true point-to-segment distance:
    1. Project P onto the infinite line through A and B → parameter t.
    2. Clamp t to [0, 1] so the closest point stays on the segment.
    3. Return distance from P to that clamped closest point.

    If A == B (zero-length segment), returns distance A→P.
    """
    abx, aby = bx - ax, by - ay
    ab_len_sq = abx ** 2 + aby ** 2
    if ab_len_sq < 1e-9:
        return math.sqrt((px - ax) ** 2 + (py - ay) ** 2)
    # Scalar projection of AP onto AB, normalised by |AB|^2
    t = ((px - ax) * abx + (py - ay) * aby) / ab_len_sq
    t = max(0.0, min(1.0, t))          # clamp to segment
    closest_x = ax + t * abx
    closest_y = ay + t * aby
    return math.sqrt((px - closest_x) ** 2 + (py - closest_y) ** 2)


def _compute_local_residuals(df: pd.DataFrame, max_time_gap_s: float) -> np.ndarray:
    """For each fix, compute perpendicular distance to the prev→next segment.

    Only computed when BOTH neighbouring time gaps are within max_time_gap_s.
    Returns an array of residuals (NaN where the test is not applicable).
    """
    utm_x, utm_y = _TO_UTM.transform(df["longitude"].values, df["latitude"].values)
    n = len(df)
    residuals = np.full(n, np.nan)

    # Per-vehicle grouping so we don't mix trajectories across vehicles
    for _, grp in df.groupby("vehicle_id"):
        idx = grp.index.tolist()
        for k in range(1, len(idx) - 1):
            i_prev = idx[k - 1]
            i_curr = idx[k]
            i_next = idx[k + 1]

            prev_gap = df.at[i_curr, "dt_seconds"]
            next_gap = df.at[i_next,  "dt_seconds"]

            # Require both gaps to be positive and within the time limit.
            # Zero/negative gaps (duplicate timestamps) make the local
            # geometry undefined; large gaps mean the vehicle may have turned.
            if not (0 < prev_gap <= max_time_gap_s and 0 < next_gap <= max_time_gap_s):
                continue

            residuals[i_curr] = _perpendicular_distance_m(
                utm_x[i_curr], utm_y[i_curr],
                utm_x[i_prev], utm_y[i_prev],
                utm_x[i_next], utm_y[i_next],
            )

    return residuals


def filter_gps_noise(
    df: pd.DataFrame,
    max_speed_kmh: float = MAX_SPEED_KMH,
    max_time_gap_seconds: float = MAX_TIME_GAP_S,
    local_outlier_threshold_m: float = LOCAL_OUTLIER_M,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove GPS noise observations using two complementary tests.

    GPS noise filtering asks:
        "Is this observation inconsistent with the vehicle's own trajectory?"
    It does NOT use road distance — that belongs to change detection.

    ── Test 1: GPS speed outlier ────────────────────────────────────────────────
    Existence of an erroneous middle fix X causes the calculated speed from the previous fix
    to X, and from X to the next fix, to become unrealistically high compared
    with the surrounding trajectory.

    A fix is flagged as a likely GPS speed outlier when:
      - the calculated speed from the previous fix to the current fix is
        unrealistically high (> max_speed_kmh)  AND
      - the calculated speed from the current fix to the next fix is also
        unrealistically high (> max_speed_kmh)

    Requiring both conditions avoids removing a legitimate fix where only the
    speed from the previous fix is high (e.g. a long sampling gap after a
    fast road section).

    ── Test 2: isolated local trajectory outlier ────────────────────────────────
    For each fix, construct the straight-line segment from its previous fix to
    its next fix.  Measure the perpendicular distance from the current fix to
    that segment (local_trajectory_residual_m).

    Only applied when both neighbouring time gaps <= max_time_gap_seconds.
    A large gap means the vehicle may have turned — the three points no longer
    form a meaningful local segment.

    A fix is an isolated outlier when:
      - local_trajectory_residual_m > local_outlier_threshold_m
      - AND its neighbours are NOT similarly displaced
        (if they are, the cluster is a persistent deviation, not noise)

    "Neighbours not similarly displaced" is checked conservatively:
    if the previous OR next residual also exceeds the threshold, the current
    fix is part of a multi-point cluster and is preserved.

    ── Diagnostic columns added ─────────────────────────────────────────────────
    next_speed_kmh, prev_time_gap_s, next_time_gap_s,
    local_trajectory_residual_m,
    speed_spike_flag, local_outlier_flag,
    gps_noise_flag, noise_reason

    Returns (cleaned_df, removed_df).
    """
    df = df.copy()

    # ── Derived fields for both tests ─────────────────────────────────────────
    df["next_speed_kmh"] = (
        df.groupby("vehicle_id")["derived_speed_kmh"].shift(-1).fillna(0.0)
    )
    df["prev_time_gap_s"] = df["dt_seconds"]
    df["next_time_gap_s"] = (
        df.groupby("vehicle_id")["dt_seconds"].shift(-1).fillna(0.0)
    )

    # ── Test 1: GPS speed outlier ─────────────────────────────────────────────
    # speed_from_previous_kmh is already stored as derived_speed_kmh (dist/dt).
    # speed_to_next_kmh is the derived_speed_kmh of the following fix.
    unrealistic_speed_from_previous = df["derived_speed_kmh"].fillna(0.0) > max_speed_kmh
    unrealistic_speed_to_next       = df["next_speed_kmh"]                > max_speed_kmh
    df["speed_spike_flag"] = unrealistic_speed_from_previous & unrealistic_speed_to_next

    # ── Test 2: local trajectory outlier ─────────────────────────────────────
    residuals = _compute_local_residuals(df, max_time_gap_seconds)
    df["local_trajectory_residual_m"] = residuals

    exceeds_threshold = pd.Series(residuals, index=df.index) > local_outlier_threshold_m

    # Check whether the current fix is an isolated trajectory outlier or part of
    # a broader multi-point deviation.
    #
    # Example residual patterns:
    #
    #   5 m   6 m   40 m   7 m   5 m
    #   
    #   The third fix has a large residual while its neighbors are normal.
    #   This looks like one isolated bad GPS observation.
    #
    #   5 m   35 m   40 m   32 m   6 m
    #              
    #   The third fix is large, but its neighboring residuals are also large.
    #   This suggests a more persistent multi-point deviation, so do not remove
    #   the current fix automatically as isolated GPS noise.
    #
    # shift(1) gets the previous fix's residual within the same vehicle trajectory;
    # shift(-1) gets the next fix's residual.
    prev_exceeds = (
        df.groupby("vehicle_id")["local_trajectory_residual_m"]
        .shift(1)
        .fillna(0.0) > local_outlier_threshold_m
    )
    next_exceeds = (
        df.groupby("vehicle_id")["local_trajectory_residual_m"]
        .shift(-1)
        .fillna(0.0) > local_outlier_threshold_m
    )
    neighbours_also_displaced = prev_exceeds | next_exceeds

    # Isolated = exceeds threshold AND no neighbour also exceeds it
    df["local_outlier_flag"] = exceeds_threshold & ~neighbours_also_displaced

    # ── Final noise decision ──────────────────────────────────────────────────
    df["gps_noise_flag"] = df["speed_spike_flag"] | df["local_outlier_flag"]

    def _reason(row: pd.Series) -> Optional[str]:
        if row["speed_spike_flag"] and row["local_outlier_flag"]:
            return "multiple_flags"
        if row["speed_spike_flag"]:
            return "unrealistic_speed_between_consecutive_fixes"
        if row["local_outlier_flag"]:
            return "isolated_local_trajectory_outlier"
        return None

    df["noise_reason"] = df.apply(_reason, axis=1)

    removed = df[df["gps_noise_flag"]].copy()
    cleaned = df[~df["gps_noise_flag"]].copy()

    logger.info(
        "Noise filter: removed %d observations "
        "(speed_spike=%d, local_outlier=%d)",
        len(removed),
        df["speed_spike_flag"].sum(),
        df["local_outlier_flag"].sum(),
    )
    return cleaned, removed


# ── Optional trajectory smoother ──────────────────────────────────────────────

def smooth_trajectory(
    df: pd.DataFrame,
    window: int = 5,
) -> pd.DataFrame:
    """Apply a rolling median smoother to lat/lon per vehicle.

    This is a display aid only — do NOT use smoothed coordinates to decide
    whether a map change exists.  The raw (filtered) trajectory is the input
    to change detection.

    Uses a rolling median (robust to outliers) rather than a mean.
    Window=5 means each smoothed point is the median of itself ± 2 neighbours.

    Adds columns: latitude_smooth, longitude_smooth.
    """
    df = df.copy()
    df["latitude_smooth"] = (
        df.groupby("vehicle_id")["latitude"]
        .transform(lambda s: s.rolling(window, center=True, min_periods=1).median())
    )
    df["longitude_smooth"] = (
        df.groupby("vehicle_id")["longitude"]
        .transform(lambda s: s.rolling(window, center=True, min_periods=1).median())
    )
    return df
