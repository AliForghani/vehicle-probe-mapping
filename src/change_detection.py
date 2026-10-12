"""
GPS noise vs candidate map change detection.

Core principle:
  One vehicle + one unusual observation   → likely GPS noise
  Many independent vehicles + same spatial deviation + repeated over time
                                          → candidate real map change

IMPORTANT: ground_truth_type is NEVER used here.
Detection is based solely on distance-to-road and multi-vehicle aggregation.
"""

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import Point
from shapely.ops import nearest_points

logger = logging.getLogger(__name__)

_TO_UTM = Transformer.from_crs("EPSG:4326", "EPSG:32618", always_xy=True)


def calculate_road_residuals(
    df: pd.DataFrame,
    roads_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """For every probe observation, compute its distance to the nearest road (metres).

    This is the primary signal for change detection.
    Uses UTM projection for accurate metric distance.
    """
    df = df.copy()

    road_geoms_utm = [
        gpd.GeoDataFrame(geometry=[g], crs="EPSG:4326")
        .to_crs("EPSG:32618")
        .geometry.iloc[0]
        for g in roads_gdf.geometry
    ]

    dists = []
    for row in df.itertuples():
        pt_utm = Point(_TO_UTM.transform(row.longitude, row.latitude))
        min_dist = min(road_geom.distance(pt_utm) for road_geom in road_geoms_utm)
        dists.append(round(min_dist, 2))

    df["dist_to_road_m"] = dists
    return df


def detect_candidate_map_changes(
    df: pd.DataFrame,
    dist_threshold_m: float = 18.0,
    min_observations: int = 5,
    min_unique_vehicles: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flag H3 cells as candidate map changes.

    Algorithm:
    1. Flag individual observations where dist_to_road_m > dist_threshold_m
       (these are *suspicious* — not confirmed noise, not confirmed new road)
    2. Aggregate suspicious observations by H3 cell (res 11)
    3. Require a cell to have:
       - at least min_observations suspicious fixes
       - at least min_unique_vehicles independent vehicles
    4. Cells meeting both thresholds are flagged as candidate map changes

    Returns:
      (candidate_cells_df, suspicious_obs_df)
    """
    if "dist_to_road_m" not in df.columns:
        raise ValueError("Run calculate_road_residuals() first.")

    suspicious = df[df["dist_to_road_m"] > dist_threshold_m].copy()
    logger.info(
        "Suspicious observations (dist > %.1f m): %d / %d total (%.1f%%)",
        dist_threshold_m,
        len(suspicious),
        len(df),
        100 * len(suspicious) / max(len(df), 1),
    )

    if suspicious.empty:
        return pd.DataFrame(), suspicious

    # Aggregate by H3 cell
    agg = (
        suspicious.groupby("h3_cell_res11")
        .agg(
            observations=("vehicle_id", "count"),
            unique_vehicles=("vehicle_id", "nunique"),
            avg_dist_m=("dist_to_road_m", "mean"),
            max_dist_m=("dist_to_road_m", "max"),
            avg_lat=("latitude", "mean"),
            avg_lon=("longitude", "mean"),
        )
        .reset_index()
        .rename(columns={"h3_cell_res11": "h3_cell"})
    )

    # Apply multi-vehicle threshold: the key quality gate
    candidate_cells = agg[
        (agg["observations"] >= min_observations)
        & (agg["unique_vehicles"] >= min_unique_vehicles)
    ].copy()

    candidate_cells["candidate_road_change"] = True

    logger.info(
        "Candidate change cells: %d (out of %d suspicious cells)",
        len(candidate_cells),
        len(agg),
    )
    return candidate_cells, suspicious


def validate_candidate_change(
    candidate_cells: pd.DataFrame,
    df_with_ground_truth: pd.DataFrame,
) -> dict:
    """Compare detected candidates against synthetic ground truth.

    ground_truth_type is ONLY used here, at validation time, never inside
    the detection algorithm above.

    Returns a simple metrics dict.
    """
    if candidate_cells.empty:
        return {"detected": False, "message": "No candidate cells detected."}

    # Which H3 cells contain any candidate_new_road observations?
    true_change_cells = set(
        df_with_ground_truth[df_with_ground_truth["ground_truth_type"] == "candidate_new_road"][
            "h3_cell_res11"
        ].unique()
    )

    detected_cells = set(candidate_cells["h3_cell"].values)

    true_positives  = detected_cells & true_change_cells
    false_positives = detected_cells - true_change_cells

    # Did we catch any of the true change area?
    detected = len(true_positives) > 0

    # Unique vehicles in detected cells
    veh_in_detected = df_with_ground_truth[
        df_with_ground_truth["h3_cell_res11"].isin(detected_cells)
    ]["vehicle_id"].nunique()

    metrics = {
        "detected": detected,
        "true_positive_cells": len(true_positives),
        "false_positive_cells": len(false_positives),
        "total_candidate_cells": len(detected_cells),
        "unique_vehicles_in_detected": veh_in_detected,
        "message": (
            f"Detected {len(true_positives)} true-change cells, "
            f"{len(false_positives)} false positive cells."
        ),
    }
    logger.info("Validation: %s", metrics["message"])
    return metrics
