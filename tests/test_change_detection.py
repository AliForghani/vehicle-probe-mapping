"""Tests for change_detection.py"""

import pandas as pd
import geopandas as gpd
import pytest
from shapely.geometry import LineString

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.change_detection import (
    calculate_road_residuals,
    detect_candidate_map_changes,
    validate_candidate_change,
)
from src.h3_indexing import add_h3_cells


def _road_gdf():
    line = LineString([(-77.080, 38.886), (-77.060, 38.886)])
    return gpd.GeoDataFrame(
        {"road_id": [1], "road_name": ["Test Road"], "geometry": [line]},
        crs="EPSG:4326",
    )


def _probe_df():
    rows = []
    # 4 normal vehicles near the road
    for vid in range(4):
        for j in range(10):
            rows.append({
                "vehicle_id": f"V{vid:03d}",
                "latitude": 38.886 + 0.00003,  # ~3 m north
                "longitude": -77.075 + j * 0.001,
                "speed": 40.0,
                "ground_truth_type": "mapped_road",
            })
    # 4 vehicles on offset road ~30 m north
    for vid in range(4, 8):
        for j in range(10):
            rows.append({
                "vehicle_id": f"V{vid:03d}",
                "latitude": 38.886 + 0.00027,  # ~30 m north
                "longitude": -77.075 + j * 0.001,
                "speed": 40.0,
                "ground_truth_type": "candidate_new_road",
            })
    return pd.DataFrame(rows)


def test_residuals_small_for_on_road():
    roads = _road_gdf()
    df = _probe_df()
    df_h3 = add_h3_cells(df)
    with_residuals = calculate_road_residuals(df_h3, roads)
    on_road = with_residuals[with_residuals["ground_truth_type"] == "mapped_road"]
    assert (on_road["dist_to_road_m"] < 15).all()


def test_residuals_large_for_offset():
    roads = _road_gdf()
    df = _probe_df()
    df_h3 = add_h3_cells(df)
    with_residuals = calculate_road_residuals(df_h3, roads)
    offset = with_residuals[with_residuals["ground_truth_type"] == "candidate_new_road"]
    assert (offset["dist_to_road_m"] > 15).all()


def test_detection_finds_candidate_cells():
    roads = _road_gdf()
    df = _probe_df()
    df_h3 = add_h3_cells(df)
    with_residuals = calculate_road_residuals(df_h3, roads)
    candidates, _ = detect_candidate_map_changes(
        with_residuals,
        dist_threshold_m=15.0,
        min_observations=3,
        min_unique_vehicles=2,
    )
    assert len(candidates) > 0


def test_validation_detects_true_positive():
    roads = _road_gdf()
    df = _probe_df()
    df_h3 = add_h3_cells(df)
    with_residuals = calculate_road_residuals(df_h3, roads)
    candidates, _ = detect_candidate_map_changes(
        with_residuals,
        dist_threshold_m=15.0,
        min_observations=3,
        min_unique_vehicles=2,
    )
    metrics = validate_candidate_change(candidates, with_residuals)
    assert metrics["detected"] is True
    assert metrics["true_positive_cells"] > 0
