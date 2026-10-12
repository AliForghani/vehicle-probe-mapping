"""Tests for map_matching.py"""

import pandas as pd
import geopandas as gpd
import pytest
from datetime import datetime, timedelta
from shapely.geometry import LineString, Point

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.map_matching import get_candidates, map_match_nearest


def _simple_road_gdf():
    """A short east-west road segment near Arlington VA."""
    line = LineString([(-77.080, 38.886), (-77.060, 38.886)])
    return gpd.GeoDataFrame(
        {"road_id": [1], "road_name": ["Test Road"], "geometry": [line]},
        crs="EPSG:4326",
    )


def _probe_df(lon: float, lat: float, n: int = 3):
    rows = []
    for i in range(n):
        rows.append({
            "vehicle_id": "V001",
            "timestamp": datetime(2026, 1, 1, 8, 0, i * 2),
            "latitude": lat,
            "longitude": lon + i * 0.001,
            "speed": 40.0,
            "heading": 90.0,
        })
    return pd.DataFrame(rows)


def test_candidates_found_on_road():
    roads = _simple_road_gdf()
    pt = Point(-77.070, 38.886)   # exactly on the road
    candidates = get_candidates(pt, roads, search_radius_m=60.0)
    assert len(candidates) == 1
    assert candidates[0].road_id == 1


def test_no_candidates_far_from_road():
    roads = _simple_road_gdf()
    pt = Point(-77.070, 38.900)   # ~1.5 km north
    candidates = get_candidates(pt, roads, search_radius_m=60.0)
    assert len(candidates) == 0


def test_nearest_match_on_road():
    roads = _simple_road_gdf()
    df = _probe_df(-77.075, 38.886)
    result = map_match_nearest(df, roads, search_radius_m=60.0)
    assert "matched_road_id" in result.columns
    assert result["matched_road_id"].notna().all()


def test_nearest_match_off_road_returns_none():
    roads = _simple_road_gdf()
    df = _probe_df(-77.075, 38.900)  # 1.5 km north of road
    result = map_match_nearest(df, roads, search_radius_m=60.0)
    assert result["matched_road_id"].isna().all()
