"""Tests for preprocessing.py"""

import pandas as pd
import pytest
from datetime import datetime

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing import preprocess_probes, filter_gps_noise


def _base_df(**overrides):
    row = {
        "vehicle_id": "V001",
        "timestamp": datetime(2026, 1, 1, 8, 0, 0),
        "latitude": 38.886,
        "longitude": -77.070,
        "speed": 40.0,
        "heading": 90.0,
        "ground_truth_type": "mapped_road",
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_drops_nan_coordinates():
    df = _base_df(latitude=None)
    out = preprocess_probes(df)
    assert len(out) == 0


def test_rejects_zero_zero():
    df = _base_df(latitude=0.0, longitude=0.0)
    out = preprocess_probes(df)
    assert len(out) == 0


def test_rejects_out_of_range_lat():
    df = _base_df(latitude=95.0)
    out = preprocess_probes(df)
    assert len(out) == 0


def test_keeps_valid_row():
    df = _base_df()
    out = preprocess_probes(df)
    assert len(out) == 1


def test_removes_exact_duplicates():
    df = pd.concat([_base_df(), _base_df()], ignore_index=True)
    out = preprocess_probes(df)
    assert len(out) == 1


def test_derived_speed_columns_added():
    df = _base_df()
    out = preprocess_probes(df)
    assert "dt_seconds" in out.columns
    assert "dist_m" in out.columns
    assert "derived_speed_kmh" in out.columns


def test_impossible_speed_removed():
    """A derived speed of 300 km/h should be filtered out."""
    rows = []
    t0 = datetime(2026, 1, 1, 8, 0, 0)
    t1 = datetime(2026, 1, 1, 8, 0, 1)  # 1 second later
    rows.append({
        "vehicle_id": "V001", "timestamp": t0,
        "latitude": 38.886, "longitude": -77.070,
        "speed": 40.0, "heading": 90.0, "ground_truth_type": "mapped_road",
    })
    # Move 300 m in 1 second → 1080 km/h derived speed
    rows.append({
        "vehicle_id": "V001", "timestamp": t1,
        "latitude": 38.886, "longitude": -77.067,  # ~250 m east
        "speed": 40.0, "heading": 90.0, "ground_truth_type": "mapped_road",
    })
    df = pd.DataFrame(rows)
    pre = preprocess_probes(df)
    cleaned, removed = filter_gps_noise(pre)
    # The impossible-speed observation should be removed
    assert len(removed) >= 1
