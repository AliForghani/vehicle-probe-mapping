"""Tests for h3_indexing.py"""

import pandas as pd
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.h3_indexing import latlng_to_h3, cell_to_parent, add_h3_cells, aggregate_by_h3


def test_latlng_returns_string():
    cell = latlng_to_h3(38.886, -77.070, 11)
    assert isinstance(cell, str)
    assert len(cell) > 0


def test_parent_resolution_is_coarser():
    cell_11 = latlng_to_h3(38.886, -77.070, 11)
    cell_10 = cell_to_parent(cell_11, 10)
    cell_9  = cell_to_parent(cell_11, 9)
    # Coarser resolutions have shorter (fewer-character) cell IDs in h3 v4
    # Or we can check that the resolution is correct via h3.get_resolution
    import h3
    assert h3.get_resolution(cell_11) == 11
    assert h3.get_resolution(cell_10) == 10
    assert h3.get_resolution(cell_9)  == 9


def test_parent_hierarchy_consistent():
    """The res-10 parent of cell_11 must equal parent of cell_10."""
    cell_11 = latlng_to_h3(38.886, -77.070, 11)
    cell_10_direct  = cell_to_parent(cell_11, 10)
    cell_9_via_11   = cell_to_parent(cell_11, 9)
    cell_9_via_10   = cell_to_parent(cell_10_direct, 9)
    assert cell_9_via_11 == cell_9_via_10


def test_add_h3_cells_adds_columns():
    df = pd.DataFrame([
        {"vehicle_id": "V001", "latitude": 38.886, "longitude": -77.070, "speed": 40.0},
    ])
    out = add_h3_cells(df)
    assert "h3_cell_res11" in out.columns
    assert "h3_cell_res10" in out.columns
    assert "h3_cell_res9"  in out.columns


def test_aggregate_counts_unique_vehicles():
    df = pd.DataFrame([
        {"vehicle_id": "V001", "latitude": 38.886, "longitude": -77.070, "speed": 40.0},
        {"vehicle_id": "V001", "latitude": 38.886, "longitude": -77.070, "speed": 42.0},
        {"vehicle_id": "V002", "latitude": 38.886, "longitude": -77.070, "speed": 38.0},
    ])
    df = add_h3_cells(df)
    agg = aggregate_by_h3(df, resolution=11)
    # All three points are in roughly the same cell
    assert agg["unique_vehicles"].max() == 2  # V001 and V002
    assert agg["observations"].max() >= 2
