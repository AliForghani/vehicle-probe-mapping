"""Tests for preprocessing.py"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import pytest
from datetime import datetime, timedelta

from src.preprocessing import preprocess_probes, filter_gps_noise


# ── Helpers ────────────────────────────────────────────────────────────────────

def _row(vid, t, lat, lon, speed=40.0):
    return {"vehicle_id": vid, "timestamp": t,
            "latitude": lat, "longitude": lon,
            "speed": speed, "heading": 90.0, "ground_truth_type": "mapped_road"}


def _t(offset_s: int):
    return datetime(2026, 1, 1, 8, 0, 0) + timedelta(seconds=offset_s)


def _pre(rows):
    """Convenience: build DataFrame → preprocess."""
    return preprocess_probes(pd.DataFrame(rows))


# ── preprocess_probes ──────────────────────────────────────────────────────────

def test_drops_nan_coordinates():
    df = pd.DataFrame([_row("V1", _t(0), None, -77.07)])
    assert len(preprocess_probes(df)) == 0


def test_rejects_zero_zero():
    df = pd.DataFrame([_row("V1", _t(0), 0.0, 0.0)])
    assert len(preprocess_probes(df)) == 0


def test_rejects_out_of_range_lat():
    df = pd.DataFrame([_row("V1", _t(0), 95.0, -77.07)])
    assert len(preprocess_probes(df)) == 0


def test_keeps_valid_row():
    df = pd.DataFrame([_row("V1", _t(0), 38.886, -77.07)])
    assert len(preprocess_probes(df)) == 1


def test_removes_exact_duplicates():
    r = _row("V1", _t(0), 38.886, -77.07)
    df = pd.DataFrame([r, r])
    assert len(preprocess_probes(df)) == 1


def test_derived_speed_columns_added():
    df = pd.DataFrame([_row("V1", _t(0), 38.886, -77.07)])
    out = preprocess_probes(df)
    assert {"dt_seconds", "dist_m", "derived_speed_kmh"}.issubset(out.columns)


# ── filter_gps_noise: Test 1 — impossible speed spike ─────────────────────────

def test_impossible_speed_spike_removed():
    """
    fix0 ──normal──► fix1(SPIKE) ──normal──► fix2
    fix1: ~250m east in 1 second → ~900 km/h arrival AND ~900 km/h departure.
    Expected: fix1 removed, fix0 and fix2 kept.
    """
    rows = [
        _row("V1", _t(0),  38.886,  -77.070),   # fix0 — normal
        _row("V1", _t(1),  38.886,  -77.067),   # fix1 — ~250m east in 1s ≈ 900 km/h
        _row("V1", _t(2),  38.8861, -77.0699),  # fix2 — back to 8m in 1s (normal)
    ]
    pre = _pre(rows)
    cleaned, removed = filter_gps_noise(pre, max_speed_kmh=250.0)
    assert len(removed) == 1
    assert len(cleaned) == 2
    assert removed.iloc[0]["noise_reason"] == "unrealistic_speed_between_consecutive_fixes"


# ── filter_gps_noise: Test 2 — persistent cluster NOT removed ─────────────────

def test_persistent_offset_cluster_preserved():
    """
    Several consecutive points follow a parallel path ~30m north.
    These should NOT be removed — they are a candidate map change.
    """
    # Normal road at lat ~38.886
    # Offset road ~30m north ≈ lat 38.886 + 0.00027
    rows = []
    for i in range(6):
        rows.append(_row("V1", _t(i * 2), 38.886 + 0.00027, -77.075 + i * 0.0002))

    pre = _pre(rows)
    cleaned, removed = filter_gps_noise(
        pre, local_outlier_threshold_m=25.0, max_time_gap_seconds=30.0
    )
    assert len(removed) == 0, f"Should not remove persistent cluster, removed: {len(removed)}"


# ── filter_gps_noise: Test 3 — normal GPS scatter NOT removed ─────────────────

def test_normal_scatter_not_removed():
    """
    Points with ~3m GPS scatter around a straight trajectory.
    None should be flagged as noise.
    """
    rows = []
    for i in range(8):
        # Tiny lat variation simulates GPS noise
        lat = 38.886 + i * 0.000009 + (0.00002 if i == 3 else 0.0)
        rows.append(_row("V1", _t(i * 2), lat, -77.075 + i * 0.0002))

    pre = _pre(rows)
    cleaned, removed = filter_gps_noise(
        pre, max_speed_kmh=250.0, local_outlier_threshold_m=25.0
    )
    assert len(removed) == 0


# ── filter_gps_noise: Test 4 — large time gap not flagged ─────────────────────

def test_large_time_gap_not_flagged():
    """
    fix0 and fix2 are far apart in space but connected by a 300-second gap.
    fix1 is at an unusual position but the gaps are too large to apply
    the local-trajectory test, so it must not be removed on that basis.
    We use a low enough speed that the speed test also does not trigger.
    """
    rows = [
        _row("V1", _t(0),   38.886,  -77.070),   # fix0
        _row("V1", _t(300), 38.887,  -77.060),   # fix1 — 300s gap (too large for local test)
        _row("V1", _t(600), 38.888,  -77.050),   # fix2 — another 300s gap
    ]
    pre = _pre(rows)
    cleaned, removed = filter_gps_noise(
        pre, max_time_gap_seconds=30.0, local_outlier_threshold_m=25.0
    )
    # Derived speed over 300s / ~900m ≈ 10.8 km/h — well below MAX_SPEED_KMH
    assert len(removed) == 0


# ── filter_gps_noise: Test 5 — impossible speed spike (explicit) ──────────────

def test_speed_spike_flag_set():
    """Verify the diagnostic column speed_spike_flag is True for the spike fix."""
    rows = [
        _row("V1", _t(0), 38.886,  -77.070),
        _row("V1", _t(1), 38.886,  -77.067),   # spike
        _row("V1", _t(2), 38.8861, -77.0699),
    ]
    pre = _pre(rows)
    cleaned, removed = filter_gps_noise(pre, max_speed_kmh=250.0)
    assert removed.iloc[0]["speed_spike_flag"] == True


# ── filter_gps_noise: isolated single outlier ─────────────────────────────────

def test_single_isolated_local_outlier_removed():
    """
    One point jumps 40m perpendicular to an otherwise straight trajectory.
    Its neighbours are on the straight line → isolated → should be removed.
    """
    # Straight trajectory heading east along lat=38.886
    # fix3 jumps ~40m north (≈ 0.00036 degrees lat)
    rows = [
        _row("V1", _t(0),  38.886,       -77.080),
        _row("V1", _t(2),  38.886,       -77.078),
        _row("V1", _t(4),  38.886,       -77.076),
        _row("V1", _t(6),  38.886 + 0.00036, -77.074),  # outlier ~40m north
        _row("V1", _t(8),  38.886,       -77.072),
        _row("V1", _t(10), 38.886,       -77.070),
    ]
    pre = _pre(rows)
    # Use tight speed limit so speed test doesn't interfere (all speeds are low here)
    cleaned, removed = filter_gps_noise(
        pre, max_speed_kmh=250.0, local_outlier_threshold_m=30.0, max_time_gap_seconds=30.0
    )
    assert len(removed) == 1
    assert removed.iloc[0]["noise_reason"] == "isolated_local_trajectory_outlier"
