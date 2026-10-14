"""
Visualization helpers for probe data analysis.

Returns matplotlib figures (used directly by Streamlit via st.pyplot()).
"""

from typing import Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd


def plot_raw_probes(df: pd.DataFrame, roads_gdf: gpd.GeoDataFrame) -> plt.Figure:
    """Map 1: raw probe data coloured by ground_truth_type."""
    fig, ax = plt.subplots(figsize=(12, 5))

    # Road
    for geom in roads_gdf.geometry:
        xs, ys = geom.xy
        ax.plot(xs, ys, "k-", linewidth=3, label="Mapped road", zorder=3)

    # Normal probes
    normal = df[df["ground_truth_type"] == "mapped_road"]
    ax.scatter(normal["longitude"], normal["latitude"],
               s=4, c="#4C9BE8", alpha=0.5, label="Normal GPS (mapped road)")

    # Candidate new road
    cand = df[df["ground_truth_type"] == "candidate_new_road"]
    ax.scatter(cand["longitude"], cand["latitude"],
               s=4, c="#E84C4C", alpha=0.7, label="Candidate new road")

    ax.set_title("Raw Probe Data — Arlington VA", fontsize=13)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    return fig


def plot_filtered_probes(
    cleaned: pd.DataFrame,
    removed: pd.DataFrame,
    roads_gdf: gpd.GeoDataFrame,
) -> plt.Figure:
    """Map 2: cleaned vs removed GPS noise observations."""
    fig, ax = plt.subplots(figsize=(12, 5))
    for geom in roads_gdf.geometry:
        xs, ys = geom.xy
        ax.plot(xs, ys, "k-", linewidth=3, zorder=3)
    ax.scatter(cleaned["longitude"], cleaned["latitude"],
               s=4, c="#4C9BE8", alpha=0.4, label="Kept")
    if not removed.empty:
        ax.scatter(removed["longitude"], removed["latitude"],
                   s=20, c="red", marker="x", label="Removed (noise)")
    ax.set_title("Preprocessing Filter: Kept vs Removed", fontsize=13)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_h3_aggregation(
    h3_agg: pd.DataFrame,
    roads_gdf: gpd.GeoDataFrame,
    color_by: str = "unique_vehicles",
) -> plt.Figure:
    """Map 3: H3 cell centres coloured by observation density or unique vehicles."""
    fig, ax = plt.subplots(figsize=(12, 5))
    for geom in roads_gdf.geometry:
        xs, ys = geom.xy
        ax.plot(xs, ys, "k-", linewidth=3, zorder=3)

    vals = h3_agg[color_by].values
    sc = ax.scatter(
        h3_agg["lng_center"], h3_agg["lat_center"],
        c=vals, cmap="YlOrRd", s=80, alpha=0.8, edgecolors="none",
    )
    plt.colorbar(sc, ax=ax, label=color_by.replace("_", " ").title())
    ax.set_title(f"H3 Aggregation — {color_by.replace('_', ' ').title()} per Cell (res 11)", fontsize=13)
    fig.tight_layout()
    return fig


def plot_map_matching(
    vehicle_df: pd.DataFrame,
    matched_df: pd.DataFrame,
    roads_gdf: gpd.GeoDataFrame,
    vehicle_id: Optional[str] = None,
) -> plt.Figure:
    """Map 4: one vehicle trajectory with nearest-road snap points.

    Draws all mapped roads so the intersection ambiguity zone is visible.
    Snap lines are coloured by which road was matched:
      - green  = Wilson Blvd (correct for east-west vehicle)
      - red    = cross street (likely a mismatch near the intersection)
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    # Draw each road with a distinct colour and label
    road_colors = ["black", "#888888"]
    for i, (geom, name) in enumerate(zip(roads_gdf.geometry, roads_gdf["road_name"])):
        xs, ys = geom.xy
        ax.plot(xs, ys, color=road_colors[i % len(road_colors)],
                linewidth=3, zorder=3, label=name)

    if vehicle_id is None:
        vehicle_id = vehicle_df["vehicle_id"].iloc[0]

    veh = vehicle_df[vehicle_df["vehicle_id"] == vehicle_id].sort_values("timestamp")
    ax.plot(veh["longitude"], veh["latitude"], "b.-", alpha=0.6,
            markersize=5, linewidth=1, label="Raw GPS trajectory")

    vm = matched_df[matched_df["vehicle_id"] == vehicle_id].dropna(
        subset=["matched_lon", "matched_lat"]
    )
    ax.scatter(vm["matched_lon"], vm["matched_lat"],
               s=15, c="orange", zorder=4, label="Snapped position (nearest road)")

    # Snap lines: red if snapped to cross street (road_id != 1), green otherwise
    wilson_road_id = roads_gdf[roads_gdf["road_name"].str.contains("Wilson")]["road_id"].iloc[0]
    for _, row in vm.iterrows():
        orig_row = veh[veh["timestamp"] == row["timestamp"]]
        if orig_row.empty:
            continue
        snapped_to_cross = (row.get("matched_road_id") != wilson_road_id)
        color = "#E84C4C" if snapped_to_cross else "#2ECC71"
        ax.plot(
            [orig_row.iloc[0]["longitude"], row["matched_lon"]],
            [orig_row.iloc[0]["latitude"],  row["matched_lat"]],
            color=color, alpha=0.5, linewidth=1.0,
        )

    # Legend entries for snap line colours
    ax.plot([], [], color="#2ECC71", linewidth=1.5, label="Snap → Wilson Blvd (correct)")
    ax.plot([], [], color="#E84C4C", linewidth=1.5, label="Snap → cross street (mismatch)")

    ax.set_title(
        f"Map Matching — Vehicle {vehicle_id}  "
        f"(red snap lines = nearest-road error near intersection)",
        fontsize=11,
    )
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig


def plot_hmm_matching(
    vehicle_df: pd.DataFrame,
    hmm_df: pd.DataFrame,
    roads_gdf: gpd.GeoDataFrame,
    vehicle_id: Optional[str] = None,
) -> plt.Figure:
    """Map 4b: same vehicle trajectory snapped by HMM / Viterbi.

    Uses hmm_matched_* columns from map_match_trajectory().
    Snap lines are coloured the same way as plot_map_matching:
      green  = Wilson Blvd (correct)
      red    = cross street (mismatch — should be rare with HMM)
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    road_colors = ["black", "#888888"]
    for i, (geom, name) in enumerate(zip(roads_gdf.geometry, roads_gdf["road_name"])):
        xs, ys = geom.xy
        ax.plot(xs, ys, color=road_colors[i % len(road_colors)],
                linewidth=3, zorder=3, label=name)

    if vehicle_id is None:
        vehicle_id = vehicle_df["vehicle_id"].iloc[0]

    veh = vehicle_df[vehicle_df["vehicle_id"] == vehicle_id].sort_values("timestamp")
    ax.plot(veh["longitude"], veh["latitude"], "b.-", alpha=0.6,
            markersize=5, linewidth=1, label="Raw GPS trajectory")

    vm = hmm_df[hmm_df["vehicle_id"] == vehicle_id].dropna(
        subset=["hmm_matched_lon", "hmm_matched_lat"]
    )
    ax.scatter(vm["hmm_matched_lon"], vm["hmm_matched_lat"],
               s=15, c="orange", zorder=4, label="Snapped position (HMM)")

    wilson_road_id = roads_gdf[roads_gdf["road_name"].str.contains("Wilson")]["road_id"].iloc[0]
    for _, row in vm.iterrows():
        orig_row = veh[veh["timestamp"] == row["timestamp"]]
        if orig_row.empty:
            continue
        snapped_to_cross = (row.get("hmm_matched_road_id") != wilson_road_id)
        color = "#E84C4C" if snapped_to_cross else "#2ECC71"
        ax.plot(
            [orig_row.iloc[0]["longitude"], row["hmm_matched_lon"]],
            [orig_row.iloc[0]["latitude"],  row["hmm_matched_lat"]],
            color=color, alpha=0.5, linewidth=1.0,
        )

    ax.plot([], [], color="#2ECC71", linewidth=1.5, label="Snap → Wilson Blvd (correct)")
    ax.plot([], [], color="#E84C4C", linewidth=1.5, label="Snap → cross street (mismatch)")

    ax.set_title(
        f"HMM / Viterbi Matching — Vehicle {vehicle_id}  "
        f"(trajectory-aware: fewer red snaps near intersection)",
        fontsize=11,
    )
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig


def plot_change_detection(
    df: pd.DataFrame,
    candidate_cells: pd.DataFrame,
    roads_gdf: gpd.GeoDataFrame,
    dist_threshold_m: float = 18.0,
) -> plt.Figure:
    """Map 5: suspicious observations and flagged candidate H3 cells."""
    fig, ax = plt.subplots(figsize=(12, 5))
    for geom in roads_gdf.geometry:
        xs, ys = geom.xy
        ax.plot(xs, ys, "k-", linewidth=3, zorder=3)

    normal = df[df["dist_to_road_m"] <= dist_threshold_m]
    ax.scatter(normal["longitude"], normal["latitude"],
               s=3, c="#4C9BE8", alpha=0.3, label="On-road obs")

    susp = df[df["dist_to_road_m"] > dist_threshold_m]
    ax.scatter(susp["longitude"], susp["latitude"],
               s=8, c="orange", alpha=0.6, label="Suspicious off-road")

    if not candidate_cells.empty:
        ax.scatter(
            candidate_cells["avg_lon"], candidate_cells["avg_lat"],
            s=120, c="red", marker="*", zorder=5,
            label=f"Candidate change cell\n(≥{dist_threshold_m}m, multi-vehicle)",
        )

    ax.set_title("Change Detection: Noise vs Candidate Map Change", fontsize=13)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def summary_bar(metrics: dict) -> plt.Figure:
    """Simple bar chart of key summary metrics."""
    labels = ["Total obs", "Unique vehicles", "Removed noise",
              "Off-road obs", "Candidate cells"]
    values = [
        metrics.get("total_obs", 0),
        metrics.get("unique_vehicles", 0),
        metrics.get("removed_noise", 0),
        metrics.get("off_road_obs", 0),
        metrics.get("candidate_cells", 0),
    ]
    colors = ["#4C9BE8", "#4C9BE8", "#E84C4C", "orange", "red"]
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(labels, values, color=colors)
    ax.bar_label(bars, fmt="%d")
    ax.set_title("Simulation Summary", fontsize=13)
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig
