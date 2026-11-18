"""
Visualization helpers for probe data analysis.

Returns Plotly figures (rendered in Streamlit via st.plotly_chart()) so maps
support zoom, pan and hover inspection.
"""

from typing import Iterable, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BLUE = "#4C9BE8"
RED = "#E84C4C"
GREEN = "#2ECC71"
ORANGE = "orange"

# Shared Plotly config for st.plotly_chart: mouse-wheel zoom on, logo off.
PLOTLY_CONFIG = {"scrollZoom": True, "displaylogo": False}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _base_figure(title: str, roads_gdf: gpd.GeoDataFrame, height: int = 500) -> go.Figure:
    """Figure with lon/lat axes at true ground aspect ratio."""
    fig = go.Figure()
    # At this latitude one degree of longitude is cos(lat) shorter than one
    # degree of latitude; lock the axes so distances/angles look right.
    _, miny, _, maxy = roads_gdf.total_bounds if len(roads_gdf) else (0, 0, 0, 0)
    mean_lat = (miny + maxy) / 2.0
    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.75)", font=dict(size=11)),
        dragmode="pan",
        hovermode="closest",
        plot_bgcolor="white",
    )
    fig.update_xaxes(title="Longitude", showgrid=True, gridcolor="#eee", zeroline=False)
    fig.update_yaxes(
        title="Latitude", showgrid=True, gridcolor="#eee", zeroline=False,
        scaleanchor="x", scaleratio=1.0 / np.cos(np.radians(mean_lat)),
    )
    return fig


def _add_roads(
    fig: go.Figure,
    roads_gdf: gpd.GeoDataFrame,
    colors: Iterable[str] = ("black",),
    show_names: bool = False,
) -> None:
    colors = list(colors)
    for i, (geom, name) in enumerate(zip(roads_gdf.geometry, roads_gdf["road_name"])):
        xs, ys = geom.xy
        fig.add_trace(go.Scatter(
            x=list(xs), y=list(ys), mode="lines",
            line=dict(color=colors[i % len(colors)], width=4),
            name=name if show_names else "Mapped road (existing map)",
            legendgroup=None if show_names else "roads",
            showlegend=show_names or i == 0,
            hovertemplate=f"{name}<extra></extra>",
        ))


def _points(df: pd.DataFrame, lon: str = "longitude", lat: str = "latitude", **kwargs) -> go.Scattergl:
    """WebGL scatter of probe points with vehicle/speed hover info when available."""
    custom_cols = [c for c in ("vehicle_id", "derived_speed_kmh", "dist_to_road_m") if c in df.columns]
    hover = "lon %{x:.5f}<br>lat %{y:.5f}"
    for i, c in enumerate(custom_cols):
        fmt = "" if c == "vehicle_id" else ":.1f"
        hover += f"<br>{c} %{{customdata[{i}]{fmt}}}"
    return go.Scattergl(
        x=df[lon], y=df[lat], mode="markers",
        customdata=df[custom_cols].values if custom_cols else None,
        hovertemplate=hover + "<extra>%{fullData.name}</extra>",
        **kwargs,
    )


def _snap_segments(pairs: Iterable[tuple]) -> tuple:
    """Flatten (x0, y0, x1, y1) pairs into one x/y list separated by None."""
    xs, ys = [], []
    for x0, y0, x1, y1 in pairs:
        xs += [x0, x1, None]
        ys += [y0, y1, None]
    return xs, ys


# ── Maps ───────────────────────────────────────────────────────────────────────

def plot_raw_probes(df: pd.DataFrame, roads_gdf: gpd.GeoDataFrame) -> go.Figure:
    """Map 1: raw probe data coloured by the simulation's ground_truth_type.

    The colours are the simulation's answer key — real data carries no such label.
    """
    fig = _base_figure("Raw Probe Data — Arlington VA", roads_gdf)

    normal = df[df["ground_truth_type"] == "mapped_road"]
    fig.add_trace(_points(normal, name="GPS fixes — vehicles on mapped road (simulated truth)",
                          marker=dict(size=4, color=BLUE, opacity=0.5)))

    cand = df[df["ground_truth_type"] == "candidate_new_road"]
    fig.add_trace(_points(cand, name="GPS fixes — vehicles on unmapped new road (simulated truth)",
                          marker=dict(size=4, color=RED, opacity=0.7)))

    _add_roads(fig, roads_gdf)
    return fig


def plot_filtered_probes(
    cleaned: pd.DataFrame,
    removed: pd.DataFrame,
    roads_gdf: gpd.GeoDataFrame,
) -> go.Figure:
    """Map 2: cleaned vs removed GPS noise observations, removed ones split by filter."""
    fig = _base_figure("Preprocessing Filter: Kept vs Removed", roads_gdf)
    fig.add_trace(_points(cleaned, name="Kept",
                          marker=dict(size=4, color=BLUE, opacity=0.4)))
    _add_roads(fig, roads_gdf)

    if removed.empty:
        return fig

    # One layer per filter (noise_reason from preprocessing.filter_gps_noise),
    # each with its own colour AND symbol so it reads without colour too.
    reasons = [
        ("unrealistic_speed_between_consecutive_fixes", "Removed: speed spike (> 200 km/h in and out)",
         "#D62728", "x"),
        ("isolated_local_trajectory_outlier", "Removed: isolated jump (> 25 m off own path)",
         "#7B3FBF", "diamond"),
        ("multiple_flags", "Removed: both filters", "black", "star"),
    ]
    reason_col = removed["noise_reason"] if "noise_reason" in removed.columns else None
    for key, label, color, symbol in reasons:
        sub = removed[reason_col == key] if reason_col is not None else removed.iloc[0:0]
        if sub.empty:
            continue
        cols = [c for c in ("vehicle_id", "derived_speed_kmh", "next_speed_kmh",
                            "local_trajectory_residual_m") if c in sub.columns]
        hover = label + "<br>vehicle %{customdata[0]}"
        if "derived_speed_kmh" in cols:
            hover += "<br>speed in %{customdata[1]:.0f} km/h<br>speed out %{customdata[2]:.0f} km/h"
        if "local_trajectory_residual_m" in cols:
            hover += f"<br>off own path %{{customdata[{cols.index('local_trajectory_residual_m')}]:.1f}} m"
        fig.add_trace(go.Scatter(
            x=sub["longitude"], y=sub["latitude"], mode="markers", name=label,
            marker=dict(size=10, color=color, symbol=symbol, line=dict(color="white", width=1)),
            customdata=sub[cols].values,
            hovertemplate=hover + "<extra></extra>",
        ))

    # Anything without a recognised reason (shouldn't happen) still gets drawn.
    if reason_col is not None:
        other = removed[~reason_col.isin([k for k, *_ in reasons])]
    else:
        other = removed
    if not other.empty:
        fig.add_trace(_points(other, name="Removed (noise)",
                              marker=dict(size=8, color="red", symbol="x")))
    return fig


def plot_h3_aggregation(
    h3_agg: pd.DataFrame,
    roads_gdf: gpd.GeoDataFrame,
    resolution: Optional[int] = None,
) -> go.Figure:
    """Map 3: two stacked maps of H3 cell centres — unique vehicles (top) and
    average speed (bottom). Axes are linked, so zoom/pan applies to both."""
    panels = [
        ("unique_vehicles", "Unique vehicles", "YlOrRd"),
        ("avg_speed", "Avg speed (km/h)", "Viridis"),
    ]
    res_txt = f" (res {resolution})" if resolution is not None else ""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=[f"{label} per H3 cell{res_txt}" for _, label, _ in panels],
    )

    hover = (
        "%{customdata[0]}<br>obs %{customdata[1]}<br>"
        "vehicles %{customdata[2]}<br>avg speed %{customdata[3]:.1f} km/h<extra></extra>"
    )
    custom = h3_agg[["h3_cell", "observations", "unique_vehicles", "avg_speed"]].values

    for row, (col, label, scale) in enumerate(panels, start=1):
        for geom, name in zip(roads_gdf.geometry, roads_gdf["road_name"]):
            xs, ys = geom.xy
            fig.add_trace(go.Scatter(
                x=list(xs), y=list(ys), mode="lines",
                line=dict(color="black", width=4),
                name="Mapped road (existing map)", legendgroup="roads",
                showlegend=(row == 1 and name == roads_gdf["road_name"].iloc[0]),
                hovertemplate=f"{name}<extra></extra>",
            ), row=row, col=1)

        fig.add_trace(go.Scattergl(
            x=h3_agg["lng_center"], y=h3_agg["lat_center"], mode="markers",
            name=label, showlegend=False,
            marker=dict(
                size=10, color=h3_agg[col], colorscale=scale, opacity=0.85,
                colorbar=dict(title=label, len=0.42, y=0.79 if row == 1 else 0.21),
            ),
            customdata=custom, hovertemplate=hover,
        ), row=row, col=1)

    # True ground aspect on the top map; the bottom map shares both axes,
    # so it gets the same ranges and zoom/pan stays in sync.
    _, miny, _, maxy = roads_gdf.total_bounds if len(roads_gdf) else (0, 0, 0, 0)
    fig.update_yaxes(scaleanchor="x", scaleratio=1.0 / np.cos(np.radians((miny + maxy) / 2.0)),
                     row=1, col=1)
    fig.update_yaxes(matches="y", row=2, col=1)
    fig.update_xaxes(showgrid=True, gridcolor="#eee", zeroline=False)
    fig.update_yaxes(title="Latitude", showgrid=True, gridcolor="#eee", zeroline=False)
    fig.update_xaxes(title="Longitude", row=2, col=1)
    fig.update_layout(
        height=800,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.75)", font=dict(size=11)),
        dragmode="pan",
        hovermode="closest",
        plot_bgcolor="white",
    )
    return fig


def _add_matching_traces(
    fig: go.Figure,
    col: int,
    vehicle_df: pd.DataFrame,
    matched_df: pd.DataFrame,
    roads_gdf: gpd.GeoDataFrame,
    vehicle_id: str,
    prefix: str,
    showlegend: bool,
) -> None:
    """Draw roads, raw trajectory, snapped points and snap lines into panel `col`.

    Snap lines are coloured by which road was matched:
      green = Wilson Blvd (correct for east-west vehicle)
      red   = cross street (mismatch near the intersection)

    Traces share a legendgroup across panels, so one legend click toggles both.
    """
    lon_col, lat_col, road_col = f"{prefix}_lon", f"{prefix}_lat", f"{prefix}_road_id"

    road_colors = ("black", "#888888")
    for i, (geom, name) in enumerate(zip(roads_gdf.geometry, roads_gdf["road_name"])):
        xs, ys = geom.xy
        fig.add_trace(go.Scatter(
            x=list(xs), y=list(ys), mode="lines",
            line=dict(color=road_colors[i % len(road_colors)], width=4),
            name=name, legendgroup=f"road-{name}", showlegend=showlegend,
            hovertemplate=f"{name}<extra></extra>",
        ), row=1, col=col)

    veh = vehicle_df[vehicle_df["vehicle_id"] == vehicle_id].sort_values("timestamp")
    vm = matched_df[matched_df["vehicle_id"] == vehicle_id].dropna(subset=[lon_col, lat_col])

    wilson_road_id = roads_gdf[roads_gdf["road_name"].str.contains("Wilson")]["road_id"].iloc[0]
    joined = vm.merge(
        veh[["timestamp", "longitude", "latitude"]], on="timestamp",
        how="inner", suffixes=("_m", ""),
    )
    on_wilson = joined[road_col] == wilson_road_id

    for mask, color, name in (
        (on_wilson, GREEN, "Snap → Wilson Blvd (correct)"),
        (~on_wilson, RED, "Snap → cross street (mismatch)"),
    ):
        sub = joined[mask]
        xs, ys = _snap_segments(zip(sub["longitude"], sub["latitude"], sub[lon_col], sub[lat_col]))
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", name=name, legendgroup=name, showlegend=showlegend,
            line=dict(color=color, width=1.5), opacity=0.6, hoverinfo="skip",
        ), row=1, col=col)

    fig.add_trace(go.Scatter(
        x=veh["longitude"], y=veh["latitude"], mode="lines+markers",
        name="Raw GPS trajectory", legendgroup="raw", showlegend=showlegend,
        line=dict(color="blue", width=1), marker=dict(size=5, color="blue"), opacity=0.6,
        hovertemplate="%{text}<br>lon %{x:.5f}<br>lat %{y:.5f}<extra>raw</extra>",
        text=veh["timestamp"].astype(str),
    ), row=1, col=col)
    fig.add_trace(go.Scatter(
        x=vm[lon_col], y=vm[lat_col], mode="markers",
        name="Snapped position", legendgroup="snapped", showlegend=showlegend,
        marker=dict(size=6, color=ORANGE),
        customdata=vm[[road_col]].values,
        hovertemplate="road %{customdata[0]}<br>lon %{x:.5f}<br>lat %{y:.5f}<extra>snapped</extra>",
    ), row=1, col=col)


def plot_matching_comparison(
    vehicle_df: pd.DataFrame,
    matched_df: pd.DataFrame,
    hmm_df: pd.DataFrame,
    roads_gdf: gpd.GeoDataFrame,
    vehicle_id: Optional[str] = None,
) -> go.Figure:
    """Map 4: nearest-road (left) vs HMM / Viterbi (right) for one vehicle.

    One figure with linked axes, so zoom/pan applies to both panels, and one
    shared legend that toggles the same layer in both.
    """
    if vehicle_id is None:
        vehicle_id = vehicle_df["vehicle_id"].iloc[0]

    fig = make_subplots(
        rows=1, cols=2, shared_xaxes=True, shared_yaxes=True, horizontal_spacing=0.03,
        subplot_titles=["Nearest-road (baseline)", "HMM / Viterbi (trajectory-aware)"],
    )
    _add_matching_traces(fig, 1, vehicle_df, matched_df, roads_gdf, vehicle_id,
                         prefix="matched", showlegend=True)
    _add_matching_traces(fig, 2, vehicle_df, hmm_df, roads_gdf, vehicle_id,
                         prefix="hmm_matched", showlegend=False)

    # True ground aspect on the left panel; the right panel shares both axes,
    # so it gets the same ranges and zoom/pan stays in sync.
    _, miny, _, maxy = roads_gdf.total_bounds if len(roads_gdf) else (0, 0, 0, 0)
    fig.update_yaxes(scaleanchor="x", scaleratio=1.0 / np.cos(np.radians((miny + maxy) / 2.0)),
                     row=1, col=1)
    fig.update_xaxes(matches="x", row=1, col=2)
    fig.update_xaxes(title="Longitude", showgrid=True, gridcolor="#eee", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#eee", zeroline=False)
    fig.update_yaxes(title="Latitude", row=1, col=1)
    # Fixes where the two methods pick different roads.
    near = matched_df[matched_df["vehicle_id"] == vehicle_id][["timestamp", "longitude", "latitude",
                                                               "matched_road_id"]]
    hmm = hmm_df[hmm_df["vehicle_id"] == vehicle_id][["timestamp", "hmm_matched_road_id"]]
    both = near.merge(hmm, on="timestamp")
    disagree = both[both["matched_road_id"] != both["hmm_matched_road_id"]]

    # Open zoomed in where it matters: the difference is a few metres and invisible
    # at full-road zoom. Double-click resets to the full road.
    if not disagree.empty:
        cx, cy = disagree["longitude"].iloc[0], disagree["latitude"].iloc[0]
        half_x, half_y = 0.0005, 0.0004            # ≈ ±45 m
        zoom_txt = "zoomed to where the methods disagree"
        for col in (1, 2):
            fig.add_trace(go.Scatter(
                x=disagree["longitude"], y=disagree["latitude"], mode="markers",
                name="Methods disagree here", legendgroup="disagree", showlegend=(col == 1),
                marker=dict(size=22, color="rgba(0,0,0,0)", line=dict(color="#D62728", width=2)),
                hovertemplate="methods disagree on this fix<extra></extra>",
            ), row=1, col=col)
    else:
        ix = (roads_gdf.geometry.iloc[0].intersection(roads_gdf.geometry.iloc[1])
              if len(roads_gdf) > 1 else None)
        if ix is not None and not ix.is_empty:
            cx, cy = ix.centroid.x, ix.centroid.y
        else:
            cx, cy = both["longitude"].mean(), both["latitude"].mean()
        half_x, half_y = 0.0015, 0.0012            # ≈ ±130 m
        zoom_txt = "zoomed to intersection"
    fig.update_xaxes(range=[cx - half_x, cx + half_x])
    fig.update_yaxes(range=[cy - half_y, cy + half_y])

    fig.update_layout(
        title=dict(text=f"Map Matching — Vehicle {vehicle_id} ({zoom_txt} — "
                        "double-click to see full road)", font=dict(size=15)),
        height=580,
        margin=dict(l=10, r=10, t=70, b=90),
        legend=dict(orientation="h", x=0, y=-0.22, font=dict(size=11)),
        dragmode="pan",
        hovermode="closest",
        plot_bgcolor="white",
    )
    return fig


def plot_change_detection(
    df: pd.DataFrame,
    candidate_cells: pd.DataFrame,
    roads_gdf: gpd.GeoDataFrame,
    dist_threshold_m: float = 18.0,
) -> go.Figure:
    """Map 5: suspicious observations and flagged candidate H3 cells."""
    fig = _base_figure("Change Detection: Noise vs Candidate Map Change", roads_gdf)

    normal = df[df["dist_to_road_m"] <= dist_threshold_m]
    fig.add_trace(_points(normal, name="On-road obs",
                          marker=dict(size=3, color=BLUE, opacity=0.3)))

    susp = df[df["dist_to_road_m"] > dist_threshold_m]
    fig.add_trace(_points(susp, name="Suspicious off-road",
                          marker=dict(size=6, color=ORANGE, opacity=0.6)))

    _add_roads(fig, roads_gdf)

    if not candidate_cells.empty:
        fig.add_trace(go.Scatter(
            x=candidate_cells["avg_lon"], y=candidate_cells["avg_lat"], mode="markers",
            name=f"Candidate change cell (≥{dist_threshold_m:.0f} m, multi-vehicle)",
            marker=dict(size=16, color="red", symbol="star", line=dict(color="darkred", width=1)),
            hovertemplate="candidate cell<br>lon %{x:.5f}<br>lat %{y:.5f}<extra></extra>",
        ))
    return fig


def plot_threshold_sweep(
    sweep: pd.DataFrame,
    current_t: float,
    noise_rule_t: Optional[float] = None,
) -> go.Figure:
    """True-change vs false-positive cells across off-road thresholds.

    sweep: DataFrame with columns threshold_m, true_positive_cells, false_positive_cells.
    Vertical reference lines mark the current slider value and the noise-only 3σ rule.
    """
    fig = go.Figure()
    for col, name, color, symbol in (
        ("true_positive_cells", "True-change cells detected", "#2F6FD0", "circle"),
        ("false_positive_cells", "False-positive cells", "#D9822B", "diamond"),
    ):
        fig.add_trace(go.Scatter(
            x=sweep["threshold_m"], y=sweep[col], mode="lines+markers", name=name,
            line=dict(color=color, width=2), marker=dict(size=8, color=color, symbol=symbol),
            hovertemplate=f"{name}: %{{y}}<extra></extra>",
        ))

    # Reference lines are drawn as legend entries (not top annotations) so their
    # labels never collide when the values are close or equal.
    refs = [(current_t, "Current setting", "solid")]
    if noise_rule_t is not None:
        refs.append((noise_rule_t, "3σ noise rule", "dash"))
    y_top = max(1, sweep[["true_positive_cells", "false_positive_cells"]].values.max()) * 1.05
    for x, label, dash in refs:
        fig.add_trace(go.Scatter(
            x=[x, x], y=[0, y_top], mode="lines",
            name=f"{label} ({x:.1f} m)",
            line=dict(color="#555", width=1.5, dash=dash),
            hovertemplate=f"{label}: {x:.1f} m<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text="Threshold sweep — detections vs false positives", font=dict(size=15)),
        height=440,
        margin=dict(l=10, r=10, t=50, b=10),
        hovermode="x unified",
        plot_bgcolor="white",
        legend=dict(orientation="h", x=0, y=-0.2, font=dict(size=11)),
    )
    fig.update_xaxes(title="Off-road threshold (m)", showgrid=True, gridcolor="#eee", zeroline=False)
    fig.update_yaxes(title="H3 cells", showgrid=True, gridcolor="#eee", zeroline=False, rangemode="tozero")
    return fig


def summary_bar(metrics: dict) -> go.Figure:
    """Simple bar chart of key summary metrics."""
    labels = ["Total obs", "Unique vehicles", "Removed noise", "Off-road obs", "Candidate cells"]
    keys = ["total_obs", "unique_vehicles", "removed_noise", "off_road_obs", "candidate_cells"]
    values = [metrics.get(k, 0) for k in keys]
    colors = [BLUE, BLUE, RED, ORANGE, "red"]
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors, text=values, textposition="outside"))
    fig.update_layout(title="Simulation Summary", yaxis_title="Count", height=400)
    return fig
