"""
Synthetic GPS probe data generation for Washington DC / Northern Virginia area.

Scenario:
  - One existing mapped road (Wilson Blvd, Arlington VA segment)
  - Normal vehicles travel along it with GPS noise
  - A fraction of vehicles repeatedly follow a parallel path ~30m north —
    simulating a new lane or geometry change not yet in the map
"""

import logging
import math
from datetime import datetime, timedelta
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import LineString, Point

logger = logging.getLogger(__name__)

# Coordinate transformers
_TO_UTM = Transformer.from_crs("EPSG:4326", "EPSG:32618", always_xy=True)
_TO_WGS = Transformer.from_crs("EPSG:32618", "EPSG:4326", always_xy=True)


def _offset_linestring(line: LineString, offset_m: float) -> LineString:
    """Offset a LineString by offset_m metres perpendicular (northward for + values).

    Uses UTM for metric accuracy, then converts back to WGS84.
    """
    coords_utm = [_TO_UTM.transform(lon, lat) for lon, lat in line.coords]
    line_utm = LineString(coords_utm)
    # parallel_offset is deprecated in Shapely 2; use offset_curve
    try:
        offset_utm = line_utm.offset_curve(offset_m)
    except AttributeError:  # shapely < 2
        offset_utm = line_utm.parallel_offset(offset_m, side="left")
    coords_wgs = [_TO_WGS.transform(x, y) for x, y in offset_utm.coords]
    return LineString([(lon, lat) for lon, lat in coords_wgs])


def generate_road_network() -> gpd.GeoDataFrame:
    """Return a GeoDataFrame of the existing mapped road segments.

    Represents a simplified segment of Wilson Blvd in Arlington, VA running
    roughly east-west.  Coordinates are WGS84 (EPSG:4326).
    """
    # Approximate Wilson Blvd segment: Courthouse → Ballston (~2.4 km)
    road_coords = [
        (-77.0820, 38.8862),
        (-77.0750, 38.8870),
        (-77.0680, 38.8875),
        (-77.0610, 38.8880),
        (-77.0540, 38.8882),
        (-77.0470, 38.8880),
    ]
    road = LineString(road_coords)

    gdf = gpd.GeoDataFrame(
        {
            "road_id": [1],
            "road_name": ["Wilson Blvd (simulated)"],
            "geometry": [road],
        },
        crs="EPSG:4326",
    )
    logger.info("Generated road network: %d segment(s)", len(gdf))
    return gdf


def _sample_point_on_line(line: LineString, rng: np.random.Generator) -> tuple[float, float]:
    """Return a random (lon, lat) point uniformly sampled along line."""
    frac = rng.uniform(0.0, 1.0)
    pt = line.interpolate(frac, normalized=True)
    return pt.x, pt.y  # lon, lat


def generate_probe_data(
    road_gdf: gpd.GeoDataFrame,
    n_vehicles: int = 80,
    points_per_vehicle: int = 60,
    noise_std_m: float = 6.0,
    candidate_road_fraction: float = 0.20,
    candidate_offset_m: float = 30.0,
    start_time: Optional[datetime] = None,
) -> pd.DataFrame:
    """Generate synthetic vehicle probe observations.

    Parameters
    ----------
    road_gdf:               existing mapped road network
    n_vehicles:             total number of synthetic vehicles
    points_per_vehicle:     GPS fixes per vehicle
    noise_std_m:            GPS noise standard deviation in metres
    candidate_road_fraction: fraction of vehicles that follow the offset path
    candidate_offset_m:     lateral offset of the candidate new road (metres)
    start_time:             base timestamp; defaults to midnight today

    Returns
    -------
    DataFrame with columns:
        vehicle_id, timestamp, latitude, longitude, speed, heading,
        ground_truth_type
    NOTE: ground_truth_type is for validation ONLY — never use inside the
    detection algorithm.
    """
    if start_time is None:
        start_time = datetime(2026, 10, 12, 6, 0, 0)

    rng = np.random.default_rng(42)
    road = road_gdf.geometry.iloc[0]

    # Build candidate (offset) road
    candidate_road = _offset_linestring(road, candidate_offset_m)

    n_candidate = int(n_vehicles * candidate_road_fraction)
    n_normal = n_vehicles - n_candidate

    records: list[dict] = []

    def _add_vehicle(vid: int, line: LineString, label: str) -> None:
        # Vehicles travel in order along the line (increasing fraction)
        fractions = np.sort(rng.uniform(0.0, 1.0, points_per_vehicle))
        for i, frac in enumerate(fractions):
            pt = line.interpolate(frac, normalized=True)
            lon_clean, lat_clean = pt.x, pt.y

            # Convert to UTM to add metric noise, then back
            utm_x, utm_y = _TO_UTM.transform(lon_clean, lat_clean)
            utm_x += rng.normal(0, noise_std_m)
            utm_y += rng.normal(0, noise_std_m)
            lon_noisy, lat_noisy = _TO_WGS.transform(utm_x, utm_y)

            # Approximate heading from previous point
            if i == 0:
                heading = 90.0  # start heading east
            else:
                prev = records[-1]
                dlat = lat_noisy - prev["latitude"]
                dlon = lon_noisy - prev["longitude"]
                heading = (math.degrees(math.atan2(dlon, dlat)) + 360) % 360

            # Approximate speed: ~40 km/h with small variation
            speed = float(rng.normal(40.0, 5.0))
            speed = max(0.0, speed)

            ts = start_time + timedelta(seconds=i * 2 + vid * 200)

            records.append(
                {
                    "vehicle_id": f"V{vid:04d}",
                    "timestamp": ts,
                    "latitude": lat_noisy,
                    "longitude": lon_noisy,
                    "speed": round(speed, 2),
                    "heading": round(heading, 2),
                    "ground_truth_type": label,
                }
            )

    for vid in range(n_normal):
        _add_vehicle(vid, road, "mapped_road")

    for vid in range(n_normal, n_vehicles):
        _add_vehicle(vid, candidate_road, "candidate_new_road")

    df = pd.DataFrame(records)
    logger.info(
        "Generated %d observations: %d normal vehicles, %d candidate-road vehicles",
        len(df),
        n_normal,
        n_candidate,
    )
    return df
