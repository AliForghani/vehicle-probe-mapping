"""
H3 spatial indexing for vehicle probe observations.

H3 assigns every lat/lon to a hexagonal cell at a chosen resolution.
It is used for:
  - grouping / aggregation (count obs per cell)
  - spatial partitioning (process data shard by shard)
  - neighbourhood analysis (ring queries)
  - coarse-to-fine drill-down (res 9 → 11)

H3 is NOT:
  - a replacement for road geometry
  - a map-matching algorithm
  - an exact distance calculator (use PostGIS for that)

Resolution guide (approximate cell edge length):
  res 9  ≈  174 m   (city-block scale)
  res 10 ≈   66 m   (large intersection)
  res 11 ≈   25 m   (lane-level — chosen for this prototype)

Resolution 11 is an *illustrative* choice for this prototype.
It is NOT a universal automotive standard.
True lane-level analysis often requires sub-metre positioning and lidar.
"""

import logging

import h3
import pandas as pd

logger = logging.getLogger(__name__)

RESOLUTIONS = [11, 10, 9]   # fine → coarse


def latlng_to_h3(lat: float, lng: float, resolution: int) -> str:
    """Return H3 cell ID for a single lat/lng at the given resolution."""
    return h3.latlng_to_cell(lat, lng, resolution)


def cell_to_parent(cell: str, parent_resolution: int) -> str:
    """Return the parent H3 cell at a coarser resolution."""
    return h3.cell_to_parent(cell, parent_resolution)


def add_h3_cells(df: pd.DataFrame) -> pd.DataFrame:
    """Add H3 cell columns at resolutions 11, 10, and 9.

    Resolution 11 is computed directly from lat/lon.
    Resolutions 10 and 9 are derived from the res-11 parent — this is
    cheaper than recomputing from lat/lon and guarantees hierarchy consistency.

    Adds columns: h3_cell_res11, h3_cell_res10, h3_cell_res9
    """
    df = df.copy()

    logger.info("Assigning H3 cells at resolutions %s …", RESOLUTIONS)
    df["h3_cell_res11"] = [
        h3.latlng_to_cell(lat, lng, 11)
        for lat, lng in zip(df["latitude"], df["longitude"])
    ]

    # Parent derivation: faster and hierarchy-consistent
    df["h3_cell_res10"] = df["h3_cell_res11"].map(lambda c: h3.cell_to_parent(c, 10))
    df["h3_cell_res9"]  = df["h3_cell_res11"].map(lambda c: h3.cell_to_parent(c, 9))

    logger.info(
        "H3 cells assigned. Unique cells — res11: %d, res10: %d, res9: %d",
        df["h3_cell_res11"].nunique(),
        df["h3_cell_res10"].nunique(),
        df["h3_cell_res9"].nunique(),
    )
    return df


def aggregate_by_h3(df: pd.DataFrame, resolution: int = 11) -> pd.DataFrame:
    """Aggregate probe observations by H3 cell.

    Returns a DataFrame with:
        h3_cell, observations, unique_vehicles, avg_speed, lat_center, lng_center

    WHY multi-vehicle aggregation matters:
      100 observations from ONE vehicle   → weak evidence (could be one confused driver)
      100 observations from 20 INDEPENDENT vehicles → strong evidence of a real feature
    """
    col = f"h3_cell_res{resolution}"
    if col not in df.columns:
        raise ValueError(f"Column {col} not found. Run add_h3_cells() first.")

    agg = (
        df.groupby(col)
        .agg(
            observations=("vehicle_id", "count"),
            unique_vehicles=("vehicle_id", "nunique"),
            avg_speed=("speed", "mean"),
        )
        .reset_index()
        .rename(columns={col: "h3_cell"})
    )

    # Add approximate centre of each cell for visualisation
    centres = [h3.cell_to_latlng(c) for c in agg["h3_cell"]]
    agg["lat_center"] = [c[0] for c in centres]
    agg["lng_center"] = [c[1] for c in centres]

    return agg


def h3_neighbors(cell: str, k: int = 1) -> list[str]:
    """Return all H3 cells within k rings of the given cell."""
    return list(h3.grid_disk(cell, k))
