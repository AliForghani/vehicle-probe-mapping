"""
PostGIS spatial queries for probe data analysis.

Key distinction:
  H3 cell ID  → grouping / aggregation / partitioning
  PostGIS geometry → exact distance / proximity / geometric calculations

All precise spatial calculations here use PostGIS geometry, NOT H3 cell IDs.
"""

import logging

import pandas as pd
from sqlalchemy import text

logger = logging.getLogger(__name__)


def query_nearby_probes(
    engine,
    lon: float,
    lat: float,
    radius_m: float = 100.0,
) -> pd.DataFrame:
    """Find all probe observations within radius_m metres of a point.

    Uses ST_DWithin on geography (spheroidal) for accurate metric distances.
    The GiST index on geom accelerates this to O(log n) rather than O(n).

    This query operates on the PostGIS geometry column — NOT the H3 cell.
    ST_DWithin with a 100m radius gives exact circle containment;
    H3 at res 11 (~25m cells) would only give an approximate hexagonal region.
    """
    sql = text("""
        SELECT
            vehicle_id,
            timestamp,
            latitude,
            longitude,
            speed,
            heading,
            h3_cell_res11,
            ST_Distance(
                geom::geography,
                ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography
            ) AS distance_m
        FROM vehicle_probes
        WHERE ST_DWithin(
            geom::geography,
            ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography,
            :radius_m
        )
        ORDER BY distance_m;
    """)
    with engine.connect() as conn:
        result = conn.execute(sql, {"lon": lon, "lat": lat, "radius_m": radius_m})
        return pd.DataFrame(result.fetchall(), columns=result.keys())


def aggregate_probes_by_h3(engine, resolution: int = 11) -> pd.DataFrame:
    """Aggregate probe counts and statistics per H3 cell.

    H3 grouping is ideal here because we want:
    - per-cell observation density
    - per-cell unique vehicle counts
    - per-cell average speed

    WHY unique vehicles matter:
      100 observations from 1 vehicle  → weak (one driver's habit or loop)
      100 observations from 20 vehicles → strong (real road feature)
    """
    col = f"h3_cell_res{resolution}"
    sql = text(f"""
        SELECT
            {col}       AS h3_cell,
            COUNT(*)    AS observations,
            COUNT(DISTINCT vehicle_id) AS unique_vehicles,
            AVG(speed)  AS avg_speed
        FROM vehicle_probes
        GROUP BY {col}
        ORDER BY observations DESC;
    """)
    with engine.connect() as conn:
        result = conn.execute(sql)
        return pd.DataFrame(result.fetchall(), columns=result.keys())


def calculate_road_residuals(engine) -> pd.DataFrame:
    """For every probe observation, compute its perpendicular distance to the nearest road.

    Uses ST_Distance with geography cast for accurate metres.
    ST_ClosestPoint gives the foot of the perpendicular (useful for visualization).

    This is the key signal for change detection:
    - small residual  → vehicle is on the mapped road (or GPS noise around it)
    - large residual  → vehicle is persistently off the mapped road
    """
    sql = text("""
        SELECT
            vp.vehicle_id,
            vp.timestamp,
            vp.latitude,
            vp.longitude,
            vp.h3_cell_res11,
            vp.geom,
            MIN(
                ST_Distance(vp.geom::geography, r.geom::geography)
            ) AS dist_to_road_m
        FROM vehicle_probes vp
        CROSS JOIN LATERAL (
            SELECT geom
            FROM roads
            ORDER BY vp.geom <-> geom
            LIMIT 5
        ) r
        GROUP BY vp.vehicle_id, vp.timestamp, vp.latitude, vp.longitude,
                 vp.h3_cell_res11, vp.geom;
    """)
    with engine.connect() as conn:
        result = conn.execute(sql)
        return pd.DataFrame(result.fetchall(), columns=result.keys())


def find_road_candidates_for_point(
    engine,
    lon: float,
    lat: float,
    search_radius_m: float = 50.0,
) -> pd.DataFrame:
    """Return road segments within search_radius_m of a point.

    Used as the first step in map matching: collect candidates before scoring.
    """
    sql = text("""
        SELECT
            road_id,
            road_name,
            ST_Distance(
                geom::geography,
                ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography
            ) AS dist_m,
            ST_AsText(
                ST_ClosestPoint(
                    geom,
                    ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)
                )
            ) AS closest_point_wkt
        FROM roads
        WHERE ST_DWithin(
            geom::geography,
            ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography,
            :radius_m
        )
        ORDER BY dist_m;
    """)
    with engine.connect() as conn:
        result = conn.execute(sql, {"lon": lon, "lat": lat, "radius_m": search_radius_m})
        return pd.DataFrame(result.fetchall(), columns=result.keys())
