"""
PostGIS database operations: table creation, data loading, index creation.

Uses SQLAlchemy + GeoAlchemy2 for ORM-style geometry handling, and
psycopg2 as the underlying driver.

Connection is configured via environment variables (see docker-compose.yml):
  PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD
"""

import logging
import os

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)


def _connection_url() -> str:
    host = os.getenv("PGHOST", "localhost")
    port = os.getenv("PGPORT", "5432")
    db   = os.getenv("PGDATABASE", "probes")
    user = os.getenv("PGUSER", "probes")
    pwd  = os.getenv("PGPASSWORD", "probes")
    return f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}"


def get_engine():
    """Return a SQLAlchemy engine connected to PostGIS."""
    url = _connection_url()
    engine = create_engine(url, pool_pre_ping=True)
    return engine


def create_tables(engine) -> None:
    """Create vehicle_probes and roads tables if they do not exist."""
    ddl = open(
        os.path.join(os.path.dirname(__file__), "..", "sql", "create_tables.sql")
    ).read()
    with engine.begin() as conn:
        conn.execute(text(ddl))
    logger.info("Tables created (or already exist).")


def create_spatial_indexes(engine) -> None:
    """Create GiST spatial indexes on geometry columns.

    GiST (Generalized Search Tree) is a PostgreSQL index structure optimised
    for spatial data.  Without it, every ST_DWithin / ST_Distance call does a
    full sequential scan — fine for thousands of rows, catastrophic at millions.

    Index layout:
        idx_vehicle_geom  — accelerates proximity queries on probe points
        idx_roads_geom    — accelerates road-segment lookups during map matching
    """
    ddl = open(
        os.path.join(os.path.dirname(__file__), "..", "sql", "create_indexes.sql")
    ).read()
    with engine.begin() as conn:
        conn.execute(text(ddl))
    logger.info("GiST spatial indexes created (or already exist).")


def load_probes_to_postgis(df: pd.DataFrame, engine) -> None:
    """Load the probe DataFrame into the vehicle_probes table.

    Converts lat/lon to PostGIS geometry on the fly.
    """
    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(row.longitude, row.latitude) for row in df.itertuples()],
        crs="EPSG:4326",
    )
    # GeoDataFrame.to_postgis writes geometry as WKB which PostGIS understands
    gdf.to_postgis(
        "vehicle_probes",
        engine,
        if_exists="append",
        index=False,
        chunksize=1000,
    )
    logger.info("Loaded %d probe observations to PostGIS.", len(gdf))


def load_roads_to_postgis(road_gdf: gpd.GeoDataFrame, engine) -> None:
    """Load road network GeoDataFrame into the roads table."""
    road_gdf.to_postgis(
        "roads",
        engine,
        if_exists="append",
        index=False,
    )
    logger.info("Loaded %d road segments to PostGIS.", len(road_gdf))


def truncate_tables(engine) -> None:
    """Clear all rows from vehicle_probes and roads (for re-runs)."""
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE vehicle_probes, roads RESTART IDENTITY;"))
    logger.info("Tables truncated.")
