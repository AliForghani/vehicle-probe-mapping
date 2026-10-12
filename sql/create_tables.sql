-- Enable PostGIS extension (idempotent)
CREATE EXTENSION IF NOT EXISTS postgis;

-- Vehicle probe observations
CREATE TABLE IF NOT EXISTS vehicle_probes (
    id              BIGSERIAL PRIMARY KEY,
    vehicle_id      TEXT        NOT NULL,
    timestamp       TIMESTAMPTZ NOT NULL,
    latitude        DOUBLE PRECISION NOT NULL,
    longitude       DOUBLE PRECISION NOT NULL,
    speed           REAL,
    heading         REAL,
    h3_cell_res11   TEXT,
    h3_cell_res10   TEXT,
    h3_cell_res9    TEXT,
    -- WGS84 geometry; exact spatial ops use this column, not H3
    geom            GEOMETRY(POINT, 4326)
);

-- Existing mapped road network
CREATE TABLE IF NOT EXISTS roads (
    road_id     BIGSERIAL PRIMARY KEY,
    road_name   TEXT,
    geom        GEOMETRY(LINESTRING, 4326)
);
