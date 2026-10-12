-- ============================================================
-- Example PostGIS queries demonstrating the core concepts
-- ============================================================

-- 1. Find all observations within 100 m of a location
--    Uses geometry column + GiST index — NOT H3 cell
SELECT
    vehicle_id,
    timestamp,
    latitude,
    longitude,
    speed,
    ST_Distance(
        geom::geography,
        ST_SetSRID(ST_MakePoint(-77.05, 38.88), 4326)::geography
    ) AS distance_m
FROM vehicle_probes
WHERE ST_DWithin(
    geom::geography,
    ST_SetSRID(ST_MakePoint(-77.05, 38.88), 4326)::geography,
    100
)
ORDER BY distance_m;

-- 2. Aggregate observations per H3 cell (res 11)
--    H3 is ideal for this grouping / density analysis
SELECT
    h3_cell_res11,
    COUNT(*)                    AS observations,
    COUNT(DISTINCT vehicle_id)  AS unique_vehicles,
    AVG(speed)                  AS avg_speed_kmh
FROM vehicle_probes
GROUP BY h3_cell_res11
ORDER BY unique_vehicles DESC;

-- 3. Distance from each probe to the nearest road
--    LATERAL join: for each probe, find closest road, compute distance
SELECT
    vp.vehicle_id,
    vp.timestamp,
    vp.latitude,
    vp.longitude,
    nearest.road_name,
    ST_Distance(vp.geom::geography, nearest.geom::geography) AS dist_to_road_m,
    ST_AsText(ST_ClosestPoint(nearest.geom, vp.geom))         AS snap_point
FROM vehicle_probes vp
CROSS JOIN LATERAL (
    SELECT road_name, geom
    FROM roads
    ORDER BY vp.geom <-> geom   -- KNN operator, uses GiST
    LIMIT 1
) nearest;

-- 4. Candidate map change detection:
--    H3 cells with >15m average distance from road, observed by 3+ vehicles
SELECT
    vp.h3_cell_res11,
    COUNT(*)                    AS observations,
    COUNT(DISTINCT vp.vehicle_id) AS unique_vehicles,
    AVG(
        ST_Distance(vp.geom::geography, roads.geom::geography)
    ) AS avg_dist_m
FROM vehicle_probes vp
CROSS JOIN LATERAL (
    SELECT geom FROM roads ORDER BY vp.geom <-> geom LIMIT 1
) roads
GROUP BY vp.h3_cell_res11
HAVING
    AVG(ST_Distance(vp.geom::geography, roads.geom::geography)) > 15
    AND COUNT(DISTINCT vp.vehicle_id) >= 3
ORDER BY unique_vehicles DESC;

-- 5. Probe trajectory for a single vehicle (for map matching input)
SELECT vehicle_id, timestamp, latitude, longitude, speed, heading, geom
FROM vehicle_probes
WHERE vehicle_id = 'V0000'
ORDER BY timestamp;
