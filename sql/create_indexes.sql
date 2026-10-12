-- GiST spatial indexes
--
-- GiST (Generalized Search Tree) is the PostgreSQL index type for spatial data.
-- Without these, every ST_DWithin / ST_Distance call scans the entire table.
-- With them, the planner uses R-tree-like bounding-box pruning: O(log n) lookups.
--
-- H3 cell columns get plain B-tree indexes for fast GROUP BY and equality joins.
-- GiST is not appropriate for H3 because H3 cell IDs are text, not geometries.

-- Probe point geometry index
CREATE INDEX IF NOT EXISTS idx_vehicle_geom
    ON vehicle_probes USING GIST (geom);

-- Road geometry index (used during map matching candidate lookup)
CREATE INDEX IF NOT EXISTS idx_roads_geom
    ON roads USING GIST (geom);

-- B-tree indexes on H3 columns for GROUP BY aggregation queries
CREATE INDEX IF NOT EXISTS idx_probes_h3_res11
    ON vehicle_probes (h3_cell_res11);

CREATE INDEX IF NOT EXISTS idx_probes_h3_res10
    ON vehicle_probes (h3_cell_res10);

-- Compound index for common trajectory reconstruction pattern
CREATE INDEX IF NOT EXISTS idx_probes_vehicle_time
    ON vehicle_probes (vehicle_id, timestamp);
