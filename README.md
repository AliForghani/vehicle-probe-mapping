# vehicle-probe-spatial-indexing

A GitHub-ready Python + PostGIS prototype demonstrating how vehicle GPS/probe data
is processed, spatially indexed, map-matched, and analyzed to detect candidate road
network changes — all in a small, locally runnable dataset designed to be explainable
in a technical interview.

---

## The Problem

Automotive mapping companies (HERE, TomTom, Apple Maps, Google Maps) continuously
collect billions of GPS/probe observations from vehicles. From this data they:

- Keep the road map up to date (new roads, geometry changes, lane additions)
- Detect speed limit changes and congestion patterns
- Infer turn restrictions and road classifications

**The core challenge:** a single GPS point far from the mapped road could be noise
or could indicate a real map change. Telling these apart requires spatial indexing,
statistical aggregation over many independent vehicles, and map matching.

---

## Architecture

```
GPS observation
      │
      ├── lat/lon → H3 cell ID
      │             ↓
      │         aggregation
      │         partitioning
      │         neighbourhood analysis
      │
      └── POINT geometry → PostGIS
                            ↓
                        GiST index
                            ↓
                    ST_DWithin
                    ST_Distance
                    ST_ClosestPoint
```

### H3 indexing vs PostGIS/GiST indexing — key distinction

| | H3 | PostGIS + GiST |
|---|---|---|
| **What it stores** | Hexagonal cell ID (text) | Actual geometry (POINT, LINESTRING) |
| **Good for** | Grouping, aggregation, partitioning, density maps | Exact distance, proximity queries, map matching |
| **Not for** | Precise distance calculation | Coarse spatial partitioning |
| **Index type** | B-tree on text column | GiST on geometry column |
| **Example** | "Group all probes by H3 cell, count per cell" | "Find all probes within 100 m of this point" |

**H3 cell assignment ≠ GiST index.**

```
H3:      GPS point → H3 cell ID → grouping / aggregation / partitioning
PostGIS: GPS point → geometry  → GiST index → efficient exact spatial queries
```

---

## Synthetic Data Scenario — Arlington VA

### Existing mapped road
A simplified segment of Wilson Blvd in Arlington VA (~2.4 km east-west).

### Normal vehicle trajectories
~80% of vehicles travel approximately along the mapped road with realistic
GPS noise (σ ≈ 6 m by default).

### Candidate new road
~20% of vehicles repeatedly travel along a **parallel path ~30 m north** of the
mapped road. This path does **not exist** in the current road network, simulating
a new lane, newly constructed road, or changed road geometry.

> **Why 30 m?**  
> A 30 m offset makes the distinction between GPS noise and a real map change
> easy to visualize and teach. True lane-level map inference often involves
> separations of 3–5 m, requiring centimetre-precision positioning (RTK-GPS,
> lidar odometry) and far more sophisticated algorithms.

---

## Project Structure

```
vehicle-probe-mapping/
├── app.py                 Streamlit dashboard
├── run_demo.py            End-to-end demo (no database required)
├── docker-compose.yml     PostGIS container
├── requirements.txt
├── src/
│   ├── generate_data.py   Synthetic road + probe generation
│   ├── preprocessing.py   GPS cleaning and derived fields
│   ├── h3_indexing.py     H3 cell assignment and aggregation
│   ├── database.py        PostGIS load/query helpers
│   ├── spatial_queries.py ST_DWithin, residuals, aggregation
│   ├── map_matching.py    Nearest-road + HMM/Viterbi matcher
│   ├── change_detection.py Distance residuals + candidate detection
│   └── visualization.py   Matplotlib figures for Streamlit
├── sql/
│   ├── create_tables.sql
│   ├── create_indexes.sql
│   └── example_queries.sql
├── tests/
│   ├── test_preprocessing.py
│   ├── test_h3.py
│   ├── test_map_matching.py
│   └── test_change_detection.py
└── data/optional_sample_data/   Output figures from run_demo.py
```

---

## How to Run

### Quick demo (no database)

```bash
pip install -r requirements.txt
python run_demo.py
# Figures saved to data/optional_sample_data/
```

### Streamlit dashboard

```bash
streamlit run app.py
```

### With PostGIS (full pipeline)

```bash
docker compose up -d
python -c "
from src.database import get_engine, create_tables, create_spatial_indexes, \
     load_probes_to_postgis, load_roads_to_postgis
from src.generate_data import generate_road_network, generate_probe_data
from src.preprocessing import preprocess_probes, filter_gps_noise
from src.h3_indexing import add_h3_cells

roads = generate_road_network()
raw   = generate_probe_data(roads)
pre   = preprocess_probes(raw)
clean, _ = filter_gps_noise(pre)
h3    = add_h3_cells(clean)

engine = get_engine()
create_tables(engine)
create_spatial_indexes(engine)
load_roads_to_postgis(roads, engine)
load_probes_to_postgis(h3, engine)
print('Loaded.')
"
```

### Tests

```bash
pytest tests/ -v
```

---

## Core Functions

| Function | Module | Purpose |
|---|---|---|
| `generate_road_network()` | generate_data | Mapped road as GeoDataFrame |
| `generate_probe_data()` | generate_data | Synthetic vehicle GPS traces |
| `preprocess_probes()` | preprocessing | Clean, sort, derive dt/dist/speed |
| `filter_gps_noise()` | preprocessing | Remove impossible-speed / isolated jumps |
| `add_h3_cells()` | h3_indexing | Assign H3 res 9/10/11 cell IDs |
| `aggregate_by_h3()` | h3_indexing | Count obs/vehicles/speed per cell |
| `query_nearby_probes()` | spatial_queries | ST_DWithin proximity query |
| `calculate_road_residuals()` | change_detection | Distance to nearest road per obs |
| `detect_candidate_map_changes()` | change_detection | Multi-vehicle off-road aggregation |
| `map_match_nearest()` | map_matching | Baseline nearest-road snap |
| `map_match_trajectory()` | map_matching | HMM / Viterbi trajectory matching |

---

## H3 Resolution Notes

| Resolution | Approx cell edge | Useful for |
|---|---|---|
| 9 | ~174 m | City-block scale density |
| 10 | ~66 m | Intersection-level aggregation |
| 11 | ~25 m | Approximate lane-level grouping |

Resolution 11 is an **illustrative** choice for this prototype.
It is **not** a universal automotive standard.

---

## Map Matching

### Baseline: nearest road
For each GPS point, find the geometrically closest road segment.
Fast but fails near parallel roads, on-ramps, and intersections.

### HMM / Viterbi trajectory-aware matching

```
Hidden Markov Model

Hidden states   = which road segment the vehicle is actually on
Observations    = noisy GPS fixes

Emission prob   = P(GPS point | road segment)
                  ∝ Gaussian(distance to road)

Transition prob = P(road_t | road_{t-1})
                  ∝ exp(-|route_distance - GPS_distance| / beta)

Viterbi selects the most-likely road SEQUENCE, not greedy best-per-point.
```

---

## GPS Noise vs Real Map Change

```
one vehicle + one unusual fix        → likely GPS noise

many independent vehicles
+ same spatial deviation (> threshold)
+ repeated over multiple time windows
+ consistent direction of travel     → candidate map change
```

---

## Interview Question 1 — Automotive-Scale Architecture

> "Suppose we receive **billions** of GPS observations. How do you design the pipeline?"

```
Vehicle telemetry
      ↓
Stream ingestion         Kafka / Kinesis / Pub/Sub
      ↓
Schema validation        Protobuf / Avro schema registry
      ↓
Quality filtering        Bad coordinates, impossible speed, duplicate detection
      ↓
Time + geo partitioning  Partition by date + H3 res 5–7 region
      ↓
Distributed storage      Parquet on S3 / GCS / ADLS
      ↓
Trajectory reconstruction  Group by vehicle_id, sort by time
      ↓
Map matching             Valhalla / OSRM / custom HMM at scale (Spark)
      ↓
Spatial aggregation      H3 hexagon counts, speed, heading
      ↓
Change detection         Multi-vehicle, multi-time-window aggregation
      ↓
Validation               Compare against known ground truth / surveyed data
      ↓
Incremental map update   Candidate → review → merge
```

Key principles at scale:
- Partition by H3 cell to avoid global spatial joins
- Idempotent pipeline: re-processing a partition produces identical output
- Separate streaming (Kafka) and batch (Spark) paths
- Data quality monitoring at ingest

This prototype processes thousands of records locally.
At production scale the same logical pipeline runs on Spark/Databricks with
H3-partitioned Parquet files.

---

## Interview Question 2 — Noise vs New Road

> "How do you tell GPS noise from an actual new road or lane?"

| Evidence | Interpretation |
|---|---|
| 1 vehicle, 1 fix 30 m off road | GPS multipath — noise |
| 1 vehicle, every fix 30 m off road | Driver off-road or map error |
| 20 vehicles, every fix ~30 m off road, same direction | Candidate new road |

Signals considered: distance from road, unique vehicle count (independent),
heading consistency, persistence across time windows, spatial clustering.

---

## Interview Question 3 — Map Matching

> "How do you map-match noisy vehicle GPS?"

1. Find candidate road segments within search radius (ST_DWithin)
2. Score each candidate: Gaussian emission model on perpendicular distance
3. Viterbi over full trajectory: select most-likely road sequence, not greedy per-point

---

## Limitations

- Dataset is **synthetic**.
- Candidate road is intentionally **25–40 m** offset for clarity. Real lane-level
  changes can be 3–5 m and require RTK-GPS or lidar.
- H3 resolution 11 is **illustrative** — not a universal standard.
- H3 is **not** a replacement for road geometry.
- H3 and GiST solve **different problems** and both are needed.
- HMM map matching here is **simplified** (no heading model, no road class prior).
- Real automotive localisation uses IMU, wheel odometry, and lidar.
- Production fleet-scale processing requires distributed infrastructure.
- Real map updates require human review and stronger statistical validation.
