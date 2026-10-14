"""
Streamlit dashboard — vehicle-probe-spatial-indexing

Demonstrates the full pipeline interactively for interview/portfolio use.
Run with: streamlit run app.py
"""

import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

logging.basicConfig(level=logging.WARNING)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vehicle Probe Spatial Indexing",
    page_icon="🗺️",
    layout="wide",
)

st.title("🗺️ Vehicle Probe Spatial Indexing")
st.markdown(
    "**GPS noise vs candidate map change** — Arlington VA synthetic dataset  \n"
    "Uses H3 for aggregation and PostGIS geometry for precise spatial operations."
)

# ── Sidebar controls ───────────────────────────────────────────────────────────
st.sidebar.header("Simulation Parameters")

n_vehicles = st.sidebar.slider("Number of vehicles", 20, 200, 80, step=10)
noise_std_m = st.sidebar.slider("GPS noise std dev (metres)", 2.0, 20.0, 6.0, step=1.0)
candidate_pct = st.sidebar.slider("% vehicles on candidate new road", 5, 50, 20, step=5)
offset_m = st.sidebar.slider("Candidate road offset (metres)", 15.0, 60.0, 30.0, step=5.0)
h3_resolution = st.sidebar.selectbox("H3 resolution for aggregation", [9, 10, 11], index=2)
dist_threshold_m = st.sidebar.slider("Off-road threshold for detection (m)", 10.0, 40.0, 18.0, step=1.0)
min_unique_vehicles = st.sidebar.slider("Min unique vehicles for candidate flag", 2, 10, 3)
points_per_vehicle = st.sidebar.slider("GPS fixes per vehicle", 20, 100, 60, step=10)

run_btn = st.sidebar.button("▶ Run Simulation", type="primary")

# ── Session state: persist results across re-runs (e.g. radio button clicks) ──
if "results" not in st.session_state:
    st.session_state["results"] = None

# ── Technical concepts panel (sidebar) ────────────────────────────────────────
with st.sidebar.expander("📖 How H3 vs PostGIS/GiST differ"):
    st.markdown("""
**H3 cell ID**
- Every lat/lon → hexagonal cell ID
- Use for: grouping, aggregation, partitioning
- **Not** used for precise distance calculations

**PostGIS geometry**
- Stores actual POINT / LINESTRING geometry
- Enables: `ST_DWithin`, `ST_Distance`, `ST_ClosestPoint`

**GiST index**
- Database tree structure on geometry columns
- Turns spatial queries from O(n) → O(log n)

```
GPS point
 ├── H3 cell ID → aggregation
 └── POINT geometry → GiST → ST_DWithin
```
""")

# ── Run pipeline ───────────────────────────────────────────────────────────────
if run_btn:
    with st.spinner("Running simulation …"):
        from src.generate_data import generate_road_network, generate_probe_data
        from src.preprocessing import preprocess_probes, filter_gps_noise
        from src.h3_indexing import add_h3_cells, aggregate_by_h3
        from src.map_matching import map_match_nearest, map_match_trajectory
        from src.change_detection import (
            calculate_road_residuals,
            detect_candidate_map_changes,
            validate_candidate_change,
        )

        roads_gdf = generate_road_network()
        raw_df = generate_probe_data(
            roads_gdf,
            n_vehicles=n_vehicles,
            points_per_vehicle=points_per_vehicle,
            noise_std_m=noise_std_m,
            candidate_road_fraction=candidate_pct / 100.0,
            candidate_offset_m=offset_m,
        )

        pre_df = preprocess_probes(raw_df)
        cleaned_df, removed_df = filter_gps_noise(pre_df)
        h3_df = add_h3_cells(cleaned_df)
        h3_agg = aggregate_by_h3(h3_df, resolution=h3_resolution)
        residual_df = calculate_road_residuals(h3_df, roads_gdf)
        candidate_cells, suspicious_df = detect_candidate_map_changes(
            residual_df,
            dist_threshold_m=dist_threshold_m,
            min_observations=5,
            min_unique_vehicles=min_unique_vehicles,
        )
        val_metrics = validate_candidate_change(candidate_cells, residual_df, dist_threshold_m=dist_threshold_m)

        # Pick the Wilson Blvd vehicle with the most fixes near the intersection
        # (lon ≈ -77.0645). Use nearest_road_id from residual_df — no ground truth needed.
        INTERSECTION_LON = -77.0645
        wilson_road_id = roads_gdf[roads_gdf["road_name"].str.contains("Wilson")]["road_id"].iloc[0]
        wilson_fixes = residual_df[residual_df["nearest_road_id"] == wilson_road_id]
        fixes_near = wilson_fixes.groupby("vehicle_id").apply(
            lambda g: (abs(g["longitude"] - INTERSECTION_LON) < 0.005).sum()
        )
        sample_vid = fixes_near.idxmax()

        
        sample_df = cleaned_df[cleaned_df["vehicle_id"] == sample_vid]

        # now input one vehicle fixes
        matched_df     = map_match_nearest(sample_df, roads_gdf)
        hmm_matched_df = map_match_trajectory(sample_df, roads_gdf)

        # Store everything in session_state so re-runs (radio clicks, tab switches)
        # don't reset back to the landing page
        st.session_state["results"] = dict(
            roads_gdf=roads_gdf,
            h3_df=h3_df,
            cleaned_df=cleaned_df,
            removed_df=removed_df,
            h3_agg=h3_agg,
            residual_df=residual_df,
            candidate_cells=candidate_cells,
            suspicious_df=suspicious_df,
            val_metrics=val_metrics,
            sample_vid=sample_vid,
            sample_df=sample_df,
            matched_df=matched_df,
            hmm_matched_df=hmm_matched_df,
            offset_m=offset_m,
            h3_resolution=h3_resolution,
            dist_threshold_m=dist_threshold_m,
            min_unique_vehicles=min_unique_vehicles,
        )

    st.success("Simulation complete.")

if st.session_state["results"] is not None:
    from src.visualization import (
        plot_raw_probes,
        plot_filtered_probes,
        plot_h3_aggregation,
        plot_map_matching,
        plot_hmm_matching,
        plot_change_detection,
    )

    r = st.session_state["results"]

    # ── Summary metrics row ────────────────────────────────────────────────
    st.subheader("Summary Metrics")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total obs", len(r["cleaned_df"]))
    c2.metric("Unique vehicles", r["cleaned_df"]["vehicle_id"].nunique())
    c3.metric("Removed noise", len(r["removed_df"]))
    c4.metric("Off-road obs", len(r["suspicious_df"]))
    c5.metric("Candidate H3 cells", len(r["candidate_cells"]))
    c6.metric("Vehicles → candidate area", r["val_metrics"].get("unique_vehicles_in_detected", 0))

    # ── Tabs ───────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "1 · Raw Probes",
        "2 · Filtered",
        "3 · H3 Aggregation",
        "4 · Map Matching",
        "5 · Change Detection",
        "6 · H3 Agg Table",
    ])

    with tab1:
        st.markdown(
            "**Blue** = normal GPS observations on the mapped road  \n"
            "**Red** = vehicles following the candidate new road (~{:.0f} m offset)".format(r["offset_m"])
        )
        st.pyplot(plot_raw_probes(r["h3_df"], r["roads_gdf"]))

    with tab2:
        st.markdown(
            "**Blue** = kept after noise filter  \n"
            "**Red ×** = removed (impossible speed or isolated jump)"
        )
        st.pyplot(plot_filtered_probes(r["cleaned_df"], r["removed_df"], r["roads_gdf"]))

    with tab3:
        # Radio button re-runs the script — results stay in session_state so the
        # plot redraws with the new colour choice without losing the simulation data
        color_choice = st.radio(
            "Colour cells by:", ["unique_vehicles", "observations", "avg_speed"],
            horizontal=True,
        )
        st.markdown(
            f"H3 resolution {r['h3_resolution']} — {len(r['h3_agg'])} unique cells.  \n"
            "**Key insight:** unique vehicle count is stronger evidence than raw count."
        )
        st.pyplot(plot_h3_aggregation(r["h3_agg"], r["roads_gdf"], color_by=color_choice))

    with tab4:
        st.markdown(
            f"**Vehicle `{r['sample_vid']}`** travels east along Wilson Blvd and passes "
            f"through the N Courthouse Rd intersection (lon ≈ −77.0645).  \n"
            "Near the intersection, GPS fixes fall within the search radius of **both** roads.  \n"
            "**Left:** nearest-road matcher — may snap to the cross street (red lines = wrong road).  \n"
            "**Right:** HMM / Viterbi — trajectory continuity keeps it on Wilson Blvd (all-green = correct)."
        )
        st.caption("Nearest-road (baseline)")
        st.pyplot(plot_map_matching(
            r["sample_df"], r["matched_df"], r["roads_gdf"], r["sample_vid"]
        ))
        st.caption("HMM / Viterbi (trajectory-aware)")
        st.pyplot(plot_hmm_matching(
            r["sample_df"], r["hmm_matched_df"], r["roads_gdf"], r["sample_vid"]
        ))

    with tab5:
        st.markdown(
            f"**Orange** = suspicious observations (>{r['dist_threshold_m']:.0f} m from road)  \n"
            f"**Red ★** = H3 cells flagged as candidate map changes "
            f"(≥5 obs, ≥{r['min_unique_vehicles']} independent vehicles)  \n\n"
            + ("✅ " + r["val_metrics"]["message"] if r["val_metrics"]["detected"]
               else "⚠️ " + r["val_metrics"]["message"])
        )
        st.pyplot(plot_change_detection(r["residual_df"], r["candidate_cells"], r["roads_gdf"], r["dist_threshold_m"]))

    with tab6:
        st.markdown("Per-H3-cell aggregation table (sorted by unique vehicles)")
        if not r["h3_agg"].empty:
            display_cols = ["h3_cell", "observations", "unique_vehicles", "avg_speed"]
            st.dataframe(
                r["h3_agg"][display_cols].sort_values("unique_vehicles", ascending=False),
                use_container_width=True,
            )

else:
    # Landing state
    st.info("👈 Adjust parameters in the sidebar, then click **▶ Run Simulation**.")

    st.markdown("""
### What this project demonstrates

| Concept | What it does here |
|---|---|
| **H3 indexing** | Assigns every GPS fix to a hexagonal cell for grouping/aggregation |
| **PostGIS geometry** | Stores exact POINT geometry; used for `ST_DWithin`, `ST_Distance`, `ST_ClosestPoint` |
| **GiST spatial index** | Accelerates proximity queries from O(n) to O(log n) |
| **Nearest-road matching** | Baseline: snap each GPS point to closest road |
| **HMM map matching** | Considers whole trajectory; picks most-likely road sequence via Viterbi |
| **Change detection** | Flags H3 cells with persistent multi-vehicle off-road deviation |

### GPS noise vs real map change

```
one vehicle + one anomalous fix     → probably GPS noise
many independent vehicles
+ same spatial deviation
+ repeated over time               → candidate new road / lane
```

### Interview architecture question

```
Vehicle telemetry
    → stream ingestion (Kafka / Kinesis)
    → schema validation
    → quality filtering
    → time + H3 partitioning
    → distributed object storage (Parquet / S3)
    → trajectory reconstruction
    → map matching
    → spatial aggregation (H3)
    → change detection
    → validation → incremental map update
```
""")
