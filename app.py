"""
Streamlit dashboard — vehicle-probe-spatial-indexing

Demonstrates the full pipeline interactively.
Run with: streamlit run app.py
"""

import logging
import os

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
    "Uses H3 for aggregation and exact road geometry (in metres) for distances — all in Python."
)

# ── Sidebar controls ───────────────────────────────────────────────────────────
st.sidebar.header("Simulation Parameters")

n_vehicles = st.sidebar.slider("Number of vehicles", 20, 200, 80, step=10)
candidate_pct = st.sidebar.slider("% vehicles on unmapped new road", 5, 50, 20, step=5)
offset_m = st.sidebar.slider("Unmapped road offset north of Wilson Blvd (metres)", 15.0, 60.0, 30.0, step=5.0)
points_per_vehicle = st.sidebar.slider("GPS fixes per vehicle", 20, 100, 60, step=10)
noise_std_m = st.sidebar.slider("GPS noise std dev (metres)", 2.0, 20.0, 6.0, step=1.0)
h3_resolution = st.sidebar.selectbox("H3 resolution (tab 3 maps)", [9, 10, 11], index=2)
st.sidebar.caption("Change-detection settings are in the **5 · Change Detection** tab.")

with st.sidebar.expander("ℹ️ How the data is simulated"):
    st.markdown("""
**1. The map** has two existing roads:
- **Wilson Blvd**, running about 3 km east–west.
- **N Courthouse Rd**, about 800 m north–south, crossing Wilson near the middle.

**2. A hidden unmapped road** is a copy of Wilson Blvd shifted **north** by the offset
slider (30 m by default). It runs the full length of Wilson and is **never added to the map**.

**3. The vehicles:**
- The "% vehicles on unmapped new road" slider decides how many drive the hidden road
  (20% by default). The rest drive Wilson Blvd. **No vehicle drives Courthouse Rd.**
- Each vehicle drives its whole road west → east at its own steady speed (about 40 km/h)
  and records a fixed number of GPS fixes (60 by default) evenly over the trip — about
  every 4–5 s at the defaults.
- Each fix gets random GPS error: a normal (bell-curve) offset with a standard deviation
  of 6 m by default, in both x and y.
- Each fix also gets a reported speed, a heading, and a timestamp that matches where the
  vehicle is — so position, time and speed agree.
- Each fix is tagged with its true label, `mapped_road` or `candidate_new_road`.
  That's the answer key.
""")

run_btn = st.sidebar.button("▶ Run Simulation", type="primary")

# ── Session state: persist results across re-runs (e.g. radio button clicks) ──
if "results" not in st.session_state:
    st.session_state["results"] = None

# ── Run pipeline ───────────────────────────────────────────────────────────────
if run_btn:
    with st.spinner("Running simulation …"):
        from src.generate_data import generate_road_network, generate_probe_data
        from src.preprocessing import preprocess_probes, filter_gps_noise
        from src.h3_indexing import add_h3_cells, aggregate_by_h3
        from src.map_matching import map_match_nearest, map_match_trajectory
        from src.change_detection import calculate_road_residuals

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
        # Distance to nearest mapped road for every fix. The detection step itself
        # runs live in the Change Detection tab so its settings apply instantly.
        residual_df = calculate_road_residuals(h3_df, roads_gdf)

        # ── Pick the sample vehicle for the Map Matching tab ──────────────────
        # Goal: a vehicle on Wilson Blvd where the two methods actually DISAGREE.
        # Uses only distances and matcher outputs — no ground truth.
        wilson_road_id = roads_gdf[roads_gdf["road_name"].str.contains("Wilson")]["road_id"].iloc[0]

        # 1. On-road vehicles: their fixes are typically within a few metres of a road
        #    (vehicles on the unmapped road sit ~offset metres away).
        median_dist = residual_df.groupby("vehicle_id")["dist_to_road_m"].median()
        on_road_vids = median_dist[median_dist < 10.0].index

        # 2. Candidates: on-road vehicles with fixes whose closest road is NOT Wilson
        #    (i.e. nearest-road would snap them to the cross street). Most such fixes first.
        on_road = residual_df[residual_df["vehicle_id"].isin(on_road_vids)]
        cross_counts = (
            on_road[on_road["nearest_road_id"] != wilson_road_id]
            .groupby("vehicle_id").size().sort_values(ascending=False)
        )
        candidates = list(cross_counts.index[:10])
        if not candidates:  # no close calls this run — fall back to any on-road vehicle
            candidates = list(on_road_vids[:1]) or [residual_df["vehicle_id"].iloc[0]]

        # 3. Run both matchers on the candidates; keep the one with most disagreements
        #    (nearest-road says cross street, HMM says Wilson).
        best = None
        for vid in candidates:
            vdf = cleaned_df[cleaned_df["vehicle_id"] == vid]
            m_near = map_match_nearest(vdf, roads_gdf)
            m_hmm = map_match_trajectory(vdf, roads_gdf)
            both = m_near[["timestamp", "matched_road_id"]].merge(
                m_hmm[["timestamp", "hmm_matched_road_id"]], on="timestamp"
            )
            n_disagree = int((
                (both["matched_road_id"] != wilson_road_id)
                & (both["hmm_matched_road_id"] == wilson_road_id)
            ).sum())
            if best is None or n_disagree > best[0]:
                best = (n_disagree, vid, vdf, m_near, m_hmm)

        n_disagree, sample_vid, sample_df, matched_df, hmm_matched_df = best

        # Store everything in session_state so re-runs (radio clicks, tab switches)
        # don't reset back to the landing page
        st.session_state["results"] = dict(
            roads_gdf=roads_gdf,
            raw_df=raw_df,
            h3_df=h3_df,
            cleaned_df=cleaned_df,
            removed_df=removed_df,
            h3_agg=h3_agg,
            residual_df=residual_df,
            sample_vid=sample_vid,
            n_disagree=n_disagree,
            sample_df=sample_df,
            matched_df=matched_df,
            hmm_matched_df=hmm_matched_df,
            offset_m=offset_m,
            h3_resolution=h3_resolution,
            noise_std_m=noise_std_m,
        )

    st.success("Simulation complete.")

if st.session_state["results"] is not None:
    from src.visualization import (
        PLOTLY_CONFIG,
        plot_raw_probes,
        plot_filtered_probes,
        plot_h3_aggregation,
        plot_matching_comparison,
        plot_change_detection,
        plot_threshold_sweep,
    )
    from src.change_detection import detect_candidate_map_changes, validate_candidate_change

    r = st.session_state["results"]

    def show(fig):
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    st.caption("🔍 Scroll to zoom, drag to pan, double-click to reset view.")

    # ── Summary metrics row ────────────────────────────────────────────────
    st.subheader("Summary Metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total obs", len(r["cleaned_df"]))
    c2.metric("Unique vehicles", r["cleaned_df"]["vehicle_id"].nunique())
    c3.metric("Removed noise", len(r["removed_df"]))

    # ── Tabs ───────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "1 · Raw Probes",
        "2 · Filtered",
        "3 · H3 Aggregation",
        "4 · Map Matching",
        "5 · Change Detection",
    ])

    with tab1:
        st.markdown(
            "All GPS fixes as received, **before** noise filtering. Colours show which path each "
            "vehicle actually drove — this is the simulation's ground truth, which real data "
            "does not have.  \n"
            "**Blue** = fixes from vehicles driving the mapped road (already in the existing map)  \n"
            "**Red** = fixes from vehicles driving an unmapped new road "
            "(~{:.0f} m offset, not in the map)".format(r["offset_m"])
        )
        show(plot_raw_probes(r["raw_df"], r["roads_gdf"]))

        with st.expander("ℹ️ Terminology: mapped vs unmapped vs candidate"):
            st.markdown("""
"Mapped" is always relative to the **current published map version**.

| Stage | Term | In this demo |
|---|---|---|
| Road is in the map used today | **Mapped road** (base map) | Wilson Blvd, N Courthouse Rd |
| Road exists in reality but not in the map | **Unmapped road** | The offset path the red vehicles drive |
| Change detection flags evidence for it | **Candidate change** — still *not* mapped | Red ★ cells in tab 5 |
| Validated and added in a map release | **Mapped road** in the next map version | Not covered — no map-update step |
""")

    with tab2:
        st.markdown(
            "**Blue** = kept after noise filter  \n"
            "**Red ×** = removed by speed spike · **Purple ◆** = removed by isolated jump · "
            "**Black ★** = removed by both"
        )
        st.markdown("""
**Noise vs glitch** — *noise* is the small scatter (a few metres) every GPS fix has around the
true position. A *glitch* is a fix whose position is **far off** where the car really was — tens
to hundreds of metres or more — for one or a few readings (signals bouncing off buildings,
tunnels, receiver start-up). A fix is treated as a glitch and removed if **either** of two
filters catches it: **speed spike** or **isolated jump** (below). Everyday noise is kept.

**Filters used** — each fix is checked against the vehicle's **own trajectory**, not the road map:
- **Basic cleaning** — drop missing or impossible coordinates (e.g. 0,0) and exact duplicates.
  These have no usable position, so they are not shown on the map.
- **Speed spike** — removed if the implied speed is > 200 km/h **both** from the previous fix
  and to the next fix (a single jump out and back).
  *Speed in* = straight-line distance (metres) from the previous fix to this fix ÷ time between
  them; *speed out* = the same from this fix to the next fix.
  If only **one** of the two is high, we can't say this fix is the glitch — the neighbour may be,
  or there may be a gap in the data — so the fix is kept.
- **Isolated jump** — removed if it lies > 25 m off the straight line between its previous and
  next fix **and** its neighbours are not also off (one stray point, not a real path change).

⚠️ Both filters only catch **single** glitched fixes. A run of several off-path fixes in a row is
kept on purpose: it may be a real path the map doesn't have (e.g. an unmapped road), which is
for change detection to judge — not something to delete as noise.
""")
        show(plot_filtered_probes(r["cleaned_df"], r["removed_df"], r["roads_gdf"]))

    with tab3:
        st.markdown(
            f"H3 resolution {r['h3_resolution']} — {len(r['h3_agg'])} unique cells.  \n"
            "**Top:** unique vehicles per cell. **Bottom:** average speed per cell. "
            "The two maps zoom and pan together.  \n"
            "**Key insight:** the top map uses unique vehicles instead of raw fix count, "
            "because one vehicle can send many fixes from the same spot. "
            "Raw counts are in the **H3 aggregation table** below."
        )
        show(plot_h3_aggregation(
            r["h3_agg"], r["roads_gdf"], resolution=r["h3_resolution"]
        ))

        st.markdown("""
#### What this tab tells us
- **Busy cells away from the mapped road are a first hint of an unmapped road.** A band of
  cells running parallel to the road (~{:.0f} m off) with many **unique vehicles** and normal
  driving **avg speed** means steady traffic where the map has no road — not random GPS noise,
  which would scatter thinly across cells. Change detection (tab 5) tests this formally.
- **H3 says *where to look*, geometry decides *what it is*.** Cells are a cheap way to
  aggregate and spot hotspots, but a cell can contain both the road and nearby off-road fixes.
  Exact distances to roads (map matching, change detection) still need the real geometry.
""".format(r["offset_m"]))

        with st.expander("📋 H3 aggregation table (sorted by unique vehicles)"):
            if not r["h3_agg"].empty:
                display_cols = ["h3_cell", "observations", "unique_vehicles", "avg_speed"]
                st.dataframe(
                    r["h3_agg"][display_cols].sort_values("unique_vehicles", ascending=False),
                    use_container_width=True,
                )
            else:
                st.write("No H3 cells.")

    with tab4:
        st.info(
            "**What is map matching?** Map matching does **not** question the road map — it "
            "takes the existing roads as given and only decides **which existing road each GPS fix "
            "belongs to**, and where on that road the vehicle really was (orange snapped points).  \n"
            "• **Nearest-road:** each fix goes to the closest road, one fix at a time. Because it "
            "looks at each fix on its own, it can assign a fix to the wrong road — mostly near "
            "intersections, parallel roads and ramps.  \n"
            "• **HMM / Viterbi:** uses the whole trajectory to pick the most likely *sequence* "
            "of roads, avoiding implausible jumps between roads. It is the standard approach and "
            "usually much more accurate, because a fix only makes sense as part of a believable "
            "road sequence — but it is not guaranteed correct.  \n"
            "⚠️ This demo shows **one vehicle** and does not measure accuracy; see *In production* "
            "below.  \n"
            "Discovering *new* roads is a separate step — see **5 · Change Detection**."
        )
        n_dis = r.get("n_disagree", 0)
        if n_dis > 0:
            pick_note = (
                f"This vehicle was **chosen because the two methods disagree** on it "
                f"({n_dis} fix{'es' if n_dis != 1 else ''}: nearest-road says Courthouse, HMM says "
                "Wilson). For most vehicles they agree.  \n"
            )
        else:
            pick_note = (
                "⚠️ In this run no vehicle showed a disagreement between the two methods, so both "
                "panels look the same. Change a simulation setting (e.g. more vehicles or more GPS "
                "noise) and run again.  \n"
            )
        st.markdown(
            f"**Vehicle `{r['sample_vid']}`** travels east along Wilson Blvd and passes "
            f"through the N Courthouse Rd intersection (lon ≈ −77.0645).  \n"
            + pick_note +
            "Near the intersection, GPS fixes fall within the search radius of **both** roads. "
            "The chart opens zoomed on the fix where the methods disagree (red circle).  \n"
            "**Left:** nearest-road matcher — may snap to the cross street (red lines = wrong road).  \n"
            "**Right:** HMM / Viterbi — trajectory continuity keeps it on Wilson Blvd (green = correct)."
        )
        show(plot_matching_comparison(
            r["sample_df"], r["matched_df"], r["hmm_matched_df"], r["roads_gdf"], r["sample_vid"]
        ))

        st.markdown("""
#### In production
For illustration, this demo matches one vehicle — it takes the GPS fixes of **a single vehicle**
and assigns each fix to one of the existing roads. A production system would:

- **Match every trajectory** — all vehicles and trips, in batch (e.g. Spark, partitioned by
  vehicle/trip and time) or in streaming with a sliding-window Viterbi.
- **Use the road graph for transitions** — shortest-path distance along the network instead of
  straight-line distance, plus heading, speed, one-way and turn restrictions.
- **Measure accuracy** — compare against labelled ground-truth drives and report % of fixes and
  % of road segments matched correctly, especially near intersections.
- **Output a confidence per match** — low-confidence or unmatched stretches are not discarded;
  they feed **change detection** as evidence the map may be wrong.
""")

        with st.expander("ℹ️ Why use the road graph for HMM transitions"):
            st.markdown("""
The HMM **transition score** judges whether moving from road A at one fix to road B at the
next fix is believable.

**What this demo does** — compares the **straight-line** distance between the two raw GPS fixes
with the **straight-line** distance between the two snapped points; similar = believable.
This works here because there are only two roads and they cross, so any straight-line move
between them is also drivable.

**Why straight-line fails in real networks** — two roads 20 m apart that **aren't connected**
(highway vs. service road, two sides of a divided road): the straight-line jump is tiny, so it
looks fine, but a car would really have to drive to an interchange and back — maybe 2 km.

**What production does instead**
1. **Distance along the roads (shortest path)** — actual driving distance between the snapped
   points through the network; much longer than the GPS distance → unlikely. Fixes most
   parallel-road errors.
2. **Heading** — direction of travel should match the road's direction (also separates the two
   sides of a divided road).
3. **Speed** — driving distance ÷ time between fixes gives an implied speed; 300 km/h → reject.
4. **One-way streets** — routes going the wrong way are not allowed.
5. **Turn restrictions** — routes using a banned turn are not allowed.

**In short:** the demo asks *"is the jump between roads short?"*; production asks
*"could a car actually have driven that way in this much time?"*
""")

        st.markdown("**More production steps**")

        with st.expander("ℹ️ Find candidate roads with a spatial index"):
            st.markdown("""
For each fix, the matcher first needs the roads in a specified proximity, for example within
60 m. This demo checks **every road**,
which is fine for 2 roads but far too slow for millions. A spatial index (tree of nested
boxes) skips most of the map in a few steps, so only nearby roads get an exact distance check.

- **Database:** PostGIS **GiST index** + `ST_DWithin(road.geom, fix.geom, 60)`.
- **Python:** an **R-tree**, e.g. `roads_gdf.sindex` or Shapely's `STRtree`.
""")

        with st.expander("ℹ️ Split trips before matching"):
            st.markdown("""
**In production:** split trips first — break trajectories at long time gaps or ignition
off/on before matching.

For each vehicle, sort its fixes by time and start a **new trip** wherever the trajectory
clearly breaks; each trip is then map-matched on its own. Otherwise the HMM would try to connect
the last fix of the morning drive to the first fix of the evening drive, forcing nonsense
road sequences.

**Typical trip-break signals**
- **Time gap** — no fixes for longer than a threshold (often ~5–10 min).
- **Distance jump** — two consecutive fixes too far apart for the time between them.
- **Long stop** — speed ≈ 0 at the same place for many minutes (parked).
- **Ignition off/on** — when the data includes it.

**How ignition is known**
- **Usually reported, not inferred** — connected cars / telematics devices read it from the
  vehicle bus (CAN / OBD-II) and send it as a field or as trip-start / trip-end events.
- **If missing, inferred** — most devices only send GPS while the engine is on, so a long gap
  usually means ignition off; a long stationary period also counts as a stop.

**In this project:** no trip splitting and no ignition field — each synthetic vehicle is one
continuous drive with fixes a few seconds apart, so it isn't needed here.
""")

    with tab5:
        st.info(
            "**What is change detection?** Map matching asks *which existing road is this fix on?* "
            "Change detection asks **is there a road here that the map is missing?** — without "
            "using the simulation's answer key.  \n"
            "1. Measure each fix's distance to the nearest mapped road. If a fix is 30 m from "
            "Wilson Blvd but 3 m from N Courthouse Rd, its distance is 3 m.  \n"
            "2. Fixes farther than the threshold are **suspicious** (unexplained — not yet noise "
            "or new road).  \n"
            "3. Group suspicious fixes by H3 cell.  \n"
            "4. A cell becomes a **candidate change** (red ★) only with enough suspicious fixes "
            "from enough **different vehicles** — one vehicle off-road is likely noise; many "
            "independent vehicles in the same place is evidence.  \n"
            "5. Only *after* detection, candidates are checked against the simulation's truth "
            "(✅ / ⚠️ message below).  \n"
            "A candidate is still **not** a mapped road — validation and the map update are not "
            "part of this demo."
        )
        # ── Detection settings (live: changing them re-runs only this step) ──
        st.markdown("#### Detection settings")
        st.caption(
            "📝 Detection uses **no simulation truth** — only distance to the map and vehicle "
            "counts, exactly what production has. The settings are chosen by weighing **false "
            "alarms** (reviewer time) against **missed changes** (stale map). The ✅ / ⚠️ message "
            "and the threshold sweep below compare each setting with the simulation truth as "
            "**extra information for this demo only**. In production that trade-off is measured "
            "on a smaller real-world truth instead: GPS noise on known roads, reviewer-checked "
            "candidates, or surveyed areas."
        )
        s1, s2, s3, s4 = st.columns(4)
        dist_threshold_m = s1.slider("Off-road threshold (m)", 10.0, 40.0, 18.0, step=1.0,
                                     key="cd_threshold")
        min_observations = s2.slider("Min suspicious fixes per cell", 2, 20, 5, key="cd_min_obs")
        min_unique_vehicles = s3.slider("Min unique vehicles per cell", 2, 10, 3, key="cd_min_veh")
        cd_resolution = s4.selectbox("H3 resolution", [9, 10, 11], index=2, key="cd_res",
                                     help="Cell size for grouping: 11 ≈ 25 m, 10 ≈ 65 m, 9 ≈ 175 m.")

        candidate_cells, suspicious_df = detect_candidate_map_changes(
            r["residual_df"],
            dist_threshold_m=dist_threshold_m,
            min_observations=min_observations,
            min_unique_vehicles=min_unique_vehicles,
            resolution=cd_resolution,
        )
        val_metrics = validate_candidate_change(
            candidate_cells, r["residual_df"],
            dist_threshold_m=dist_threshold_m, resolution=cd_resolution,
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Suspicious (off-road) fixes", len(suspicious_df))
        m2.metric("Candidate H3 cells", len(candidate_cells))
        m3.metric("Vehicles → candidate area", val_metrics.get("unique_vehicles_in_detected", 0))

        st.markdown(
            f"**Orange** = suspicious observations (>{dist_threshold_m:.0f} m from road)  \n"
            f"**Red ★** = H3 cells (res {cd_resolution}) flagged as candidate map changes "
            f"(≥{min_observations} suspicious fixes, ≥{min_unique_vehicles} independent vehicles)  \n\n"
            + ("✅ " + val_metrics["message"] if val_metrics["detected"]
               else "⚠️ " + val_metrics["message"])
        )
        show(plot_change_detection(r["residual_df"], candidate_cells, r["roads_gdf"], dist_threshold_m))

        # ── Threshold sweep: re-run detection for every threshold, other settings fixed ──
        st.markdown("#### Which threshold is best?")
        sweep_rows = []
        for t in range(10, 41):
            cc, _ = detect_candidate_map_changes(
                r["residual_df"], dist_threshold_m=float(t),
                min_observations=min_observations, min_unique_vehicles=min_unique_vehicles,
                resolution=cd_resolution,
            )
            vm = validate_candidate_change(cc, r["residual_df"], dist_threshold_m=float(t),
                                           resolution=cd_resolution)
            sweep_rows.append({
                "threshold_m": t,
                "true_positive_cells": vm.get("true_positive_cells", 0),
                "false_positive_cells": vm.get("false_positive_cells", 0),
            })
        sweep_df = pd.DataFrame(sweep_rows)

        sigma = r["noise_std_m"]
        noise_rule_t = 3 * sigma

        show(plot_threshold_sweep(sweep_df, dist_threshold_m, noise_rule_t))
        st.markdown(f"""
Each point re-runs detection at that threshold, with the other settings above unchanged.
- **Lower threshold** → more true-change cells caught, but more noise passes and false
  positives rise. **Higher threshold** → fewer false positives, but the unmapped road is missed.
- **3σ noise rule ({noise_rule_t:.1f} m)** — uses only the GPS noise (σ = {sigma:.0f} m).
  Of fixes from vehicles truly on the mapped road, only about **3 in 1,000** land farther
  than this by chance (≈ 99.7% of a bell curve lies within 3σ) and get wrongly flagged as
  suspicious. These wrongly flagged fixes are scattered (one car here, another there), so they
  rarely give any cell enough different vehicles to become a candidate change.
  This is what you can do in reality.
- The multi-vehicle gate filters scattered noise, so the whole pipeline can often afford a
  somewhat **lower** threshold. In practice the point is chosen from this
  kind of curve on labelled data, weighing a false alarm (reviewer time) against a missed
  change (stale map).
""")

        st.markdown("""
#### In production
This demo flags candidates from distance to the closest road (no matter whether the car is
actually driving on that road) and vehicle counts only. A production system would:

- **Check that a fix fits a road, not just that a road is near** — use heading (an east–west
  car is not explained by a north–south road) or the distance to the road the HMM matched,
  instead of the distance to whichever road is closest.
- **Set the threshold from measured noise** — e.g. the 99th percentile of distances for fixes
  on known roads, per device type or area, rather than a fixed number.
- **Require persistence over time** — the same deviation across several days or weeks, so a
  one-off detour, closure or event doesn't become a candidate.
- **Validate before updating the map** — review candidates with imagery, street-level photos
  or a human editor; only then is the road added in the next map version.
""")

        with st.expander("ℹ️ Blind spot: closest road at the intersection"):
            st.markdown("""
If a fix is 30 m from Wilson Blvd but 3 m from N Courthouse Rd, its distance is **3 m**.

The unmapped road runs north of Wilson and **crosses Courthouse Rd**. Near that crossing,
its fixes are close to Courthouse, so they look on-road and are not flagged. Zoom into the
intersection: the orange band breaks and turns blue, and there is a **gap in the red stars**
— even though those cars are driving east–west on the unmapped road, not on Courthouse.

See the first bullet above for how production avoids this.
""")


else:
    # Landing state
    st.info("👈 Adjust parameters in the sidebar, then click **▶ Run Simulation**.")

    st.markdown("""
### What this project demonstrates

| Concept | What it does here |
|---|---|
| **GPS glitch filtering** | Removes single fixes that jump impossibly far from the vehicle's own path |
| **H3 indexing** | Assigns every GPS fix to a hexagonal cell for grouping/aggregation |
| **Nearest-road matching** | Baseline: snap each GPS point to closest road |
| **HMM map matching** | Considers whole trajectory; picks most-likely road sequence via Viterbi |
| **Change detection** | Flags H3 cells where many independent vehicles are off the mapped road |

### GPS noise vs real map change

```
one vehicle + one anomalous fix     → probably GPS noise
many independent vehicles
+ same spatial deviation
+ repeated over time               → candidate new road / lane
```

### Production architecture

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

In production, road and probe geometry would typically live in a spatial database such as
**PostGIS**, with a **GiST spatial index** so proximity queries (`ST_DWithin`, `ST_Distance`)
stay fast at scale. This demo does the same calculations in Python on a small dataset.
""")
