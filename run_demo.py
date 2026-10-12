"""
run_demo.py — end-to-end demo without a PostGIS connection.

Runs the full pipeline in-memory:
  1. Generate synthetic road network and probe data
  2. Preprocess and filter GPS noise
  3. Assign H3 cells
  4. Compute road residuals
  5. Detect candidate map changes
  6. Validate against ground truth
  7. Save summary figures

Usage:
  python run_demo.py
"""

import logging
import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for scripted runs

from src.generate_data import generate_road_network, generate_probe_data
from src.preprocessing import preprocess_probes, filter_gps_noise
from src.h3_indexing import add_h3_cells, aggregate_by_h3
from src.map_matching import map_match_nearest
from src.change_detection import calculate_road_residuals, detect_candidate_map_changes, validate_candidate_change
from src.visualization import (
    plot_raw_probes,
    plot_filtered_probes,
    plot_h3_aggregation,
    plot_map_matching,
    plot_change_detection,
    summary_bar,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    out_dir = "data/optional_sample_data"
    os.makedirs(out_dir, exist_ok=True)

    # ── 1. Generate data ────────────────────────────────────────────────────
    roads_gdf = generate_road_network()
    raw_df = generate_probe_data(
        roads_gdf,
        n_vehicles=80,
        points_per_vehicle=60,
        noise_std_m=6.0,
        candidate_road_fraction=0.20,   # 20% of vehicles follow offset road
        candidate_offset_m=30.0,
    )
    logger.info("Raw observations: %d", len(raw_df))

    # ── 2. Preprocess ───────────────────────────────────────────────────────
    pre_df = preprocess_probes(raw_df)
    cleaned_df, removed_df = filter_gps_noise(pre_df)
    logger.info(
        "After preprocessing: %d kept, %d removed", len(cleaned_df), len(removed_df)
    )

    # ── 3. H3 cells ─────────────────────────────────────────────────────────
    h3_df = add_h3_cells(cleaned_df)
    h3_agg = aggregate_by_h3(h3_df, resolution=11)
    logger.info("H3 aggregation: %d unique cells at res 11", len(h3_agg))

    # ── 4. Road residuals ───────────────────────────────────────────────────
    residual_df = calculate_road_residuals(h3_df, roads_gdf)

    # ── 5. Change detection ─────────────────────────────────────────────────
    candidate_cells, suspicious_df = detect_candidate_map_changes(
        residual_df,
        dist_threshold_m=18.0,
        min_observations=5,
        min_unique_vehicles=3,
    )

    # ── 6. Validation ───────────────────────────────────────────────────────
    metrics = validate_candidate_change(candidate_cells, residual_df)
    logger.info("Validation: %s", metrics["message"])

    # ── 7. Map matching (one sample vehicle) ────────────────────────────────
    sample_vehicle = cleaned_df["vehicle_id"].iloc[0]
    sample_df = cleaned_df[cleaned_df["vehicle_id"] == sample_vehicle]
    matched_df = map_match_nearest(sample_df, roads_gdf)

    # ── 8. Figures ──────────────────────────────────────────────────────────
    figures = {
        "01_raw_probes.png":      plot_raw_probes(h3_df, roads_gdf),
        "02_filtered.png":        plot_filtered_probes(cleaned_df, removed_df, roads_gdf),
        "03_h3_agg.png":          plot_h3_aggregation(h3_agg, roads_gdf),
        "04_map_matching.png":    plot_map_matching(sample_df, matched_df, roads_gdf, sample_vehicle),
        "05_change_detect.png":   plot_change_detection(residual_df, candidate_cells, roads_gdf),
        "06_summary.png":         summary_bar({
            "total_obs": len(cleaned_df),
            "unique_vehicles": cleaned_df["vehicle_id"].nunique(),
            "removed_noise": len(removed_df),
            "off_road_obs": len(suspicious_df),
            "candidate_cells": len(candidate_cells),
        }),
    }
    for fname, fig in figures.items():
        path = os.path.join(out_dir, fname)
        fig.savefig(path, dpi=120, bbox_inches="tight")
        logger.info("Saved: %s", path)

    print("\n── Summary ──────────────────────────────────")
    print(f"  Total observations:          {len(cleaned_df)}")
    print(f"  Unique vehicles:             {cleaned_df['vehicle_id'].nunique()}")
    print(f"  Removed noise:               {len(removed_df)}")
    print(f"  Suspicious off-road obs:     {len(suspicious_df)}")
    print(f"  Candidate change cells:      {len(candidate_cells)}")
    print(f"  Validation: {metrics['message']}")
    print(f"\n  Figures saved to: {out_dir}/")


if __name__ == "__main__":
    main()
