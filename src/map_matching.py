"""
Map matching: snap noisy GPS observations to the road network.

Two approaches are implemented:

1. map_match_nearest()  — baseline: nearest road segment per point
   Fast, simple, but fails near parallel roads / ramps / intersections.

2. map_match_trajectory() — simplified HMM / Viterbi-style
   Considers the whole trajectory sequence.
   Emission: how close is this GPS point to the candidate road?
   Transition: is it physically plausible to move from candidate A to candidate B?
   Selects the most-likely road sequence, not greedy best-at-each-step.

Key insight:
  Map matching is NOT "find the nearest road per point."
  It is "find the most likely road SEQUENCE given the complete noisy trajectory."
"""

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points

logger = logging.getLogger(__name__)

_TO_UTM = Transformer.from_crs("EPSG:4326", "EPSG:32618", always_xy=True)


@dataclass
class RoadCandidate:
    road_id: int
    road_name: str
    geometry: LineString
    dist_m: float
    closest_point: Point


def _dist_m(p1: Point, p2: Point) -> float:
    """Euclidean distance between two UTM-projected points (metres)."""
    x1, y1 = _TO_UTM.transform(p1.x, p1.y)
    x2, y2 = _TO_UTM.transform(p2.x, p2.y)
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def _point_to_road_dist(pt: Point, road: LineString) -> tuple[float, Point]:
    """Return (distance_m, closest_point) from pt to road using UTM."""
    # Project to UTM for metric accuracy
    pt_utm = Point(_TO_UTM.transform(pt.x, pt.y))
    road_utm = LineString([_TO_UTM.transform(x, y) for x, y in road.coords])
    closest_utm = nearest_points(road_utm, pt_utm)[0]
    dist = pt_utm.distance(closest_utm)
    # Back-project closest point to WGS84
    lon, lat = Transformer.from_crs("EPSG:32618", "EPSG:4326", always_xy=True).transform(
        closest_utm.x, closest_utm.y
    )
    return dist, Point(lon, lat)


def get_candidates(
    gps_point: Point,
    roads_gdf,
    search_radius_m: float = 60.0,
) -> list[RoadCandidate]:
    """Find road segments within search_radius_m of a GPS point.

    Returns a list of RoadCandidate objects sorted by distance.
    An empty list means no road within the search radius — unmatched point.
    """
    candidates = []
    for row in roads_gdf.itertuples():
        dist, closest = _point_to_road_dist(gps_point, row.geometry)
        if dist <= search_radius_m:
            candidates.append(
                RoadCandidate(
                    road_id=row.road_id,
                    road_name=row.road_name,
                    geometry=row.geometry,
                    dist_m=dist,
                    closest_point=closest,
                )
            )
    candidates.sort(key=lambda c: c.dist_m)
    return candidates


# ── Baseline: nearest-road per point ─────────────────────────────────────────

def map_match_nearest(
    vehicle_df: pd.DataFrame,
    roads_gdf,
    search_radius_m: float = 60.0,
) -> pd.DataFrame:
    """Nearest-road map matching (baseline).

    For each GPS observation, snap to the closest road segment.

    Fails when:
    - Two parallel roads are within the search radius (wrong one may win)
    - GPS noise near a ramp makes a service road appear closer
    - Intersection areas where multiple segments are equidistant

    Returns vehicle_df with added columns:
        matched_road_id, matched_road_name, match_dist_m,
        matched_lon, matched_lat
    """
    results = []
    for row in vehicle_df.itertuples():
        pt = Point(row.longitude, row.latitude)
        candidates = get_candidates(pt, roads_gdf, search_radius_m)
        if candidates:
            best = candidates[0]
            results.append({
                "matched_road_id": best.road_id,
                "matched_road_name": best.road_name,
                "match_dist_m": round(best.dist_m, 2),
                "matched_lon": best.closest_point.x,
                "matched_lat": best.closest_point.y,
            })
        else:
            results.append({
                "matched_road_id": None,
                "matched_road_name": None,
                "match_dist_m": None,
                "matched_lon": None,
                "matched_lat": None,
            })
    match_df = pd.DataFrame(results, index=vehicle_df.index)
    return pd.concat([vehicle_df, match_df], axis=1)


# ── HMM / Viterbi-style trajectory-aware matching ────────────────────────────

def _emission_prob(dist_m: float, sigma_m: float = 15.0) -> float:
    """Gaussian emission: probability that a GPS point with dist_m error
    came from this road candidate.

    Higher sigma = more tolerant of noisy GPS.
    """
    return math.exp(-(dist_m ** 2) / (2 * sigma_m ** 2))


def _transition_score(
    prev_candidate: RoadCandidate,
    curr_candidate: RoadCandidate,
    gps_dist_m: float,
) -> float:
    """Transition score: how plausible is moving from prev_candidate to curr_candidate?

    Uses the "great circle / route distance" heuristic from Newson & Krumm (2009):
    the actual road distance should be close to the GPS straight-line distance.

    Simplified here: penalise candidate pairs where the snap-point jump is
    disproportionately large compared to GPS movement.
    """
    snap_dist = _dist_m(prev_candidate.closest_point, curr_candidate.closest_point)
    diff = abs(snap_dist - gps_dist_m)
    # Exponential penalty for mismatch
    beta = 30.0  # metres
    return math.exp(-diff / beta)


def map_match_trajectory(
    vehicle_df: pd.DataFrame,
    roads_gdf,
    search_radius_m: float = 60.0,
    sigma_m: float = 15.0,
) -> pd.DataFrame:
    """Simplified HMM / Viterbi trajectory-aware map matching.

    For the whole trajectory of one vehicle:
    1. At each GPS point, collect candidate road segments within search_radius_m.
    2. Score each candidate with an emission probability (Gaussian distance model).
    3. Use Viterbi dynamic programming: at each step carry the best accumulated
       score for each candidate, factoring in the transition from the previous step.
    4. Traceback to recover the most-likely road sequence.

    This is conceptually equivalent to a Hidden Markov Model where:
    - Hidden states   = which road segment the vehicle is actually on
    - Observations    = the noisy GPS point
    - Emission prob   = how well the GPS point fits the road candidate
    - Transition prob = how plausible the candidate-to-candidate move is

    Returns vehicle_df with matched columns appended.
    """
    # Process one vehicle at a time to maintain trajectory order
    all_results: list[dict] = []

    for vehicle_id, grp in vehicle_df.groupby("vehicle_id"):
        grp = grp.sort_values("timestamp").reset_index(drop=True)
        n = len(grp)

        # Collect candidates for every GPS point
        candidates_per_step: list[list[RoadCandidate]] = []
        for row in grp.itertuples():
            pt = Point(row.longitude, row.latitude)
            cands = get_candidates(pt, roads_gdf, search_radius_m)
            if not cands:
                # No road nearby — add a dummy "unmatched" placeholder
                cands = [RoadCandidate(-1, "unmatched", LineString(), 999.0, pt)]
            candidates_per_step.append(cands)

        # Viterbi: viterbi[t][i] = (best_log_score, prev_candidate_index)
        NEG_INF = float("-inf")
        viterbi: list[list[tuple[float, int]]] = []

        # Initialise at t=0
        init_scores = []
        for cand in candidates_per_step[0]:
            log_em = math.log(max(_emission_prob(cand.dist_m, sigma_m), 1e-12))
            init_scores.append((log_em, -1))
        viterbi.append(init_scores)

        # Forward pass
        for t in range(1, n):
            step_scores = []
            row_t = grp.iloc[t]
            row_prev = grp.iloc[t - 1]
            gps_pt_t = Point(row_t.longitude, row_t.latitude)
            gps_pt_prev = Point(row_prev.longitude, row_prev.latitude)
            gps_dist = _dist_m(gps_pt_prev, gps_pt_t)

            for curr_cand in candidates_per_step[t]:
                log_em = math.log(max(_emission_prob(curr_cand.dist_m, sigma_m), 1e-12))
                best_score = NEG_INF
                best_prev_idx = 0
                for prev_idx, prev_cand in enumerate(candidates_per_step[t - 1]):
                    prev_score = viterbi[t - 1][prev_idx][0]
                    if prev_score == NEG_INF:
                        continue
                    tr = _transition_score(prev_cand, curr_cand, gps_dist)
                    log_tr = math.log(max(tr, 1e-12))
                    score = prev_score + log_tr + log_em
                    if score > best_score:
                        best_score = score
                        best_prev_idx = prev_idx
                step_scores.append((best_score, best_prev_idx))
            viterbi.append(step_scores)

        # Traceback
        matched_indices: list[int] = [0] * n
        best_last = max(range(len(viterbi[-1])), key=lambda i: viterbi[-1][i][0])
        matched_indices[n - 1] = best_last
        for t in range(n - 2, -1, -1):
            matched_indices[t] = viterbi[t + 1][matched_indices[t + 1]][1]

        # Build result rows
        for t, row in enumerate(grp.itertuples()):
            cand = candidates_per_step[t][matched_indices[t]]
            all_results.append({
                "vehicle_id": row.vehicle_id,
                "timestamp": row.timestamp,
                "latitude": row.latitude,
                "longitude": row.longitude,
                "hmm_matched_road_id": cand.road_id,
                "hmm_matched_road_name": cand.road_name,
                "hmm_match_dist_m": round(cand.dist_m, 2),
                "hmm_matched_lon": cand.closest_point.x,
                "hmm_matched_lat": cand.closest_point.y,
            })

    return pd.DataFrame(all_results)
