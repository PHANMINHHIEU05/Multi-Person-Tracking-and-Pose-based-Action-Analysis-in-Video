"""
utils/trajectory_merge.py
=========================
Offline (post-run) trajectory-based tracklet stitching.

After the online tracker has written tracks.csv, this module:
  1. Loads the CSV and groups rows by stable_track_id into *segments*
     (a segment is a contiguous run of frames for the same id; a gap of
     more than `internal_gap` frames within the same stable_id splits it
     into two segments).
  2. For every pair (segment A end, segment B start) where:
       - A.id != B.id
       - gap = B.start_frame - A.end_frame ∈ [1, max_gap_frames]
       - distance between A's last position and B's first position ≤ max_join_dist
       - direction change ≤ max_dir_change_deg
  3. Scores each candidate pair with a join cost; picks the best match
     greedily (cheapest edge first).
  4. Re-writes stable_track_id in the output CSV for merged segments;
     adds is_merged_offline column.
  5. Returns the number of merges performed.

This step runs entirely on the CSV (no GPU / frame data needed).
"""

from __future__ import annotations

import csv
import math
import os
from typing import Dict, List, Optional, Tuple


# --------------------------------------------------------------------------- #
#  Data structures
# --------------------------------------------------------------------------- #
class Segment:
    """One contiguous run of frames for a single stable_track_id."""

    def __init__(self, uid: int):
        self.uid        = uid        # stable_track_id value
        self.rows: List[dict] = []   # raw CSV rows (mutable)

    # -- Computed properties ------------------------------------------------ #
    @property
    def start_frame(self) -> int:
        return int(self.rows[0]["frame_idx"])

    @property
    def end_frame(self) -> int:
        return int(self.rows[-1]["frame_idx"])

    @property
    def length(self) -> int:
        return len(self.rows)

    @property
    def last_cx(self) -> float:
        return float(self.rows[-1]["cx"])

    @property
    def last_cy(self) -> float:
        return float(self.rows[-1]["cy"])

    @property
    def first_cx(self) -> float:
        return float(self.rows[0]["cx"])

    @property
    def first_cy(self) -> float:
        return float(self.rows[0]["cy"])

    @property
    def last_velocity(self) -> Tuple[float, float]:
        """Estimate velocity from last two points of segment."""
        if len(self.rows) < 2:
            return 0.0, 0.0
        r2, r1 = self.rows[-1], self.rows[-2]
        df = max(1, int(r2["frame_idx"]) - int(r1["frame_idx"]))
        vx = (float(r2["cx"]) - float(r1["cx"])) / df
        vy = (float(r2["cy"]) - float(r1["cy"])) / df
        return vx, vy

    @property
    def first_velocity(self) -> Tuple[float, float]:
        """Estimate velocity from first two points of segment."""
        if len(self.rows) < 2:
            return 0.0, 0.0
        r1, r2 = self.rows[0], self.rows[1]
        df = max(1, int(r2["frame_idx"]) - int(r1["frame_idx"]))
        vx = (float(r2["cx"]) - float(r1["cx"])) / df
        vy = (float(r2["cy"]) - float(r1["cy"])) / df
        return vx, vy


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def _angle_between(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
    """Angle in degrees between two 2-D vectors."""
    m1 = math.hypot(*v1)
    m2 = math.hypot(*v2)
    if m1 < 1e-6 or m2 < 1e-6:
        return 0.0
    cos_a = (v1[0] * v2[0] + v1[1] * v2[1]) / (m1 * m2)
    cos_a = max(-1.0, min(1.0, cos_a))
    return math.degrees(math.acos(cos_a))


def _join_cost(
    seg_a: Segment,
    seg_b: Segment,
    max_join_dist: float,
    max_dir_deg:   float,
) -> Optional[float]:
    """
    Compute join cost for (end of A, start of B).

    Returns None if the pair is outside gating thresholds.

    Cost ∈ [0, 1]: lower = better candidate.
    """
    gap = seg_b.start_frame - seg_a.end_frame
    if gap < 1:
        return None

    # Spatial: use velocity-predicted position
    vxa, vya = seg_a.last_velocity
    pred_cx = seg_a.last_cx + vxa * gap
    pred_cy = seg_a.last_cy + vya * gap

    dist_pred = math.hypot(seg_b.first_cx - pred_cx,
                           seg_b.first_cy - pred_cy)
    dist_last = math.hypot(seg_b.first_cx - seg_a.last_cx,
                           seg_b.first_cy - seg_a.last_cy)
    dist = min(dist_pred, dist_last)
    if dist > max_join_dist:
        return None

    # Direction continuity
    vxb, vyb = seg_b.first_velocity
    dir_change = _angle_between((vxa, vya), (vxb, vyb))
    if dir_change > max_dir_deg:
        return None

    # Normalised cost
    w_dist = dist / max(max_join_dist, 1.0)
    w_dir  = dir_change / max(max_dir_deg, 1.0)
    speed_a = math.hypot(vxa, vya)
    speed_b = math.hypot(vxb, vyb)
    speed_diff = abs(speed_a - speed_b) / (max(speed_a, speed_b, 1e-6))

    cost = 0.5 * w_dist + 0.3 * w_dir + 0.2 * speed_diff
    return float(cost)


# --------------------------------------------------------------------------- #
#  Segment extraction
# --------------------------------------------------------------------------- #
def _extract_segments(
    rows: List[dict],
    internal_gap: int = 3,
    min_length:   int = 8,
) -> List[Segment]:
    """
    Group rows by stable_track_id; split into contiguous segments.

    Segments shorter than `min_length` frames are dropped.
    """
    by_id: Dict[int, List[dict]] = {}
    for r in rows:
        tid = int(r.get("stable_track_id", r.get("track_id", 0)))
        by_id.setdefault(tid, []).append(r)

    segments: List[Segment] = []
    for uid, id_rows in by_id.items():
        id_rows.sort(key=lambda r: int(r["frame_idx"]))
        seg = Segment(uid)
        seg.rows.append(id_rows[0])
        for r in id_rows[1:]:
            gap = int(r["frame_idx"]) - int(seg.rows[-1]["frame_idx"])
            if gap <= internal_gap:
                seg.rows.append(r)
            else:
                if seg.length >= min_length:
                    segments.append(seg)
                seg = Segment(uid)
                seg.rows.append(r)
        if seg.length >= min_length:
            segments.append(seg)

    return segments


# --------------------------------------------------------------------------- #
#  Greedy matching
# --------------------------------------------------------------------------- #
def _greedy_merge(
    segments:      List[Segment],
    max_gap:       int,
    max_join_dist: float,
    max_dir_deg:   float,
) -> List[Tuple[Segment, Segment, float]]:
    """
    Return a list of (seg_a, seg_b, cost) pairs chosen greedily.

    Algorithm: enumerate all valid (A-end, B-start) pairs, sort by cost,
    greedily pick non-conflicting pairs (each segment can only be used once
    as a 'tail' or 'head').
    """
    candidates: List[Tuple[float, Segment, Segment]] = []

    # Index segments by end_frame for efficient lookup
    by_end: Dict[int, List[Segment]] = {}
    for s in segments:
        by_end.setdefault(s.end_frame, []).append(s)

    for seg_b in segments:
        for delta in range(1, max_gap + 1):
            ef = seg_b.start_frame - delta
            for seg_a in by_end.get(ef, []):
                if seg_a.uid == seg_b.uid:
                    continue
                cost = _join_cost(seg_a, seg_b, max_join_dist, max_dir_deg)
                if cost is not None:
                    candidates.append((cost, seg_a, seg_b))

    candidates.sort(key=lambda x: x[0])

    used_as_tail: set = set()
    used_as_head: set = set()
    chosen: List[Tuple[Segment, Segment, float]] = []

    for cost, seg_a, seg_b in candidates:
        aid = id(seg_a)
        bid = id(seg_b)
        if aid in used_as_tail or bid in used_as_head:
            continue
        chosen.append((seg_a, seg_b, cost))
        used_as_tail.add(aid)
        used_as_head.add(bid)

    return chosen


# --------------------------------------------------------------------------- #
#  Public entry point
# --------------------------------------------------------------------------- #
def run_offline_merge(
    tracks_csv:       str,
    output_csv:       str,
    max_gap_frames:   int   = 45,
    max_join_dist_px: float = 160.0,
    max_dir_change:   float = 60.0,
    min_seg_length:   int   = 8,
) -> int:
    """
    Read *tracks_csv*, merge tracklets, write *output_csv*.

    Returns the number of merge operations performed.

    Parameters
    ----------
    tracks_csv       : path to online-phase tracks CSV
    output_csv       : path for merged output CSV
    max_gap_frames   : max gap between segment end and start to consider merge
    max_join_dist_px : max distance between predicted and actual next position
    max_dir_change   : max heading angle change (degrees) at join point
    min_seg_length   : discard segments shorter than this
    """
    if not os.path.exists(tracks_csv):
        return 0

    # Load all rows
    with open(tracks_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if not rows:
        return 0

    # Extract segments
    segments = _extract_segments(rows, min_length=min_seg_length)

    # Greedy matching
    merges = _greedy_merge(segments, max_gap_frames, max_join_dist_px, max_dir_change)

    # Build stable_id re-map: old_uid -> new_uid (always keep lower ID)
    uid_remap: Dict[int, int] = {}

    def resolve_uid(u: int) -> int:
        visited: set = set()
        while u in uid_remap and u not in visited:
            visited.add(u)
            u = uid_remap[u]
        return u

    for seg_a, seg_b, _ in merges:
        ra = resolve_uid(seg_a.uid)
        rb = resolve_uid(seg_b.uid)
        if ra == rb:
            continue
        # Keep the smaller (earlier) ID as root
        root  = min(ra, rb)
        other = max(ra, rb)
        uid_remap[other] = root

    if not merges:
        # Still write output with is_merged_offline=0
        pass

    # Build set of orig IDs that were merged (→ lost their identity)
    merged_away: set = set()
    for seg_a, seg_b, _ in merges:
        ra = seg_a.uid
        rb = seg_b.uid
        root  = min(resolve_uid(ra), resolve_uid(rb))
        other = max(resolve_uid(ra), resolve_uid(rb))
        if root != other:
            merged_away.add(other)

    # Write output CSV
    out_fields = list(fieldnames)
    if "is_merged_offline" not in out_fields:
        out_fields.append("is_merged_offline")
    if "stable_track_id" not in out_fields and "track_id" in out_fields:
        out_fields = [
            "stable_track_id" if f == "track_id" else f for f in out_fields
        ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            orig_id = int(row.get("stable_track_id", row.get("track_id", 0)))
            new_id  = resolve_uid(orig_id)
            out_row = dict(row)
            if "stable_track_id" in out_row:
                out_row["stable_track_id"] = new_id
            elif "track_id" in out_row:
                out_row["stable_track_id"] = new_id
            out_row["is_merged_offline"] = int(new_id != orig_id)
            writer.writerow(out_row)

    return len(merges)
