"""
utils/stitching.py
==================
Online re-association layer for stable multi-person tracking.

Architecture
------------
  TrackState      – per-raw-ID state: tentative/confirmed, velocity EMA,
                    appearance feature, age counters.
  MemoryBank      – pool of recently-ended TrackState objects (TTL-bounded).
  OnlineReassociator – given a new detection, searches the MemoryBank with
                    position gating + appearance scoring, returns best match.
  IDRemapper      – maintains raw_id → stable_id mapping with chain resolution
                    and cycle prevention.

Scoring formula (per-candidate in bank search)
----------------------------------------------
  σ           = 120 px  (spatial decay constant – generous)
  pos_score   = exp(−min(dist_to_pred, dist_to_last) / σ)   ∈ (0, 1]
  app_score   = hist_similarity(feat_new, feat_bank)          ∈ [0, 1]
  final_score = α * pos_score + (1−α) * app_score            ∈ [0, 1]

Gating (all must pass):
  1. gap <= max_gap_frames
  2. dist <= max_center_dist  OR  edge-edge scenario  OR  iou >= 0.03
  3. app_score >= min_app_sim   (very lenient ~0.30)

Edge-proximity logic:
  If the bank record was last seen near an edge AND the new detection is
  also near an edge, the spatial gate is essentially ignored – we allow
  the entire frame distance.  Appearance score becomes the primary
  discriminator.

Accept if final_score >= final_score_thresh.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from .appearance import hist_similarity

# --------------------------------------------------------------------------- #
#  EMA alpha defaults
# --------------------------------------------------------------------------- #
_VEL_ALPHA  = 0.4   # velocity EMA  (higher = more reactive)
_CONF_ALPHA = 0.2   # conf EMA      (lower   = smoother)
_SIGMA_PX   = 120.0 # spatial decay constant (generous – wider = less positional penalty)

# Edge-proximity: if a track disappears/appears within this many pixels of
# the frame border, it's likely exiting/entering.  In that case we ignore
# the spatial gate and rely on appearance.
_EDGE_MARGIN_PX = 80


# --------------------------------------------------------------------------- #
#  Track State
# --------------------------------------------------------------------------- #
@dataclass
class TrackState:
    """
    State maintained for one active raw_track_id.

    Lifecycle
    ---------
    TENTATIVE → becomes CONFIRMED when the track has been observed for
    at least `confirm_min_frames` frames, its bboxes are tall enough, and
    its average confidence is high enough.
    """

    raw_id:           int
    stable_id:        int           # = raw_id initially; updated by IDRemapper
    first_seen_frame: int
    last_seen_frame:  int

    # Geometry
    x1: int = 0
    y1: int = 0
    x2: int = 0
    y2: int = 0
    cx: float = 0.0
    cy: float = 0.0
    w:  int = 0
    h:  int = 0

    # Kinematics
    vx: float = 0.0
    vy: float = 0.0

    # Quality
    conf:        float = 0.0
    age_frames:  int   = 0          # frames since first seen
    seen_frames: int   = 0          # frames actually detected

    # Appearance
    feat: Optional[np.ndarray] = field(default=None, repr=False)
    # Number of feature observations blended so far
    feat_count: int = 0

    # Status
    status: str = "tentative"       # "tentative" | "confirmed"

    # Frame dimensions (set once for edge detection)
    frame_w: int = 1280
    frame_h: int = 720

    # ── Update ─────────────────────────────────────────────────────────────── #
    def update(
        self,
        frame_idx: int,
        x1: int, y1: int, x2: int, y2: int,
        conf: float,
        feat: Optional[np.ndarray],
    ):
        """Update state with a new detection on *frame_idx*."""
        new_cx = (x1 + x2) / 2.0
        new_cy = (y1 + y2) / 2.0

        # EMA velocity (only when seen on consecutive or near frames)
        gap = max(1, frame_idx - self.last_seen_frame)
        raw_vx = (new_cx - self.cx) / gap
        raw_vy = (new_cy - self.cy) / gap
        self.vx = _VEL_ALPHA * raw_vx + (1 - _VEL_ALPHA) * self.vx
        self.vy = _VEL_ALPHA * raw_vy + (1 - _VEL_ALPHA) * self.vy

        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.cx, self.cy = new_cx, new_cy
        self.w = x2 - x1
        self.h = y2 - y1
        self.conf       = _CONF_ALPHA * conf + (1 - _CONF_ALPHA) * self.conf
        self.last_seen_frame = frame_idx
        self.seen_frames += 1
        self.age_frames  = frame_idx - self.first_seen_frame + 1

        # Update appearance feature (blend toward new observation)
        if feat is not None:
            if self.feat is None:
                self.feat = feat.copy()
            else:
                # Adaptive alpha: strong weight early, decays as we accumulate
                alpha_feat = max(0.15, 0.5 / (1 + self.feat_count * 0.3))
                self.feat = alpha_feat * feat + (1 - alpha_feat) * self.feat
            self.feat_count += 1

    def promote_if_ready(
        self,
        confirm_min_frames:   int,
        confirm_min_bbox_h:   int,
        confirm_min_avg_conf: float,
    ):
        if self.status == "confirmed":
            return
        if (
            self.seen_frames >= confirm_min_frames
            and self.h >= confirm_min_bbox_h
            and self.conf >= confirm_min_avg_conf
        ):
            self.status = "confirmed"

    def is_near_edge(self) -> bool:
        """Return True if the person bbox is close to any frame border."""
        return (
            self.x1 <= _EDGE_MARGIN_PX
            or self.y1 <= _EDGE_MARGIN_PX
            or self.x2 >= self.frame_w - _EDGE_MARGIN_PX
            or self.y2 >= self.frame_h - _EDGE_MARGIN_PX
        )

    def predicted_center(self, gap: int = 1) -> Tuple[float, float]:
        """Extrapolate centre by *gap* frames using current velocity."""
        return self.cx + self.vx * gap, self.cy + self.vy * gap

    def as_bank_record(self) -> dict:
        """Serialise to a MemoryBank record dict."""
        near_edge = self.is_near_edge()
        return {
            "stable_id":        self.stable_id,
            "raw_id":           self.raw_id,
            "status_at_end":    self.status,
            "last_seen_frame":  self.last_seen_frame,
            "first_seen_frame": self.first_seen_frame,
            "seen_frames":      self.seen_frames,
            "cx": self.cx,
            "cy": self.cy,
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "vx": self.vx,
            "vy": self.vy,
            "feat":       self.feat.copy() if self.feat is not None else None,
            "feat_count": self.feat_count,
            "near_edge":  near_edge,
            # Confirmed tracks get higher priority; tentative with many
            # feature observations also get decent priority
            "priority": (
                1.0 if self.status == "confirmed"
                else min(0.8, 0.2 + self.feat_count * 0.1)
            ),
        }


# --------------------------------------------------------------------------- #
#  Geometry helpers
# --------------------------------------------------------------------------- #
def _iou(ax1: int, ay1: int, ax2: int, ay2: int,
         bx1: int, by1: int, bx2: int, by2: int) -> float:
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw  = max(0, ix2 - ix1)
    ih  = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)


def _is_near_edge_bbox(
    x1: int, y1: int, x2: int, y2: int,
    frame_w: int, frame_h: int,
) -> bool:
    """Check if a bbox is near any frame border."""
    return (
        x1 <= _EDGE_MARGIN_PX
        or y1 <= _EDGE_MARGIN_PX
        or x2 >= frame_w - _EDGE_MARGIN_PX
        or y2 >= frame_h - _EDGE_MARGIN_PX
    )


# --------------------------------------------------------------------------- #
#  Memory Bank
# --------------------------------------------------------------------------- #
class MemoryBank:
    """
    Stores recently-ended TrackState records, keyed by stable_id.

    TTL is expressed in frames (converted from seconds at construction time).
    When a stable_id re-enters the bank (same id seen again after a gap),
    the record is updated in-place so the latest observation is preserved.

    Quality gate: tracks with zero features OR very short tentative tracks
    are rejected to avoid polluting the bank with spurious detections.
    """

    def __init__(self, ttl_frames: int = 450):
        self.ttl_frames = ttl_frames
        # stable_id -> record dict (see TrackState.as_bank_record)
        self._bank: Dict[int, dict] = {}

    # -- Public API --------------------------------------------------------- #

    def add(self, record: dict):
        """Add or update a record for record["stable_id"].

        Rejects tracks with no feature information (feat_count == 0)
        or that were tentative and seen for < 3 frames – these are
        likely spurious face-only detections.
        """
        sid = record["stable_id"]

        # Quality gate: require at least 1 feature observation
        # AND at least 3 seen frames for tentative tracks
        if record.get("feat") is None or record.get("feat_count", 0) < 1:
            return
        if record.get("status_at_end") == "tentative" and record.get("seen_frames", 0) < 3:
            return

        if sid in self._bank:
            # Keep the record with the most recent last_seen_frame
            if record["last_seen_frame"] >= self._bank[sid]["last_seen_frame"]:
                self._bank[sid] = record
        else:
            self._bank[sid] = record

    def remove(self, stable_id: int):
        self._bank.pop(stable_id, None)

    def expire(self, current_frame: int):
        """Evict records whose TTL has elapsed."""
        stale = [
            sid for sid, rec in self._bank.items()
            if current_frame - rec["last_seen_frame"] > self.ttl_frames
        ]
        for sid in stale:
            del self._bank[sid]

    def records(self):
        """Iterate over all bank records."""
        return self._bank.values()

    def __len__(self):
        return len(self._bank)


# --------------------------------------------------------------------------- #
#  Online Reassociator
# --------------------------------------------------------------------------- #
class OnlineReassociator:
    """
    Given a new detection (possibly a brand-new raw_id or a returning one),
    search the MemoryBank for the best matching ended track.

    Returns (best_stable_id, score) or (None, 0.0).

    Edge-proximity logic
    --------------------
    If the bank record was last seen near an edge AND the new detection is
    also near an edge, the spatial gate is essentially ignored – we allow
    the entire frame distance.  Appearance score becomes the primary
    discriminator.
    """

    def __init__(
        self,
        max_gap_frames:      int   = 200,
        max_center_dist:     int   = 400,
        min_iou_merge:       float = 0.03,
        alpha_pos:           float = 0.45,
        min_app_sim:         float = 0.30,
        final_score_thresh:  float = 0.35,
        frame_w:             int   = 1280,
        frame_h:             int   = 720,
    ):
        self.max_gap     = max_gap_frames
        self.max_dist    = max_center_dist
        self.min_iou     = min_iou_merge
        self.alpha       = alpha_pos
        self.min_app_sim = min_app_sim
        self.thresh      = final_score_thresh
        self.frame_w     = frame_w
        self.frame_h     = frame_h

    def search(
        self,
        frame_idx: int,
        cx: float,
        cy: float,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        feat: Optional[np.ndarray],
        bank: MemoryBank,
        exclude_stable_ids: Optional[set] = None,
    ) -> Tuple[Optional[int], float]:
        """
        Find the best matching MemoryBank record for a detection at
        (cx, cy, bbox).

        Parameters
        ----------
        exclude_stable_ids : stable IDs already matched in this frame
                             (to prevent double-assignment)

        Returns
        -------
        (stable_id, score)  or  (None, 0.0)
        """
        best_sid   = None
        best_score = 0.0

        det_near_edge = _is_near_edge_bbox(
            x1, y1, x2, y2, self.frame_w, self.frame_h
        )

        for rec in bank.records():
            sid = rec["stable_id"]
            if exclude_stable_ids and sid in exclude_stable_ids:
                continue

            # ── Gate 1: temporal ────────────────────────────────────────────
            gap = frame_idx - rec["last_seen_frame"]
            if gap < 1 or gap > self.max_gap:
                continue

            # ── Gate 2: spatial ─────────────────────────────────────────────
            # Use velocity-predicted position if available
            pred_cx = rec["cx"] + rec["vx"] * gap
            pred_cy = rec["cy"] + rec["vy"] * gap

            dist_pred = math.hypot(cx - pred_cx, cy - pred_cy)
            dist_last = math.hypot(cx - rec["cx"], cy - rec["cy"])
            dist      = min(dist_pred, dist_last)

            iou = _iou(x1, y1, x2, y2,
                        rec["x1"], rec["y1"], rec["x2"], rec["y2"])

            # Edge-edge scenario: person left near an edge, reappears near
            # edge → skip spatial gate (rely on appearance)
            edge_both = det_near_edge and rec.get("near_edge", False)

            spatial_ok = (
                edge_both
                or dist <= self.max_dist
                or iou >= self.min_iou
            )
            if not spatial_ok:
                continue

            # ── Appearance score ─────────────────────────────────────────────
            app_score = hist_similarity(feat, rec["feat"])
            if app_score < self.min_app_sim:
                continue

            # ── Position score (exponential decay) ──────────────────────────
            # For edge-edge, clamp dist so pos_score doesn't destroy final
            eff_dist = min(dist, 200.0) if edge_both else dist
            pos_score = math.exp(-eff_dist / _SIGMA_PX)

            # ── Final score ──────────────────────────────────────────────────
            if edge_both:
                # In edge-edge scenario, appearance is king
                score = 0.15 * pos_score + 0.85 * app_score
            else:
                score = self.alpha * pos_score + (1.0 - self.alpha) * app_score

            # Weight by priority (confirmed tracks score higher)
            score *= rec.get("priority", 1.0)

            # Slight boost for tracks with rich feature history
            feat_count = rec.get("feat_count", 0)
            if feat_count >= 5:
                score *= 1.05

            if score > best_score and score >= self.thresh:
                best_score = score
                best_sid   = sid

        return best_sid, best_score


# --------------------------------------------------------------------------- #
#  ID Remapper
# --------------------------------------------------------------------------- #
class IDRemapper:
    """
    Maintains the raw_id → stable_id mapping.

    Rules
    -----
    * stable_id is the earliest confirmed (or tentative) ID in the chain.
    * Path compression: resolve() always returns the root.
    * Cycle prevention: we refuse to add a mapping that would create a cycle.
    * Monotonically-assigned new stable IDs: first observation of raw_id
      assigns stable_id = raw_id (identity mapping) until stitching merges it.
    """

    def __init__(self):
        # raw_id -> stable_id (single hop; resolve() follows chains)
        self._remap: Dict[int, int] = {}
        self.stitch_count: int = 0

    def resolve(self, raw_id: int) -> int:
        """Follow chain to root, applying path compression."""
        visited: set = set()
        cur = raw_id
        path = []
        while cur in self._remap and cur not in visited:
            visited.add(cur)
            path.append(cur)
            cur = self._remap[cur]
        # Path compression
        for node in path:
            self._remap[node] = cur
        return cur

    def merge(self, new_raw_id: int, target_stable_id: int):
        """
        Map *new_raw_id* → *target_stable_id* (cycle-safe).

        Does nothing if adding this mapping would create a cycle.
        """
        root_target = self.resolve(target_stable_id)
        root_new    = self.resolve(new_raw_id)
        if root_new == root_target:
            return  # already same root
        # Cycle check: would mapping root_new → root_target create cycle?
        cur = root_target
        visited: set = set()
        while cur in self._remap and cur not in visited:
            visited.add(cur)
            cur = self._remap[cur]
            if cur == root_new:
                return  # cycle detected, skip
        self._remap[new_raw_id] = root_target
        self.stitch_count += 1

    def is_known(self, raw_id: int) -> bool:
        """Return True if raw_id has ever been merged into another ID."""
        return raw_id in self._remap
