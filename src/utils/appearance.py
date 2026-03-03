"""
utils/appearance.py
===================
Upper-body appearance feature extraction and similarity computation.

Upper-body ROI definition
--------------------------
Given a full-person bounding box (x1, y1, x2, y2) we crop the upper 45 % of
the height.  This is robust when:
  - Only the head/face is initially visible → we still get a small but
    consistent feature region.
  - Full body is visible → we get torso + head which ignores legs (more stable
    across walking/running).

Clamping is applied to keep the crop within frame bounds.

Feature
-------
2-D HSV histogram on (H, S) channels.  We ignore V (brightness) because it
changes significantly with lighting variation.  The histogram is L1-normalised
so it represents a colour probability distribution.

Similarity
----------
We use the inverse of the Bhattacharyya distance:
  similarity = 1 - bhattacharyya_distance(h1, h2) ∈ [0, 1]
This is better-calibrated than CORREL for non-Gaussian histograms.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
#  ROI extraction
# --------------------------------------------------------------------------- #
def extract_upper_body_roi(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    upper_frac: float = 0.45,
) -> Optional[np.ndarray]:
    """
    Return the upper *upper_frac* portion of the person bbox as a BGR crop.

    Returns None if the crop would be empty (e.g. tiny bbox or out-of-bounds).

    Parameters
    ----------
    frame       : full BGR frame
    x1,y1,x2,y2: person bounding box (pixels, may be outside frame)
    upper_frac  : what fraction of height to keep, from y1 downward
    """
    fh, fw = frame.shape[:2]

    # Clamp bbox to frame
    cx1 = max(0, int(x1))
    cy1 = max(0, int(y1))
    cx2 = min(fw - 1, int(x2))
    cy2 = min(fh - 1, int(y2))

    bh = cy2 - cy1
    bw = cx2 - cx1
    if bh < 4 or bw < 4:
        return None

    roi_y2 = cy1 + max(4, int(bh * upper_frac))
    roi_y2 = min(roi_y2, fh - 1)

    crop = frame[cy1:roi_y2, cx1:cx2]
    if crop.size == 0:
        return None
    return crop


# --------------------------------------------------------------------------- #
#  Histogram computation
# --------------------------------------------------------------------------- #
def compute_hsv_hist(
    roi: Optional[np.ndarray],
    h_bins: int = 30,
    s_bins: int = 32,
) -> Optional[np.ndarray]:
    """
    Compute L1-normalised 2-D (H, S) histogram as a flat float32 array.

    Returns None if roi is None or too small.
    """
    if roi is None or roi.size == 0:
        return None
    if roi.shape[0] < 2 or roi.shape[1] < 2:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        [h_bins, s_bins],
        [0, 180, 0, 256],
    )
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
    return hist.flatten().astype(np.float32)


# --------------------------------------------------------------------------- #
#  Convenience: extract + compute in one call
# --------------------------------------------------------------------------- #
def extract_feature(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    h_bins: int = 30,
    s_bins: int = 32,
) -> Optional[np.ndarray]:
    """Single-call helper: ROI → normalised HSV histogram or None."""
    roi = extract_upper_body_roi(frame, x1, y1, x2, y2)
    return compute_hsv_hist(roi, h_bins, s_bins)


# --------------------------------------------------------------------------- #
#  Similarity
# --------------------------------------------------------------------------- #
def hist_similarity(
    h1: Optional[np.ndarray],
    h2: Optional[np.ndarray],
) -> float:
    """
    Bhattacharyya-based similarity ∈ [0, 1]:  0 = no overlap, 1 = identical.

    Uses cv2.HISTCMP_BHATTACHARYYA which returns a *distance* in [0, 1].
    We flip it:  similarity = 1 - distance.
    Both arrays must be 1-D float32 of the same length.
    """
    if h1 is None or h2 is None:
        return 0.0
    a = h1.astype(np.float32).ravel()
    b = h2.astype(np.float32).ravel()
    if a.shape != b.shape or a.size == 0:
        return 0.0
    dist = float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))
    return float(np.clip(1.0 - dist, 0.0, 1.0))
