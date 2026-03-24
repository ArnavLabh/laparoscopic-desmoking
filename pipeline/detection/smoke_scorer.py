"""
Multi-signal heuristic smoke detector.
Based on inter-channel discrepancy (IC) and dark channel prior (DC)
from Pan et al. (2022) DeSmoke-LAP, with saturation as a third signal.

No training required. Use this until smoke_classifier.py is trained.
"""

import cv2
import numpy as np
from collections import deque


def ic_score(frame: np.ndarray) -> float:
    """
    Inter-channel discrepancy (Pan et al. 2022, Eq. 6).
    Low value = high smoke — channels converge to same gray under smoke.
    frame: BGR uint8
    """
    f = frame.astype(np.float32) / 255.0
    b, g, r = f[..., 0], f[..., 1], f[..., 2]
    psi = np.abs(r - g) + np.abs(g - b) + np.abs(b - r)
    return float(psi.mean())


def dc_score(frame: np.ndarray, patch_size: int = 15) -> float:
    """
    Dark channel prior (He et al. 2010).
    High value = more haze/smoke.
    """
    f = frame.astype(np.float32) / 255.0
    dark = np.min(f, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark = cv2.erode(dark, kernel)
    return float(dark.mean())


def saturation_score(frame: np.ndarray) -> float:
    """
    Saturation std in HSV.
    Smoke flattens and desaturates the image — low std = more smoke.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return float(hsv[..., 1].std())


def compute_smoke_score(frame: np.ndarray) -> dict:
    """
    Combines IC, DC, and saturation into a single smoke score in [0, 1].
    Bounds recalibrated from TLH_2 signal output:
        IC range observed:  ~0.36 – 1.09  (lower = more smoke)
        DC range observed:  ~0.005 – 0.52 (higher = more smoke)
        SAT range observed: ~25 – 74      (less reliable, kept low weight)

    Note: scores still overlap between clear/hazy — use CNN for hard decisions.
    This function is best used for visualization and soft blending, not thresholding.
    """
    ic  = ic_score(frame)
    dc  = dc_score(frame)
    sat = saturation_score(frame)

    # Recalibrated bounds from actual data
    ic_norm  = 1.0 - np.clip((ic - 0.3) / (1.1 - 0.3), 0.0, 1.0)  # 0.3–1.1 observed range
    dc_norm  = np.clip(dc / 0.55, 0.0, 1.0)                          # 0–0.55 observed range
    sat_norm = 1.0 - np.clip(sat / 75.0, 0.0, 1.0)                  # 0–75 observed range

    # DC carries the most discriminative signal — bump its weight
    score = 0.4 * ic_norm + 0.5 * dc_norm + 0.1 * sat_norm

    return {
        "ic": ic,
        "dc": dc,
        "sat": sat,
        "smoke_score": float(score),
    }



class TemporalSmokeDetector:
    """
    Per-frame scorer with sliding window temporal smoothing.
    Prevents hard switches on single noisy frames.

    Usage:
        detector = TemporalSmokeDetector(window=5, threshold=0.5)
        is_smoke, score = detector.update(frame)
    """

    def __init__(self, window: int = 5, threshold: float = 0.5):
        self.buffer = deque(maxlen=window)
        self.threshold = threshold

    def update(self, frame: np.ndarray) -> tuple[bool, float]:
        """
        Returns (is_smoky: bool, smoothed_score: float).
        Score is the mean of the last `window` frames.
        """
        metrics = compute_smoke_score(frame)
        self.buffer.append(metrics["smoke_score"])
        smoothed = float(np.mean(self.buffer))
        return smoothed > self.threshold, smoothed

    def reset(self):
        """Call between videos to clear temporal state."""
        self.buffer.clear()