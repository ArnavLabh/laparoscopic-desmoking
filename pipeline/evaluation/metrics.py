"""
Image quality evaluation metrics.
PSNR, SSIM, and Delta-E (CIE76) for comparing enhanced vs original frames.
"""

import cv2
import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio


def compute_psnr(original: np.ndarray, enhanced: np.ndarray) -> float:
    """
    Peak Signal-to-Noise Ratio in dB.
    Higher = better. >30dB is generally acceptable, >40dB is excellent.
    Both inputs: BGR uint8.
    """
    return float(peak_signal_noise_ratio(original, enhanced, data_range=255))


def compute_ssim(original: np.ndarray, enhanced: np.ndarray) -> float:
    """
    Structural Similarity Index. Range [0, 1], higher = better.
    Measures luminance, contrast and structure together.
    Both inputs: BGR uint8.
    """
    orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    enh_gray  = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    return float(structural_similarity(orig_gray, enh_gray, data_range=255))


def compute_delta_e(original: np.ndarray, enhanced: np.ndarray) -> float:
    """
    CIE76 Delta-E color difference.
    Measures perceptual color shift introduced by enhancement.
    Lower = better. <2 is imperceptible, <5 is acceptable for clinical use.
    Both inputs: BGR uint8.
    """
    orig_lab = cv2.cvtColor(original, cv2.COLOR_BGR2Lab).astype(np.float32)
    enh_lab  = cv2.cvtColor(enhanced, cv2.COLOR_BGR2Lab).astype(np.float32)
    diff = np.sqrt(np.sum((orig_lab - enh_lab) ** 2, axis=2))
    return float(diff.mean())


def evaluate_frame(original: np.ndarray,
                   enhanced: np.ndarray) -> dict:
    """
    Run all three metrics on a single frame pair.
    Returns dict with psnr, ssim, delta_e.

    original: the input frame (hazy)
    enhanced: the output frame (after desmoking)

    Note: PSNR and SSIM here compare enhanced vs original hazy frame,
    not vs a ground truth clean frame. Use for tracking enhancement
    strength, not absolute quality, until ground truth pairs are available.
    """
    return {
        "psnr":    compute_psnr(original, enhanced),
        "ssim":    compute_ssim(original, enhanced),
        "delta_e": compute_delta_e(original, enhanced),
    }