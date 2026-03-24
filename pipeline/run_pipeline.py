"""
End-to-end laparoscopic video enhancement pipeline.

Usage:
    python -m pipeline.run_pipeline --input video.mp4 --output output.mp4

Flow:
    Frame extraction → Smoke detection → Enhancement → Evaluation → Reconstruction
"""

import cv2
import os
import argparse
import json
import numpy as np
import torch

from pipeline.detection.smoke_classifier import load_classifier, predict_frame
from pipeline.enhancement.desmoke import load_generator, desmoke_frame
from pipeline.evaluation.metrics import evaluate_frame


def extract_frames(video_path: str):
    """
    Generator that yields (frame_index, frame) tuples from a video file.
    Avoids loading all frames into memory at once.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps        = cap.get(cv2.CAP_PROP_FPS)
    total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {total} frames @ {fps:.1f}fps  ({width}x{height})")

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield idx, frame
        idx += 1

    cap.release()


def get_video_properties(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    props = {
        "fps":    cap.get(cv2.CAP_PROP_FPS),
        "width":  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "total":  int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return props


def run_pipeline(
    input_path:      str,
    output_path:     str,
    classifier_path: str = "weights/smoke_classifier_finetuned.pth",
    generator_path:  str = "weights/G_hazy2clear_lite.pth",
    confidence_thresh: float = 0.80,
    save_metrics:    bool = True,
):
    """
    Main pipeline function.

    confidence_thresh: only enhance frames where classifier confidence >= this.
    Frames below threshold are treated as clear and passed through unchanged.
    This avoids enhancing borderline frames where the classifier is uncertain.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load models
    print("Loading classifier...")
    classifier = load_classifier(classifier_path, device=device)

    print("Loading generator...")
    generator = load_generator(generator_path, device=device)

    # Video properties
    props = get_video_properties(input_path)
    fps, width, height = props["fps"], props["width"], props["height"]
    total = props["total"]

    # Output video writer
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Metrics log
    metrics_log = []
    smoky_count, enhanced_count = 0, 0

    print(f"\nProcessing {total} frames...\n")

    for idx, frame in extract_frames(input_path):
        # --- Smoke detection ---
        label, confidence = predict_frame(classifier, frame, device=device)
        is_smoky = (label == 1) and (confidence >= confidence_thresh)

        if is_smoky:
            smoky_count += 1
            # --- Enhancement ---
            enhanced = desmoke_frame(generator, frame, device=device)
            enhanced_count += 1

            # --- Evaluation ---
            metrics = evaluate_frame(original=frame, enhanced=enhanced)
            metrics_log.append({
                "frame":      idx,
                "smoky":      True,
                "confidence": round(confidence, 4),
                **{k: round(v, 4) for k, v in metrics.items()},
            })

            writer.write(enhanced)

            print(
                f"Frame {idx:05d} | SMOKY  conf={confidence:.2f} | "
                f"PSNR={metrics['psnr']:.1f}  "
                f"SSIM={metrics['ssim']:.3f}  "
                f"ΔE={metrics['delta_e']:.2f}"
            )

        else:
            # Clear frame — pass through unchanged
            metrics_log.append({
                "frame":      idx,
                "smoky":      False,
                "confidence": round(confidence, 4),
                "psnr":       None,
                "ssim":       None,
                "delta_e":    None,
            })
            writer.write(frame)

            if idx % 50 == 0:  # only print every 50 clear frames to reduce noise
                print(f"Frame {idx:05d} | CLEAR  conf={confidence:.2f}")

    writer.release()

    # --- Summary ---
    enhanced_metrics = [m for m in metrics_log if m["smoky"]]
    print(f"\n{'='*50}")
    print(f"Total frames:     {total}")
    print(f"Smoky detected:   {smoky_count} ({100*smoky_count/total:.1f}%)")
    print(f"Enhanced:         {enhanced_count}")

    if enhanced_metrics:
        avg_psnr    = np.mean([m["psnr"]    for m in enhanced_metrics])
        avg_ssim    = np.mean([m["ssim"]    for m in enhanced_metrics])
        avg_delta_e = np.mean([m["delta_e"] for m in enhanced_metrics])
        print(f"\nAverage metrics on enhanced frames:")
        print(f"  PSNR:    {avg_psnr:.2f} dB")
        print(f"  SSIM:    {avg_ssim:.4f}")
        print(f"  Delta-E: {avg_delta_e:.2f}")
        print(f"\nDelta-E interpretation:")
        print(f"  <2  = imperceptible color shift (ideal)")
        print(f"  <5  = acceptable for clinical use")
        print(f"  >5  = noticeable color distortion (review needed)")

    # Save metrics JSON
    if save_metrics:
        metrics_path = output_path.replace(".mp4", "_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics_log, f, indent=2)
        print(f"\nMetrics saved to: {metrics_path}")

    print(f"Output video:     {output_path}")
    print(f"{'='*50}")

    return metrics_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",       required=True,  help="Input video path")
    parser.add_argument("--output",      required=True,  help="Output video path")
    parser.add_argument("--classifier",  default="weights/smoke_classifier_finetuned.pth")
    parser.add_argument("--generator",   default="weights/G_hazy2clear_lite.pth")
    parser.add_argument("--confidence",  type=float, default=0.80,
                        help="Minimum classifier confidence to trigger enhancement")
    parser.add_argument("--no-metrics",  action="store_true")
    args = parser.parse_args()

    run_pipeline(
        input_path       = args.input,
        output_path      = args.output,
        classifier_path  = args.classifier,
        generator_path   = args.generator,
        confidence_thresh= args.confidence,
        save_metrics     = not args.no_metrics,
    )