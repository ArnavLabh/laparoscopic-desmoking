"""
Laparoscopic Video Enhancement Pipeline — Streamlit Demo App
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import json
import time
import torch
from PIL import Image
import requests
import re

from pipeline.detection.smoke_classifier import load_classifier, predict_frame
from pipeline.enhancement.desmoke import load_generator, desmoke_frame
from pipeline.evaluation.metrics import evaluate_frame
from pipeline.utils.download_weights import ensure_weights

ensure_weights()

# Page config

st.set_page_config(
    page_title="Laparoscopic Desmoking v0.1",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS

st.markdown("""
<style>
    .stApp { background-color: #0a0e14; color: #c8d0e0; }
    .block-container { padding-top: 1rem !important; padding-bottom: 0.5rem !important; }
    div[data-testid="stVerticalBlock"] > div { margin-bottom: 0.2rem !important; }

    [data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid #1e2733;
    }
    [data-testid="stSidebar"] .block-container { padding-top: 0.75rem !important; }

    [data-testid="stMetric"] {
        background-color: #0d1a26;
        border: 1px solid #1a3a5c;
        border-radius: 6px;
        padding: 8px 12px !important;
    }
    [data-testid="stMetricLabel"] { color: #5a8a9f; font-size: 0.7rem; }
    [data-testid="stMetricValue"] { color: #00d4ff; font-weight: 600; font-size: 1rem; }

    h1 {
        color: #00d4ff; font-family: 'Courier New', monospace;
        font-size: 1.4rem; margin-bottom: 0 !important;
    }
    h2 {
        color: #5a8a9f; font-size: 0.95rem; font-weight: 400;
        letter-spacing: 0.1em; margin-top: 0 !important; margin-bottom: 0.3rem !important;
    }
    h3 {
        color: #00b4d8; font-size: 0.85rem;
        margin-bottom: 0.2rem !important; margin-top: 0.3rem !important;
    }

    .stProgress > div > div { background-color: #00d4ff; }
    .stProgress { margin-bottom: 0.2rem !important; }
    hr { border-color: #1e2733; margin: 0.4rem 0 !important; }

    .stButton > button {
        background-color: #0d3a5c; color: #00d4ff;
        border: 1px solid #1a6a9f; border-radius: 4px;
        font-family: monospace; width: 100%; padding: 0.4rem !important;
    }
    .stButton > button:hover { background-color: #1a5c8a; border-color: #00d4ff; }

    [data-testid="stFileUploader"] {
        border: 1px dashed #1e2733; border-radius: 6px; padding: 4px;
    }
    .stSlider { padding-top: 0 !important; padding-bottom: 0 !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        font-family: monospace; font-size: 0.8rem;
        padding: 4px 12px; color: #5a8a9f;
    }
    .stTabs [aria-selected="true"] { color: #00d4ff !important; }
</style>
""", unsafe_allow_html=True)


# Helpers

@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = load_classifier("weights/smoke_classifier_finetuned.pth", device=device)
    generator  = load_generator("weights/G_hazy2clear_lite.pth", device=device)
    return classifier, generator, device


def frames_to_time(frames: int, fps: float) -> str:
    """Convert frame number to MM:SS string."""
    total_sec = int(frames / fps)
    m, s = divmod(total_sec, 60)
    return f"{m}:{s:02d}"


def time_to_frames(time_str: str, fps: float) -> int:
    """Convert MM:SS string to frame number."""
    try:
        parts = time_str.strip().split(":")
        if len(parts) == 2:
            return int((int(parts[0]) * 60 + int(parts[1])) * fps)
        return int(float(parts[0]) * fps)
    except Exception:
        return 0


def resolve_video_url(url: str) -> str:
    gdrive_match = re.search(r"drive\.google\.com/file/d/([a-zA-Z0-9_-]+)", url)
    if gdrive_match:
        file_id = gdrive_match.group(1)
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    return url


def download_video_from_url(url: str) -> str:
    resolved = resolve_video_url(url)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    with requests.get(resolved, stream=True, timeout=30) as r:
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=8192):
            tmp.write(chunk)
    tmp.flush()
    return tmp.name


def read_frame_at(video_path: str, frame_idx: int):
    """Read a single frame from a video file by index."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


# Session state init

for key, val in [
    ("review_idx", 0),
    ("output_video_path", None),
    ("metrics_log", None),
    ("processing_done", False),
    ("tfile_path", None),
    ("total_frames", 0),
    ("fps", 25.0),
    ("start_frame", 0),
    ("end_frame", 0),
]:
    if key not in st.session_state:
        st.session_state[key] = val


# Sidebar

with st.sidebar:
    st.markdown("## ⚙ PIPELINE CONFIG")
    st.markdown("---")

    confidence_thresh = st.slider(
        "Detection Confidence Threshold",
        min_value=0.50, max_value=0.99,
        value=0.80, step=0.01,
        help="Frames below this confidence pass through unchanged"
    )

    st.markdown("---")
    st.markdown("## 📊 METRIC THRESHOLDS")
    st.markdown("""
    <div style='font-family:monospace; font-size:0.72rem; color:#5a8a9f; line-height:1.8'>
    PSNR &nbsp; >30 dB &nbsp; acceptable<br>
    PSNR &nbsp; >40 dB &nbsp; excellent<br>
    SSIM &nbsp; >0.90 &nbsp;&nbsp; good<br>
    ΔE &nbsp;&nbsp;&nbsp; &lt;2.0 &nbsp;&nbsp; imperceptible<br>
    ΔE &nbsp;&nbsp;&nbsp; &lt;5.0 &nbsp;&nbsp; clinical safe
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-family:monospace; font-size:0.68rem; color:#2a4a5a; line-height:1.7'>
    SMOKE CLASSIFIER<br>
    MobileNetV2 · v0.1<br>
    val_acc = 89.7%<br><br>
    ENHANCEMENT MODEL<br>
    CycleGAN Lite · v0.1<br>
    50 epochs · 128×128
    </div>
    """, unsafe_allow_html=True)


# Header

st.markdown("# 🔬 LAPAROSCOPIC DESMOKING PIPELINE")
st.markdown("## SURGICAL SMOKE DETECTION & REMOVAL · v0.1")
st.markdown("---")

# Upload

tab1, tab2 = st.tabs(["📁  UPLOAD FILE", "🔗  PASTE URL"])

tfile_path = st.session_state.tfile_path

with tab1:
    uploaded = st.file_uploader(
        "Upload laparoscopic video (max 200MB)",
        type=["mp4", "avi", "mov"],
        help="For larger files use the URL tab"
    )
    if uploaded is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded.read())
        tfile.flush()
        tfile_path = tfile.name
        st.session_state.tfile_path = tfile_path
        st.session_state.processing_done = False

with tab2:
    st.markdown("""
    <div style='font-family:monospace; font-size:0.72rem; color:#5a8a9f; margin-bottom:4px'>
    ACCEPTED: Google Drive share link · Direct video URL (Dropbox etc.)
    </div>
    """, unsafe_allow_html=True)
    video_url = st.text_input(
        "Paste video URL",
        placeholder="https://drive.google.com/file/d/... or https://dl.dropbox.com/..."
    )
    if video_url:
        with st.spinner("Fetching video..."):
            try:
                tfile_path = download_video_from_url(video_url)
                st.session_state.tfile_path = tfile_path
                st.session_state.processing_done = False
                st.success("Video fetched successfully.")
            except Exception as e:
                st.error(f"Could not fetch video: {e}")
                tfile_path = None

if tfile_path is None:
    st.markdown("""
    <div style='text-align:center; padding:40px; color:#2a4a5a;
                border:1px dashed #1e2733; border-radius:8px; margin-top:12px'>
        <div style='font-size:1.8rem'>📹</div>
        <div style='font-family:monospace; font-size:0.85rem; margin-top:8px'>
            AWAITING VIDEO INPUT
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Video info

cap = cv2.VideoCapture(tfile_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
cap.release()

if total_frames < 2:
    st.error("Could not read video. If using a URL, ensure it is a direct download link.")
    st.stop()

st.session_state.total_frames = total_frames
st.session_state.fps = fps

st.markdown("---")
vi1, vi2, vi3 = st.columns(3)
vi1.metric("TOTAL FRAMES", total_frames)
vi2.metric("FPS", f"{fps:.1f}")
vi3.metric("DURATION", frames_to_time(total_frames, fps))

# Time-based range selector

st.markdown("#### SELECT CLIP RANGE")

duration_sec = int(total_frames / fps)

time_col1, time_col2 = st.columns(2)
with time_col1:
    start_time_input = st.text_input(
        "Start time (MM:SS)", value="0:00", placeholder="0:00"
    )
with time_col2:
    default_end = frames_to_time(min(int(fps * 10), total_frames), fps)
    end_time_input = st.text_input(
        "End time (MM:SS)",
        value=default_end,
        placeholder=frames_to_time(total_frames, fps),
        help=f"Max: {frames_to_time(total_frames, fps)}"
    )

# Slider for quick scrubbing — shows seconds
scrub = st.slider(
    "Timeline scrubber",
    min_value=0,
    max_value=max(duration_sec, 1),
    value=(0, min(10, duration_sec)),
    step=1,
    format="%d s",
    label_visibility="collapsed"
)

# Slider takes priority over text input
start_frame = int(scrub[0] * fps)
end_frame   = min(int(scrub[1] * fps), total_frames)

if end_frame <= start_frame:
    end_frame = min(start_frame + int(fps), total_frames)

frames_to_process = end_frame - start_frame

sc1, sc2, sc3, sc4 = st.columns(4)
sc1.metric("START",       frames_to_time(start_frame, fps))
sc2.metric("END",         frames_to_time(end_frame,   fps))
sc3.metric("CLIP LENGTH", frames_to_time(frames_to_process, fps))
sc4.metric("FRAMES",      frames_to_process)

st.markdown("---")

# Run button

run = st.button("▶  RUN ENHANCEMENT PIPELINE")

if not run and not st.session_state.processing_done:
    st.stop()

# Processing

if run:
    st.session_state.processing_done = False
    st.session_state.start_frame = start_frame
    st.session_state.end_frame   = end_frame

    with st.spinner("Loading models..."):
        classifier, generator, device = load_models()

    st.markdown(
        f'<div style="font-family:monospace; font-size:0.72rem; color:#5a8a9f; margin-bottom:4px">'
        f'DEVICE: {device.upper()} &nbsp;|&nbsp; '
        f'CLASSIFIER: smoke_classifier_finetuned.pth &nbsp;|&nbsp; '
        f'GENERATOR: G_hazy2clear_lite.pth</div>',
        unsafe_allow_html=True
    )
    st.markdown("---")

    progress_bar = st.progress(0)
    status_text  = st.empty()

    frame_cols = st.columns(2)
    with frame_cols[0]:
        st.markdown("### ORIGINAL")
        orig_display = st.empty()
    with frame_cols[1]:
        st.markdown("### OUTPUT")
        output_label = st.empty()
        enh_display  = st.empty()

    st.markdown("---")
    st.markdown("### LIVE METRICS")
    mc = st.columns(4)
    m_frame   = mc[0].empty()
    m_psnr    = mc[1].empty()
    m_ssim    = mc[2].empty()
    m_delta_e = mc[3].empty()

    st.markdown("---")
    st.markdown("### FRAME LOG")
    log_container = st.empty()

    cap = cv2.VideoCapture(tfile_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    metrics_log = []
    log_lines   = []
    smoky_count = 0

    output_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(output_tmp.name, fourcc, fps, (w, h))

    for idx in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break

        label, confidence = predict_frame(classifier, frame, device=device)
        is_smoky = (label == 1) and (confidence >= confidence_thresh)

        if is_smoky:
            smoky_count  += 1
            enhanced      = desmoke_frame(generator, frame, device=device)
            metrics       = evaluate_frame(frame, enhanced)
            output_frame  = enhanced
            status        = "SMOKY"
        else:
            enhanced      = frame.copy()
            metrics       = {"psnr": None, "ssim": None, "delta_e": None}
            output_frame  = frame
            status        = "CLEAR"
            time.sleep(0.08)  # hold passthrough frames long enough to be visible

        writer.write(output_frame)
        metrics_log.append({
            "frame":      idx,
            "time":       frames_to_time(idx, fps),
            "smoky":      is_smoky,
            "confidence": round(confidence, 4),
            **{k: round(v, 4) if v is not None else None for k, v in metrics.items()}
        })

        done_count = idx - start_frame + 1
        progress_bar.progress(done_count / frames_to_process)
        status_text.markdown(
            f'<div style="font-family:monospace; font-size:0.75rem; color:#5a8a9f">'
            f'{frames_to_time(idx, fps)} &nbsp;|&nbsp; '
            f'FRAME {done_count}/{frames_to_process} &nbsp;|&nbsp; '
            f'SMOKY: {smoky_count}</div>',
            unsafe_allow_html=True
        )

        orig_display.image(cv2.cvtColor(frame,    cv2.COLOR_BGR2RGB), use_container_width=True)
        enh_display.image( cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB), use_container_width=True)

        if is_smoky:
            output_label.markdown(
                '<div style="font-family:monospace; font-size:0.72rem; '
                'background:#1a2e1a; border:1px solid #2d7a2d; border-radius:4px; '
                'padding:3px 8px; color:#69db69; display:inline-block; margin-bottom:4px">'
                '✦ SMOKE REMOVED</div>',
                unsafe_allow_html=True
            )
        else:
            output_label.markdown(
                '<div style="font-family:monospace; font-size:0.72rem; '
                'background:#1a1a2e; border:1px solid #2d2d7a; border-radius:4px; '
                'padding:3px 8px; color:#6b6bff; display:inline-block; margin-bottom:4px">'
                '→ PASSING UNTOUCHED</div>',
                unsafe_allow_html=True
            )

        m_frame.metric("TIME",  frames_to_time(idx, fps))
        if is_smoky and metrics["psnr"] is not None:
            m_psnr.metric("PSNR",  f"{metrics['psnr']:.1f} dB")
            m_ssim.metric("SSIM",  f"{metrics['ssim']:.3f}")
            m_delta_e.metric("ΔE", f"{metrics['delta_e']:.2f}")
        else:
            m_psnr.metric("PSNR",  "—")
            m_ssim.metric("SSIM",  "—")
            m_delta_e.metric("ΔE", "—")

        psnr_str = f"{metrics['psnr']:.1f}dB" if metrics["psnr"] else "  —  "
        de_str   = f"ΔE={metrics['delta_e']:.2f}" if metrics["delta_e"] else "ΔE= —"
        log_lines.append(
            f"[{frames_to_time(idx, fps):>5}] {status:5s} | "
            f"conf={confidence:.2f} | PSNR={psnr_str:>8} | {de_str}"
        )
        log_container.code("\n".join(log_lines[-10:]), language=None)

    cap.release()
    writer.release()

    st.session_state.output_video_path = output_tmp.name
    st.session_state.metrics_log       = metrics_log
    st.session_state.processing_done   = True
    st.session_state.review_idx        = 0

    # Summary
    st.markdown("---")
    st.markdown("### PIPELINE SUMMARY")

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("FRAMES PROCESSED",  frames_to_process)
    s2.metric("SMOKY DETECTED",    smoky_count)
    s3.metric("SMOKE %",           f"{100*smoky_count/max(frames_to_process,1):.1f}%")
    s4.metric("CLEAR (UNCHANGED)", frames_to_process - smoky_count)

    enhanced_m = [m for m in metrics_log if m["smoky"] and m["psnr"] is not None]
    if enhanced_m:
        st.markdown("---")
        st.markdown("### AVERAGE QUALITY METRICS")
        a1, a2, a3 = st.columns(3)
        avg_psnr    = np.mean([m["psnr"]    for m in enhanced_m])
        avg_ssim    = np.mean([m["ssim"]    for m in enhanced_m])
        avg_delta_e = np.mean([m["delta_e"] for m in enhanced_m])
        a1.metric("AVG PSNR",    f"{avg_psnr:.2f} dB")
        a2.metric("AVG SSIM",    f"{avg_ssim:.4f}")
        a3.metric("AVG DELTA-E", f"{avg_delta_e:.2f}")

        if avg_delta_e < 2:
            de_status, de_color = "✓ IMPERCEPTIBLE — clinically safe",        "#69db69"
        elif avg_delta_e < 5:
            de_status, de_color = "⚠ ACCEPTABLE — within clinical tolerance", "#ffd700"
        else:
            de_status, de_color = "✗ NOTICEABLE — review enhancement output", "#ff6b6b"

        st.markdown(
            f'<div style="font-family:monospace; font-size:0.8rem; '
            f'color:{de_color}; margin-top:4px">'
            f'COLOR FIDELITY: {de_status}</div>',
            unsafe_allow_html=True
        )

# Post-processing frame review

if st.session_state.processing_done and st.session_state.metrics_log:

    st.markdown("---")
    st.markdown("### FRAME REVIEW")

    metrics_log     = st.session_state.metrics_log
    output_path     = st.session_state.output_video_path
    total_processed = len(metrics_log)

    # Navigation controls
    ctrl1, ctrl2, ctrl3, ctrl4, ctrl5 = st.columns([1, 1, 3, 1, 1])
    with ctrl1:
        if st.button("⏮", help="First frame"):
            st.session_state.review_idx = 0
    with ctrl2:
        if st.button("◀", help="Previous frame"):
            st.session_state.review_idx = max(0, st.session_state.review_idx - 1)
    with ctrl3:
        review_scrub = st.slider(
            "Review",
            min_value=0,
            max_value=total_processed - 1,
            value=st.session_state.review_idx,
            format="Frame %d",
            label_visibility="collapsed",
            key="review_scrub"
        )
        st.session_state.review_idx = review_scrub
    with ctrl4:
        if st.button("▶", help="Next frame"):
            st.session_state.review_idx = min(total_processed - 1, st.session_state.review_idx + 1)
    with ctrl5:
        if st.button("⏭", help="Last frame"):
            st.session_state.review_idx = total_processed - 1

    ridx       = st.session_state.review_idx
    frame_meta = metrics_log[ridx]
    actual_frame = frame_meta["frame"]

    st.markdown(
        f'<div style="font-family:monospace; font-size:0.75rem; color:#5a8a9f; text-align:center; margin:4px 0">'
        f'TIME: {frame_meta["time"]} &nbsp;|&nbsp; '
        f'FRAME {ridx + 1}/{total_processed} &nbsp;|&nbsp; '
        f'{"⚠ SMOKY" if frame_meta["smoky"] else "✓ CLEAR"} &nbsp;|&nbsp; '
        f'CONF: {frame_meta["confidence"]:.2f}</div>',
        unsafe_allow_html=True
    )

    orig_frame = read_frame_at(tfile_path, actual_frame)
    out_frame  = read_frame_at(output_path, ridx)

    if orig_frame is not None and out_frame is not None:
        rev_cols = st.columns(2)
        with rev_cols[0]:
            st.markdown("### ORIGINAL")
            st.image(cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        with rev_cols[1]:
            st.markdown("### OUTPUT")
            if frame_meta["smoky"]:
                st.markdown(
                    '<div style="font-family:monospace; font-size:0.72rem; '
                    'background:#1a2e1a; border:1px solid #2d7a2d; border-radius:4px; '
                    'padding:3px 8px; color:#69db69; display:inline-block; margin-bottom:4px">'
                    '✦ SMOKE REMOVED</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div style="font-family:monospace; font-size:0.72rem; '
                    'background:#1a1a2e; border:1px solid #2d2d7a; border-radius:4px; '
                    'padding:3px 8px; color:#6b6bff; display:inline-block; margin-bottom:4px">'
                    '→ PASSING UNTOUCHED</div>',
                    unsafe_allow_html=True
                )
            st.image(cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB), use_container_width=True)

    if frame_meta["smoky"] and frame_meta.get("psnr") is not None:
        rm1, rm2, rm3 = st.columns(3)
        rm1.metric("PSNR",  f"{frame_meta['psnr']:.1f} dB")
        rm2.metric("SSIM",  f"{frame_meta['ssim']:.4f}")
        rm3.metric("ΔE",    f"{frame_meta['delta_e']:.2f}")

    # Downloads
    st.markdown("---")
    dl1, dl2 = st.columns(2)
    with dl1:
        with open(output_path, "rb") as f:
            st.download_button(
                label="⬇  DOWNLOAD ENHANCED VIDEO",
                data=f,
                file_name="enhanced_output.mp4",
                mime="video/mp4"
            )
    with dl2:
        st.download_button(
            label="⬇  DOWNLOAD METRICS JSON",
            data=json.dumps(metrics_log, indent=2),
            file_name="pipeline_metrics.json",
            mime="application/json"
        )

st.markdown("""
<div style='font-family:monospace; font-size:0.65rem; color:#1e2733;
            text-align:center; margin-top:8px'>
PIPELINE v0.1 · SMOKE CLASSIFIER MobileNetV2 · ENHANCEMENT CycleGAN LITE
</div>
""", unsafe_allow_html=True)
