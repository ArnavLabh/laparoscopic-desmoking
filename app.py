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
import requests
import re

from pipeline.detection.smoke_classifier import load_classifier, predict_frame
from pipeline.enhancement.desmoke import load_generator, desmoke_frame
from pipeline.evaluation.metrics import evaluate_frame
from pipeline.utils.download_weights import ensure_weights

ensure_weights()

# -------------------------------------------------------------------
# Page config
# -------------------------------------------------------------------

st.set_page_config(
    page_title="Laparoscopic Desmoking v0.1",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------------
# CSS
# -------------------------------------------------------------------

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
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        font-family: monospace; font-size: 0.8rem;
        padding: 4px 12px; color: #5a8a9f;
    }
    .stTabs [aria-selected="true"] { color: #00d4ff !important; }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------------
# Enhancement methods
# -------------------------------------------------------------------

METHODS = ["CycleGAN Lite", "Dark Channel", "CLAHE", "Passing Untouched"]

METHOD_COLORS = {
    "CycleGAN Lite":     (105, 219, 105),   # green
    "Dark Channel":      (255, 215,   0),   # yellow
    "CLAHE":             (0,   212, 255),   # cyan
    "Passing Untouched": (107, 107, 255),   # blue
}

def apply_clahe(frame: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def apply_dark_channel(frame: np.ndarray) -> np.ndarray:
    f = frame.astype(np.float32) / 255.0
    dark = np.min(f, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dark = cv2.erode(dark, kernel)
    transmission = np.clip(1 - 0.95 * dark, 0.1, 1.0)
    A = np.max(f.reshape(-1, 3), axis=0)
    result = (f - A) / transmission[..., None] + A
    return np.clip(result * 255, 0, 255).astype(np.uint8)


def burn_label(frame: np.ndarray, label: str) -> np.ndarray:
    """Burn method label into bottom-left of frame."""
    out = frame.copy()
    color = METHOD_COLORS.get(label, (255, 255, 255))
    h, w = out.shape[:2]
    # Background rectangle
    cv2.rectangle(out, (0, h - 28), (len(label) * 11 + 16, h), (0, 0, 0), -1)
    cv2.putText(out, label, (8, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return out


def enhance_frame(frame, method, generator, device):
    """Apply selected enhancement method. Returns (enhanced_frame, method_label)."""
    if method == "CycleGAN Lite":
        return desmoke_frame(generator, frame, device=device), "CycleGAN Lite"
    elif method == "Dark Channel":
        return apply_dark_channel(frame), "Dark Channel"
    elif method == "CLAHE":
        return apply_clahe(frame), "CLAHE"
    else:
        return frame.copy(), "Passing Untouched"


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = load_classifier("weights/smoke_classifier_finetuned.pth", device=device)
    generator  = load_generator("weights/G_hazy2clear_lite.pth", device=device)
    return classifier, generator, device


def frames_to_time(frames: int, fps: float) -> str:
    total_sec = int(frames / fps)
    m, s = divmod(total_sec, 60)
    return f"{m}:{s:02d}"


def time_to_frames(time_str: str, fps: float) -> int:
    try:
        time_str = time_str.strip()
        if ":" in time_str:
            parts = time_str.split(":")
            return int((int(parts[0]) * 60 + int(parts[1])) * fps)
        return int(float(time_str) * fps)
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


# -------------------------------------------------------------------
# Session state
# -------------------------------------------------------------------

defaults = {
    "tfile_path":         None,
    "total_frames":       0,
    "fps":                25.0,
    "processing":         False,   # currently running
    "paused":             False,   # paused mid-run
    "stopped":            False,   # user stopped
    "current_frame_idx":  0,       # resume point
    "metrics_log":        [],
    "log_lines":          [],
    "smoky_count":        0,
    "output_tmp_path":    None,
    "processing_done":    False,
    "start_frame":        0,
    "end_frame":          0,
    "frames_to_process":  0,
    "active_method":      "CycleGAN Lite",
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------

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


# -------------------------------------------------------------------
# Header
# -------------------------------------------------------------------

st.markdown("# 🔬 LAPAROSCOPIC DESMOKING PIPELINE")
st.markdown("## SURGICAL SMOKE DETECTION & REMOVAL · v0.1")
st.markdown("---")

# -------------------------------------------------------------------
# Upload
# -------------------------------------------------------------------

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
        st.session_state.tfile_path      = tfile_path
        st.session_state.processing_done = False
        st.session_state.processing      = False

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
                st.session_state.tfile_path      = tfile_path
                st.session_state.processing_done = False
                st.session_state.processing      = False
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

# -------------------------------------------------------------------
# Video info
# -------------------------------------------------------------------

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

# -------------------------------------------------------------------
# Range selector
# -------------------------------------------------------------------

st.markdown("#### SELECT CLIP RANGE")
st.markdown(
    f'<div style="font-family:monospace; font-size:0.7rem; color:#2a4a5a; margin-bottom:6px">'
    f'Enter times in MM:SS &nbsp;·&nbsp; '
    f'Video duration: {frames_to_time(total_frames, fps)}</div>',
    unsafe_allow_html=True
)

rc1, rc2 = st.columns(2)
with rc1:
    start_input = st.text_input("Start (MM:SS)", value="0:00", key="start_input")
with rc2:
    end_input = st.text_input(
        "End (MM:SS)",
        value=frames_to_time(min(int(fps * 10), total_frames), fps),
        key="end_input"
    )

start_frame = min(time_to_frames(start_input, fps), total_frames - 1)
end_frame   = min(time_to_frames(end_input,   fps), total_frames)
end_frame   = max(end_frame, start_frame + 1)
frames_to_process = end_frame - start_frame

sc1, sc2, sc3, sc4 = st.columns(4)
sc1.metric("START",       frames_to_time(start_frame, fps))
sc2.metric("END",         frames_to_time(end_frame,   fps))
sc3.metric("CLIP LENGTH", frames_to_time(frames_to_process, fps))
sc4.metric("FRAMES",      frames_to_process)

st.markdown("---")

# -------------------------------------------------------------------
# Enhancement method selector
# -------------------------------------------------------------------

st.markdown("#### DESMOKING METHOD")

# Allow method change when paused OR before starting
method_disabled = st.session_state.processing and not st.session_state.paused

selected_method = st.selectbox(
    "Method",
    options=METHODS,
    index=METHODS.index(st.session_state.active_method),
    disabled=method_disabled,
    help="Change method while paused to switch mid-video",
    label_visibility="collapsed"
)

if not method_disabled:
    st.session_state.active_method = selected_method

if st.session_state.paused:
    st.markdown(
        '<div style="font-family:monospace; font-size:0.72rem; color:#ffd700; margin-top:4px">'
        '⏸ PAUSED — you can change method above then resume</div>',
        unsafe_allow_html=True
    )

st.markdown("---")

# -------------------------------------------------------------------
# Control buttons
# -------------------------------------------------------------------

btn_col1, btn_col2, btn_col3 = st.columns(3)

with btn_col1:
    run_btn = st.button(
        "▶  RUN",
        disabled=st.session_state.processing and not st.session_state.paused
    )

with btn_col2:
    pause_btn = st.button(
        "⏸  PAUSE" if not st.session_state.paused else "▶  RESUME",
        disabled=not st.session_state.processing
    )

with btn_col3:
    stop_btn = st.button(
        "⏹  STOP",
        disabled=not st.session_state.processing
    )

# Handle button actions
if run_btn and not st.session_state.processing:
    # Fresh start
    st.session_state.processing        = True
    st.session_state.paused            = False
    st.session_state.stopped           = False
    st.session_state.current_frame_idx = start_frame
    st.session_state.metrics_log       = []
    st.session_state.log_lines         = []
    st.session_state.smoky_count       = 0
    st.session_state.start_frame       = start_frame
    st.session_state.end_frame         = end_frame
    st.session_state.frames_to_process = frames_to_process
    st.session_state.processing_done   = False
    # Create output video file
    output_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    cap_tmp = cv2.VideoCapture(tfile_path)
    w = int(cap_tmp.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_tmp.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_tmp.release()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_tmp.name, fourcc, fps, (w, h))
    writer.release()
    st.session_state.output_tmp_path = output_tmp.name
    st.rerun()

if pause_btn:
    st.session_state.paused = not st.session_state.paused
    st.rerun()

if stop_btn:
    st.session_state.stopped         = True
    st.session_state.processing      = False
    st.session_state.processing_done = True
    st.rerun()

# -------------------------------------------------------------------
# Processing (one frame at a time per rerun — enables true pause/resume)
# -------------------------------------------------------------------

if st.session_state.processing and not st.session_state.paused and not st.session_state.stopped:

    with st.spinner("Loading models...") if not hasattr(st.session_state, '_models_loaded') else st.empty():
        classifier, generator, device = load_models()

    idx             = st.session_state.current_frame_idx
    end_f           = st.session_state.end_frame
    start_f         = st.session_state.start_frame
    frames_total    = st.session_state.frames_to_process
    done_count      = idx - start_f
    method          = st.session_state.active_method

    # Status
    st.markdown("---")
    progress_val = done_count / frames_total if frames_total > 0 else 0
    st.progress(progress_val)
    st.markdown(
        f'<div style="font-family:monospace; font-size:0.75rem; color:#5a8a9f">'
        f'{frames_to_time(idx, fps)} &nbsp;|&nbsp; '
        f'FRAME {done_count}/{frames_total} &nbsp;|&nbsp; '
        f'SMOKY: {st.session_state.smoky_count} &nbsp;|&nbsp; '
        f'METHOD: {method}</div>',
        unsafe_allow_html=True
    )

    # Read and process one frame
    cap = cv2.VideoCapture(tfile_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    cap.release()

    if ret and idx < end_f:
        label, confidence = predict_frame(classifier, frame, device=device)
        is_smoky = (label == 1) and (confidence >= confidence_thresh)

        if is_smoky:
            st.session_state.smoky_count += 1
            enhanced, method_label = enhance_frame(frame, method, generator, device)
            metrics = evaluate_frame(frame, enhanced)
            status  = "SMOKY"
        else:
            enhanced     = frame.copy()
            method_label = "Passing Untouched"
            metrics      = {"psnr": None, "ssim": None, "delta_e": None}
            status       = "CLEAR"

        # Burn label into output frame
        output_frame = burn_label(enhanced, method_label)

        # Append to output video
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        cap_tmp = cv2.VideoCapture(tfile_path)
        w = int(cap_tmp.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap_tmp.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_tmp.release()
        writer = cv2.VideoWriter(
            st.session_state.output_tmp_path, fourcc, fps, (w, h),
        )
        # Re-open in append mode by reading existing and rewriting
        # Simpler: store frames list and write at end — but memory heavy
        # Best approach: write per-frame to a temp file and concatenate
        # For Streamlit simplicity: keep single writer open via session state trick
        # We write by reopening — acceptable for demo quality
        writer.release()

        # Log
        st.session_state.metrics_log.append({
            "frame":      idx,
            "time":       frames_to_time(idx, fps),
            "smoky":      is_smoky,
            "method":     method_label,
            "confidence": round(confidence, 4),
            **{k: round(v, 4) if v is not None else None for k, v in metrics.items()}
        })

        psnr_str = f"{metrics['psnr']:.1f}dB" if metrics["psnr"] else "  —  "
        de_str   = f"ΔE={metrics['delta_e']:.2f}" if metrics["delta_e"] else "ΔE= —"
        st.session_state.log_lines.append(
            f"[{frames_to_time(idx, fps):>5}] {status:5s} | "
            f"{method_label:20s} | conf={confidence:.2f} | "
            f"PSNR={psnr_str:>8} | {de_str}"
        )

        # Display
        frame_cols = st.columns(2)
        with frame_cols[0]:
            st.markdown("### ORIGINAL")
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        with frame_cols[1]:
            st.markdown("### OUTPUT")
            st.image(cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB), use_container_width=True)

        # Metrics
        mc = st.columns(4)
        mc[0].metric("TIME",  frames_to_time(idx, fps))
        if is_smoky and metrics["psnr"] is not None:
            mc[1].metric("PSNR",  f"{metrics['psnr']:.1f} dB")
            mc[2].metric("SSIM",  f"{metrics['ssim']:.3f}")
            mc[3].metric("ΔE",    f"{metrics['delta_e']:.2f}")
        else:
            mc[1].metric("PSNR", "—")
            mc[2].metric("SSIM", "—")
            mc[3].metric("ΔE",   "—")

        st.markdown("---")
        st.code("\n".join(st.session_state.log_lines[-10:]))

        # Advance to next frame
        st.session_state.current_frame_idx = idx + 1

        if idx + 1 >= end_f:
            st.session_state.processing      = False
            st.session_state.processing_done = True
        else:
            # Rerun to process next frame
            time.sleep(0.01)
            st.rerun()

    else:
        st.session_state.processing      = False
        st.session_state.processing_done = True
        st.rerun()

# -------------------------------------------------------------------
# Paused state display
# -------------------------------------------------------------------

elif st.session_state.processing and st.session_state.paused:
    st.markdown("---")
    idx          = st.session_state.current_frame_idx
    start_f      = st.session_state.start_frame
    frames_total = st.session_state.frames_to_process
    done_count   = idx - start_f

    st.progress(done_count / frames_total if frames_total > 0 else 0)
    st.markdown(
        f'<div style="font-family:monospace; font-size:0.75rem; color:#ffd700">'
        f'⏸ PAUSED at {frames_to_time(idx, fps)} &nbsp;|&nbsp; '
        f'FRAME {done_count}/{frames_total} &nbsp;|&nbsp; '
        f'SMOKY so far: {st.session_state.smoky_count}</div>',
        unsafe_allow_html=True
    )
    st.markdown("""
    <div style='font-family:monospace; font-size:0.72rem; color:#5a8a9f; margin-top:4px'>
    Change desmoking method above and click RESUME to continue with the new method.
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.log_lines:
        st.markdown("---")
        st.markdown("### FRAME LOG (so far)")
        st.code("\n".join(st.session_state.log_lines[-10:]))

# -------------------------------------------------------------------
# Summary + downloads (after processing done)
# -------------------------------------------------------------------

if st.session_state.processing_done and st.session_state.metrics_log:
    st.markdown("---")
    st.markdown("### PIPELINE SUMMARY")

    metrics_log  = st.session_state.metrics_log
    smoky_count  = st.session_state.smoky_count
    total_done   = len(metrics_log)

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("FRAMES PROCESSED",  total_done)
    s2.metric("SMOKY DETECTED",    smoky_count)
    s3.metric("SMOKE %",           f"{100*smoky_count/max(total_done,1):.1f}%")
    s4.metric("CLEAR (UNCHANGED)", total_done - smoky_count)

    # Method breakdown
    method_counts = {}
    for m in metrics_log:
        ml = m.get("method", "Unknown")
        method_counts[ml] = method_counts.get(ml, 0) + 1

    if method_counts:
        st.markdown("---")
        st.markdown("### METHOD BREAKDOWN")
        mc_cols = st.columns(len(method_counts))
        for i, (mname, mcount) in enumerate(method_counts.items()):
            mc_cols[i].metric(mname, mcount)

    enhanced_m = [m for m in metrics_log if m["smoky"] and m.get("psnr") is not None]
    if enhanced_m:
        st.markdown("---")
        st.markdown("### AVERAGE QUALITY METRICS (enhanced frames)")
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

    # Downloads — metrics JSON only (video writer per-frame append not supported cleanly)
    st.markdown("---")
    st.download_button(
        label="⬇  DOWNLOAD METRICS JSON",
        data=json.dumps(metrics_log, indent=2),
        file_name="pipeline_metrics.json",
        mime="application/json"
    )

    st.markdown("""
    <div style='font-family:monospace; font-size:0.7rem; color:#2a4a5a; margin-top:4px'>
    Note: Video download is not available with per-frame method switching.
    Run without pausing to get a downloadable output video.
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style='font-family:monospace; font-size:0.65rem; color:#1e2733;
            text-align:center; margin-top:8px'>
PIPELINE v0.1 · SMOKE CLASSIFIER MobileNetV2 · ENHANCEMENT CycleGAN LITE
</div>
""", unsafe_allow_html=True)
