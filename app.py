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
    div[data-testid="stVerticalBlock"] > div { margin-bottom: 0.15rem !important; }

    [data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid #1e2733;
    }
    [data-testid="stSidebar"] .block-container { padding-top: 0.75rem !important; }

    [data-testid="stMetric"] {
        background-color: #0d1a26;
        border: 1px solid #1a3a5c;
        border-radius: 6px;
        padding: 6px 10px !important;
    }
    [data-testid="stMetricLabel"] { color: #5a8a9f; font-size: 0.68rem; }
    [data-testid="stMetricValue"] { color: #00d4ff; font-weight: 600; font-size: 0.95rem; }

    h1 {
        color: #00d4ff; font-family: 'Courier New', monospace;
        font-size: 1.4rem; margin-bottom: 0 !important;
    }
    h2 {
        color: #5a8a9f; font-size: 0.9rem; font-weight: 400;
        letter-spacing: 0.1em; margin-top: 0 !important; margin-bottom: 0.2rem !important;
    }
    h3 {
        color: #00b4d8; font-size: 0.82rem;
        margin-bottom: 0.1rem !important; margin-top: 0.2rem !important;
    }

    .stProgress > div > div { background-color: #00d4ff; }
    .stProgress { margin-bottom: 0.1rem !important; margin-top: 0.1rem !important; }
    hr { border-color: #1e2733; margin: 0.3rem 0 !important; }

    .stButton > button {
        background-color: #0d3a5c; color: #00d4ff;
        border: 1px solid #1a6a9f; border-radius: 4px;
        font-family: monospace; width: 100%; padding: 0.35rem !important;
    }
    .stButton > button:hover { background-color: #1a5c8a; border-color: #00d4ff; }

    [data-testid="stFileUploader"] {
        border: 1px dashed #1e2733; border-radius: 6px; padding: 4px;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        font-family: monospace; font-size: 0.78rem;
        padding: 3px 10px; color: #5a8a9f;
    }
    .stTabs [aria-selected="true"] { color: #00d4ff !important; }

    /* Tighten selectbox */
    div[data-baseweb="select"] { margin-top: 0 !important; }
</style>
""", unsafe_allow_html=True)


# Enhancement methods

METHODS = ["CycleGAN Lite", "Dark Channel", "CLAHE", "Passing Untouched"]

METHOD_COLORS = {
    "CycleGAN Lite":     (105, 219, 105),
    "Dark Channel":      (255, 215,   0),
    "CLAHE":             (  0, 212, 255),
    "Passing Untouched": (107, 107, 255),
}

METHOD_HEX = {
    "CycleGAN Lite":     "#69db69",
    "Dark Channel":      "#ffd700",
    "CLAHE":             "#00d4ff",
    "Passing Untouched": "#6b6bff",
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


def enhance_frame(frame, method, generator, device):
    if method == "CycleGAN Lite":
        return desmoke_frame(generator, frame, device=device), "CycleGAN Lite"
    elif method == "Dark Channel":
        return apply_dark_channel(frame), "Dark Channel"
    elif method == "CLAHE":
        return apply_clahe(frame), "CLAHE"
    else:
        return frame.copy(), "Passing Untouched"


def burn_label(frame: np.ndarray, label: str) -> np.ndarray:
    out = frame.copy()
    color = METHOD_COLORS.get(label, (255, 255, 255))
    h = out.shape[0]
    cv2.rectangle(out, (0, h - 28), (len(label) * 11 + 16, h), (0, 0, 0), -1)
    cv2.putText(out, label, (8, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return out


# Helpers

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


# Session state

defaults = {
    "tfile_path":      None,
    "stop_flag":       False,
    "processing_done": False,
    "metrics_log":     [],
    "output_path":     None,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# Sidebar

with st.sidebar:
    st.markdown("## ⚙ CONFIG")
    st.markdown("---")

    confidence_thresh = st.slider(
        "Smoke Detection Threshold",
        min_value=0.50, max_value=0.99,
        value=0.80, step=0.01,
    )

    st.markdown("---")
    st.markdown("## 📊 METRIC GUIDE")
    st.markdown("""
    <div style='font-family:monospace; font-size:0.7rem; color:#5a8a9f; line-height:1.9'>
    PSNR &nbsp; >30 dB &nbsp; acceptable<br>
    PSNR &nbsp; >40 dB &nbsp; excellent<br>
    SSIM &nbsp; >0.90 &nbsp;&nbsp; good<br>
    ΔE &nbsp;&nbsp;&nbsp; &lt;2.0 &nbsp;&nbsp; imperceptible<br>
    ΔE &nbsp;&nbsp;&nbsp; &lt;5.0 &nbsp;&nbsp; clinical safe
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-family:monospace; font-size:0.67rem; color:#2a4a5a; line-height:1.8'>
    CLASSIFIER · MobileNetV2 · v0.1<br>
    val_acc = 89.7%<br>
    ENHANCER · CycleGAN Lite · v0.1<br>
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
        st.session_state.tfile_path      = tfile_path
        st.session_state.processing_done = False

with tab2:
    st.markdown("""
    <div style='font-family:monospace; font-size:0.7rem; color:#5a8a9f; margin-bottom:4px'>
    Google Drive share link · Direct video URL (Dropbox etc.)
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
                st.success("Video fetched successfully.")
            except Exception as e:
                st.error(f"Could not fetch video: {e}")
                tfile_path = None

if tfile_path is None:
    st.markdown("""
    <div style='text-align:center; padding:32px; color:#2a4a5a;
                border:1px dashed #1e2733; border-radius:8px; margin-top:10px'>
        <div style='font-size:1.6rem'>📹</div>
        <div style='font-family:monospace; font-size:0.82rem; margin-top:6px'>
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
    st.error("Could not read video. Ensure URL is a direct download link, not a preview page.")
    st.stop()

st.session_state.tfile_path = tfile_path

st.markdown("---")
vi1, vi2, vi3 = st.columns(3)
vi1.metric("TOTAL FRAMES", total_frames)
vi2.metric("FPS", f"{fps:.1f}")
vi3.metric("DURATION", frames_to_time(total_frames, fps))

# Clip range + method (side by side to save space)

st.markdown("---")
cfg1, cfg2, cfg3, cfg4 = st.columns([1, 1, 0.2, 1.8])

with cfg1:
    start_input = st.text_input(
        "Start (MM:SS)",
        value="0:00",
        help=f"Video duration: {frames_to_time(total_frames, fps)}"
    )
with cfg2:
    end_input = st.text_input(
        "End (MM:SS)",
        value=frames_to_time(min(int(fps * 10), total_frames), fps),
    )
with cfg3:
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)  # spacer
with cfg4:
    selected_method = st.selectbox(
        "Desmoking Method",
        options=METHODS,
        index=0,
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

# Run button

st.markdown("---")
if st.button("▶  RUN ENHANCEMENT PIPELINE"):
    st.session_state.stop_flag       = False
    st.session_state.processing_done = False
    st.session_state.metrics_log     = []
    run = True
else:
    run = False

if not run:
    st.stop()

# Load models

with st.spinner("Loading models..."):
    classifier, generator, device = load_models()

st.markdown(
    f'<div style="font-family:monospace; font-size:0.7rem; color:#5a8a9f; margin-bottom:2px">'
    f'DEVICE: {device.upper()} &nbsp;·&nbsp; METHOD: {selected_method}</div>',
    unsafe_allow_html=True
)

# Processing layout

progress_bar = st.progress(0)
status_text  = st.empty()

# Original | Output side by side
frame_cols = st.columns(2)
with frame_cols[0]:
    st.markdown("### ORIGINAL")
    orig_display = st.empty()
with frame_cols[1]:
    st.markdown("### OUTPUT")
    enh_display = st.empty()

# Metrics + method label directly below frames (no separate section header)
metric_row = st.columns(5)
m_time    = metric_row[0].empty()
m_method  = metric_row[1].empty()
m_psnr    = metric_row[2].empty()
m_ssim    = metric_row[3].empty()
m_delta_e = metric_row[4].empty()

# Stop button below output
st.markdown("<div style='margin-top:6px'></div>", unsafe_allow_html=True)
stop_col, _ = st.columns([1, 3])
with stop_col:
    if st.button("⏹  STOP PROCESSING"):
        st.session_state.stop_flag = True

st.markdown("---")
st.markdown("""
<div style='font-family:monospace; font-size:0.7rem; color:#2a4a5a'>FRAME LOG</div>
""", unsafe_allow_html=True)
log_container = st.empty()

# Processing loop

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

    if st.session_state.stop_flag:
        status_text.markdown(
            '<div style="font-family:monospace; font-size:0.72rem; color:#ff6b6b">'
            '⏹ STOPPED</div>',
            unsafe_allow_html=True
        )
        break

    ret, frame = cap.read()
    if not ret:
        break

    label, confidence = predict_frame(classifier, frame, device=device)
    is_smoky = (label == 1) and (confidence >= confidence_thresh)

    if is_smoky:
        smoky_count += 1
        enhanced, method_label = enhance_frame(frame, selected_method, generator, device)
        metrics = evaluate_frame(frame, enhanced)
        status  = "SMOKY"
    else:
        enhanced     = frame.copy()
        method_label = "Passing Untouched"
        metrics      = {"psnr": None, "ssim": None, "delta_e": None}
        status       = "CLEAR"
        time.sleep(0.04)

    output_frame = burn_label(enhanced, method_label)
    writer.write(output_frame)

    metrics_log.append({
        "frame":      idx,
        "time":       frames_to_time(idx, fps),
        "smoky":      is_smoky,
        "method":     method_label,
        "confidence": round(confidence, 4),
        **{k: round(v, 4) if v is not None else None for k, v in metrics.items()}
    })

    done_count = idx - start_frame + 1
    progress_bar.progress(done_count / frames_to_process)
    status_text.markdown(
        f'<div style="font-family:monospace; font-size:0.72rem; color:#5a8a9f">'
        f'{frames_to_time(idx, fps)} &nbsp;·&nbsp; '
        f'{done_count}/{frames_to_process} frames &nbsp;·&nbsp; '
        f'Smoky: {smoky_count}</div>',
        unsafe_allow_html=True
    )

    orig_display.image(cv2.cvtColor(frame,        cv2.COLOR_BGR2RGB), use_container_width=True)
    enh_display.image( cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB), use_container_width=True)

    # Metrics row inline — method label + quality scores together
    color_hex = METHOD_HEX.get(method_label, "#ffffff")
    m_time.metric("TIME", frames_to_time(idx, fps))
    m_method.metric("METHOD", method_label)
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
        f"{method_label:20s} | conf={confidence:.2f} | "
        f"PSNR={psnr_str:>8} | {de_str}"
    )
    log_container.code("\n".join(log_lines[-8:]), language=None)

cap.release()
writer.release()

st.session_state.metrics_log     = metrics_log
st.session_state.output_path     = output_tmp.name
st.session_state.processing_done = True

# Summary

st.markdown("---")
st.markdown("### SUMMARY")

total_done = len(metrics_log)
s1, s2, s3, s4 = st.columns(4)
s1.metric("PROCESSED",  total_done)
s2.metric("SMOKY",      smoky_count)
s3.metric("SMOKE %",    f"{100*smoky_count/max(total_done,1):.1f}%")
s4.metric("CLEAR",      total_done - smoky_count)

enhanced_m = [m for m in metrics_log if m["smoky"] and m.get("psnr") is not None]
if enhanced_m:
    st.markdown("---")
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
        f'<div style="font-family:monospace; font-size:0.78rem; '
        f'color:{de_color}; margin-top:2px">'
        f'COLOR FIDELITY: {de_status}</div>',
        unsafe_allow_html=True
    )

# Downloads

st.markdown("---")
dl1, dl2 = st.columns(2)
with dl1:
    with open(output_tmp.name, "rb") as f:
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
<div style='font-family:monospace; font-size:0.62rem; color:#1e2733;
            text-align:center; margin-top:6px'>
PIPELINE v0.1 · MobileNetV2 CLASSIFIER · CycleGAN LITE ENHANCER
</div>
""", unsafe_allow_html=True)
