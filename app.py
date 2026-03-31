"""
Laparoscopic Video Enhancement Pipeline — Streamlit Demo App
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import json
import torch
from PIL import Image
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
# CSS — dark medical monitor feel, tighter spacing
# -------------------------------------------------------------------

st.markdown("""
<style>
    /* Base */
    .stApp { background-color: #0a0e14; color: #c8d0e0; }

    /* Remove default top padding */
    .block-container { padding-top: 1rem !important; padding-bottom: 0.5rem !important; }

    /* Tighten element spacing */
    div[data-testid="stVerticalBlock"] > div { margin-bottom: 0.2rem !important; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid #1e2733;
    }
    [data-testid="stSidebar"] .block-container { padding-top: 0.75rem !important; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #0d1a26;
        border: 1px solid #1a3a5c;
        border-radius: 6px;
        padding: 8px 12px !important;
    }
    [data-testid="stMetricLabel"] { color: #5a8a9f; font-size: 0.7rem; }
    [data-testid="stMetricValue"] { color: #00d4ff; font-weight: 600; font-size: 1rem; }

    /* Headers — tighter margins */
    h1 {
        color: #00d4ff;
        font-family: 'Courier New', monospace;
        font-size: 1.4rem;
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }
    h2 {
        color: #5a8a9f;
        font-size: 0.95rem;
        font-weight: 400;
        letter-spacing: 0.1em;
        margin-top: 0 !important;
        margin-bottom: 0.3rem !important;
    }
    h3 {
        color: #00b4d8;
        font-size: 0.85rem;
        margin-bottom: 0.2rem !important;
        margin-top: 0.3rem !important;
    }

    /* Progress bar */
    .stProgress > div > div { background-color: #00d4ff; }
    .stProgress { margin-bottom: 0.2rem !important; }

    /* Divider */
    hr { border-color: #1e2733; margin: 0.4rem 0 !important; }

    /* Button */
    .stButton > button {
        background-color: #0d3a5c;
        color: #00d4ff;
        border: 1px solid #1a6a9f;
        border-radius: 4px;
        font-family: monospace;
        width: 100%;
        padding: 0.4rem !important;
    }
    .stButton > button:hover {
        background-color: #1a5c8a;
        border-color: #00d4ff;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        border: 1px dashed #1e2733;
        border-radius: 6px;
        padding: 4px;
    }

    /* Slider tighter */
    .stSlider { padding-top: 0 !important; padding-bottom: 0 !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        font-family: monospace;
        font-size: 0.8rem;
        padding: 4px 12px;
        color: #5a8a9f;
    }
    .stTabs [aria-selected="true"] { color: #00d4ff !important; }

    /* Image captions */
    .frame-label {
        color: #5a8a9f;
        font-family: 'Courier New', monospace;
        font-size: 0.7rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------------
# Model loading
# -------------------------------------------------------------------

@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = load_classifier("weights/smoke_classifier_finetuned.pth", device=device)
    generator  = load_generator("weights/G_hazy2clear_lite.pth", device=device)
    return classifier, generator, device


# -------------------------------------------------------------------
# URL helpers
# -------------------------------------------------------------------

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

tfile_path = None

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
fps          = cap.get(cv2.CAP_PROP_FPS) or 25
cap.release()

if total_frames < 2:
    st.error("Could not read video. If using a URL, ensure it is a direct download link.")
    st.stop()

st.markdown("---")

# Video info row
vi1, vi2, vi3 = st.columns(3)
vi1.metric("TOTAL FRAMES", total_frames)
vi2.metric("FPS", f"{fps:.1f}")
vi3.metric("DURATION", f"{total_frames/fps:.1f}s")

# -------------------------------------------------------------------
# Timeline scrubber
# -------------------------------------------------------------------

st.markdown("#### SELECT FRAME RANGE")

scrub = st.slider(
    "Timeline",
    min_value=0,
    max_value=total_frames,
    value=(0, min(300, total_frames)),
    step=1,
    format="Frame %d",
    label_visibility="collapsed"
)

start_frame, end_frame = scrub

if end_frame <= start_frame:
    st.error("End frame must be greater than start frame.")
    st.stop()

frames_to_process = end_frame - start_frame

sc1, sc2, sc3 = st.columns(3)
sc1.metric("START", start_frame)
sc2.metric("END",   end_frame)
sc3.metric("TO PROCESS", frames_to_process)

st.markdown("---")

# -------------------------------------------------------------------
# Run button
# -------------------------------------------------------------------

run = st.button("▶  RUN ENHANCEMENT PIPELINE")

if not run:
    st.stop()

# -------------------------------------------------------------------
# Load models
# -------------------------------------------------------------------

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

# -------------------------------------------------------------------
# Live processing layout
# -------------------------------------------------------------------

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

# -------------------------------------------------------------------
# Processing
# -------------------------------------------------------------------

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
        smoky_count += 1
        enhanced     = desmoke_frame(generator, frame, device=device)
        metrics      = evaluate_frame(frame, enhanced)
        output_frame = enhanced
        status       = "SMOKY"
    else:
        enhanced     = frame.copy()
        metrics      = {"psnr": None, "ssim": None, "delta_e": None}
        output_frame = frame
        status       = "CLEAR"

    writer.write(output_frame)
    metrics_log.append({
        "frame": idx, "smoky": is_smoky,
        "confidence": round(confidence, 4),
        **{k: round(v, 4) if v is not None else None for k, v in metrics.items()}
    })

    # Progress
    progress_bar.progress((idx - start_frame + 1) / frames_to_process)
    status_text.markdown(
        f'<div style="font-family:monospace; font-size:0.75rem; color:#5a8a9f">'
        f'FRAME {idx - start_frame + 1}/{frames_to_process} &nbsp;|&nbsp; '
        f'SMOKY: {smoky_count}</div>',
        unsafe_allow_html=True
    )

    # Frame display
    orig_display.image(cv2.cvtColor(frame,    cv2.COLOR_BGR2RGB), use_container_width=True)
    enh_display.image( cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB), use_container_width=True)

    # Output label indicator
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

    # Metrics
    m_frame.metric("FRAME", f"{idx - start_frame + 1}/{frames_to_process}")
    if is_smoky and metrics["psnr"] is not None:
        m_psnr.metric("PSNR",  f"{metrics['psnr']:.1f} dB")
        m_ssim.metric("SSIM",  f"{metrics['ssim']:.3f}")
        m_delta_e.metric("ΔE", f"{metrics['delta_e']:.2f}")
    else:
        m_psnr.metric("PSNR",  "—")
        m_ssim.metric("SSIM",  "—")
        m_delta_e.metric("ΔE", "—")

    # Log
    psnr_str = f"{metrics['psnr']:.1f}dB" if metrics["psnr"] else "  —  "
    de_str   = f"ΔE={metrics['delta_e']:.2f}" if metrics["delta_e"] else "ΔE= —"
    log_lines.append(
        f"[{idx+1:05d}] {status:5s} | conf={confidence:.2f} | "
        f"PSNR={psnr_str:>8} | {de_str}"
    )
    log_container.code("\n".join(log_lines[-10:]), language=None)

cap.release()
writer.release()

# -------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------

st.markdown("---")
st.markdown("### PIPELINE SUMMARY")

s1, s2, s3, s4 = st.columns(4)
s1.metric("FRAMES PROCESSED",  frames_to_process)
s2.metric("SMOKY DETECTED",    smoky_count)
s3.metric("SMOKE %",           f"{100*smoky_count/frames_to_process:.1f}%")
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
        de_status, de_color = "✓ IMPERCEPTIBLE — clinically safe",         "#69db69"
    elif avg_delta_e < 5:
        de_status, de_color = "⚠ ACCEPTABLE — within clinical tolerance",  "#ffd700"
    else:
        de_status, de_color = "✗ NOTICEABLE — review enhancement output",  "#ff6b6b"

    st.markdown(
        f'<div style="font-family:monospace; font-size:0.8rem; color:{de_color}; margin-top:4px">'
        f'COLOR FIDELITY: {de_status}</div>',
        unsafe_allow_html=True
    )

# -------------------------------------------------------------------
# Downloads
# -------------------------------------------------------------------

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
<div style='font-family:monospace; font-size:0.65rem; color:#1e2733;
            text-align:center; margin-top:8px'>
PIPELINE v0.1 · SMOKE CLASSIFIER MobileNetV2 · ENHANCEMENT CycleGAN LITE
</div>
""", unsafe_allow_html=True)

# Cleanup
try:
    os.unlink(tfile_path)
    os.unlink(output_tmp.name)
except Exception:
    pass
