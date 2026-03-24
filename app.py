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

# Page config — must be first Streamlit call

st.set_page_config(
    page_title="Laparoscopic Desmoking v0.1",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Custom CSS — dark medical monitor feel


st.markdown("""
<style>
    /* Base */
    .stApp {
        background-color: #0a0e14;
        color: #c8d0e0;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid #1e2733;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #0d1a26;
        border: 1px solid #1a3a5c;
        border-radius: 6px;
        padding: 12px;
    }
    [data-testid="stMetricLabel"] { color: #5a8a9f; font-size: 0.75rem; }
    [data-testid="stMetricValue"] { color: #00d4ff; font-weight: 600; }

    /* Headers */
    h1 { color: #00d4ff; font-family: 'Courier New', monospace; font-size: 1.6rem; }
    h2 { color: #5a8a9f; font-size: 1.1rem; font-weight: 400; letter-spacing: 0.1em; }
    h3 { color: #00b4d8; font-size: 0.95rem; }

    /* Status badges */
    .badge-smoky {
        background: #3d1a1a;
        border: 1px solid #8b2020;
        color: #ff6b6b;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-family: monospace;
    }
    .badge-clear {
        background: #1a3d1a;
        border: 1px solid #2d7a2d;
        color: #69db69;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-family: monospace;
    }

    /* Frame label */
    .frame-label {
        color: #5a8a9f;
        font-family: 'Courier New', monospace;
        font-size: 0.7rem;
        text-align: center;
        margin-top: 4px;
    }

    /* Progress bar */
    .stProgress > div > div { background-color: #00d4ff; }

    /* Divider */
    hr { border-color: #1e2733; }

    /* Button */
    .stButton > button {
        background-color: #0d3a5c;
        color: #00d4ff;
        border: 1px solid #1a6a9f;
        border-radius: 4px;
        font-family: monospace;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #1a5c8a;
        border-color: #00d4ff;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        border: 1px dashed #1e2733;
        border-radius: 6px;
        padding: 8px;
    }

    /* Chart */
    .stPlotlyChart { border: 1px solid #1e2733; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)



# Model loading — cached so they only load once


@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = load_classifier("weights/smoke_classifier_finetuned.pth", device=device)
    generator  = load_generator("weights/G_hazy2clear_lite.pth", device=device)
    return classifier, generator, device


def resolve_video_url(url: str) -> str:
    """
    Converts a Google Drive share link to a direct download URL.
    Passes through direct video URLs unchanged.
    """
    # Google Drive share link pattern
    gdrive_match = re.search(r"drive\.google\.com/file/d/([a-zA-Z0-9_-]+)", url)
    if gdrive_match:
        file_id = gdrive_match.group(1)
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    return url  # assume direct URL otherwise


def download_video_from_url(url: str) -> str:
    """
    Downloads video from URL to a temp file.
    Returns path to temp file.
    """
    resolved = resolve_video_url(url)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    with requests.get(resolved, stream=True, timeout=30) as r:
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=8192):
            tmp.write(chunk)
    tmp.flush()
    return tmp.name

# Sidebar


with st.sidebar:
    st.markdown("## ⚙ PIPELINE CONFIG")
    st.markdown("---")

    confidence_thresh = st.slider(
        "Detection Confidence Threshold",
        min_value=0.50, max_value=0.99,
        value=0.80, step=0.01,
        help="Frames below this confidence are passed through unchanged"
    )

    max_frames = st.number_input(
        "Max Frames to Process (0 = all)",
        min_value=0, max_value=5000,
        value=300, step=50,
        help="Limit frames for faster demo. Set 0 for full video."
    )

    st.markdown("---")
    st.markdown("## 📊 METRIC THRESHOLDS")

    st.markdown("""
    <div style='font-family:monospace; font-size:0.75rem; color:#5a8a9f; line-height:2'>
    PSNR &nbsp;&nbsp; >30 dB &nbsp;&nbsp; acceptable<br>
    PSNR &nbsp;&nbsp; >40 dB &nbsp;&nbsp; excellent<br>
    SSIM &nbsp;&nbsp; >0.90 &nbsp;&nbsp;&nbsp; good<br>
    ΔE &nbsp;&nbsp;&nbsp;&nbsp; &lt;2.0 &nbsp;&nbsp;&nbsp; imperceptible<br>
    ΔE &nbsp;&nbsp;&nbsp;&nbsp; &lt;5.0 &nbsp;&nbsp;&nbsp; clinical safe
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-family:monospace; font-size:0.7rem; color:#2a4a5a'>
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
    <div style='font-family:monospace; font-size:0.75rem; color:#5a8a9f; margin-bottom:8px'>
    ACCEPTED SOURCES: Google Drive share link · Direct video file URL (Dropbox etc.)
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
    <div style='text-align:center; padding:60px; color:#2a4a5a;
                border:1px dashed #1e2733; border-radius:8px; margin-top:20px;'>
        <div style='font-size:2rem'>📹</div>
        <div style='font-family:monospace; margin-top:12px'>
            AWAITING VIDEO INPUT
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# Save uploaded file to temp

cap = cv2.VideoCapture(tfile_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps          = cap.get(cv2.CAP_PROP_FPS) or 25
cap.release()

frames_to_process = total_frames if max_frames == 0 else min(max_frames, total_frames)

col1, col2, col3 = st.columns(3)
col1.metric("TOTAL FRAMES", total_frames)
col2.metric("FPS", f"{fps:.1f}")
col3.metric("DURATION", f"{total_frames/fps:.1f}s")

st.markdown("---")
st.markdown("### FRAME RANGE")

range_col1, range_col2 = st.columns(2)
with range_col1:
    start_frame = st.number_input(
        "Start Frame", min_value=0,
        max_value=total_frames - 1, value=0, step=1
    )
with range_col2:
    end_frame = st.number_input(
        "End Frame", min_value=1,
        max_value=total_frames,
        value=min(300, total_frames), step=1
    )

if end_frame <= start_frame:
    st.error("End frame must be greater than start frame.")
    st.stop()

frames_to_process = end_frame - start_frame

r1, r2, r3 = st.columns(3)
r1.metric("START FRAME", start_frame)
r2.metric("END FRAME",   end_frame)
r3.metric("FRAMES TO PROCESS", frames_to_process)

st.markdown("---")


# Run pipeline button


run = st.button("▶  RUN ENHANCEMENT PIPELINE")

if not run:
    st.stop()


# Load models


with st.spinner("Loading models..."):
    classifier, generator, device = load_models()

st.markdown(f"""
<div style='font-family:monospace; font-size:0.75rem; color:#5a8a9f'>
DEVICE: {device.upper()} &nbsp;|&nbsp;
CLASSIFIER: smoke_classifier_finetuned.pth &nbsp;|&nbsp;
GENERATOR: G_hazy2clear_lite.pth
</div>
""", unsafe_allow_html=True)
st.markdown("---")


# Live processing


# Layout
progress_bar  = st.progress(0)
status_text   = st.empty()
frame_cols    = st.columns(2)

with frame_cols[0]:
    st.markdown("### ORIGINAL")
    orig_display = st.empty()

with frame_cols[1]:
    st.markdown("### ENHANCED")
    enh_display = st.empty()

st.markdown("---")
st.markdown("### LIVE METRICS")
metric_cols = st.columns(4)
m_frame   = metric_cols[0].empty()
m_psnr    = metric_cols[1].empty()
m_ssim    = metric_cols[2].empty()
m_delta_e = metric_cols[3].empty()

st.markdown("---")
st.markdown("### FRAME LOG")
log_container = st.empty()

# Processing state
cap = cv2.VideoCapture(tfile_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

processed_frames = []
metrics_log      = []
log_lines        = []
smoky_count      = 0

output_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter(output_tmp.name, fourcc, fps, (w, h))

for idx in range(start_frame, end_frame):
    ret, frame = cap.read()
    if not ret:
        break

    # Detect
    label, confidence = predict_frame(classifier, frame, device=device)
    is_smoky = (label == 1) and (confidence >= confidence_thresh)

    if is_smoky:
        smoky_count += 1
        enhanced = desmoke_frame(generator, frame, device=device)
        metrics  = evaluate_frame(frame, enhanced)
        output_frame = enhanced
        status = "SMOKY"
        badge  = f'<span class="badge-smoky">⚠ SMOKE</span>'
    else:
        enhanced     = frame.copy()
        metrics      = {"psnr": None, "ssim": None, "delta_e": None}
        output_frame = frame
        status = "CLEAR"
        badge  = f'<span class="badge-clear">✓ CLEAR</span>'

    writer.write(output_frame)
    metrics_log.append({
        "frame": idx, "smoky": is_smoky,
        "confidence": round(confidence, 4),
        **{k: round(v, 4) if v is not None else None for k, v in metrics.items()}
    })

    # Update UI every frame
    progress_bar.progress((idx + 1) / frames_to_process)
    status_text.markdown(
        f'<div style="font-family:monospace; font-size:0.8rem; color:#5a8a9f">'
        f'PROCESSING FRAME {idx+1}/{frames_to_process} &nbsp;|&nbsp; '
        f'SMOKY DETECTED: {smoky_count}</div>',
        unsafe_allow_html=True
    )

    # Frame display
    orig_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    enh_rgb  = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    orig_display.image(orig_rgb, use_container_width=True)
    enh_display.image(enh_rgb,  use_container_width=True)

    # Metrics display
    m_frame.metric("FRAME", f"{idx+1}/{frames_to_process}")
    if is_smoky and metrics["psnr"] is not None:
        m_psnr.metric("PSNR",    f"{metrics['psnr']:.1f} dB")
        m_ssim.metric("SSIM",    f"{metrics['ssim']:.3f}")
        m_delta_e.metric("ΔE",   f"{metrics['delta_e']:.2f}")
    else:
        m_psnr.metric("PSNR",    "—")
        m_ssim.metric("SSIM",    "—")
        m_delta_e.metric("ΔE",   "—")

    # Log (last 10 lines)
    psnr_str = f"{metrics['psnr']:.1f}dB" if metrics["psnr"] else "  —  "
    de_str   = f"ΔE={metrics['delta_e']:.2f}" if metrics["delta_e"] else "ΔE= —"
    log_lines.append(
        f"[{idx+1:05d}] {status:5s} | conf={confidence:.2f} | "
        f"PSNR={psnr_str:>8} | {de_str}"
    )
    log_container.code("\n".join(log_lines[-10:]), language=None)

cap.release()
writer.release()


# Summary


st.markdown("---")
st.markdown("### PIPELINE SUMMARY")

enhanced_m = [m for m in metrics_log if m["smoky"] and m["psnr"] is not None]

s1, s2, s3, s4 = st.columns(4)
s1.metric("FRAMES PROCESSED",  frames_to_process)
s2.metric("SMOKY DETECTED",    smoky_count)
s3.metric("SMOKE %",           f"{100*smoky_count/frames_to_process:.1f}%")
s4.metric("CLEAR (UNCHANGED)", frames_to_process - smoky_count)

if enhanced_m:
    st.markdown("---")
    st.markdown("### AVERAGE QUALITY METRICS (enhanced frames only)")
    a1, a2, a3 = st.columns(3)
    avg_psnr    = np.mean([m["psnr"]    for m in enhanced_m])
    avg_ssim    = np.mean([m["ssim"]    for m in enhanced_m])
    avg_delta_e = np.mean([m["delta_e"] for m in enhanced_m])
    a1.metric("AVG PSNR",    f"{avg_psnr:.2f} dB")
    a2.metric("AVG SSIM",    f"{avg_ssim:.4f}")
    a3.metric("AVG DELTA-E", f"{avg_delta_e:.2f}")

    # Delta-E clinical interpretation
    if avg_delta_e < 2:
        de_status = "✓ IMPERCEPTIBLE — clinically safe"
        de_color  = "#69db69"
    elif avg_delta_e < 5:
        de_status = "⚠ ACCEPTABLE — within clinical tolerance"
        de_color  = "#ffd700"
    else:
        de_status = "✗ NOTICEABLE — review enhancement output"
        de_color  = "#ff6b6b"

    st.markdown(
        f'<div style="font-family:monospace; font-size:0.85rem; '
        f'color:{de_color}; margin-top:8px">COLOR FIDELITY: {de_status}</div>',
        unsafe_allow_html=True
    )


# Download


st.markdown("---")
with open(output_tmp.name, "rb") as f:
    st.download_button(
        label="⬇  DOWNLOAD ENHANCED VIDEO",
        data=f,
        file_name="enhanced_output.mp4",
        mime="video/mp4"
    )

metrics_json = json.dumps(metrics_log, indent=2)
st.download_button(
    label="⬇  DOWNLOAD METRICS JSON",
    data=metrics_json,
    file_name="pipeline_metrics.json",
    mime="application/json"
)

st.markdown("---")
st.markdown("""
<div style='font-family:monospace; font-size:0.7rem; color:#1e2733; text-align:center'>
PIPELINE v0.1 · SMOKE CLASSIFIER MobileNetV2 · ENHANCEMENT CycleGAN LITE
</div>
""", unsafe_allow_html=True)

# Cleanup temp files
try:
    os.unlink(tfile_path)
    os.unlink(output_tmp.name)
except Exception:
    pass