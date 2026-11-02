"""
Unsupervised Traffic Video Anomaly Detection
Masked Spatiotemporal Autoencoder — Streamlit Demo App
"""

import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
import tempfile
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter, gaussian_filter1d

#Constants
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 128
CLIP_LEN = 16
CKPT     = "ucsd_masked_v3_best.pth"

#Model Definition (must exactly match training)

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            # Block 1: (B,1,16,128,128) → (B,32,16,64,64)
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, stride=(1,2,2), padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True),
            # Block 2: → (B,64,8,32,32)
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # Block 3: → (B,128,4,16,16)
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # Block 4: → (B,256,2,8,8)
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.enc(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dec = nn.Sequential(
            # Block 1: (B,256,2,8,8) → (B,128,4,16,16)
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # Block 2: → (B,64,8,32,32)
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # Block 3: → (B,32,16,64,64)
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True),
            # Block 4: → (B,1,16,128,128)
            nn.ConvTranspose3d(32, 1, kernel_size=(1,4,4),
                               stride=(1,2,2), padding=(0,1,1)),
            nn.Tanh()
        )

    def forward(self, x):
        return self.dec(x)


class ConvAE3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        return self.decoder(self.encoder(x))


#Helper functions

def preprocess_frame(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    return (gray.astype(np.float32) / 255.0 - 0.5) / 0.5, gray


def extract_frames(video_path, max_frames=200):
    cap = cv2.VideoCapture(video_path)
    norm_frames, raw_frames = [], []
    while len(norm_frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        norm, raw = preprocess_frame(frame)
        norm_frames.append(norm)
        raw_frames.append(raw)
    cap.release()
    return norm_frames, raw_frames


@st.cache_resource(show_spinner="Loading model…")
def load_model(ckpt_path):
    model = ConvAE3D().to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    epoch = ckpt.get("epoch", "?")
    loss  = ckpt.get("loss",  float("nan"))
    return model, epoch, loss


def score_clip(model, norm_frames, stride=4, top_k_pct=5,
               spatial_sigma=2, temporal_sigma=3):
    """
    Slide a window of CLIP_LEN over norm_frames.
    Returns per-frame anomaly scores and per-frame error maps.
    """
    n = len(norm_frames)
    frame_scores = np.zeros(n, dtype=np.float32)
    frame_counts = np.zeros(n, dtype=np.float32)
    frame_errmaps = [None] * n  # will store last written error map per frame

    clips, starts = [], []
    for s in range(0, n - CLIP_LEN + 1, stride):
        clips.append(np.stack(norm_frames[s:s + CLIP_LEN]))
        starts.append(s)

    if not clips:
        return frame_scores, frame_errmaps

    with torch.no_grad():
        for i in range(0, len(clips), 8):
            batch_np = np.stack(clips[i:i + 8])
            batch = torch.tensor(batch_np, dtype=torch.float32).unsqueeze(1).to(DEVICE)
            recon = model(batch)
            err   = (recon - batch).pow(2).squeeze(1).cpu().numpy()  # (B,T,H,W)

            for j, s in enumerate(starts[i:i + 8]):
                if j >= err.shape[0]:
                    break
                for t in range(CLIP_LEN):
                    emap = gaussian_filter(err[j, t], sigma=spatial_sigma)
                    k    = max(1, int(top_k_pct / 100.0 * emap.size))
                    sc   = np.partition(emap.ravel(), -k)[-k:].mean()
                    frame_scores[s + t] += sc
                    frame_counts[s + t] += 1
                    frame_errmaps[s + t] = emap

    valid = frame_counts > 0
    frame_scores[valid] /= frame_counts[valid]
    frame_scores = gaussian_filter1d(frame_scores, sigma=temporal_sigma)

    # Normalise to [0, 1]
    mn, mx = frame_scores.min(), frame_scores.max()
    frame_scores = (frame_scores - mn) / (mx - mn + 1e-8)

    return frame_scores, frame_errmaps


def make_heatmap_overlay(raw_gray, errmap, alpha=0.45):
    if errmap is None:
        return cv2.cvtColor(raw_gray, cv2.COLOR_GRAY2RGB)
    h_norm = (errmap - errmap.min()) / (errmap.max() - errmap.min() + 1e-8)
    heat   = cv2.applyColorMap((h_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    orig   = cv2.cvtColor(raw_gray, cv2.COLOR_GRAY2RGB)
    return cv2.addWeighted(orig, 1 - alpha, heat, alpha, 0)


#Page config
st.set_page_config(
    page_title="Traffic Anomaly Detector",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 Unsupervised Traffic Video Anomaly Detection")
st.markdown(
    "**Masked Spatiotemporal Autoencoder** — trained only on normal footage. "
    "High reconstruction error signals an anomaly. "
    "Per-pixel error maps provide intrinsic spatial explainability."
)

#Sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    ckpt_path = st.text_input(
        "Checkpoint path",
        value=CKPT,
        help="Absolute path to the .pth checkpoint on your Colab/Drive"
    )

    threshold = st.slider(
        "Anomaly threshold", 0.0, 1.0, 0.50, 0.01,
        help="Frames above this score are flagged as anomalies"
    )
    stride = st.selectbox(
        "Inference stride (frames)", [2, 4, 8], index=1,
        help="Smaller = more accurate but slower"
    )
    top_k_pct = st.slider(
        "Top-k% scoring (%)", 1, 20, 5,
        help="Percentage of highest-error pixels used for frame score"
    )
    show_heatmaps = st.checkbox("Show per-frame heatmaps", value=True)
    max_heatmap_frames = st.slider(
        "Max heatmap frames shown", 4, 32, 16,
        help="How many frames to display in the heatmap grid"
    ) if show_heatmaps else 16

    st.markdown("---")
    st.markdown("**Model:** ConvAE3D (6.27M params)  \n**Input:** 16 × 128 × 128  \n**Masking:** 50% spatiotemporal patches")

#Load model
if not Path(ckpt_path).exists():
    st.error(
        f" Checkpoint not found at `{ckpt_path}`.  \n"
        "Update the path in the sidebar or make sure Google Drive is mounted."
    )
    st.stop()

model, ckpt_epoch, ckpt_loss = load_model(ckpt_path)
st.sidebar.success(f" Model loaded  \nEpoch {ckpt_epoch} | Loss {ckpt_loss:.6f}  \nDevice: **{DEVICE}**")

#Upload 
uploaded = st.file_uploader(
    "Upload a video clip (.mp4 / .avi / .mov)",
    type=["mp4", "avi", "mov"]
)

if not uploaded:
    st.info(" Upload a video above to begin inference.")
    st.stop()

# Extract frames
with tempfile.NamedTemporaryFile(suffix=Path(uploaded.name).suffix, delete=False) as f:
    f.write(uploaded.read())
    video_path = f.name

with st.spinner("Extracting frames…"):
    norm_frames, raw_frames = extract_frames(video_path)

n_frames = len(norm_frames)
st.write(f"📽️ **{uploaded.name}** — {n_frames} frames extracted")

if n_frames < CLIP_LEN:
    st.error(f"Need at least {CLIP_LEN} frames. Got {n_frames}. Upload a longer clip.")
    st.stop()

# Run inference
with st.spinner(f"Running inference on {n_frames} frames (stride={stride})…"):
    frame_scores, frame_errmaps = score_clip(
        model, norm_frames,
        stride=stride,
        top_k_pct=top_k_pct,
        spatial_sigma=2,
        temporal_sigma=3
    )

n_anomaly = int((frame_scores > threshold).sum())
pct_anomaly = 100 * n_anomaly / n_frames

# Summary metrics
st.markdown("---")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total frames",    n_frames)
c2.metric("Anomaly frames",  n_anomaly)
c3.metric("Anomaly %",       f"{pct_anomaly:.1f}%")
c4.metric("Peak score",      f"{frame_scores.max():.3f}")

#  Score timeline 
st.subheader("📊 Frame-level anomaly score timeline")
fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(frame_scores, color="#E67E22", lw=1.8, label="Anomaly score")
ax.axhline(threshold, color="red", lw=1.5, ls="--", label=f"Threshold = {threshold:.2f}")
ax.fill_between(range(n_frames), frame_scores, threshold,
                where=frame_scores > threshold,
                alpha=0.3, color="red", label="Anomalous region")
ax.set_xlabel("Frame index")
ax.set_ylabel("Normalised anomaly score")
ax.set_xlim(0, n_frames - 1)
ax.set_ylim(-0.05, 1.05)
ax.legend(fontsize=9)
ax.grid(alpha=0.25)
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# Heatmap grid
if show_heatmaps:
    st.subheader("🔥 Per-frame spatial heatmaps (reconstruction error)")
    st.caption(
        "Blue = low error (normal). Yellow/Red = high error (anomalous region). "
        "Heatmap overlaid on original grayscale frame."
    )

    # Pick evenly spaced frames, bias toward high-score frames
    display_indices = np.linspace(0, n_frames - 1, max_heatmap_frames, dtype=int)
    n_cols = 8
    rows = [display_indices[i:i + n_cols] for i in range(0, len(display_indices), n_cols)]

    for row_indices in rows:
        cols = st.columns(len(row_indices))
        for col, fi in zip(cols, row_indices):
            overlay = make_heatmap_overlay(raw_frames[fi], frame_errmaps[fi])
            sc = frame_scores[fi]
            label = f"⚠ f{fi}" if sc > threshold else f"✓ f{fi}"
            col.image(
                overlay[..., ::-1],
                caption=f"{label}\n{sc:.3f}",
                use_container_width=True
            )

# Frame-by-frame inspector
st.markdown("---")
st.subheader("🔍 Frame inspector")
fi = st.slider("Select frame", 0, n_frames - 1, 0)

col_orig, col_heat, col_info = st.columns([1, 1, 1])

with col_orig:
    st.markdown("**Original frame**")
    st.image(raw_frames[fi], clamp=True, use_container_width=True)

with col_heat:
    st.markdown("**Error heatmap overlay**")
    overlay = make_heatmap_overlay(raw_frames[fi], frame_errmaps[fi])
    st.image(overlay[..., ::-1], use_container_width=True)

with col_info:
    sc = frame_scores[fi]
    status = "🔴 ANOMALY" if sc > threshold else "🟢 Normal"
    st.markdown("**Diagnosis**")
    st.markdown(f"### {status}")
    st.metric("Anomaly score", f"{sc:.4f}")
    st.metric("Threshold", f"{threshold:.2f}")
    st.progress(float(sc))
    if frame_errmaps[fi] is not None:
        em = frame_errmaps[fi]
        st.caption(
            f"Error map — min: {em.min():.4f} | "
            f"max: {em.max():.4f} | "
            f"mean: {em.mean():.4f}"
        )
