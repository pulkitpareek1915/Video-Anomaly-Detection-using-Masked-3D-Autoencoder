# Video Anomaly Detection using Masked 3D Autoencoder

An unsupervised video anomaly detection system built in PyTorch. A 3D convolutional autoencoder is trained **only on normal footage** with masked spatiotemporal pretraining. At test time, frames the model cannot reconstruct well are flagged as anomalous, and per-pixel reconstruction error maps show *where* the anomaly is, with no extra components or labels.

An interactive Streamlit app lets you upload a video and inspect anomaly scores and heatmaps frame by frame.

---

## Highlights

- **Unsupervised:** trained on normal clips only; no anomaly labels needed.
- **Masked autoencoding:** 50% of spatiotemporal patches are zeroed during training, so the model must learn normal motion patterns to rebuild the clip.
- **Explainable output:** per-pixel squared-error heatmaps localize the anomaly at zero extra parameters.
- **Controlled ablation:** masking improves frame-level ROC-AUC from 73.3% to 76.2% with identical architecture, optimizer and hyperparameters.
- **Deployed demo:** Streamlit app with score timeline, heatmap grid and frame inspector.

---

## Tech Stack

| Area | Tools |
| --- | --- |
| Language | Python |
| Deep learning | PyTorch, 3D Convolutional Autoencoder, Transposed Convolutions, Batch Normalization, LeakyReLU |
| Learning methods | Unsupervised learning, masked autoencoding (self-supervised pretraining) |
| Training | AdamW, warmup + cosine LR decay, gradient clipping, custom weighted loss |
| Video / signal processing | OpenCV, NumPy, SciPy (Gaussian smoothing) |
| Evaluation | scikit-learn (ROC-AUC), Matplotlib |
| Demo | Streamlit |
| Compute | Google Colab (T4 GPU, about 40 minutes for 40 epochs) |

---

## Pipeline

```
Video  ->  frames  ->  grayscale 128x128  ->  16-frame clips (sliding window, stride 4)
   ->  3D conv encoder  ->  latent  ->  3D transposed-conv decoder  ->  reconstructed clip
   ->  per-pixel squared error  ->  Gaussian smoothing  ->  top-5% mean per frame
   ->  average over overlapping clips  ->  temporal smoothing (sigma = 3)
   ->  frame anomaly score (0-1)  +  JET heatmap overlay
```

### 1. Data preparation
- Dataset: **UCSD Ped2**. Frames are converted to grayscale, resized to 128x128 and normalized.
- 16-frame sliding-window clips (stride 4) with temporal jitter and spatial augmentation during training.

### 2. Model
- 3D convolutional autoencoder with about **6.27M parameters**.
- Encoder: stacked `Conv3d` blocks with `BatchNorm3d` and `LeakyReLU(0.2)` that downsample space and time (channels 32, 64, 128, 256).
- Decoder: `ConvTranspose3d` blocks that rebuild the full 16x128x128 clip.

### 3. Masked pretraining
- 8x8 spatial patches (2 frames deep) are randomly masked at a 50% ratio during training only. No masking at test time.
- Custom weighted loss: masked-patch reconstruction (0.6), temporal consistency (0.3) and visible-patch reconstruction (0.1).
- AdamW (lr 2e-4, weight decay 1e-4), 10-epoch warmup then cosine decay, gradient clipping at 0.5, batch size 8, 40 epochs. The best-loss checkpoint is saved.

### 4. Anomaly scoring
- Per-pixel squared error is smoothed spatially (Gaussian), and each frame's score is the mean of its top 5% error pixels.
- Scores are averaged across overlapping clips, smoothed over time (sigma = 3) and normalized to [0, 1].

### 5. Explainability
- The same error maps are overlaid on the original frame with a JET colormap to show where the model failed to reconstruct.

---

## Results

Frame-level ROC-AUC on **UCSD Ped2**:

| Method | AUC |
| --- | --- |
| Ours, 0% masking (baseline) | 73.3% |
| **Ours, 50% masking (proposed)** | **76.2%** |

Masking contributes **+2.9% AUC** under an otherwise identical setup.

For context, published methods report higher scores on Ped2 (Hasan et al., CVPR 2016: 90.0%; Liu et al., CVPR 2018: 95.4%; AED-MAE, CVPR 2024: 99.3%). This project is a lightweight, from-scratch implementation focused on the masking ablation and explainability, not a state-of-the-art result.

---

## Streamlit Demo

Upload an `.mp4`, `.avi` or `.mov` clip. The app shows:
- Frame-level anomaly score timeline with an adjustable threshold
- Per-frame reconstruction-error heatmaps
- Interactive frame inspector (original vs heatmap overlay)

Sidebar controls include the anomaly threshold, top-k error percentage, number of heatmap frames and checkpoint path.

---

## Repository Structure

```
Video-Anomaly-Detection/
├── app.py                          # Streamlit demo app
├── notebooks/
│   └── Main.ipynb                  # Full training and evaluation notebook (Colab)
├── requirements.txt
├── checkpoints/
│   └── ucsd_masked_v3_best.pth     # Pretrained checkpoint
├── samples/                        # Sample test videos
└── Figs/                           # Result figures (ROC, heatmaps, ablation, loss curve)
```

---

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app loads `checkpoints/ucsd_masked_v3_best.pth` by default. To retrain, open `notebooks/Main.ipynb` in Google Colab with a GPU runtime and download UCSD Ped2 from the link below.

Dataset: [UCSD Ped2](http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm)

---

## Limitations and Future Work

- Evaluated on UCSD Ped2 only; a ShanghaiTech data loader is included in the notebook, but no ShanghaiTech results are reported.
- AUC is well below recent published methods. Likely improvements: a larger or transformer-based encoder, longer training, color input and multi-scale scoring.
- Frame-level scoring only; no object-level localization or tracking.

---

## Author

[Pulkit Pareek], The LNM Institute of Information Technology, Jaipur
