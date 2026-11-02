# Unsupervised Video Anomaly Detection
### Masked Spatiotemporal 3D Convolutional Autoencoder · PyTorch · Streamlit

A lightweight unsupervised video anomaly detection system trained only on normal footage. Frames the model fails to reconstruct well are flagged as anomalous. Per-pixel reconstruction error maps provide spatial explainability with no extra components.

---

## Results

| Method                            | UCSD Ped2 AUC |
|-----------------------------------|---------------|
| Hasan et al. (CVPR 2016)          | 90.0% |
| Liu et al. (CVPR 2018)            | 95.4% |
| AED-MAE (CVPR 2024)               | 99.3% |
| Ours — 0% masking (baseline)      | 73.3% |
| **Ours — 50% masking (proposed)** | **76.2%**|

---

## How it works

- **Architecture** — 3D Convolutional Autoencoder (~6.27M parameters). Input: 16-frame grayscale clip at 128×128.
- **Masked pretraining** — During training, 50% of spatiotemporal patches are randomly zeroed. The model reconstructs the full clip from corrupted input, building a strong prior over normal motion. No masking at test time.
- **Anomaly scoring** — Per-frame score = mean of top-5% pixel errors after Gaussian smoothing, averaged across overlapping clips (stride 4), smoothed temporally (σ=3).
- **Heatmaps** — Per-pixel squared error overlaid with JET colormap. Zero extra parameters.

---

## Repository structure

```
video-anomaly-detection/
├── app.py                          # Streamlit demo app
├── Main.ipynb                      # Full training notebook (Colab)
├── requirements.txt
├── checkpoints/
│   └── ucsd_masked_v3_best.pth     # Pretrained checkpoint
├── samples/
│   ├── test001_final.mp4           # Sample test video
│   └── test010_final.mp4           # Sample test video
└── Figs/                           # Result figures
```

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit app

```bash
streamlit run app.py
```

Upload any `.mp4 / .avi / .mov` clip. The app shows:
- Frame-level anomaly score timeline
- Per-frame reconstruction error heatmaps
- Interactive frame inspector with original vs heatmap overlay

The app loads the checkpoint from `checkpoints/ucsd_masked_v3_best.pth` by default (configurable in the sidebar).

### 3. Training

The full training pipeline is in `Main.ipynb`. Runs on Google Colab T4 GPU (~40 min for 40 epochs on UCSD Ped2).

![Training Loss Curve](Figs/Training_Loss_Curve.png)
![ROC-AUC](Figs/ROC-AUC_USCD_Ped2.png)


---

## Datasets

- **UCSD Ped2** — http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm
- **ShanghaiTech** — https://svip-lab.github.io/dataset/campus_dataset.html

---

## Ablation

| Masking ratio   | AUC |
|-----------------|-----|
| 0% (baseline)   |73.3%|
| 50% (proposed)  |76.2%|

Both models use identical architecture, optimiser, and hyperparameters. The +2.9% improvement is solely from the masked pretraining objective.

---

## Demo

![Anomaly Score Timeline](Figs/Anomaly_Score_Timeline.png)
![Heatmap Grid](Figs/Heatmaps.png)
![Frame Inspector](Figs/Frame_Inspector.png)

---


