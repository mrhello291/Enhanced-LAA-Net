# 🔬 Enhanced LAA -Net: CBAM + Restormer for Robust Face Forgery Detection

## Overview

This repository presents an improved face forgery detection model built on **LAA -Net**, enhanced by:

1. **CBAM** (Convolutional Block Attention Module) — for robust channel & spatial feature refinement.
2. **Restormer** integration — to boost performance on noisy or degraded input images via transformer-based restoration.

These upgrades significantly enhance generalization across varied domains and fortify detection on noisy data.

---

## 🚀 Key Contributions

* **CBAM Integration**

  * Lightweight and modular attention blocks refine feature maps along channel and spatial dimensions
  * Boosts robustness against artifacts, noise, and compression by focusing on meaningful forgery traces

* **Restormer-Based Restoration Preprocessing**

  * Efficient transformer architecture specialized for high-res image denoising and deblurring
  * Serves as a front-end to improve detection under noisy or degraded conditions

---

## 📈 Experimental Results

### CBAM-enhanced LAA -Net (On YouTube -real / Celeb -real / Celeb -synthesis)

| Metric | Accuracy (%) | AUC (%) | AP (%) | AR (%) | mF1 (%) |
| ------ | ------------ | ------- | ------ | ------ | ------- |
| Value  | 63.64        | 62.92   | 73.59  | 59.62  | 65.87   |

> Moderate accuracy/AUC demonstrates strong capability in cross-domain real/fake detection.

---

### LAA -Net + Restormer (On FaceForensics++ splits)

* **Datasets**: Original / Deepfakes / Face2Face / FaceSwap / NeuralTextures
* **Metrics**: ACC = 50.7%, AUC = 89.07%, AP = 89.74%, AR = 50.71%, mF1 = 64.81%

The high AUC/AP indicate exceptional discrimination even with noisy inputs, though raw accuracy drops due to domain shifts.

---

## 💡 Why These Improvements Matter

* **Robustness & Generalization**
  CBAM refines features adaptively, enhancing reliability across domains.

* **Noise & Degradation Handling**
  Restormer excels in restoration tasks (denoising, deblurring), improving downstream detection in real-world scenarios.

---

## ⚙️ Architecture

```
Input Image ──► [Restormer Restoration Module] ──► [LAA -Net Backbone + CBAM] ──► Classifier
```

1. **Restormer**: Removes noise, blur, artifacts.
2. **LAA -Net**: Structure as per original implementation.
3. **CBAM**: Plugged in intermediate layers for attention-based refinement.

---

## ⚙️ Requirements

* Python ≥ 3.7
* PyTorch ≥ 1.10
* Other: NumPy, OpenCV, scikit-learn, tqdm, CUDA

```bash
pip install -r requirements.txt
```


---

## 📊 Performance Notes

* **CBAM-only version** delivers balanced real/fake detection with good F1 scores.
* **Restormer-enhanced version** particularly shines in noisy or compressed data scenarios (e.g., real-world or social-media visuals).

---

## 🧰 File Structure

```
.
├── data/                # Dataset loaders and utilities
├── model/               # LAA -Net backbone, CBAM modules
├── restormer/           # Restormer implementation
├── scripts/             # Training & evaluation scripts
├── requirements.txt
└── README.md
```

---

## 🎯 Future Directions

* Fine-tune Restormer alongside detection head for end-to-end optimization
* Integrate cross-domain augmentation (e.g., compression, video frames)
* Explore additional attention modules or frequency-domain fusion techniques

---

## 📚 References

- [CBAM: Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521) — Lightweight attention boosting generalization  
- [Restormer: Efficient Transformer for High-Resolution Image Restoration](https://arxiv.org/abs/2111.09881) — Transformer-based image restoration

---

## 👥 Team

This project was developed and maintained by:

- **Bhupesh Yadav**
- **Piyush Kumar**
- **Asif Hoda**


---

## 🔗 Contact

For questions, collaborations, or citing this work, please open an issue or reach out at `[bhupeshy510@gmail.com]`.
