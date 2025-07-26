# Multimodal Fusion Framework Using Contrastive Learning for Exposure Keratopathy

**Gyutae Oh<sup>1,*</sup>**, **Yeokyoung Won<sup>2,3*</sup>**, **Donghui Lim<sup>2,3,4,†</sup>**, **Jitae Shin<sup>1,†</sup>**

<sup>1</sup>Department of Electrical and Computer Engineering, Sungkyunkwan University  
<sup>2</sup>SNU Eye Clinic, Seoul 06035, Republic of Korea  
<sup>3</sup>Department of Ophthalmology, Samsung Medical Center, Sungkyunkwan University School of Medicine, Seoul 06351, Republic of Korea  
<sup>4</sup>Samsung Advanced Institute for Health Sciences and Technology, Sungkyunkwan University, Seoul 03063, Republic of Korea

*These authors contributed equally as co-first authors.*  
†*These authors jointly supervised this work as co-corresponding authors.*

---

## 📧 Contact
- **Gyutae Oh**: alswo740012@g.skku.edu
- **Yeokyoung Won**: wyk900105@hanmail.net  
- **Donghui Lim**: ldhlse@gmail.com
- **Jitae Shin**: jtshin@skku.edu

[📄 Paper](https://github.com/GYUGYUT/Multimodal-Fusion-Framework-Using-Contrastive-Learning-for-Exposure-Keratopathy) | [💻 Code](https://github.com/GYUGYUT/Multimodal-Fusion-Framework-Using-Contrastive-Learning-for-Exposure-Keratopathy)

---

## 🎯 Abstract

**A multimodal fusion framework for automated grading of exposure keratopathy using four types of anterior segment images (broad-beam, slit-beam, scatter, blue-light). Achieves high diagnostic performance with contrastive learning-based feature fusion, even with single-modality input at inference.**

<p align="center">
  <img src="./image/Figure1.png" alt="Framework Overview" width="80%"><br>
  <em>Figure 1. Overall architecture showing the general workflow and example images for the broad-beam, slit-beam, blue-light, and scatter modalities. Note: Due to data privacy restrictions, similar images from [30] (CC BY 4.0) are used as alternatives with prior permission.</em>
</p>

<p align="center">
  <img src="./image/Figure2.png" alt="Feature Visualization" width="80%"><br>
  <em>Figure 2. UMAP visualization of features before and after training with our proposed method for each independently trained backbone, grouped by modality and grade. Identical colors represent the same grade, and identical shapes represent the same modality.</em>
</p>

---

## 🚀 Overview

### Problem Statement
Accurate diagnosis of exposure keratopathy requires multiple illumination-based anterior segment images (broad-beam, slit-beam, scatter, and blue-light). However, acquiring all modalities is often impractical in real-world clinical settings.

### Our Solution
We propose a **two-stage multimodal fusion framework** that performs multimodal contrastive learning during training but requires only a single broad-beam image at inference.

### 🧠 Methodology

#### **Stage 1: Contrastive Multimodal Representation Learning**
- **Grade Based Learning (GBL)**: Encourages learning of grade-specific features
- **Beam Based Learning (BBL)**: Promotes modality-specific feature learning
- **Cross-modal encoding**: Broad-beam image encodes pathological cues from other three modalities

#### **Stage 2: Dynamic Feature Fusion (DFF)**
- Fuses feature vectors from Stage 1 via learnable scalar weight (α)
- Achieves high classification accuracy using only broad-beam input

### 📈 Key Results

Compared to single-modality baselines:
- **+16% improvement** in accuracy
- **+13% improvement** in F1 score

This framework enables efficient, high-performance diagnosis with minimal imaging requirements — making it well-suited for real-world deployment in medical environments.

---

## 🛠️ Installation

```bash
pip install -r requirements.txt
```

---

## 📊 Data Preparation

⚠️ **Important Notice**: The study was approved by the Institutional Review Board (IRB) of the Samsung Medical Center (IRB no. 2024-12-004). Due to privacy regulations, the dataset cannot be publicly released.

### Setup Instructions
1. Prepare your own dataset following the required structure
2. Create an `.xlsx` file containing data split information
3. Update the correct paths in `run_train.sh`
4. Modify `smc_dataloader` if necessary based on your dataset structure

### Example Configuration (`run_train.sh`)
```bash
BROAD_TRAIN="grade_photo1_train.xlsx"  # Replace with your train file
IMAGE_FOLDER="SMC_New_original"        # Replace with your image folder
```

---

## 🚀 Usage

```bash
bash run_train.sh
```

**Customization Options:**
- Modify data paths, backbone, and hyperparameters in `run_train.sh`
- Training and evaluation results will be saved in the `result/` folder

---

## 📋 Results

### Table 2. Comparison of Individual and Combined Training Results

| Train Method | Train Data | Accuracy | Specificity | Sensitivity | F1 Score |
|:-------------|:-----------|:---------|:------------|:-----------|:---------|
| Supervised   | Broad-beam | 0.7045   | 0.9008      | 0.7015      | 0.7049   |
| Supervised   | Slit-beam  | 0.6692   | 0.8787      | 0.6686      | 0.6510   |
| Supervised   | Blue-light | 0.6447   | 0.8758      | 0.6407      | 0.6368   |
| Supervised   | Scatter    | 0.6957   | 0.8900      | 0.6685      | 0.6729   |
| Supervised   | Total      | 0.6865   | 0.8880      | 0.6783      | 0.6789   |
| **Ours**     | Stage 1:Total / Stage2:Broad | **0.8409** | **0.9462** | **0.8382** | **0.8379** |

### Table 3. Ablation Study: Performance Evaluation by Removing GBL, BBL, DFF

| GBL | BBL | DFF | Accuracy | Specificity | Sensitivity | F1 Score |
|:----|:----|:----|:---------|:------------|:-----------|:---------|
| ✓   | ✓   |     | 0.7500   | 0.9154      | 0.7492      | 0.7563   |
| ✓   |     | ✓   | 0.7727   | 0.9235      | 0.7700      | 0.7710   |
|     | ✓   | ✓   | 0.7727   | 0.9232      | 0.7678      | 0.7685   |
| ✓   | ✓   | ✓   | 0.7954   | 0.9315      | 0.7969      | 0.7958   |
|     |     | ✓   | 0.6590   | 0.8858      | 0.6640      | 0.6634   |
| ✓   | ✓   | ✓   | **0.8409** | **0.9462** | **0.8382** | **0.8167** |

---

## 📚 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{Oh2024MultimodalFusion,
  title={Multimodal Fusion Framework Using Contrastive Learning for Exposure Keratopathy},
  author={Gyutae Oh and Yeokyoung Won and Donghui Lim and Jitae Shin},
  journal={MICCAI OMIA Workshop},
  year={2024}
}
```

---

## 📞 Contact

For questions and collaborations, please reach out to:

- **Gyutae Oh** (alswo740012@g.skku.edu) - Department of Electrical and Computer Engineering, Sungkyunkwan University
- **Yeokyoung Won** (wyk900105@hanmail.net) - SNU Eye Clinic
- **Donghui Lim** (ldhlse@gmail.com) - Samsung Medical Center
- **Jitae Shin** (jtshin@skku.edu) - Department of Electrical and Computer Engineering, Sungkyunkwan University

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

</div>