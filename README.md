<p align="center">

  <h1 align="center">Multimodal Fusion Framework Using Contrastive Learning for Exposure Keratopathy</h1>
  <p align="center">
    <b>Gyutae Oh</b> (Department of Electrical and Computer Engineering, Sungkyunkwan University, alswo740012@g.skku.edu) ·
    <b>Yeokyoung Won</b> (Department of Ophthalmology, Samsung Medical Center, Sungkyunkwan University School of Medicine, wyk900105@hanmail.net) ·
    <b>Donghui Lim</b> (Department of Ophthalmology, Samsung Medical Center, Sungkyunkwan University School of Medicine, ldhlse@gmail.com) ·
    <b>Jitae Shin</b> (Department of Electrical and Computer Engineering, Sungkyunkwan University, jtshin@skku.edu)
  </p>
  <h3 align="center"><a href="https://github.com/GYUGYUT/Multimodal-FusionFramework-Using-Contrastive-Learning-for-Exposure-Keratopathy">[Paper/Code GitHub]</a></h3>
</p>

<p align="center">
  <img src="./image/Figure1.png" alt="Framework Overview" width="80%"><br>
  <i>Figure 1. Overview of the proposed multimodal fusion framework for exposure keratopathy grading.</i>
</p>

<p align="center">
  <img src="./image/Figure2.png" alt="Feature Visualization" width="80%"><br>
  <i>Figure 2. UMAP visualization of learned features by modality and grade.</i>
</p>

<p align="center">
<b>A multimodal fusion framework for automated grading of exposure keratopathy using four types of anterior segment images (broad-beam, slit-beam, scatter, blue-light). Achieves high diagnostic performance with contrastive learning-based feature fusion, even with single-modality input at inference.</b>
</p>
<br>

# Overview

This repository implements a two-stage multimodal fusion framework for automated grading of exposure keratopathy using four complementary anterior segment imaging modalities (broad-beam, slit-beam, scatter, blue-light).

- **Stage 1:** Grade Based Learning & Beam Based Learning (contrastive learning-based multimodal feature fusion)
- **Stage 2:** Dynamic Feature Fusion (utilizes only broad-beam input at inference, leveraging multimodal information)
- Achieves over 16% improvement in F1 score and accuracy compared to single-modality baselines; maintains high performance in real clinical settings with only a single image at inference.

# Installation
```bash
pip install -r requirements.txt
```

# Data Preparation
- Prepare Excel (.xlsx) files for each modality containing image IDs and grade (label) information.
- Image files should be stored in the `SMC_New_original` folder (file names must match the 'Detailed ID' column in the Excel files).
- **You must provide your own data. Edit the Excel file names and image folder path in `run_train.sh` to match your dataset.**
- Example (see `run_train.sh`):
  - `BROAD_TRAIN="grade_photo1_train.xlsx"` (replace with your own train file)
  - `IMAGE_FOLDER="SMC_New_original"` (replace with your own image folder)

# Usage
```bash
bash run_train.sh
```
- You can modify data paths, backbone, and hyperparameters in `run_train.sh`.
- Training and evaluation results will be saved in the `result/` folder.

# Main Results
| Method           | Accuracy | F1 Score |
|------------------|----------|----------|
| Single (Broad)   | 0.7045   | 0.7049   |
| Single (Slit)    | 0.6692   | 0.6510   |
| Single (Blue)    | 0.6447   | 0.6368   |
| Single (Scatter) | 0.6957   | 0.6729   |
| **Ours**         | **0.8409** | **0.8379** |

# Citation
```bibtex
@article{Oh2024MultimodalFusion,
  title={Multimodal Fusion Framework Using Contrastive Learning for Exposure Keratopathy},
  author={Gyutae Oh and Yeokyoung Won and Donghui Lim and Jitae Shin},
  journal={MICCAI OMIA Workshop},
  year={2024}
}
```

# Contact
- alswo740012@g.skku.edu (Gyutae Oh)
- wyk900105@hanmail.net (Yeokyoung Won)
- ldhlse@gmail.com (Donghui Lim)
- jtshin@skku.edu (Jitae Shin)