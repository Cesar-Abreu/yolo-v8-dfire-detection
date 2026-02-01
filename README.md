# YOLOv8 Pruning for Wildfire Detection (D-Fire Dataset)

This repository contains experimental Python code developed to support the **yolo-v8-dfire-detection**:

> **“Fire Detection System Based on Convolutional Neural Networks Optimized Through Layer Removal”**  
> César Neto de Nery Abreu, M.Sc.  
> Federal Center for Technological Education of Minas Gerais (CEFET-MG), 2024

The code implements **training, evaluation, and layer-level pruning strategies** applied to **YOLOv8** models for **wildfire and smoke detection**, with a focus on **reducing computational cost (FLOPs)** while preserving predictive performance.

---

## 📌 Project Motivation

Modern convolutional neural networks achieve high accuracy in fire detection tasks but often require significant computational resources, limiting their deployment on **low-power or edge devices**.

This project investigates whether **removing entire convolutional layers** from a pretrained YOLOv8 model can:

- Reduce FLOPs and inference cost  
- Maintain competitive detection performance (mAP, F1-score)  
- Enable deployment on energy-constrained and remote environments  

All experiments were conducted using the **D-Fire dataset**, following the methodology described in the dissertation.

---

## 📂 Repository Contents

- `Yolov8_prune_dfire.ipynb`  
  Main experimental notebook containing:
  - Dataset preparation
  - YOLOv8 training
  - Multiple layer pruning strategies
  - Model retraining and evaluation

This repository is **research-oriented** and intended for **reproducibility and experimentation**, not as a production-ready package.

---

## 🧠 Methods Implemented

The notebook evaluates several **layer removal (pruning) strategies**, including:

1. **Baseline Training**
   - Transfer learning using YOLOv8 pretrained weights
   - Training on the D-Fire wildfire dataset

2. **Random Layer Removal**
   - Convolutional layers removed at random
   - Used as a control baseline

3. **Ranking-Based Layer Removal**
   - Layers ranked by importance metrics
   - Least important layers removed iteratively

4. **Linear Correlation Method**
   - Identifies layers with highly correlated outputs
   - Removes redundant layers while preserving information flow

Each pruned model is retrained and evaluated to analyze the trade-off between:
- **Detection performance** (mAP, F1-score)
- **Computational cost** (FLOPs)

---

## 📊 Evaluation Metrics

The following metrics are used consistently with the dissertation:

- **mAP (Mean Average Precision)**
- **F1-score**
- **GFLOPs**
- **Inference complexity**

Comparisons are made between the original YOLOv8 model and multiple pruned variants.

---

## 🧰 Requirements

The experiments were conducted using:

- Python 3.8+
- PyTorch 1.13.1
- Torchvision 0.14.1
- Ultralytics YOLOv8 (`ultralytics==8.0.90`)
- torch-pruning
- gdown

Example installation:

```bash
pip install ultralytics==8.0.90
pip install torch==1.13.1 torchvision==0.14.1
pip install torch-pruning gdown

This code directly supports the experiments and results presented in the dissertation and the related publication:

“Version 8 of YOLO for Wildfire Detection”
Presented at Deep Learning Theory and Applications (DeLTA 2024)
Published in Communications in Computer and Information Science (CCIS, vol. 2172)
