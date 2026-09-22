# AdverSense — Weekend Execution Guide

This document outlines the commands to execute the pipeline for the AdverSense Lab Mode and Fast Adversarial Training (Fast AT) workflow.

---

## ⚡ Execution Flow & Dependencies

> [!IMPORTANT]
> - `generate_attacks.py` can run **in parallel** with `train_adversarial.py`.
> - `train_detector.py` and Flask (`app.py`) depend strictly on `generate_attacks.py` (which generates `attacks/detector_adv_success.pt` and `attacks/{ATTACK}_index.json`), **not** on `train_adversarial.py`.
> - `train_adversarial.py` trains the Fast AT model and saves exclusively to `models/victim_cnn_at.pth`. It does **not** overwrite `models/victim_cnn.pth`.
> - **Flask must be restarted after each model change.**

---

## Exact Commands in Order

### 1. (Terminal 1) Fast Adversarial Training (Independent)
Trains the robust model using Wong et al. Fast AT ($\epsilon=0.3$, random-start FGSM) warm-started from `models/victim_cnn.pth`.
```bash
python3 train_adversarial.py
```
*Wait until saved to `models/victim_cnn_at.pth`.*

---

### 2. (Terminal 2) Generate Attacks & Lab Indices
Generates attacks on `models/victim_cnn.pth`, writing tagged PNGs, bulk `.pt` tensors, `attacks/{ATTACK}_index.json`, and `attacks/detector_adv_success.pt`.
```bash
python3 generate_attacks.py
```
*Wait until JSON index files (`attacks/*_index.json`) and `attacks/detector_adv_success.pt` are written.*

---

### 3. (Terminal 2) Train Detector CNN
Trains the security guard detector on balanced clean and successful adversarial samples across FGSM, PGD, and DeepFool.
```bash
python3 train_detector.py
```
*Verifies clean False Positive Rate (FPR $\le$ 15%) and saves `models/detector_cnn.pth`.*

---

### 4. (Terminal 2) Launch AdverSense Flask Server
Starts the web application with both file upload analysis and Lab Mode pre-generated attack evaluation.
```bash
python3 app.py
```
Open **http://localhost:5000** in your web browser.

> [!WARNING]
> **Always restart Flask (`Ctrl+C` then `python3 app.py`) after regenerating attacks or retraining models.**

---

## Acceptance Verification

1. **Lab Mode**:
   - Select attack `PGD` or `DeepFool`.
   - Choose any indexed sample (e.g., `idx 312 · true 0 → pred 2`).
   - Click **Run lab sample →**.
   - Output shows ground truth digit from MNIST dataset matching the index record, independent of pixel heuristics or detector misses.
2. **Upload Mode**:
   - Drag & drop any image file (e.g. `deepfool 3.png`, `fgsm 5.png`, `pgd 0.png`, or arbitrary images).
   - Click **Analyse →** to run through the standard upload pipeline.
