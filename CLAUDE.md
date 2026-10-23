# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains a Retinopathy of Prematurity (ROP) detection system using deep learning. The main component is a PyTorch‑based classifier that analyzes retinal images to detect pathological ROP signs.

## High‑Level Architecture

- **Data Layer** – Images are stored under `dataset/images/` (or the Kaggle‑specific subfolders). Labels are encoded in the filename (`DG` codes) and parsed by the training script.
- **Preprocessing Pipeline** – Implemented in `ROPDataset.preprocess_image()`. It converts BGR→RGB, extracts red/green channels, applies bilateral filtering, and performs CLAHE contrast enhancement. This yields a normalized 224×224 RGB tensor.
- **Model Layer** – A ResNet‑50 backbone (pre‑trained on ImageNet) with a custom head (`Dropout → Linear → Sigmoid`) for binary classification (Normal vs. ROP). Training, evaluation, and model checkpointing are orchestrated in `rop_training/train.py`.

## Development Commands

### 1. Set up the environment
```bash
# Create a virtual environment (optional but recommended)
python -m venv .venv
# Activate the environment (Linux/macOS)
source .venv/bin/activate
# Activate on Windows PowerShell
.venv\Scripts\Activate.ps1

# Install runtime and development dependencies
pip install -r requirements.txt
# (If `requirements.txt` is missing, install the core libs manually)
# pip install torch torchvision opencv-python numpy pandas scikit-learn
```

### 2. Run the training script
```bash
python rop_training/train.py
```
The script will:
1. Locate all `*.jpg` images under the default dataset path.
2. Filter files with clear Normal/ROP labels.
3. Perform a stratified train‑test split.
4. Train a ResNet‑50 for one epoch (quick sanity check) and report test accuracy.
5. Save the trained weights to `rop_resnet50_model.pth`.

### 3. Quick sanity run (limited data)
The script already limits training to the first 40 training images and 10 test images. To run an even faster check, edit the slice indices in `train.py` (e.g., `train_dataset = ROPDataset(X_train[:10], …)`). No extra command is required.

### 4. Lint / static analysis (optional)
If you have `ruff` or `flake8` installed, you can lint the codebase:
```bash
ruff .   # or flake8 .
```
These tools help keep the Python code style consistent.

### 5. Model inference (example snippet)
```python
import torch
from torchvision import models, transforms
from PIL import Image

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
# Replace the head with the saved checkpoint
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.5),
    torch.nn.Linear(model.fc.in_features, 1),
    torch.nn.Sigmoid()
)
model.load_state_dict(torch.load('rop_resnet50_model.pth'))
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

img = Image.open('path/to/sample.jpg')
img_tensor = preprocess(img).unsqueeze(0)
with torch.no_grad():
    prob = model(img_tensor).item()
print('ROP probability:', prob)
```

## Repository Layout (high level)
```
├── dataset/                 # Raw image data, organized in subfolders
├── rop_training/           # Training script (`train.py`)
├── rop_resnet50_model.pth  # Saved model weights (generated after training)
└── CLAUDE.md              # This guidance file
```

Feel free to expand the commands section with additional tooling (e.g., `torchvision` utilities, hyper‑parameter sweeps) as the project evolves.
