"""Model loading, preprocessing and prediction for the ROP Streamlit app.

The preprocessing here must stay identical to ROPDataset.preprocess_image in
rop-notebook.ipynb, otherwise the model sees different images than it was trained on.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

DEFAULT_IMG_SIZE = 448
DEFAULT_THRESHOLD = 0.5

_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_config(config_path):
    """Read the threshold and image size saved by the notebook; fall back to defaults."""
    path = Path(config_path)
    if path.exists():
        cfg = json.loads(path.read_text(encoding="utf-8"))
        return {
            "threshold": float(cfg.get("threshold", DEFAULT_THRESHOLD)),
            "img_size": int(cfg.get("img_size", DEFAULT_IMG_SIZE)),
            "from_file": True,
        }
    return {"threshold": DEFAULT_THRESHOLD, "img_size": DEFAULT_IMG_SIZE, "from_file": False}


def load_model(weights_path, device="cpu"):
    """ResNet-50 with the same head as in training (Dropout -> Linear -> Sigmoid)."""
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 1),
        nn.Sigmoid(),
    )
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device).eval()
    return model


def decode_image(data):
    """Raw bytes of an uploaded file -> BGR image (like cv2.imread), or None if unreadable."""
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def preprocess(img_bgr, img_size):
    """
    Same steps as training: resize, keep red + green channels, bilateral filter, CLAHE.
    Returns an RGB uint8 image.
    """
    img = cv2.resize(img_bgr, (img_size, img_size))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    r, g, _ = cv2.split(img_rgb)
    img_rg = cv2.merge([r, g, np.zeros_like(r)])

    img_filtered = cv2.bilateralFilter(img_rg, d=9, sigmaColor=3, sigmaSpace=0.1)

    lab = cv2.cvtColor(img_filtered, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)


@torch.no_grad()
def predict(model, images_rgb, device="cpu", batch_size=8):
    """Preprocessed RGB images -> list of ROP probabilities (0 to 1)."""
    probs = []
    for start in range(0, len(images_rgb), batch_size):
        batch = torch.stack([_to_tensor(img) for img in images_rgb[start:start + batch_size]]).to(device)
        probs.extend(model(batch).view(-1).cpu().tolist())
    return probs
