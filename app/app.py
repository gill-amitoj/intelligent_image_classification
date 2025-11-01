# ----------------------------------------------------------
# Flask App for Real-time Inference with Grad-CAM
# ----------------------------------------------------------
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
from flask import Flask, jsonify, redirect, render_template, request, send_from_directory, url_for
import torch
from PIL import Image
from torchvision import transforms

# Ensure project root is on sys.path so `src` is importable when running as a script
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import build_model
from src.grad_cam import GradCAM, overlay_heatmap_on_image

# Config
MODELS_DIR = ROOT / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "resnet18_best.pth"
CLASSES_PATH = MODELS_DIR / "classes.json"
UPLOAD_DIR = ROOT / "app" / "static" / "uploads"
OUTPUT_DIR = ROOT / "app" / "static" / "outputs"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)


def load_classes() -> List[str]:
    if CLASSES_PATH.exists():
        with open(CLASSES_PATH, "r", encoding="utf-8") as f:
            idx_to_class: Dict[str, str] = json.load(f)
        return [idx_to_class[str(i)] for i in range(len(idx_to_class))]
    return []


def load_model(device: torch.device):
    classes = load_classes()
    if not DEFAULT_MODEL_PATH.exists():
        return None, classes
    model = build_model(num_classes=len(classes) if classes else 2)
    state = torch.load(DEFAULT_MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, classes


def preprocess(img: Image.Image, img_size: int = 224) -> torch.Tensor:
    t = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return t(img).unsqueeze(0)


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return redirect(url_for("index"))

    file = request.files["image"]
    if file.filename == "":
        return redirect(url_for("index"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes = load_model(device)
    if model is None:
        return render_template("error.html", message="Model file not found. Train the model first.")

    # Save upload
    filename = file.filename
    save_path = UPLOAD_DIR / filename
    file.save(save_path)

    # Inference
    img = Image.open(save_path).convert("RGB")
    x = preprocess(img).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred_idx = int(probs.argmax())
        pred_class = classes[pred_idx] if classes else str(pred_idx)
        confidence = float(probs[pred_idx])

    # Grad-CAM
    target_layer = getattr(model, "layer4")[-1]
    cam = GradCAM(model, target_layer)
    heatmap = cam.generate(x, class_idx=pred_idx)
    overlay = overlay_heatmap_on_image(np.array(img.resize((224, 224))), heatmap, alpha=0.45)
    out_name = f"cam_{Path(filename).stem}_{pred_class}.jpg"
    out_path = OUTPUT_DIR / out_name

    # Save via OpenCV since overlay is BGR
    import cv2

    cv2.imwrite(str(out_path), overlay)
    cam.remove_hooks()

    return render_template(
        "result.html",
        original_image=url_for("static", filename=f"uploads/{filename}"),
        cam_image=url_for("static", filename=f"outputs/{out_name}"),
        pred_class=pred_class,
        confidence=f"{confidence*100:.2f}%",
    )


if __name__ == "__main__":
    # Bind to localhost by default to avoid macOS "Local Network" permission prompts.
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 5000))
    app.run(host=host, port=port, debug=True)
