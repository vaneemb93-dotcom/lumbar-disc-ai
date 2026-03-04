import os, glob
import numpy as np
import cv2
from PIL import Image

import streamlit as st
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import gdown

st.set_page_config(page_title="Lumbar Disc AI Demo", layout="wide")

DRIVE_FOLDER_URL = st.secrets.get("DRIVE_FOLDER_URL", "")
DET_FILENAME = st.secrets.get("DET_FILENAME", "detector_fasterrcnn_discs_best.pt")
CLF_FILENAME = st.secrets.get("CLF_FILENAME", "modelo_hidratacion_v3_resnet18_detstyle.pth")
SCORE_THR = float(st.secrets.get("SCORE_THR", 0.5))

DEVICE = "cpu"
WEIGHTS_DIR = "weights"

def ensure_3ch(img_np):
    if img_np.ndim == 2:
        img_np = np.stack([img_np]*3, axis=-1)
    if img_np.shape[-1] == 4:
        img_np = img_np[..., :3]
    return img_np

def to_tensor_01(img_np):
    img = img_np.astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1)

def sort_top_to_bottom(boxes):
    yc = (boxes[:, 1] + boxes[:, 3]) / 2.0
    return boxes[np.argsort(yc)]

def build_detector(num_classes=2):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def build_classifier(num_classes=2):
    model = torchvision.models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model

def download_weights_from_folder():
    if not DRIVE_FOLDER_URL:
        st.error("Falta DRIVE_FOLDER_URL en Secrets.")
        st.stop()

    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    # Descarga la carpeta pública desde Drive
    gdown.download_folder(DRIVE_FOLDER_URL, output=WEIGHTS_DIR, quiet=False, use_cookies=False)

    det_candidates = glob.glob(os.path.join(WEIGHTS_DIR, "**", DET_FILENAME), recursive=True)
    clf_candidates = glob.glob(os.path.join(WEIGHTS_DIR, "**", CLF_FILENAME), recursive=True)

    if not det_candidates:
        st.error(f"No encontré el detector '{DET_FILENAME}' dentro de la carpeta descargada.")
        st.stop()
    if not clf_candidates:
        st.error(f"No encontré el clasificador '{CLF_FILENAME}' dentro de la carpeta descargada.")
        st.stop()

    return det_candidates[0], clf_candidates[0]

@st.cache_resource
def load_models():
    det_path, clf_path = download_weights_from_folder()

    sd_det = torch.load(det_path, map_location=DEVICE)
    sd_clf = torch.load(clf_path, map_location=DEVICE)

    # Inferir num_classes del detector
    num_classes = 2
    k = "roi_heads.box_predictor.cls_score.weight"
    if isinstance(sd_det, dict) and k in sd_det:
        num_classes = sd_det[k].shape[0]

    detector = build_detector(num_classes=num_classes)
    detector.load_state_dict(sd_det, strict=True)
    detector.to(DEVICE).eval()

    # Inferir clases del clasificador
    clf_classes = 2
    if isinstance(sd_clf, dict) and "fc.weight" in sd_clf:
        clf_classes = sd_clf["fc.weight"].shape[0]

    classifier = build_classifier(num_classes=clf_classes)
    classifier.load_state_dict(sd_clf, strict=True)
    classifier.to(DEVICE).eval()

    return detector, classifier

DETECTOR, CLASSIFIER = load_models()

def detect_discs(pil_img):
    img_np = ensure_3ch(np.array(pil_img))
    x = to_tensor_01(img_np)
    with torch.no_grad():
        out = DETECTOR([x.to(DEVICE)])[0]
    boxes = out["boxes"].cpu().numpy()
    scores = out["scores"].cpu().numpy()
    boxes = boxes[scores >= SCORE_THR]
    if len(boxes) == 0:
        return np.zeros((0,4), dtype=np.float32)
    return sort_top_to_bottom(boxes)

def classify_crop(crop_np):
    crop_np = ensure_3ch(crop_np)
    crop_np = cv2.resize(crop_np, (224, 224))
    x = crop_np.astype(np.float32)/255.0
    x = torch.from_numpy(x).permute(2,0,1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = CLASSIFIER(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    # Asumimos clase 1 = Hydrated (si está invertido, lo cambiamos)
    p_hyd = float(probs[1]) if probs.shape[0] > 1 else float(probs[0])
    label = "Hydrated" if p_hyd >= 0.5 else "Desiccated"
    return label, p_hyd

def run(pil_img):
    img_np = ensure_3ch(np.array(pil_img))
    overlay = img_np.copy()

    boxes = detect_discs(pil_img)
    if len(boxes) == 0:
        return overlay, ["No discs detected. Probá otra slice o bajá SCORE_THR."]

    if len(boxes) > 5:
        boxes = boxes[:5]

    lines = []
    for i, b in enumerate(boxes, start=1):
        x1,y1,x2,y2 = [int(v) for v in b]
        x1=max(0,x1); y1=max(0,y1)
        x2=min(overlay.shape[1]-1,x2); y2=min(overlay.shape[0]-1,y2)

        crop = img_np[y1:y2, x1:x2]
        if crop.size == 0:
            lines.append(f"Disc {i}: empty crop")
            continue

        label, p = classify_crop(crop)
        color = (0,255,0) if label=="Hydrated" else (255,0,0)
        cv2.rectangle(overlay, (x1,y1), (x2,y2), color, 2)
        cv2.putText(overlay, f"D{i} {label[:3]} {p:.2f}", (x1, max(18,y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        lines.append(f"Disc {i}: {label} (P_hyd={p:.2f})")

    return overlay, lines

st.title("🦴 Lumbar Disc AI — Sagittal T2 Demo")
uploaded = st.file_uploader("Subí una sagital T2 (JPG/PNG)", type=["jpg","jpeg","png"])

if uploaded:
    pil = Image.open(uploaded).convert("RGB")
    overlay, lines = run(pil)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Input")
        st.image(pil, use_container_width=True)
    with c2:
        st.subheader("Overlay + Report")
        st.image(overlay, use_container_width=True)
        st.code("\n".join(lines))
else:
    st.info("Subí una imagen para correr el pipeline.")
