import io
import os
from typing import Optional, List
from contextlib import asynccontextmanager

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pydicom
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights

# =========================
# CONFIG (FIXED)
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224

# Threshold fijo (NO se expone en la web ni en query params)
THRESHOLD = 0.35

# Agregación fija:
# - "max": score_global = max(p_malignant_i)  (recomendado para screening/alerta)
AGGREGATION = "max"

LABEL_ID_TO_NAME = {0: "BENIGN", 1: "MALIGNANT"}


# =========================
# DICOM → PIL
# =========================
def load_dicom_as_pil_from_bytes(dcm_bytes: bytes) -> Image.Image:
    ds = pydicom.dcmread(io.BytesIO(dcm_bytes), force=True)

    if not hasattr(ds, "PixelData"):
        raise ValueError("DICOM has no PixelData")

    img = ds.pixel_array.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    img = img * slope + intercept

    # robust normalization
    lo, hi = np.percentile(img, (1, 99))
    img = np.clip(img, lo, hi)
    img = (img - lo) / (hi - lo + 1e-6)

    # simple crop around breast region
    mask = img > 0.05
    if mask.any():
        ys, xs = np.where(mask)
        img = img[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]

    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img).convert("L")


# =========================
# Torch transforms
# =========================
tfm_eval = T.Compose(
    [
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.25]),
    ]
)


# =========================
# Model
# =========================
class ResNetBinaryTransfer(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = resnet18(weights=ResNet18_Weights.DEFAULT)

        # convert conv1 to 1-channel
        w = self.net.conv1.weight.data.clone()
        self.net.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        self.net.conv1.weight.data = w.mean(dim=1, keepdim=True)

        # binary head (2 logits)
        self.net.fc = nn.Linear(self.net.fc.in_features, 2)

    def forward(self, x):
        return self.net(x)


def load_model() -> nn.Module:
    here = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.join(here, "artifacts", "best_model.pt")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError("❌ No encuentro artifacts/best_model.pt")

    model = ResNetBinaryTransfer().to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    # expected checkpoint format: {"model": state_dict, ...}
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        # fallback if user saved just state_dict
        model.load_state_dict(ckpt)

    model.eval()
    return model


# =========================
# FastAPI lifespan
# =========================
MODEL: Optional[nn.Module] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL
    MODEL = load_model()
    print(f"✅ Model loaded on {DEVICE}")
    yield
    MODEL = None


app = FastAPI(title="Pink Alert – Mammography AI", lifespan=lifespan)


# =========================
# WEB UI (ONLY SUMMARY)
# =========================
@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Pink Alert</title>

<style>
body {
    font-family: Arial, sans-serif;
    background: linear-gradient(135deg, #fde2e8, #f8bbd0);
}
.container {
    max-width: 560px;
    margin: 70px auto;
    background: white;
    padding: 28px;
    border-radius: 16px;
    box-shadow: 0 25px 50px rgba(0,0,0,0.25);
}
h1 {
    text-align: center;
    color: #c2185b;
    margin-bottom: 8px;
}
.subtitle {
    text-align: center;
    color: #666;
    margin-bottom: 18px;
}
.small {
    font-size: 13px;
    color: #777;
    margin-top: -8px;
    text-align: center;
}
input[type=file] {
    width: 100%;
    margin: 14px 0 14px 0;
}
button {
    width: 100%;
    padding: 14px;
    background: #c2185b;
    color: white;
    border: none;
    border-radius: 10px;
    font-size: 16px;
    cursor: pointer;
}
button:hover {
    background: #ad1457;
}
.result {
    margin-top: 18px;
    padding: 16px;
    border-radius: 12px;
    font-size: 18px;
    display: none;
    text-align: center;
}
.benign {
    background: #e8f5e9;
    color: #2e7d32;
    border: 2px solid #66bb6a;
}
.malignant {
    background: #fdecea;
    color: #c62828;
    border: 2px solid #ef5350;
    animation: pulse 1.2s infinite;
}
@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(239,83,80,0.5); }
    70% { box-shadow: 0 0 0 12px rgba(239,83,80,0); }
}
.prob {
    font-size: 14px;
    margin-top: 8px;
    color: #444;
}
.error {
    background: #fff3cd;
    border: 2px solid #ffecb5;
    color: #856404;
}
</style>
</head>

<body>
<div class="container">
    <h1>Pink Alert</h1>
    <div class="subtitle">AI-based mammography analysis</div>
    <div class="small">Upload exactly 4 DICOM (.dcm) files (e.g. L-CC, R-CC, L-MLO, R-MLO)</div>

    <input type="file" id="fileInput" accept=".dcm" multiple>

    <button onclick="analyze()">Analyze (4 images)</button>

    <div id="result" class="result"></div>
</div>

<script>
const fileInput = document.getElementById("fileInput");
const resultDiv = document.getElementById("result");

function fmtPct(x) {
    return (x * 100).toFixed(1) + "%";
}

async function analyze() {
    const files = fileInput.files;

    if (!files || files.length !== 4) {
        alert("Please select exactly 4 DICOM (.dcm) files");
        return;
    }

    resultDiv.style.display = "none";
    resultDiv.className = "result";

    const formData = new FormData();
    for (const f of files) {
        formData.append("files", f);
    }

    try {
        const res = await fetch(`/predict4`, {
            method: "POST",
            body: formData
        });

        const data = await res.json();

        if (!res.ok) {
            resultDiv.style.display = "block";
            resultDiv.className = "result error";
            resultDiv.innerHTML = `⚠️ <strong>Error</strong><div class="prob">${data.detail || "Unknown error"}</div>`;
            return;
        }

        const isMal = data.overall_prediction === "MALIGNANT";
        resultDiv.style.display = "block";
        resultDiv.className = "result " + (isMal ? "malignant" : "benign");

        if (isMal) {
            resultDiv.innerHTML = `
                ⚠️ <strong>MALIGNANT (overall)</strong>
                <div class="prob">Overall probability: ${fmtPct(data.overall_prob_malignant)}</div>
            `;
        } else {
            resultDiv.innerHTML = `
                ✅ <strong>BENIGN (overall)</strong>
                <div class="prob">Overall probability: ${fmtPct(data.overall_prob_malignant)}</div>
            `;
        }

    } catch (e) {
        resultDiv.style.display = "block";
        resultDiv.className = "result error";
        resultDiv.innerHTML = `⚠️ <strong>Network/Error</strong><div class="prob">${String(e)}</div>`;
    }
}
</script>

</body>
</html>
"""


# =========================
# RESPONSES (ONLY SUMMARY)
# =========================
class Predict4SummaryResponse(BaseModel):
    overall_prediction: str
    overall_prob_malignant: float


# =========================
# HELPERS
# =========================
def infer_prob_malignant_from_bytes(dcm_bytes: bytes) -> float:
    pil = load_dicom_as_pil_from_bytes(dcm_bytes)
    x = tfm_eval(pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(MODEL(x), dim=1)[0]
        return float(probs[1])


def aggregate_overall_max(probs: List[float]) -> float:
    # AGGREGATION fija a "max"
    return float(max(probs)) if probs else 0.0


# =========================
# API: EXACTLY 4 images -> SUMMARY ONLY
# =========================
@app.post("/predict4", response_model=Predict4SummaryResponse)
async def predict4(files: List[UploadFile] = File(...)):
    if MODEL is None:
        raise HTTPException(500, "Model not loaded")

    if len(files) != 4:
        raise HTTPException(400, f"Please upload exactly 4 files. Received: {len(files)}")

    probs_list: List[float] = []

    for f in files:
        dcm_bytes = await f.read()
        try:
            p_mal = infer_prob_malignant_from_bytes(dcm_bytes)
        except Exception as e:
            raise HTTPException(400, f"Invalid DICOM file '{f.filename}': {str(e)}")

        probs_list.append(float(p_mal))

    overall_prob = aggregate_overall_max(probs_list)
    overall_pred = 1 if overall_prob >= THRESHOLD else 0

    return Predict4SummaryResponse(
        overall_prediction=LABEL_ID_TO_NAME[int(overall_pred)],
        overall_prob_malignant=float(overall_prob),
    )
