import io
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from PIL import Image
import pydicom
import torchvision.transforms as T
import uvicorn

from train_agent import MultiModalNet, IMG_SIZE, TEXT_COLS
from agent_policy import mc_dropout_predict, policy_decision

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="Mammography AI Agent")

ckpt = torch.load("artifacts/best_model.pt", map_location=DEVICE)
vocab = ckpt["vocab"]

model = MultiModalNet(vocab_size=len(vocab)).to(DEVICE)
model.load_state_dict(ckpt["model"])
model.eval()

tfm = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.25])
])

def encode_text(payload: dict):
    text = " ".join([str(payload.get(c, "")) for c in TEXT_COLS]).lower()
    toks = [t for t in text.replace("_", " ").split() if t]
    ids = [vocab.get(t, vocab["<unk>"]) for t in toks][:64]
    if len(ids) == 0:
        ids = [vocab["<unk>"]]
    return torch.tensor([ids], dtype=torch.long)

class MetaPayload(BaseModel):
    breast_density: float
    assessment: float
    subtlety: float

    abnormality_type: str | None = ""
    calc_type: str | None = ""
    calc_distribution: str | None = ""
    image_view: str | None = ""
    left_or_right_breast: str | None = ""

@app.post("/agent/predict")
async def predict(file: UploadFile = File(...), meta: MetaPayload = None):
    raw = await file.read()

    ds = pydicom.dcmread(io.BytesIO(raw), force=True)
    img_arr = ds.pixel_array.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    img_arr = img_arr * slope + intercept

    lo, hi = np.percentile(img_arr, (1, 99))
    img_arr = np.clip(img_arr, lo, hi)
    img_arr = (img_arr - lo) / (hi - lo + 1e-6)
    img_arr = (img_arr * 255).astype(np.uint8)

    pil = Image.fromarray(img_arr).convert("L")

    x_img = tfm(pil).unsqueeze(0).to(DEVICE)
    x_meta = torch.tensor([[meta.breast_density, meta.assessment, meta.subtlety]],
                          dtype=torch.float32).to(DEVICE)

    payload = {
        "abnormality type": meta.abnormality_type or "",
        "calc type": meta.calc_type or "",
        "calc distribution": meta.calc_distribution or "",
        "image view": meta.image_view or "",
        "left or right breast": meta.left_or_right_breast or ""
    }
    x_txt = encode_text(payload).to(DEVICE)

    mean_p, unc = mc_dropout_predict(model, x_img, x_meta, x_txt, mc_runs=12)
    mean_p = float(mean_p[0])
    unc = float(unc[0])

    action = policy_decision(mean_p, unc)

    return {
        "prob_malignant": mean_p,
        "uncertainty": unc,
        "action": action
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
