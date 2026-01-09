import os
import glob
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights

import pydicom
from PIL import Image

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


# =========================
# CONFIG
# =========================
CSV_TRAIN = r"D:\Clase\Ingeneria\3.Maila\1.Sehilekoa\PBL\Image\manifest-ZkhPvrLo5216730872708713142\pbl5_AI\calc_case_description_train_set.csv"
CSV_TEST  = r"D:\Clase\Ingeneria\3.Maila\1.Sehilekoa\PBL\Image\manifest-ZkhPvrLo5216730872708713142\pbl5_AI\calc_case_description_test_set.csv"

IMAGES_ROOT = r"D:\Clase\Ingeneria\3.Maila\1.Sehilekoa\PBL\Image\manifest-ZkhPvrLo5216730872708713142\pbl5_AI\CBIS-DDSM"

USE_PATH_COL = "image file path"

BATCH_SIZE = 8
NUM_EPOCHS = 12

# Fine-tuning 2 fases
FREEZE_EPOCHS = 3         # épocas entrenando solo la cabeza (fc)
LR_HEAD = 1e-3            # LR más alto para la cabeza
LR_ALL  = 2e-4            # LR más bajo para toda la red

WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2

IMG_SIZE = 224
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABEL_MAP = {
    "MALIGNANT": 1,
    "BENIGN": 0,
    "BENIGN_WITHOUT_CALLBACK": 0
}


# =========================
# Reproducibility
# =========================
def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_all(SEED)

if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True


# =========================
# Resolve full dicom path (robusto)
# =========================
def resolve_full_dicom_from_csv(rel_path: str) -> str | None:
    rel = str(rel_path).replace("/", os.sep).replace("\\", os.sep)

    marker = f"CBIS-DDSM{os.sep}"
    if marker in rel:
        rel = rel.split(marker, 1)[1]

    case_folder = rel.split(os.sep, 1)[0]
    case_dir = os.path.join(IMAGES_ROOT, case_folder)

    if not os.path.isdir(case_dir):
        return None

    full_candidates = glob.glob(
        os.path.join(case_dir, "**", "*full mammogram images*", "*.dcm"),
        recursive=True
    )
    if full_candidates:
        full_candidates.sort()
        for p in full_candidates:
            if os.path.basename(p).lower() == "1-1.dcm":
                return p
        return full_candidates[0]

    any_candidates = glob.glob(os.path.join(case_dir, "**", "*.dcm"), recursive=True)
    if any_candidates:
        any_candidates.sort()
        for p in any_candidates:
            if os.path.basename(p).lower() == "1-1.dcm":
                return p
        return any_candidates[0]

    return None


# =========================
# DICOM loader -> PIL (grayscale) + CROP
# =========================
def load_dicom_as_pil(dcm_path: str) -> Image.Image:
    ds = pydicom.dcmread(dcm_path, force=True)
    img = ds.pixel_array.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    img = img * slope + intercept

    # normalización robusta
    lo, hi = np.percentile(img, (1, 99))
    img = np.clip(img, lo, hi)
    img = (img - lo) / (hi - lo + 1e-6)  # -> [0,1]

    # ---- CROP simple de la región útil (quita fondo negro)
    mask = img > 0.05
    if mask.any():
        ys, xs = np.where(mask)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        img = img[y0:y1+1, x0:x1+1]

    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img).convert("L")


# =========================
# Dataset (solo imagen)
# =========================
class CBISImageDataset(Dataset):
    def __init__(self, df, tfm):
        self.df = df.reset_index(drop=True)
        self.tfm = tfm

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        dcm_path = str(row["abs_path"])

        img = load_dicom_as_pil(dcm_path)
        img = self.tfm(img)

        y = torch.tensor(LABEL_MAP[str(row["pathology"])], dtype=torch.long)
        return img, y


# =========================
# Model (Transfer Learning ResNet18 -> 1 canal) canal1 solo grises
# =========================
class ResNetBinaryTransfer(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = resnet18(weights=ResNet18_Weights.DEFAULT)

        # adaptar a 1 canal promediando pesos RGB
        w_rgb = self.net.conv1.weight.data.clone()  # [64,3,7,7]
        self.net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.net.conv1.weight.data = w_rgb.mean(dim=1, keepdim=True)

        self.net.fc = nn.Linear(self.net.fc.in_features, 2)

    def forward(self, x):
        return self.net(x)


# =========================
# Eval
# =========================
@torch.no_grad()
def eval_epoch(model, loader, ce):
    model.eval()
    losses = []
    ys = []
    preds = []

    for img, y in loader:
        img, y = img.to(DEVICE), y.to(DEVICE)
        logits = model(img)
        loss = ce(logits, y)
        losses.append(loss.item())

        p = torch.argmax(logits, dim=1)
        ys.extend(y.detach().cpu().numpy().tolist())
        preds.extend(p.detach().cpu().numpy().tolist())

    acc = accuracy_score(ys, preds)
    f1 = f1_score(ys, preds, zero_division=0)
    return float(np.mean(losses)), acc, f1, ys, preds


def set_requires_grad_backbone(model: ResNetBinaryTransfer, train_backbone: bool):
    """
    train_backbone=False -> congela todo excepto fc
    train_backbone=True  -> entrena todo
    """
    for name, p in model.net.named_parameters():
        if "fc" in name:
            p.requires_grad = True
        else:
            p.requires_grad = train_backbone


def make_optimizer(model, lr):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=WEIGHT_DECAY)


def train():
    here = os.path.dirname(os.path.abspath(__file__))
    SAVE_PATH = os.path.join(here, "artifacts", "best_model.pt")
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    print("Checkpoint se guardará en:", SAVE_PATH)
    print("DEVICE =", DEVICE)

    df_train = pd.read_csv(CSV_TRAIN)
    df_test  = pd.read_csv(CSV_TEST)

    df_train = df_train.dropna(subset=[USE_PATH_COL, "pathology", "patient_id"])
    df_test  = df_test.dropna(subset=[USE_PATH_COL, "pathology", "patient_id"])

    df_train["abs_path"] = df_train[USE_PATH_COL].apply(resolve_full_dicom_from_csv)
    df_test["abs_path"]  = df_test[USE_PATH_COL].apply(resolve_full_dicom_from_csv)

    before_tr, before_te = len(df_train), len(df_test)
    df_train = df_train.dropna(subset=["abs_path"]).copy()
    df_test  = df_test.dropna(subset=["abs_path"]).copy()

    print(f"Resolved train: {len(df_train)}/{before_tr}")
    print(f"Resolved test : {len(df_test)}/{before_te}")

    gss = GroupShuffleSplit(test_size=0.15, n_splits=1, random_state=SEED)
    tr_idx, va_idx = next(gss.split(df_train, groups=df_train["patient_id"]))
    tr_df = df_train.iloc[tr_idx].copy()
    va_df = df_train.iloc[va_idx].copy()

    print("Val pathology counts:\n", va_df["pathology"].value_counts())

    # ---- Class weights desde TRAIN split
    y_train = tr_df["pathology"].map(lambda x: LABEL_MAP[str(x)]).astype(int).values
    counts = np.bincount(y_train, minlength=2)
    weights = np.where(counts > 0, 1.0 / counts, 0.0)
    weights = weights / weights.sum() * 2.0  # normaliza para que sum ~2
    class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

    print("Train class counts:", counts.tolist())
    print("Class weights:", class_weights.detach().cpu().numpy().tolist())

    ce = nn.CrossEntropyLoss(weight=class_weights)

    tfm_train = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.RandomApply([T.RandomRotation(7)], p=0.5),
        T.RandomApply([T.ColorJitter(contrast=0.15)], p=0.5),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.25]),
    ])
    tfm_eval = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.25]),
    ])

    ds_tr = CBISImageDataset(tr_df, tfm_train)
    ds_va = CBISImageDataset(va_df, tfm_eval)
    ds_te = CBISImageDataset(df_test, tfm_eval)

    pin = (DEVICE == "cuda")
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=pin)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin)
    dl_te = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin)

    model = ResNetBinaryTransfer().to(DEVICE)

    # =========================
    # FASE 1: congelar backbone, entrenar solo fc
    # =========================
    set_requires_grad_backbone(model, train_backbone=False)
    opt = make_optimizer(model, lr=LR_HEAD)
    print(f"FASE 1: entrenando solo fc durante {FREEZE_EPOCHS} épocas (lr={LR_HEAD})")

    best_f1 = -1.0
    patience = 4
    bad = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        # pasar a fase 2 cuando toque
        if epoch == FREEZE_EPOCHS + 1:
            set_requires_grad_backbone(model, train_backbone=True)
            opt = make_optimizer(model, lr=LR_ALL)
            print(f"FASE 2: entrenando TODO el modelo (lr={LR_ALL})")

        model.train()
        total = 0.0
        n = 0

        for img, y in dl_tr:
            img, y = img.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(set_to_none=True)

            logits = model(img)
            loss = ce(logits, y)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bs = y.size(0)
            total += float(loss.item()) * bs
            n += bs

        tr_loss = total / max(1, n)
        va_loss, va_acc, va_f1, _, _ = eval_epoch(model, dl_va, ce)

        print(f"[{epoch:02d}] train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  val_acc={va_acc:.4f}  val_f1={va_f1:.4f}")

        # Guardado: siempre en epoch 1 y luego si mejora F1
        should_save = (epoch == 1) or (va_f1 > best_f1 + 1e-4)
        if should_save:
            best_f1 = max(best_f1, va_f1)
            bad = 0
            torch.save({
                "model": model.state_dict(),
                "img_size": IMG_SIZE,
                "note": "transfer+crop+classweights+2phase-finetune"
            }, SAVE_PATH)
            print("✅ Guardado checkpoint:", SAVE_PATH)
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

    print("Existe best_model.pt?", os.path.exists(SAVE_PATH))

    # Test final
    ckpt = torch.load(SAVE_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])

    te_loss, te_acc, te_f1, ys, preds = eval_epoch(model, dl_te, ce)
    print(f"TEST: loss={te_loss:.4f} acc={te_acc:.4f} f1={te_f1:.4f}")
    print("Confusion matrix:\n", confusion_matrix(ys, preds))
    print(classification_report(ys, preds, digits=4))


if __name__ == "__main__":
    train()
