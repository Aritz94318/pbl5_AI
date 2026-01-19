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
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    recall_score,
)

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
FREEZE_EPOCHS = 5
LR_HEAD = 1e-3
LR_ALL  = 1e-4

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
# THRESHOLD / BALANCED CONFIG (NEW)
# =========================
TARGET_RECALL_MAL = 0.95        # mantiene sensibilidad alta
THRESH_MODE = "max_balanced"    # maximiza balanced accuracy bajo ese constraint
THRESH_GRID = 500

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

    # Prioriza full mammogram
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

    # Fallback: cualquier dicom
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

    # crop simple (quita fondo negro)
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
# Model (Transfer Learning ResNet18 -> 1 canal)
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
# Eval (prob maligno + AUC + recall_malignant by argmax)
# =========================
@torch.no_grad()
def eval_epoch(model, loader, loss_fn):
    model.eval()
    losses = []
    ys = []
    preds = []
    p_mal_list = []

    for img, y in loader:
        img, y = img.to(DEVICE), y.to(DEVICE)
        logits = model(img)

        loss = loss_fn(logits, y)
        losses.append(float(loss.item()))

        probs = torch.softmax(logits, dim=1)[:, 1]  # prob maligno
        p_mal_list.extend(probs.detach().cpu().numpy().tolist())

        p = torch.argmax(logits, dim=1)
        ys.extend(y.detach().cpu().numpy().tolist())
        preds.extend(p.detach().cpu().numpy().tolist())

    acc = accuracy_score(ys, preds)
    f1 = f1_score(ys, preds, zero_division=0)

    try:
        auc = roc_auc_score(ys, p_mal_list)
    except Exception:
        auc = float("nan")

    rec_mal = recall_score(ys, preds, pos_label=1, zero_division=0)

    # NEW: devolvemos también p_mal_list
    return float(np.mean(losses)), acc, f1, auc, rec_mal, ys, preds, p_mal_list


# =========================
# THRESHOLD utilities (NEW)
# =========================
def metrics_from_cm(cm: np.ndarray):
    # cm = [[tn, fp],
    #       [fn, tp]]
    tn, fp, fn, tp = cm.ravel()
    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    rec_mal = tp / max(1, (tp + fn))         # sensibilidad maligno
    spec = tn / max(1, (tn + fp))            # especificidad (benigno)
    prec_mal = tp / max(1, (tp + fp))        # precision maligno
    f1_mal = 2 * prec_mal * rec_mal / max(1e-12, (prec_mal + rec_mal))
    bal_acc = 0.5 * (spec + rec_mal)         # balanced accuracy
    return acc, prec_mal, rec_mal, spec, f1_mal, bal_acc


def find_best_threshold_on_val(
    ys_val: list[int],
    p_mal_val: list[float],
    target_recall_mal: float = 0.95,
    mode: str = "max_balanced",   # "max_balanced" | "max_spec" | "max_f1"
    n_grid: int = 500
):
    probs = np.array(p_mal_val, dtype=np.float32)
    ys = np.array(ys_val, dtype=np.int64)

    lo = float(np.clip(probs.min(), 0.0, 1.0))
    hi = float(np.clip(probs.max(), 0.0, 1.0))
    if hi - lo < 1e-6:
        thr = 0.5
        preds = (probs >= thr).astype(int)
        cm = confusion_matrix(ys, preds, labels=[0, 1])
        acc, prec, rec, spec, f1, bal = metrics_from_cm(cm)
        return thr, cm, {"acc": acc, "prec_mal": prec, "rec_mal": rec, "spec": spec, "f1_mal": f1, "bal_acc": bal}

    thresholds = np.linspace(lo, hi, n_grid)

    best_thr = None
    best_score = -1.0
    best_cm = None
    best_metrics = None

    for thr in thresholds:
        preds = (probs >= thr).astype(int)
        cm = confusion_matrix(ys, preds, labels=[0, 1])
        if cm.shape != (2, 2):
            continue

        acc, prec, rec, spec, f1, bal = metrics_from_cm(cm)

        # constraint: recall maligno mínimo
        if rec + 1e-12 < target_recall_mal:
            continue

        if mode == "max_balanced":
            score = bal
        elif mode == "max_spec":
            score = spec
        elif mode == "max_f1":
            score = f1
        else:
            raise ValueError("mode must be: max_balanced, max_spec, max_f1")

        if score > best_score:
            best_score = score
            best_thr = float(thr)
            best_cm = cm
            best_metrics = {"acc": acc, "prec_mal": prec, "rec_mal": rec, "spec": spec, "f1_mal": f1, "bal_acc": bal}

    # fallback: si no hay threshold que cumpla el target, maximizamos balanced sin constraint
    if best_thr is None:
        best_thr = 0.5
        best_score = -1.0
        for thr in thresholds:
            preds = (probs >= thr).astype(int)
            cm = confusion_matrix(ys, preds, labels=[0, 1])
            if cm.shape != (2, 2):
                continue
            acc, prec, rec, spec, f1, bal = metrics_from_cm(cm)
            score = bal
            if score > best_score:
                best_score = score
                best_thr = float(thr)
                best_cm = cm
                best_metrics = {"acc": acc, "prec_mal": prec, "rec_mal": rec, "spec": spec, "f1_mal": f1, "bal_acc": bal}

    return best_thr, best_cm, best_metrics


def print_threshold_report(name: str, ys: list[int], p_mal_list: list[float], threshold: float):
    preds = [1 if p >= threshold else 0 for p in p_mal_list]
    cm = confusion_matrix(ys, preds, labels=[0, 1])
    print(f"\n==== {name} (threshold={threshold:.4f}) ====")
    print("Confusion matrix:\n", cm)
    print(classification_report(ys, preds, digits=4))


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
    # Guardado
    here = os.path.dirname(os.path.abspath(__file__))
    SAVE_PATH = os.path.join(here, "artifacts", "best_model.pt")
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    print("Checkpoint se guardará en:", SAVE_PATH)
    print("DEVICE =", DEVICE)

    # Leer CSV
    df_train = pd.read_csv(CSV_TRAIN)
    df_test  = pd.read_csv(CSV_TEST)

    # Limpieza básica
    df_train = df_train.dropna(subset=[USE_PATH_COL, "pathology", "patient_id"])
    df_test  = df_test.dropna(subset=[USE_PATH_COL, "pathology", "patient_id"])

    # Resolver paths DICOM
    df_train["abs_path"] = df_train[USE_PATH_COL].apply(resolve_full_dicom_from_csv)
    df_test["abs_path"]  = df_test[USE_PATH_COL].apply(resolve_full_dicom_from_csv)

    before_tr, before_te = len(df_train), len(df_test)
    df_train = df_train.dropna(subset=["abs_path"]).copy()
    df_test  = df_test.dropna(subset=["abs_path"]).copy()

    print(f"Resolved train: {len(df_train)}/{before_tr}")
    print(f"Resolved test : {len(df_test)}/{before_te}")

    # Split train/val por paciente
    gss = GroupShuffleSplit(test_size=0.15, n_splits=1, random_state=SEED)
    tr_idx, va_idx = next(gss.split(df_train, groups=df_train["patient_id"]))
    tr_df = df_train.iloc[tr_idx].copy()
    va_df = df_train.iloc[va_idx].copy()

    print("Val pathology counts:\n", va_df["pathology"].value_counts())

    # ---- Class weights SUAVIZADOS: sqrt(1/count)
    y_train = tr_df["pathology"].map(lambda x: LABEL_MAP[str(x)]).astype(int).values
    counts = np.bincount(y_train, minlength=2)
    counts_safe = np.maximum(counts, 1)

    weights = np.sqrt(1.0 / counts_safe)
    weights = weights / weights.sum() * 2.0  # sum ~2
    class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

    print("Train class counts:", counts.tolist())
    print("Class weights (sqrt inv):", class_weights.detach().cpu().numpy().tolist())

    # =========================
    # LOSS (CHANGED): CrossEntropy + class weights (más balanceado que Focal)
    # =========================
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # ---- Transforms
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

    # ---- Datasets
    ds_tr = CBISImageDataset(tr_df, tfm_train)
    ds_va = CBISImageDataset(va_df, tfm_eval)
    ds_te = CBISImageDataset(df_test, tfm_eval)

    ds_tr_eval = CBISImageDataset(tr_df, tfm_eval)

    pin = (DEVICE == "cuda")
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=pin)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin)
    dl_te = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin)
    dl_tr_eval = DataLoader(ds_tr_eval, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin)

    # ---- Model
    model = ResNetBinaryTransfer().to(DEVICE)

    # =========================
    # FASE 1: congelar backbone, entrenar solo fc
    # =========================
    set_requires_grad_backbone(model, train_backbone=False)
    opt = make_optimizer(model, lr=LR_HEAD)
    print(f"FASE 1: entrenando solo fc durante {FREEZE_EPOCHS} épocas (lr={LR_HEAD})")

    # =========================
    # CHECKPOINT (CHANGED):
    #   Constraint: val_recall_mal >= 0.95
    #   Objetivo: max val_auc (mejor separación)
    # =========================
    best_auc = -1.0

    patience = 5
    bad = 0

    for epoch in range(1, NUM_EPOCHS + 1):
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
            loss = loss_fn(logits, y)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bs = y.size(0)
            total += float(loss.item()) * bs
            n += bs

        tr_loss = total / max(1, n)
        va_loss, va_acc, va_f1, va_auc, va_rec_mal, _, _, _ = eval_epoch(model, dl_va, loss_fn)

        print(
            f"[{epoch:02d}] train_loss={tr_loss:.4f}  "
            f"val_loss={va_loss:.4f}  val_acc={va_acc:.4f}  val_f1={va_f1:.4f}  "
            f"val_auc={va_auc:.4f}  val_recall_mal={va_rec_mal:.4f}"
        )

        improved = False
        if (va_rec_mal >= TARGET_RECALL_MAL) and (not np.isnan(va_auc)) and (va_auc > best_auc + 1e-4):
            improved = True

        if improved:
            best_auc = va_auc
            bad = 0

            torch.save({
                "model": model.state_dict(),
                "img_size": IMG_SIZE,
                "note": "transfer+crop+sqrt_classweights+CE_loss+2phase_finetune+auc_ckpt_with_recall_constraint",
                "best_val_auc": float(best_auc),
                "target_recall_mal": float(TARGET_RECALL_MAL),
            }, SAVE_PATH)
            print("✅ Guardado checkpoint:", SAVE_PATH)
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

    print("Existe best_model.pt?", os.path.exists(SAVE_PATH))

    # =========================
    # Evaluación FINAL con el mejor modelo + THRESHOLD balanceado en VAL
    # =========================
    ckpt = torch.load(SAVE_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])

    # VAL -> seleccionar threshold que mantiene recall_mal >= 0.95 y maximiza balanced accuracy
    va_loss, va_acc, va_f1, va_auc, va_rec_mal, ys_va, _, p_mal_va = eval_epoch(model, dl_va, loss_fn)

    best_thr, best_cm, best_m = find_best_threshold_on_val(
        ys_val=ys_va,
        p_mal_val=p_mal_va,
        target_recall_mal=TARGET_RECALL_MAL,
        mode=THRESH_MODE,
        n_grid=THRESH_GRID
    )

    print("\n==============================")
    print("THRESHOLD SELECCIONADO EN VAL")
    print(f"target_recall_mal={TARGET_RECALL_MAL}  mode={THRESH_MODE}  grid={THRESH_GRID}")
    print(f"best_threshold={best_thr:.4f}")
    print(f"VAL @thr: acc={best_m['acc']:.4f}  prec_mal={best_m['prec_mal']:.4f}  "
          f"rec_mal={best_m['rec_mal']:.4f}  spec={best_m['spec']:.4f}  "
          f"f1_mal={best_m['f1_mal']:.4f}  bal_acc={best_m['bal_acc']:.4f}")
    print("VAL confusion @thr:\n", best_cm)
    print("==============================\n")

    # TRAIN(EVAL)
    tr_loss_e, _, _, tr_auc_e, _, ys_tr, _, p_mal_tr = eval_epoch(model, dl_tr_eval, loss_fn)
    print(f"\nTRAIN(EVAL): loss={tr_loss_e:.4f} auc={tr_auc_e:.4f} (thresholded report)")
    print_threshold_report("TRAIN(EVAL)", ys_tr, p_mal_tr, best_thr)

    # VAL
    print(f"\nVAL: loss={va_loss:.4f} auc={va_auc:.4f} (thresholded report)")
    print_threshold_report("VAL", ys_va, p_mal_va, best_thr)

    # TEST
    te_loss, _, _, te_auc, _, ys_te, _, p_mal_te = eval_epoch(model, dl_te, loss_fn)
    print(f"\nTEST: loss={te_loss:.4f} auc={te_auc:.4f} (thresholded report)")
    print_threshold_report("TEST", ys_te, p_mal_te, best_thr)


if __name__ == "__main__":
    train()
