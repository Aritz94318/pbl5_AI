import os
import glob
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as tvt
from torchvision.models import resnet50, ResNet50_Weights

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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_TRAIN = os.path.join(BASE_DIR, "calc_case_description_train_set.csv")
CSV_TEST  = os.path.join(BASE_DIR, "calc_case_description_test_set.csv")
IMAGES_ROOT = os.path.join(BASE_DIR, "CBIS-DDSM")
# Full mammogram (cambia a "cropped image file path" si quieres comparar)
USE_PATH_COL = "image file path"

# Resolución alta
IMG_SIZE = 640

# 640 + ResNet50 consume bastante -> batch bajo
BATCH_SIZE = 32
NUM_EPOCHS = 12

FREEZE_EPOCHS = 5
LR_HEAD = 1e-3
LR_ALL  = 1e-4

WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABEL_MAP = {
    "MALIGNANT": 1,
    "BENIGN": 0,
    "BENIGN_WITHOUT_CALLBACK": 0
}

FORCE_FRESH_RUN = True

# Focal Loss
FOCAL_GAMMA = 1.5

# Threshold (clasificador final, calibrado)
TARGET_RECALL_MAL = 0.95
THRESH_MODE = "max_balanced"   # recomendado; cambia a "max_spec" si quieres priorizar especificidad
THRESH_GRID = 500

# =========================
# METADATA CONFIG (según tus columnas)
# =========================
USE_METADATA = True

# categóricas (convertiremos a IDs)
CAT_COLS = [
    "left or right breast",   # LEFT / RIGHT
    "image view",             # CC / MLO / etc.
    "calc type",
    "calc distribution",
]

# numéricas (normalizaremos)
NUM_COLS = [
    "assessment",       # BI-RADS/assessment (1..5/6)
    "breast density",   # 1..4
    "subtlety",         # 1..5
]

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
# Resolve full dicom path (prioriza full mammogram)
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
# DICOM loader -> PIL (grayscale) + crop simple
# =========================
def load_dicom_as_pil(dcm_path: str) -> Image.Image:
    ds = pydicom.dcmread(dcm_path, force=True)
    img = ds.pixel_array.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    img = img * slope + intercept

    lo, hi = np.percentile(img, (1, 99))
    img = np.clip(img, lo, hi)
    img = (img - lo) / (hi - lo + 1e-6)  # [0,1]

    mask = img > 0.05
    if mask.any():
        ys, xs = np.nonzero(mask)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        img = img[y0:y1+1, x0:x1+1]

    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img).convert("L")


# =========================
# Metadata encoders
# =========================
def build_category_maps(df: pd.DataFrame, cat_cols: list[str]) -> dict[str, dict[str, int]]:
    """
    Devuelve maps: col -> {category_string -> id}
    id 0 = unknown
    """
    maps = {}
    for c in cat_cols:
        vals = df[c].astype(str).fillna("UNK").str.strip().str.upper()
        uniq = sorted(set(vals.tolist()))
        # 0 reservado a unknown
        m = {"UNK": 0}
        nxt = 1
        for u in uniq:
            if u == "UNK":
                continue
            m[u] = nxt
            nxt += 1
        maps[c] = m
    return maps


def encode_category(value, mapping: dict[str, int]) -> int:
    if pd.isna(value):
        return 0
    s = str(value).strip().upper()
    return int(mapping.get(s, 0))


def safe_float(x) -> float:
    if pd.isna(x):
        return float("nan")
    try:
        return float(x)
    except Exception:
        s0 = str(x).strip()

        # Construye un string numérico pero:
        # - permite '-' SOLO si es el primer carácter (signo)
        out = []
        for i, ch in enumerate(s0):
            if ch.isdigit() or ch == "." or (ch == "-" and i == 0):
                out.append(ch)

        s = "".join(out)

        try:
            # caso "-" o "" -> nan
            if s in ("", "-"):
                return float("nan")
            return float(s)
        except Exception:
            return float("nan")



def compute_num_norm_stats(df: pd.DataFrame, num_cols: list[str]) -> dict[str, tuple[float, float]]:
    """
    mean/std por columna (con nan ignorados). Si std=0 -> std=1
    """
    stats = {}
    for c in num_cols:
        arr = df[c].apply(safe_float).to_numpy(dtype=np.float32)
        m = float(np.nanmean(arr)) if np.isfinite(arr).any() else 0.0
        s = float(np.nanstd(arr)) if np.isfinite(arr).any() else 1.0
        if s < 1e-6:
            s = 1.0
        stats[c] = (m, s)
    return stats


def normalize_num(x: float, mean: float, std: float) -> float:
    if not np.isfinite(x):
        return 0.0
    return (x - mean) / std


# =========================
# Dataset (imagen + metadata)
# =========================
class CBISImageMetaDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tfm, cat_maps: dict, num_stats: dict,
                 use_metadata: bool = True):
        self.df = df.reset_index(drop=True)
        self.tfm = tfm
        self.cat_maps = cat_maps
        self.num_stats = num_stats
        self.use_metadata = use_metadata

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img = load_dicom_as_pil(str(row["abs_path"]))
        img = self.tfm(img)

        y = torch.tensor(LABEL_MAP[str(row["pathology"])], dtype=torch.long)

        if not self.use_metadata:
            return img, y

        # categorical ids
        cat_ids = []
        for c in CAT_COLS:
            cat_ids.append(encode_category(row[c], self.cat_maps[c]))
        cat_ids = torch.tensor(cat_ids, dtype=torch.long)  # [n_cat]

        # numeric normalized
        nums = []
        for c in NUM_COLS:
            mean, std = self.num_stats[c]
            v = safe_float(row[c])
            nums.append(normalize_num(v, mean, std))
        nums = torch.tensor(nums, dtype=torch.float32)  # [n_num]

        return img, cat_ids, nums, y


# =========================
# Model: ResNet50 + metadata fusion
# =========================
class ResNet50Meta(nn.Module):
    def __init__(self, cat_cardinalities: list[int], num_dim: int, use_metadata: bool = True):
        super().__init__()
        self.use_metadata = use_metadata

        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)

        # 1 canal
        w_rgb = self.backbone.conv1.weight.data.clone()
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.conv1.weight.data = w_rgb.mean(dim=1, keepdim=True)

        img_feat_dim = self.backbone.fc.in_features  # 2048
        self.backbone.fc = nn.Identity()

        if self.use_metadata:
            # Embeddings por cada cat
            self.cat_embs = nn.ModuleList()
            emb_out_total = 0
            for card in cat_cardinalities:
                emb_dim = int(min(32, round(np.sqrt(card) + 1)))
                self.cat_embs.append(nn.Embedding(card, emb_dim))
                emb_out_total += emb_dim

            # pequeña MLP para números
            self.num_mlp = nn.Sequential(
                nn.Linear(num_dim, 32),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
            )

            head_in = img_feat_dim + emb_out_total + 32
        else:
            head_in = img_feat_dim

        self.head = nn.Sequential(
            nn.Linear(head_in, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        )

    def forward(self, x, cat_ids=None, nums=None):
        img_feats = self.backbone(x)  # [B,2048]

        if not self.use_metadata:
            return self.head(img_feats)

        # cat embeddings
        embs = []
        for i, emb in enumerate(self.cat_embs):
            embs.append(emb(cat_ids[:, i]))
        cat_feats = torch.cat(embs, dim=1) if len(embs) else torch.zeros((x.size(0), 0), device=x.device)

        num_feats = self.num_mlp(nums)

        feats = torch.cat([img_feats, cat_feats, num_feats], dim=1)
        return self.head(feats)


# =========================
# Focal Loss
# =========================
class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 1.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, target):
        ce = nn.functional.cross_entropy(logits, target, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        return loss.mean()


# =========================
# Temperature Scaling (calibración)
# =========================
class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_T = nn.Parameter(torch.zeros(1))  # log(T) para asegurar T>0

    def forward(self, logits):
        T = torch.exp(self.log_T) + 1e-6
        return logits / T


@torch.no_grad()
def collect_logits_and_labels(model, loader, use_metadata: bool):
    model.eval()
    all_logits = []
    all_y = []
    for batch in loader:
        if use_metadata:
            img, cat_ids, nums, y = batch
            img, cat_ids, nums = img.to(DEVICE), cat_ids.to(DEVICE), nums.to(DEVICE)
            logits = model(img, cat_ids, nums)
        else:
            img, y = batch
            img = img.to(DEVICE)
            logits = model(img)
        all_logits.append(logits.detach().cpu())
        all_y.append(y.detach().cpu())
    return torch.cat(all_logits, dim=0), torch.cat(all_y, dim=0)


def fit_temperature_on_val(val_logits: torch.Tensor, val_y: torch.Tensor, max_iter=2000):
    scaler = TemperatureScaler().to(DEVICE)
    optimizer = torch.optim.LBFGS(scaler.parameters(), lr=0.05, max_iter=max_iter)
    ce = nn.CrossEntropyLoss()

    val_logits = val_logits.to(DEVICE)
    val_y = val_y.to(DEVICE)

    def closure():
        optimizer.zero_grad()
        loss = ce(scaler(val_logits), val_y)
        loss.backward()
        return loss

    optimizer.step(closure)
    temp_t = float(torch.exp(scaler.log_T).detach().cpu().item())
    return max(temp_t, 1e-6)


# =========================
# Eval (temperature opcional)
# =========================
@torch.no_grad()
def eval_epoch(model, loader, loss_fn, use_metadata: bool, temperature: float | None = None):
    model.eval()
    losses, ys, preds, p_mal_list = [], [], [], []

    for batch in loader:
        if use_metadata:
            img, cat_ids, nums, y = batch
            img, cat_ids, nums, y = img.to(DEVICE), cat_ids.to(DEVICE), nums.to(DEVICE), y.to(DEVICE)
            logits = model(img, cat_ids, nums)
        else:
            img, y = batch
            img, y = img.to(DEVICE), y.to(DEVICE)
            logits = model(img)

        loss = loss_fn(logits, y)
        losses.append(float(loss.item()))

        logits_use = logits / max(temperature, 1e-6) if temperature is not None else logits

        probs = torch.softmax(logits_use, dim=1)[:, 1]
        p_mal_list.extend(probs.detach().cpu().numpy().tolist())

        p = torch.argmax(logits_use, dim=1)
        ys.extend(y.detach().cpu().numpy().tolist())
        preds.extend(p.detach().cpu().numpy().tolist())

    acc = accuracy_score(ys, preds)
    f1 = f1_score(ys, preds, zero_division=0)

    try:
        auc = roc_auc_score(ys, p_mal_list)
    except Exception:
        auc = float("nan")

    rec_mal = recall_score(ys, preds, pos_label=1, zero_division=0)

    return float(np.mean(losses)), acc, f1, auc, rec_mal, ys, preds, p_mal_list


# =========================
# Threshold utilities
# =========================
def metrics_from_cm(cm: np.ndarray):
    tn, fp, fn, tp = cm.ravel()
    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    rec_mal = tp / max(1, (tp + fn))
    spec = tn / max(1, (tn + fp))
    prec_mal = tp / max(1, (tp + fp))
    f1_mal = 2 * prec_mal * rec_mal / max(1e-12, (prec_mal + rec_mal))
    bal_acc = 0.5 * (spec + rec_mal)
    return acc, prec_mal, rec_mal, spec, f1_mal, bal_acc


def find_best_threshold_on_val(ys_val, p_mal_val, target_recall_mal=0.95, mode="max_spec", n_grid=500):
    probs = np.array(p_mal_val, dtype=np.float32)
    ys = np.array(ys_val, dtype=np.int64)

    lo = float(np.clip(probs.min(), 0.0, 1.0))
    hi = float(np.clip(probs.max(), 0.0, 1.0))
    thresholds = np.linspace(lo, hi, n_grid)

    best_thr, best_score, best_cm, best_metrics = None, -1.0, None, None

    for thr in thresholds:
        pred = (probs >= thr).astype(int)
        cm = confusion_matrix(ys, pred, labels=[0, 1])
        if cm.shape != (2, 2):
            continue
        acc, prec, rec, spec, f1, bal = metrics_from_cm(cm)

        if rec + 1e-12 < target_recall_mal:
            continue

        if mode == "max_spec":
            score = spec
        elif mode == "max_balanced":
            score = bal
        else:
            score = f1

        if score > best_score:
            best_score = score
            best_thr = float(thr)
            best_cm = cm
            best_metrics = {
                "acc": acc,
                "prec_mal": prec,
                "rec_mal": rec,
                "spec": spec,
                "f1_mal": f1,
                "bal_acc": bal
            }

    if best_thr is None:
        best_thr = 0.5
        pred = (probs >= best_thr).astype(int)
        best_cm = confusion_matrix(ys, pred, labels=[0, 1])
        acc, prec, rec, spec, f1, bal = metrics_from_cm(best_cm)
        best_metrics = {
            "acc": acc,
            "prec_mal": prec,
            "rec_mal": rec,
            "spec": spec,
            "f1_mal": f1,
            "bal_acc": bal
        }

    return best_thr, best_cm, best_metrics


def print_threshold_report(name, ys, p_mal_list, threshold):
    preds = [1 if p >= threshold else 0 for p in p_mal_list]
    cm = confusion_matrix(ys, preds, labels=[0, 1])
    print(f"\n==== {name} (threshold={threshold:.4f}) ====")
    print("Confusion matrix:\n", cm)
    print(classification_report(ys, preds, digits=4))


# =========================
# Freeze / optimizer
# =========================
def set_requires_grad_backbone(model: "ResNet50Meta", train_backbone: bool):
    for p in model.backbone.parameters():
        p.requires_grad = train_backbone
    for p in model.head.parameters():
        p.requires_grad = True
    if model.use_metadata:
        for p in model.cat_embs.parameters():
            p.requires_grad = True
        for p in model.num_mlp.parameters():
            p.requires_grad = True


def make_optimizer(model, lr):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=WEIGHT_DECAY)


# =========================
# Train
# =========================
def prepare_data():
    df_train = pd.read_csv(CSV_TRAIN)
    df_test = pd.read_csv(CSV_TEST)
    needed = [USE_PATH_COL, "pathology", "patient_id"] + (CAT_COLS + NUM_COLS if USE_METADATA else [])
    df_train = df_train.dropna(subset=[c for c in needed if c in df_train.columns])
    df_test = df_test.dropna(subset=[c for c in needed if c in df_test.columns])
    df_train["abs_path"] = df_train[USE_PATH_COL].apply(resolve_full_dicom_from_csv)
    df_test["abs_path"] = df_test[USE_PATH_COL].apply(resolve_full_dicom_from_csv)
    df_train = df_train.dropna(subset=["abs_path"]).copy()
    df_test = df_test.dropna(subset=["abs_path"]).copy()
    gss = GroupShuffleSplit(test_size=0.15, n_splits=1, random_state=SEED)
    tr_idx, va_idx = next(gss.split(df_train, groups=df_train["patient_id"]))
    tr_df = df_train.iloc[tr_idx].copy()
    va_df = df_train.iloc[va_idx].copy()
    return tr_df, va_df, df_test

def prepare_metadata(tr_df):
    use_metadata = USE_METADATA
    for c in CAT_COLS + NUM_COLS:
        if c not in tr_df.columns:
            print(f"⚠️ Falta columna metadata '{c}'. Desactivo metadata.")
            use_metadata = False
            break
    if use_metadata:
        cat_maps = build_category_maps(tr_df, CAT_COLS)
        num_stats = compute_num_norm_stats(tr_df, NUM_COLS)
        cat_cardinalities = [len(cat_maps[c]) for c in CAT_COLS]
        num_dim = len(NUM_COLS)
        print("✅ Metadata activada.")
        print("Cat cardinalities:", dict(zip(CAT_COLS, cat_cardinalities)))
        print("Num cols:", NUM_COLS)
    else:
        cat_maps, num_stats, cat_cardinalities, num_dim = {}, {}, [], 0
    return use_metadata, cat_maps, num_stats, cat_cardinalities, num_dim

def prepare_dataloaders(tr_df, va_df, df_test, cat_maps, num_stats, use_metadata):
    tfm_train = tvt.Compose([
        tvt.Resize((IMG_SIZE, IMG_SIZE)),
        tvt.RandomApply([tvt.RandomRotation(7)], p=0.5),
        tvt.RandomApply([tvt.ColorJitter(contrast=0.12)], p=0.5),
        tvt.ToTensor(),
        tvt.Normalize(mean=[0.5], std=[0.25]),
    ])
    tfm_eval = tvt.Compose([
        tvt.Resize((IMG_SIZE, IMG_SIZE)),
        tvt.ToTensor(),
        tvt.Normalize(mean=[0.5], std=[0.25]),
    ])
    ds_tr = CBISImageMetaDataset(tr_df, tfm_train, cat_maps, num_stats, use_metadata=use_metadata)
    ds_va = CBISImageMetaDataset(va_df, tfm_eval, cat_maps, num_stats, use_metadata=use_metadata)
    ds_te = CBISImageMetaDataset(df_test, tfm_eval, cat_maps, num_stats, use_metadata=use_metadata)
    ds_tr_eval = CBISImageMetaDataset(tr_df, tfm_eval, cat_maps, num_stats, use_metadata=use_metadata)
    pin = (DEVICE == "cuda")
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=pin)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin)
    dl_te = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin)
    dl_tr_eval = DataLoader(ds_tr_eval, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin)
    return dl_tr, dl_va, dl_te, dl_tr_eval

def get_class_weights(tr_df):
    y_train = tr_df["pathology"].map(lambda x: LABEL_MAP[str(x)]).astype(int).values
    counts = np.bincount(y_train, minlength=2)
    counts_safe = np.maximum(counts, 1)
    weights = np.sqrt(1.0 / counts_safe)
    weights = weights / weights.sum() * 2.0
    class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    return class_weights

def train_one_epoch(model, dl_tr, loss_fn, opt, use_metadata):
    model.train()
    total = 0.0
    n = 0
    for batch in dl_tr:
        opt.zero_grad(set_to_none=True)
        if use_metadata:
            img, cat_ids, nums, y = batch
            img, cat_ids, nums, y = img.to(DEVICE), cat_ids.to(DEVICE), nums.to(DEVICE), y.to(DEVICE)
            logits = model(img, cat_ids, nums)
        else:
            img, y = batch
            img, y = img.to(DEVICE), y.to(DEVICE)
            logits = model(img)
        loss = loss_fn(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        bs = y.size(0)
        total += float(loss.item()) * bs
        n += bs
    return total / max(1, n)

def save_best_checkpoint(model, use_metadata, best_auc, save_path):
    torch.save({
        "model": model.state_dict(),
        "img_size": IMG_SIZE,
        "use_metadata": bool(use_metadata),
        "cat_cols": CAT_COLS if use_metadata else [],
        "num_cols": NUM_COLS if use_metadata else [],
        "note": "full_mammogram+resnet50+img640+focal+metadata+2phase+best_by_val_auc",
        "best_val_auc": best_auc,
        "focal_gamma": float(FOCAL_GAMMA),
    }, save_path)
    print("✅ Guardado BEST checkpoint:", save_path)

def save_fallback_checkpoint(model, use_metadata, save_path):
    torch.save({
        "model": model.state_dict(),
        "img_size": IMG_SIZE,
        "use_metadata": bool(use_metadata),
        "note": "fallback_final_saved_as_best",
        "focal_gamma": float(FOCAL_GAMMA),
    }, save_path)
    print("⚠️ No hubo mejora; guardado modelo final como best_model.pt.")

def train_model(model, dl_tr, dl_va, loss_fn, use_metadata, save_path):
    best_auc = -1.0
    patience = 5
    bad = 0
    opt = make_optimizer(model, lr=LR_HEAD)
    set_requires_grad_backbone(model, train_backbone=False)
    print(f"FASE 1: head{' + metadata' if use_metadata else ''} durante {FREEZE_EPOCHS} épocas (lr={LR_HEAD})")
    for epoch in range(1, NUM_EPOCHS + 1):
        if epoch == FREEZE_EPOCHS + 1:
            set_requires_grad_backbone(model, train_backbone=True)
            opt = make_optimizer(model, lr=LR_ALL)
            print(f"FASE 2: entrenando TODO (lr={LR_ALL})")
        tr_loss = train_one_epoch(model, dl_tr, loss_fn, opt, use_metadata)
        va_loss, va_acc, va_f1, va_auc, va_rec_mal, _, _, _ = eval_epoch(
            model, dl_va, loss_fn, use_metadata=use_metadata, temperature=None
        )
        print(
            f"[{epoch:02d}] train_loss={tr_loss:.4f}  "
            f"val_loss={va_loss:.4f}  val_acc={va_acc:.4f}  val_f1={va_f1:.4f}  "
            f"val_auc={va_auc:.4f}  val_recall_mal={va_rec_mal:.4f}"
        )
        improved = (not np.isnan(va_auc)) and (va_auc > best_auc + 1e-4)
        if improved:
            best_auc = float(va_auc)
            bad = 0
            save_best_checkpoint(model, use_metadata, best_auc, save_path)
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break
    if not os.path.exists(save_path):
        save_fallback_checkpoint(model, use_metadata, save_path)

def final_evaluation(model, dl_va, dl_tr_eval, dl_te, loss_fn, use_metadata, save_path):
    ckpt = torch.load(save_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    val_logits, val_y = collect_logits_and_labels(model, dl_va, use_metadata=use_metadata)
    temp_t = fit_temperature_on_val(val_logits, val_y)
    print(f"\nTemperature scaling (fit en VAL): T={temp_t:.4f}")
    va_loss, _, _, va_auc, _, ys_va, _, p_mal_va = eval_epoch(
        model, dl_va, loss_fn, use_metadata=use_metadata, temperature=temp_t
    )
    best_thr, best_cm, best_m = find_best_threshold_on_val(
        ys_val=ys_va, p_mal_val=p_mal_va,
        target_recall_mal=TARGET_RECALL_MAL,
        mode=THRESH_MODE, n_grid=THRESH_GRID
    )
    torch.save({
        **ckpt,
        "temperature_T": float(temp_t),
        "final_threshold": float(best_thr),
        "threshold_target_recall_mal": float(TARGET_RECALL_MAL),
        "threshold_mode": str(THRESH_MODE),
    }, save_path)
    print("\n==============================")
    print("THRESHOLD SELECCIONADO EN VAL (clasificador final, calibrado)")
    print(f"target_recall_mal={TARGET_RECALL_MAL}  mode={THRESH_MODE}  grid={THRESH_GRID}")
    print(f"T={temp_t:.4f}  best_threshold={best_thr:.4f}")
    print(f"VAL @thr: acc={best_m['acc']:.4f}  prec_mal={best_m['prec_mal']:.4f}  "
          f"rec_mal={best_m['rec_mal']:.4f}  spec={best_m['spec']:.4f}  "
          f"f1_mal={best_m['f1_mal']:.4f}  bal_acc={best_m['bal_acc']:.4f}")
    print("VAL confusion @thr:\n", best_cm)
    print("==============================\n")
    tr_loss_e, _, _, tr_auc_e, _, ys_tr, _, p_mal_tr = eval_epoch(
        model, dl_tr_eval, loss_fn, use_metadata=use_metadata, temperature=temp_t
    )
    print(f"\nTRAIN(EVAL): loss={tr_loss_e:.4f} auc={tr_auc_e:.4f} (thresholded report)")
    print_threshold_report("TRAIN(EVAL)", ys_tr, p_mal_tr, best_thr)
    print(f"\nVAL: loss={va_loss:.4f} auc={va_auc:.4f} (thresholded report)")
    print_threshold_report("VAL", ys_va, p_mal_va, best_thr)
    te_loss, _, _, te_auc, _, ys_te, _, p_mal_te = eval_epoch(
        model, dl_te, loss_fn, use_metadata=use_metadata, temperature=temp_t
    )
    print(f"\nTEST: loss={te_loss:.4f} auc={te_auc:.4f} (thresholded report)")
    print_threshold_report("TEST", ys_te, p_mal_te, best_thr)

def train():
    here = os.path.dirname(os.path.abspath(__file__))
    SAVE_PATH = os.path.join(here, "artifacts", "best_model.pt")
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    if FORCE_FRESH_RUN and os.path.exists(SAVE_PATH):
        os.remove(SAVE_PATH)
        print("🧹 Borrado best_model.pt anterior para regenerarlo en esta ejecución.")
    print("DEVICE =", DEVICE)
    print("Backbone: ResNet50 | IMG_SIZE =", IMG_SIZE, "| BATCH_SIZE =", BATCH_SIZE)
    print("Loss: FocalLoss | Guardado por: VAL AUC | USE_METADATA =", USE_METADATA)
    print(f"Threshold selection (final): target_recall_mal={TARGET_RECALL_MAL} mode={THRESH_MODE}")
    tr_df, va_df, df_test = prepare_data()
    use_metadata, cat_maps, num_stats, cat_cardinalities, num_dim = prepare_metadata(tr_df)
    class_weights = get_class_weights(tr_df)
    loss_fn = FocalLoss(alpha=class_weights, gamma=FOCAL_GAMMA)
    dl_tr, dl_va, dl_te, dl_tr_eval = prepare_dataloaders(tr_df, va_df, df_test, cat_maps, num_stats, use_metadata)
    model = ResNet50Meta(cat_cardinalities=cat_cardinalities, num_dim=num_dim, use_metadata=use_metadata).to(DEVICE)
    train_model(model, dl_tr, dl_va, loss_fn, use_metadata, SAVE_PATH)
    final_evaluation(model, dl_va, dl_tr_eval, dl_te, loss_fn, use_metadata, SAVE_PATH)


if __name__ == "__main__":
    train()
