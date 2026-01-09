import os
import glob
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
import torchvision.models as models

import pydicom
from PIL import Image

from collections import Counter
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score


# =========================
# CONFIG
# =========================
CSV_TRAIN = r"D:\Clase\Ingeneria\3.Maila\1.Sehilekoa\PBL\Image\manifest-ZkhPvrLo5216730872708713142\pbl5_AI\calc_case_description_train_set.csv"
CSV_TEST  = r"D:\Clase\Ingeneria\3.Maila\1.Sehilekoa\PBL\Image\manifest-ZkhPvrLo5216730872708713142\pbl5_AI\calc_case_description_test_set.csv"


IMAGES_ROOT = r"D:\Clase\Ingeneria\3.Maila\1.Sehilekoa\PBL\Image\manifest-ZkhPvrLo5216730872708713142\pbl5_AI\CBIS-DDSM"

USE_PATH_COL = "image file path"  # FULL images
META_COLS = ["breast density", "assessment", "subtlety"]
TEXT_COLS = ["abnormality type", "calc type", "calc distribution", "image view", "left or right breast"]

BATCH_SIZE = 8
NUM_EPOCHS = 15
LR = 2e-4
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


# =========================
# NEW: Resolve real FULL .dcm from your disk structure
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
# DICOM loader
# =========================
def load_dicom_as_pil(dcm_path: str) -> Image.Image:
    ds = pydicom.dcmread(dcm_path, force=True)
    img = ds.pixel_array.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    img = img * slope + intercept

    lo, hi = np.percentile(img, (1, 99))
    img = np.clip(img, lo, hi)
    img = (img - lo) / (hi - lo + 1e-6)

    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img).convert("L")


# =========================
# Vocab builder
# =========================
def build_vocab(df: pd.DataFrame, max_vocab=5000):
    counter = Counter()
    for _, row in df.iterrows():
        text = " ".join([str(row.get(c, "")) for c in TEXT_COLS]).lower()
        toks = [t for t in text.replace("_", " ").split() if t]
        counter.update(toks)

    vocab = {"<pad>": 0, "<unk>": 1}
    for tok, _ in counter.most_common(max_vocab):
        if tok not in vocab:
            vocab[tok] = len(vocab)
    return vocab


# =========================
# Dataset
# =========================
class CBISCalcDataset(Dataset):
    def __init__(self, df, tfm, vocab):
        self.df = df.reset_index(drop=True)
        self.tfm = tfm
        self.vocab = vocab

    def __len__(self): return len(self.df)

    def _encode_text(self, row, max_len=64):
        text = " ".join([str(row.get(c, "")) for c in TEXT_COLS]).lower()
        toks = [t for t in text.replace("_", " ").split() if t]
        ids = [self.vocab.get(t, self.vocab["<unk>"]) for t in toks][:max_len]
        if len(ids) == 0:
            ids = [self.vocab["<unk>"]]
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        dcm_path = str(row["abs_path"])

        img = load_dicom_as_pil(dcm_path)
        img = self.tfm(img)

        meta = torch.tensor(row[META_COLS].values.astype("float32"))
        text_ids = self._encode_text(row)

        y = torch.tensor(LABEL_MAP[str(row["pathology"])], dtype=torch.long)
        return img, meta, text_ids, y


def collate_batch(batch):
    imgs, metas, text_ids, ys = zip(*batch)
    imgs = torch.stack(imgs, 0)
    metas = torch.stack(metas, 0)
    ys = torch.stack(ys, 0)

    max_len = max(x.size(0) for x in text_ids)
    padded = torch.zeros((len(text_ids), max_len), dtype=torch.long)
    for i, x in enumerate(text_ids):
        padded[i, :x.size(0)] = x
    return imgs, metas, padded, ys


# =========================
# Model
# =========================
class MultiModalNet(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.cnn = models.resnet18(weights=None)
        self.cnn.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        cnn_out = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity()

        self.meta = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.emb = nn.Embedding(vocab_size, 64, padding_idx=0)
        self.txt = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.head = nn.Sequential(
            nn.Linear(cnn_out + 32 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 2)
        )

    def forward(self, img, meta, text_ids):
        v_img = self.cnn(img)
        v_meta = self.meta(meta)

        e = self.emb(text_ids)
        v_txt = e.mean(dim=1)
        v_txt = self.txt(v_txt)

        x = torch.cat([v_img, v_meta, v_txt], dim=1)
        return self.head(x)


# =========================
# Train / Eval
# =========================
@torch.no_grad()
def eval_epoch(model, loader):
    model.eval()
    all_probs = []
    all_y = []
    total_loss = 0.0
    n = 0
    ce = nn.CrossEntropyLoss()

    for img, meta, txt, y in loader:
        img, meta, txt, y = img.to(DEVICE), meta.to(DEVICE), txt.to(DEVICE), y.to(DEVICE)
        logits = model(img, meta, txt)
        loss = ce(logits, y)

        probs = torch.softmax(logits, dim=1)[:, 1]
        all_probs.append(probs.detach().cpu().numpy())
        all_y.append(y.detach().cpu().numpy())

        bs = y.size(0)
        total_loss += float(loss.item()) * bs
        n += bs

    all_probs = np.concatenate(all_probs)
    all_y = np.concatenate(all_y)

    auc = roc_auc_score(all_y, all_probs) if len(np.unique(all_y)) > 1 else float("nan")
    return total_loss / max(1, n), auc


def train():
    # ======== SAVE PATH ABSOLUTO (parche)
    here = os.path.dirname(os.path.abspath(__file__))
    SAVE_PATH = os.path.join(here, "artifacts", "best_model.pt")
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    print("Checkpoint se guardará en:", SAVE_PATH)

    df_train = pd.read_csv(os.path.join(here, CSV_TRAIN))
    df_test  = pd.read_csv(os.path.join(here, CSV_TEST))

    df_train = df_train.dropna(subset=[USE_PATH_COL, "pathology", "patient_id"])
    df_test  = df_test.dropna(subset=[USE_PATH_COL, "pathology", "patient_id"])

    # resolver rutas
    df_train["abs_path"] = df_train[USE_PATH_COL].apply(resolve_full_dicom_from_csv)
    df_test["abs_path"]  = df_test[USE_PATH_COL].apply(resolve_full_dicom_from_csv)

    before_tr, before_te = len(df_train), len(df_test)
    df_train = df_train.dropna(subset=["abs_path"]).copy()
    df_test  = df_test.dropna(subset=["abs_path"]).copy()

    print(f"Resolved train: {len(df_train)}/{before_tr}")
    print(f"Resolved test : {len(df_test)}/{before_te}")

    # split por paciente
    gss = GroupShuffleSplit(test_size=0.15, n_splits=1, random_state=SEED)
    tr_idx, va_idx = next(gss.split(df_train, groups=df_train["patient_id"]))
    tr_df = df_train.iloc[tr_idx].copy()
    va_df = df_train.iloc[va_idx].copy()

    print("Val pathology counts:\n", va_df["pathology"].value_counts())

    vocab = build_vocab(tr_df)

    tfm_train = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.RandomApply([T.RandomRotation(7)], p=0.5),
        T.RandomApply([T.ColorJitter(contrast=0.15)], p=0.5),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.25])
    ])
    tfm_eval = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.25])
    ])

    ds_tr = CBISCalcDataset(tr_df, tfm_train, vocab)
    ds_va = CBISCalcDataset(va_df, tfm_eval, vocab)
    ds_te = CBISCalcDataset(df_test, tfm_eval, vocab)

    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True,
                       num_workers=NUM_WORKERS, collate_fn=collate_batch, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=NUM_WORKERS, collate_fn=collate_batch, pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=NUM_WORKERS, collate_fn=collate_batch, pin_memory=True)

    model = MultiModalNet(vocab_size=len(vocab)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    ce = nn.CrossEntropyLoss()

    best_auc = -1.0
    patience = 4
    bad = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total = 0.0
        n = 0

        for img, meta, txt, y in dl_tr:
            img, meta, txt, y = img.to(DEVICE), meta.to(DEVICE), txt.to(DEVICE), y.to(DEVICE)

            opt.zero_grad(set_to_none=True)
            logits = model(img, meta, txt)
            loss = ce(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bs = y.size(0)
            total += float(loss.item()) * bs
            n += bs

        tr_loss = total / max(1, n)
        va_loss, va_auc = eval_epoch(model, dl_va)

        print(f"[{epoch:02d}] train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  val_auc={va_auc:.4f}")

        # ======== PARCHE: guardar SI o SI en epoch 1
        should_save = (epoch == 1) or (np.isfinite(va_auc) and va_auc > best_auc + 1e-4)

        if should_save:
            if np.isfinite(va_auc):
                best_auc = max(best_auc, va_auc)
            bad = 0

            torch.save({
                "model": model.state_dict(),
                "vocab": vocab,
                "img_size": IMG_SIZE,
                "meta_cols": META_COLS,
                "text_cols": TEXT_COLS,
                "use_path_col": USE_PATH_COL
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
    te_loss, te_auc = eval_epoch(model, dl_te)
    print(f"TEST: loss={te_loss:.4f} auc={te_auc:.4f}")


if __name__ == "__main__":
    train()
