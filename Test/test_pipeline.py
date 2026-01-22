import os
import types
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from PIL import Image

# Cambia este import al nombre real de tu script:
import train_agent_simple as m


# -------------------------
# Helpers / dummies
# -------------------------
class DummyDicom:
    def __init__(self, pixel_array, slope=1.0, intercept=0.0):
        self.pixel_array = pixel_array
        self.rescale_slope = slope
        self.rescale_intercept = intercept


class DummyBackbone(nn.Module):
    """
    Simula un resnet50:
    - Tiene conv1 con pesos RGB para que el código pueda promediar a 1 canal
    - Tiene fc con atributo in_features
    - forward devuelve features [B, 2048]
    """
    def __init__(self, feat_dim=2048):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.fc = nn.Linear(feat_dim, 2)
        self._feat_dim = feat_dim

    def forward(self, x):
        b = x.shape[0]
        return torch.zeros((b, self._feat_dim), dtype=x.dtype, device=x.device)


def make_fake_image(size=(224, 224)):
    rng = np.random.default_rng(seed=42)
    arr = (rng.random(size) * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")



# -------------------------
# Tests: seed_all
# -------------------------
def test_seed_all_reproducible():
    m.seed_all(123)
    a1 = torch.rand(3)
    m.seed_all(123)
    a2 = torch.rand(3)
    assert torch.allclose(a1, a2)


# -------------------------
# Tests: resolve_full_dicom_from_csv
# -------------------------
def test_resolve_full_dicom_prefers_full_mammogram_and_1_1(tmp_path, monkeypatch):
    # Estructura:
    # IMAGES_ROOT/caseA/.../*full mammogram images*/1-1.dcm
    images_root = tmp_path / "CBIS-DDSM"
    case_dir = images_root / "caseA"
    full_dir = case_dir / "something" / "full mammogram images" / "sub"
    full_dir.mkdir(parents=True)

    target = full_dir / "1-1.dcm"
    target.write_bytes(b"fake")

    other = full_dir / "2-1.dcm"
    other.write_bytes(b"fake")

    monkeypatch.setattr(m, "IMAGES_ROOT", str(images_root))

    # rel_path contiene "CBIS-DDSM/..."
    rel_path = "CBIS-DDSM/caseA/whatever"
    out = m.resolve_full_dicom_from_csv(rel_path)

    assert out is not None
    assert os.path.basename(out).lower() == "1-1.dcm"


def test_resolve_full_dicom_fallback_any_dcm(tmp_path, monkeypatch):
    images_root = tmp_path / "CBIS-DDSM"
    case_dir = images_root / "caseB"
    any_dir = case_dir / "random" / "folder"
    any_dir.mkdir(parents=True)

    d1 = any_dir / "9-9.dcm"
    d1.write_bytes(b"x")

    monkeypatch.setattr(m, "IMAGES_ROOT", str(images_root))

    out = m.resolve_full_dicom_from_csv("caseB/something")
    assert out is not None
    assert out.endswith(".dcm")


def test_resolve_full_dicom_missing_case_folder_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "IMAGES_ROOT", str(tmp_path / "CBIS-DDSM"))
    out = m.resolve_full_dicom_from_csv("CBIS-DDSM/nope/xxx")
    assert out is None


# -------------------------
# Tests: load_dicom_as_pil
# -------------------------
def test_load_dicom_as_pil_returns_grayscale_pil(monkeypatch):
    # pixel_array con valores simulados
    px = np.linspace(0, 1000, num=256*256, dtype=np.float32).reshape(256, 256)
    ds = DummyDicom(px, slope=1.0, intercept=0.0)

    def fake_dcmread(path, force=True):
        return ds

    monkeypatch.setattr(m.pydicom, "dcmread", fake_dcmread)

    img = m.load_dicom_as_pil("fake.dcm")
    assert isinstance(img, Image.Image)
    assert img.mode == "L"
    w, h = img.size
    assert w > 0 and h > 0


# -------------------------
# Tests: metadata encoders
# -------------------------
def test_build_category_maps_and_encode_category():
    df = pd.DataFrame({
        "col": ["LEFT", "RIGHT", None, "left", "UNK"]
    })
    maps = m.build_category_maps(df, ["col"])
    assert "col" in maps
    # 0 reservado
    assert maps["col"]["UNK"] == 0
    # encode conocido / desconocido
    assert m.encode_category("LEFT", maps["col"]) != 0
    assert m.encode_category("something_else", maps["col"]) == 0
    assert m.encode_category(None, maps["col"]) == 0


def test_safe_float_parses_numbers():
    assert m.safe_float(3) == pytest.approx(3.0)
    assert m.safe_float("4.5") == pytest.approx(4.5)
    assert m.safe_float("BI-RADS 4") == pytest.approx(4.0)
    assert np.isnan(m.safe_float(None))


def test_compute_num_norm_stats_and_normalize_num():
    df = pd.DataFrame({"a": [1, 2, 3, None, "BI-RADS 4"]})
    stats = m.compute_num_norm_stats(df, ["a"])
    mean, std = stats["a"]
    assert std > 0
    # normalizar nan -> 0
    assert m.normalize_num(float("nan"), mean, std) == pytest.approx(0.0)
    # normalizar valor finito
    z = m.normalize_num(2.0, mean, std)
    assert np.isfinite(z)


# -------------------------
# Tests: Dataset __getitem__
# -------------------------
def test_dataset_getitem_with_metadata(monkeypatch):
    # Evitar leer dicom: devolvemos PIL fake
    monkeypatch.setattr(m, "load_dicom_as_pil", lambda path: make_fake_image((64, 64)))

    # transform simple: PIL -> tensor [1,H,W]
    tfm = lambda im: torch.from_numpy(np.array(im, dtype=np.float32)[None, ...] / 255.0)

    df = pd.DataFrame({
        "abs_path": ["x.dcm"],
        "pathology": ["MALIGNANT"],
        "left or right breast": ["LEFT"],
        "image view": ["CC"],
        "calc type": ["TYPEA"],
        "calc distribution": ["DIST1"],
        "assessment": [4],
        "breast density": [2],
        "subtlety": [3],
    })

    cat_maps = m.build_category_maps(df, m.CAT_COLS)
    num_stats = m.compute_num_norm_stats(df, m.NUM_COLS)

    ds = m.CBISImageMetaDataset(df, tfm, cat_maps, num_stats, use_metadata=True)
    img, cat_ids, nums, y = ds[0]

    assert isinstance(img, torch.Tensor)
    assert img.shape[0] == 1
    assert cat_ids.ndim == 1 and len(cat_ids) == len(m.CAT_COLS)
    assert nums.ndim == 1 and len(nums) == len(m.NUM_COLS)
    assert y.item() == 1


def test_dataset_getitem_without_metadata(monkeypatch):
    monkeypatch.setattr(m, "load_dicom_as_pil", lambda path: make_fake_image((64, 64)))
    tfm = lambda im: torch.from_numpy(np.array(im, dtype=np.float32)[None, ...] / 255.0)

    df = pd.DataFrame({"abs_path": ["x.dcm"], "pathology": ["BENIGN"]})
    ds = m.CBISImageMetaDataset(df, tfm, cat_maps={}, num_stats={}, use_metadata=False)
    img, y = ds[0]

    assert img.shape[0] == 1
    assert y.item() == 0


# -------------------------
# Tests: Model forward (sin descargar pesos)
# -------------------------
def test_model_forward_with_metadata(monkeypatch):
    # Parchea resnet50 para que no descargue nada
    monkeypatch.setattr(m, "resnet50", lambda weights=None: DummyBackbone(feat_dim=2048))

    # cat_cardinalities: tamaño de vocab por cada cat
    cat_card = [5, 4, 6, 3]
    model = m.ResNet50Meta(cat_cardinalities=cat_card, num_dim=3, use_metadata=True)

    x = torch.randn(2, 1, 224, 224)
    cat_ids = torch.zeros((2, len(cat_card)), dtype=torch.long)
    nums = torch.zeros((2, 3), dtype=torch.float32)

    out = model(x, cat_ids, nums)
    assert out.shape == (2, 2)


def test_model_forward_without_metadata(monkeypatch):
    monkeypatch.setattr(m, "resnet50", lambda weights=None: DummyBackbone(feat_dim=2048))
    model = m.ResNet50Meta(cat_cardinalities=[], num_dim=0, use_metadata=False)

    x = torch.randn(3, 1, 224, 224)
    out = model(x)
    assert out.shape == (3, 2)


# -------------------------
# Tests: FocalLoss
# -------------------------
def test_focal_loss_runs_and_positive():
    loss_fn = m.FocalLoss(alpha=None, gamma=1.5)
    logits = torch.tensor([[2.0, -1.0], [-1.0, 2.0]])
    y = torch.tensor([0, 1], dtype=torch.long)
    loss = loss_fn(logits, y)
    assert loss.item() > 0.0


# -------------------------
# Tests: TemperatureScaler
# -------------------------
def test_temperature_scaler_scales_logits():
    scaler = m.TemperatureScaler()
    logits = torch.tensor([[1.0, 2.0]])
    out1 = scaler(logits)

    # subimos log_T => T mayor => logits más pequeños
    with torch.no_grad():
        scaler.log_T[:] = 2.0
    out2 = scaler(logits)

    assert torch.all(out2.abs() < out1.abs())


# -------------------------
# Tests: eval_epoch / collect_logits_and_labels (dummy)
# -------------------------
class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 2)

    def forward(self, x, cat_ids=None, nums=None):
        return self.lin(x)


def test_collect_logits_and_labels_no_metadata(monkeypatch):
    # Forzar DEVICE cpu para test
    monkeypatch.setattr(m, "DEVICE", "cpu")

    model = TinyModel()
    x = torch.randn(5, 4)
    y = torch.tensor([0, 1, 0, 1, 1])
    loader = [(x, y)]

    logits, labels = m.collect_logits_and_labels(model, loader, use_metadata=False)
    assert logits.shape == (5, 2)
    assert labels.shape == (5,)


# -------------------------
# Tests: Thresholding utilities
# -------------------------
def test_metrics_from_cm():
    cm = np.array([[8, 2],
                   [1, 9]])
    acc, _, rec, spec, f1, bal = m.metrics_from_cm(cm)
    assert 0 <= acc <= 1
    assert 0 <= rec <= 1
    assert 0 <= spec <= 1
    assert 0 <= f1 <= 1
    assert 0 <= bal <= 1


def test_find_best_threshold_on_val_respects_recall_constraint():
    ys = [0, 0, 0, 1, 1, 1]
    p  = [0.1, 0.2, 0.3, 0.9, 0.8, 0.7]

    thr, cm, _ = m.find_best_threshold_on_val(
        ys_val=ys,
        p_mal_val=p,
        target_recall_mal=1.0,   # obliga a recall perfecto
        mode="max_spec",
        n_grid=50
    )

    assert 0.0 <= thr <= 1.0
    assert cm.shape == (2, 2)


def test_find_best_threshold_on_val_fallback_to_0_5_when_no_solution():
    ys = [0, 0, 0, 1, 1, 1]
    p  = [0.1, 0.2, 0.3, 0.4, 0.45, 0.49]

    thr, cm, _ = m.find_best_threshold_on_val(
        ys_val=ys,
        p_mal_val=p,
        target_recall_mal=1.1,  # <-- imposible => fuerza fallback
        mode="max_spec",
        n_grid=50
    )

    assert thr == pytest.approx(0.5)
    assert cm.shape == (2, 2)



# -------------------------
# Tests: Freeze / optimizer
# -------------------------
def test_set_requires_grad_backbone(monkeypatch):
    monkeypatch.setattr(m, "resnet50", lambda weights=None: DummyBackbone(feat_dim=2048))
    model = m.ResNet50Meta(cat_cardinalities=[3, 3, 3, 3], num_dim=3, use_metadata=True)

    m.set_requires_grad_backbone(model, train_backbone=False)
    assert all(p.requires_grad is False for p in model.backbone.parameters())
    assert all(p.requires_grad is True for p in model.head.parameters())
    assert all(p.requires_grad is True for p in model.cat_embs.parameters())
    assert all(p.requires_grad is True for p in model.num_mlp.parameters())

    m.set_requires_grad_backbone(model, train_backbone=True)
    assert all(p.requires_grad is True for p in model.backbone.parameters())


def test_make_optimizer_only_includes_trainable_params(monkeypatch):
    monkeypatch.setattr(m, "resnet50", lambda weights=None: DummyBackbone(feat_dim=2048))
    model = m.ResNet50Meta(cat_cardinalities=[], num_dim=0, use_metadata=False)

    # congela todo backbone, deja head
    m.set_requires_grad_backbone(model, train_backbone=False)
    opt = m.make_optimizer(model, lr=1e-3)

    params = []
    for g in opt.param_groups:
        params.extend(g["params"])

    # todos deben ser requires_grad=True
    assert all(p.requires_grad for p in params)
    assert len(params) > 0
