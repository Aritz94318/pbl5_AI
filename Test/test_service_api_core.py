import os
import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

import service_api as m


import pytest
import service_api as m




# -------------------------
# DICOM -> PIL
# -------------------------
def test_load_dicom_invalid_bytes_raises():
    with pytest.raises(Exception):
        m.load_dicom_as_pil_from_bytes(b"fake")


def test_load_dicom_valid_returns_grayscale(sample_dicom_bytes):
    img = m.load_dicom_as_pil_from_bytes(sample_dicom_bytes)
    assert isinstance(img, Image.Image)
    assert img.mode == "L"
    assert img.size[0] > 10


def test_tfm_eval_output_shape(sample_dicom_bytes):
    pil = m.load_dicom_as_pil_from_bytes(sample_dicom_bytes)
    x = m.tfm_eval(pil)
    assert isinstance(x, torch.Tensor)
    assert x.shape == (1, m.IMG_SIZE, m.IMG_SIZE)


# -------------------------
# Model
# -------------------------
def test_resnetbinarytransfer_forward_shape(monkeypatch):
    # Evitar descargas de weights: forzamos weights=None
    model = m.ResNetBinaryTransfer(weights=None).to("cpu").eval()
    x = torch.randn(2, 1, m.IMG_SIZE, m.IMG_SIZE)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 2)


def test_build_model_for_tests_no_download():
    model = m.build_model(for_tests=True)
    assert isinstance(model, nn.Module)
    # Debe tener conv1 de 1 canal
    assert model.net.conv1.in_channels == 1


def test_build_model_sets_eval_mode():
    model = m.build_model(for_tests=True)
    assert model.training is False


def test_load_model_raises_if_missing_ckpt(monkeypatch, tmp_path):
    # Simula que __file__ está en tmp_path (sin artifacts)
    fake_file = tmp_path / "service_api.py"
    fake_file.write_text("# fake")

    monkeypatch.setattr(m, "__file__", str(fake_file))
    with pytest.raises(FileNotFoundError):
        m.load_model()


# -------------------------
# Helpers: aggregate
# -------------------------
def test_aggregate_overall_max():
    assert m.aggregate_overall_max([0.1, 0.5, 0.2]) == pytest.approx(0.5)
    assert m.aggregate_overall_max([]) == pytest.approx(0.0)


# -------------------------
# infer_prob_malignant_from_bytes
# -------------------------
def test_infer_prob_raises_if_model_none(sample_dicom_bytes):
    old = m.MODEL
    try:
        m.MODEL = None
        with pytest.raises(RuntimeError) as e:
            m.infer_prob_malignant_from_bytes(sample_dicom_bytes)
        assert m.MODEL_NOT_LOADED_ERROR in str(e.value)
    finally:
        m.MODEL = old


def test_infer_prob_returns_float(monkeypatch, sample_dicom_bytes):
    # Modelo fake que devuelve logits fijos [B,2]
    class FakeModel(nn.Module):
        def forward(self, x):
            # logits: benign low, malignant high => softmax mal ~ alto
            return torch.tensor([[0.0, 5.0]], dtype=torch.float32)

    old = m.MODEL
    try:
        m.MODEL = FakeModel()
        p = m.infer_prob_malignant_from_bytes(sample_dicom_bytes)
        assert isinstance(p, float)
        assert 0.0 <= p <= 1.0
        assert p > 0.9
    finally:
        m.MODEL = old


# -------------------------
# load_dicom_bytes_from_url (mock httpx)
# -------------------------
@pytest.mark.asyncio
async def test_load_dicom_bytes_from_url_success(monkeypatch):
    class FakeResp:
        def raise_for_status(self):
            return None
        @property
        def content(self):
            return b"abc"

    class FakeClient:
        def __init__(self, *args, **kwargs):
            # FakeClient does not require any initialization
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc, tb):
            return False
        def get(self, url):
            return FakeResp()

    monkeypatch.setattr(m.httpx, "AsyncClient", FakeClient)
    
