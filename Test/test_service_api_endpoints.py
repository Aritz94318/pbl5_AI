import pytest
import asyncio
from fastapi.testclient import TestClient

import service_api as m


def test_home_returns_html():
    client = TestClient(m.app)
    r = client.get("/")
    assert r.status_code == 200
    assert "<html" in r.text.lower()
    assert "pink alert" in r.text.lower()


def test_predict4_returns_400_if_not_4_files(monkeypatch):
    client = TestClient(m.app)

    # Asegura que el modelo existe para no devolver 500
    old = m.MODEL
    try:
        m.MODEL = object()  # cualquier cosa no-None
        files = [("files", ("a.dcm", b"x", "application/dicom"))]  # 1 archivo
        r = client.post("/predict4", files=files)
        assert r.status_code == 400
        assert "exactly 4" in r.json()["detail"].lower()
    finally:
        m.MODEL = old


def test_predict4_returns_500_if_model_not_loaded(sample_dicom_bytes):
    client = TestClient(m.app)
    old = m.MODEL
    try:
        m.MODEL = None
        files = [("files", (f"{i}.dcm", sample_dicom_bytes, "application/dicom")) for i in range(4)]
        r = client.post("/predict4", files=files)
        assert r.status_code == 500
        assert m.MODEL_NOT_LOADED_ERROR.lower() in r.json()["detail"].lower()
    finally:
        m.MODEL = old


def test_predict4_ok_benign(monkeypatch, sample_dicom_bytes):
    client = TestClient(m.app)

    old = m.MODEL
    try:
        # Modelo "cargado"
        m.MODEL = object()

        # Forzamos inferencias bajas -> benign
        monkeypatch.setattr(m, "infer_prob_malignant_from_bytes", lambda b: 0.1)
        files = [("files", (f"{i}.dcm", sample_dicom_bytes, "application/dicom")) for i in range(4)]
        r = client.post("/predict4", files=files)

        assert r.status_code == 200
        data = r.json()
        assert data["overall_prediction"] == "BENIGN"
        assert 0.0 <= data["overall_prob_malignant"] <= 1.0
    finally:
        m.MODEL = old


def test_predict4_ok_malignant(monkeypatch, sample_dicom_bytes):
    client = TestClient(m.app)

    old = m.MODEL
    try:
        m.MODEL = object()
        monkeypatch.setattr(m, "infer_prob_malignant_from_bytes", lambda b: 0.9)
        files = [("files", (f"{i}.dcm", sample_dicom_bytes, "application/dicom")) for i in range(4)]
        r = client.post("/predict4", files=files)

        assert r.status_code == 200
        data = r.json()
        assert data["overall_prediction"] == "MALIGNANT"
        assert data["overall_prob_malignant"] == pytest.approx(0.9)
    finally:
        m.MODEL = old


@pytest.mark.asyncio
async def test_predict4_url_ok_and_forwards(monkeypatch, sample_dicom_bytes):
    client = TestClient(m.app)

    old = m.MODEL
    try:
        m.MODEL = object()

        # 1) Mock descarga bytes (sin red)
        async def fake_load_bytes(url: str) -> bytes:
            import asyncio
            await asyncio.sleep(0)
            return sample_dicom_bytes

        monkeypatch.setattr(m, "load_dicom_bytes_from_url", fake_load_bytes)

        # 2) Mock inferencia
        monkeypatch.setattr(m, "infer_prob_malignant_from_bytes", lambda b: 0.2)

        # 3) Mock forward_result para comprobar que se llama
        called = {"ok": False, "payload": None}

        async def fake_forward(payload: dict):
            
            called["ok"] = True
            called["payload"] = payload
            await asyncio.sleep(0)  # Use an async feature to avoid linter error

        monkeypatch.setattr(m, "forward_result", fake_forward)

        payload = {
            "diagnosis_id": "D123",
            "email": "a@b.com",
            "dicom_url": "http://x/1.dcm",
            "dicom_url2": "http://x/2.dcm",
            "dicom_url3": "http://x/3.dcm",
            "dicom_url4": "http://x/4.dcm",
        }

        r = client.post("/predict4-url", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert data["overall_prediction"] == "BENIGN"
        assert 0.0 <= data["overall_prob_malignant"] <= 1.0

        assert called["ok"] is True
        assert called["payload"]["diagnosis_id"] == "D123"
        assert called["payload"]["email"] == "a@b.com"
        assert called["payload"]["prediction"] in ("BENIGN", "MALIGNANT")
        assert "prob_malignant" in called["payload"]
    finally:
        m.MODEL = old


def test_predict4_url_500_if_model_not_loaded():
    client = TestClient(m.app)
    old = m.MODEL
    try:
        m.MODEL = None
        payload = {
            "diagnosis_id": "D123",
            "email": "a@b.com",
            "dicom_url": "http://x/1.dcm",
            "dicom_url2": "http://x/2.dcm",
            "dicom_url3": "http://x/3.dcm",
            "dicom_url4": "http://x/4.dcm",
        }
        r = client.post("/predict4-url", json=payload)
        assert r.status_code == 500
        assert m.MODEL_NOT_LOADED_ERROR.lower() in r.json()["detail"].lower()
    finally:
        m.MODEL = old


def test_predict4_url_400_if_processing_error(monkeypatch):
    client = TestClient(m.app)

    old = m.MODEL
    try:
        m.MODEL = object()

        async def fake_load_bytes(url: str) -> bytes:
            raise RuntimeError("boom")

        monkeypatch.setattr(m, "load_dicom_bytes_from_url", fake_load_bytes)

        payload = {
            "diagnosis_id": "D123",
            "email": "a@b.com",
            "dicom_url": "http://x/1.dcm",
            "dicom_url2": "http://x/2.dcm",
            "dicom_url3": "http://x/3.dcm",
            "dicom_url4": "http://x/4.dcm",
        }

        r = client.post("/predict4-url", json=payload)
        assert r.status_code == 400
        assert "error processing dicom" in r.json()["detail"].lower()
    finally:
        m.MODEL = old
