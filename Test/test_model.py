import torch
from service_api import build_model, IMG_SIZE, DEVICE


def test_model_build():
    model = build_model()   # ⬅️ función que SOLO crea el modelo
    assert model is not None


def test_forward_shape():
    model = build_model()
    x = torch.randn(1, 1, IMG_SIZE, IMG_SIZE).to(DEVICE)

    with torch.no_grad():
        y = model(x)

    assert y.shape == (1, 2)
