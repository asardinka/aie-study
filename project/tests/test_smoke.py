"""Smoke-тесты API без поднятия uvicorn (FastAPI TestClient)."""

from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

# До импорта приложения задаём каталог с чекпоинтами (как в docker-compose).
_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("ARTIFACTS_DIR", str(_ROOT / "artifacts"))

from src.tumor_app.api import app  # noqa: E402

client = TestClient(app)

_CLS_AX = _ROOT / "artifacts" / "classification_ax_model.pt"


def test_health() -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body.get("status") == "ok"
    assert "service" in body


def test_predict_validation_no_file() -> None:
    r = client.post("/predict", data={"plane": "ax"})
    assert r.status_code == 422


def test_predict_bad_plane() -> None:
    buf = BytesIO()
    Image.new("L", (16, 16), 200).save(buf, format="PNG")
    buf.seek(0)
    r = client.post(
        "/predict",
        data={"plane": "wrong"},
        files={"file": ("x.png", buf.getvalue(), "image/png")},
    )
    assert r.status_code == 400


@pytest.mark.skipif(not _CLS_AX.is_file(), reason="нет classification_ax_model.pt в artifacts/")
def test_predict_ok_small_image() -> None:
    buf = BytesIO()
    Image.new("L", (64, 64), 128).save(buf, format="PNG")
    buf.seek(0)
    r = client.post(
        "/predict",
        data={"plane": "ax"},
        files={"file": ("slice.png", buf.getvalue(), "image/png")},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert "predicted_class" in data
    assert data.get("plane") == "ax"
    assert data.get("image_original_b64")
    assert data.get("image_overlay_b64")
