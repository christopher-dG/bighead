from unittest.mock import Mock

import dlib
import numpy as np
import pytest

from requests import codes

from bighead import web


@pytest.fixture
def client():
    with web.app.test_client() as client:
        yield client


@web.app.route("/die")
def die():
    raise Exception("!")


def test_internal_server_error(client):
    resp = client.get("/die")
    assert resp.status_code == codes.server_error
    assert resp.is_json
    assert resp.json.get("error") == "Internal server error"


def test_not_found(client):
    resp = client.get("/abcdef", content_type="application/json")
    assert resp.status_code == codes.not_found
    assert resp.is_json
    assert resp.json.get("error") == "Not found"


def test_method_not_allowed(client):
    resp = client.get("/detect_largest_face")
    assert resp.status_code == codes.method_not_allowed
    assert resp.is_json
    assert resp.json.get("error") == "Method not allowed"


def test_detect_largest_face(client, monkeypatch):
    resp = client.post(
        "/detect_largest_face", data=b"bad", headers={"Content-Type": "image/png"}
    )
    assert resp.status_code == codes.unprocessable
    monkeypatch.setattr(
        web, "_preprocess_request", Mock(side_effect=[(b"hi", 1), ValueError("ah")])
    )
    monkeypatch.setattr(web, "_detect", Mock(return_value=({"foo": "bar"}, codes.ok)))
    resp = client.post("/detect_largest_face")
    assert resp.json == {"foo": "bar"}
    assert resp.status_code == codes.ok
    web._detect.assert_called_with(b"hi", 1)
    resp = client.post("/detect_largest_face")
    assert resp.json == {"error": "ah"}
    assert resp.status_code == codes.bad


def test_preprocess_request(client):
    with web.app.test_request_context("/detect_largest_face", data=b""), pytest.raises(
        ValueError
    ):
        web._preprocess_request()
    with web.app.test_request_context(
        "/detect_largest_face?upsample=ok", data=b"hi"
    ), pytest.raises(ValueError):
        web._preprocess_request()
    with web.app.test_request_context(
        "/detect_largest_face?upsample=-1", data=b"hi"
    ), pytest.raises(ValueError):
        web._preprocess_request()
    with web.app.test_request_context("/detect_largest_face?upsample=2", data=b"hi"):
        assert web._preprocess_request() == (b"hi", 2)


def test_error():
    assert web._error("message", codes.not_found) == (
        {"error": "message"},
        codes.not_found,
    )


def test_success():
    assert web._success(None) == ({}, codes.ok)
    box = dlib.rectangle(1, 2, 3, 4)
    assert web._success(box) == (
        {"box": {"left": 1, "right": 3, "top": 2, "bottom": 4}},
        codes.ok,
    )


def test_detect(monkeypatch):
    monkeypatch.setattr(
        web,
        "find_biggest_face",
        Mock(
            side_effect=[
                dlib.rectangle(1, 2, 3, 4),
                None,
                web.InvalidImage(RuntimeError("ahh")),
                web.TooLarge(np.random.randn(4, 4, 3)),
            ]
        ),
    )
    assert web._detect(b"hi", 1) == (
        {"box": {"left": 1, "right": 3, "top": 2, "bottom": 4}},
        codes.ok,
    )
    web.find_biggest_face.assert_called_with(data=b"hi", upsample=1)
    assert web._detect(b"hi", 1) == ({}, codes.ok)
    assert web._detect(b"hi", 1) == (
        {"error": "The image is invalid"},
        codes.unprocessable,
    )
    assert web._detect(b"hi", 1) == (
        {"error": "Image is too large: (4, 4)"},
        codes.request_entity_too_large,
    )
