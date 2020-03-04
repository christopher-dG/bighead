import os.path

from unittest.mock import patch, sentinel

import dlib
import numpy as np
import pytest

from bighead import face_detection as fd

IMAGE_DIR = os.path.join(os.path.dirname(__file__), "images")
ZERO_PATH = os.path.join(IMAGE_DIR, "zero.jpg")
ONE_PATH = os.path.join(IMAGE_DIR, "one.jpg")
TWO_PATH = os.path.join(IMAGE_DIR, "two.jpg")
BIG_PATH = os.path.join(IMAGE_DIR, "big.jpg")
with open(ONE_PATH, "rb") as f:
    ONE_BYTES = f.read()


def test_find_biggest_face():
    assert fd.find_biggest_face(path=ZERO_PATH) is None
    box = fd.find_biggest_face(path=ONE_PATH)
    assert isinstance(box, dlib.rectangle)
    assert not box.is_empty()
    assert box.left() > 300  # Ensure that we have the bigger face on the right.
    with pytest.raises(fd.TooLarge):
        fd.find_biggest_face(path=BIG_PATH)


def test_extract_biggest_face():
    assert fd.extract_biggest_face(path=ZERO_PATH) is None
    face = fd.extract_biggest_face(path=ONE_PATH)
    assert isinstance(face, np.ndarray)
    face = fd.extract_biggest_face(path=TWO_PATH)
    assert isinstance(face, np.ndarray)
    with pytest.raises(fd.TooLarge):
        fd.extract_biggest_face(path=BIG_PATH)


@patch("bighead.face_detection._load_from_data", return_value=sentinel.DATA)
@patch("bighead.face_detection._load_from_path", return_value=sentinel.PATH)
@patch("bighead.face_detection._load_from_url", return_value=sentinel.URL)
def test_load_image(load_from_url, load_from_path, load_from_data):
    assert fd._load_image(data=b"hi") is sentinel.DATA
    load_from_data.assert_called_with(b"hi")
    assert fd._load_image(path="foo") is sentinel.PATH
    load_from_path.assert_called_with("foo")
    assert fd._load_image(url="http://example.com") is sentinel.URL
    load_from_url.assert_called_with("http://example.com")
    img = np.random.randn(4, 4, 3)
    assert fd._load_image(image=img) is img
    with pytest.raises(ValueError):
        fd._load_image()


def _one_image_assertions(image):
    assert isinstance(image, np.ndarray)
    assert image.shape == (430, 800, 3)


def test_load_from_data():
    image = fd._load_from_data(ONE_BYTES)
    _one_image_assertions(image)
    with pytest.raises(fd.InvalidImage):
        fd._load_from_data(b"bad bad bad")


def test_load_from_path(tmp_path):
    image = fd._load_from_path(ONE_PATH)
    _one_image_assertions(image)
    with pytest.raises(FileNotFoundError):
        fd._load_from_path("this path does not exist")
    path = tmp_path / "test.jpg"
    with open(path, "w") as f:
        f.write("bad data")
    with pytest.raises(fd.InvalidImage):
        fd._load_from_path(str(path))


def test_load_from_url():
    with patch("requests.get") as get, pytest.raises(fd.HTTPError):
        get.return_value.raise_for_status.side_effect = fd.HTTPError()
        fd._load_from_url("http://example.com")
    with patch("requests.get") as get:
        get.return_value.configure_mock(
            content=ONE_BYTES, raise_for_status=lambda: None
        )
        image = fd._load_from_url("http://example.com")
        _one_image_assertions(image)
        get.assert_called_with("http://example.com")


def test_is_too_large():
    assert not fd._is_too_large(np.ndarray((1024, 1024, 3)))
    assert not fd._is_too_large(np.ndarray((2000, 100, 3)))
    assert fd._is_too_large(np.ndarray((1025, 1024, 3)))
    assert fd._is_too_large(np.ndarray((1024, 1025, 3)))
    assert fd._is_too_large(np.ndarray((1024, 1025, 3)))


@patch("bighead.face_detection._FACE_DETECTOR")
def test_get_bounding_box(FACE_DETECTOR):
    FACE_DETECTOR.side_effect = [
        dlib.rectangles([dlib.rectangle(1, 1, 3, 3), dlib.rectangle(1, 1, 4, 4)]),
        dlib.rectangles(),
    ]
    image = np.random.randn(4, 4, 3)
    assert fd._get_bounding_box(image, 2) == dlib.rectangle(1, 1, 4, 4)
    FACE_DETECTOR.assert_called_with(image, 2)
    assert fd._get_bounding_box(image, 1) is None


def test_crop():
    image = np.reshape(range(48), (4, 4, 3))
    rect = dlib.rectangle(2, 2, 4, 4)
    face = fd._crop(image, rect)
    assert isinstance(face, np.ndarray)
    assert face.shape == (2, 2, 3)
