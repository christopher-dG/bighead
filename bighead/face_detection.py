import os

from tempfile import mkstemp
from typing import Optional

import dlib
import numpy as np
import requests

_FACE_DETECTOR = dlib.get_frontal_face_detector()
_MAX_PIXELS = 1024 ** 2
HTTPError = requests.HTTPError


class InvalidImage(Exception):
    """
    Raised when an image is of an invalid format.

    Attributes:
        error (RuntimeError): error from dlib.load_rgb_image.
    """

    def __init__(self, error: RuntimeError) -> None:
        self.error = error
        super().__init__(*error.args)


class TooLarge(Exception):
    """
    Raised when an image is too large to process.

    Attributes:
        dims (Tuple[int, int]): dimensions (y, x) of the image.
        msg (str): Error message suitable for display.
    """

    def __init__(self, image: np.ndarray) -> None:
        self.dims = image.shape[:2]
        self.msg = f"Image is too large: {self.dims}"
        super().__init__(self.msg)


def find_biggest_face(
    *,
    data: Optional[bytes] = None,
    image: Optional[np.ndarray] = None,
    path: Optional[str] = None,
    url: Optional[str] = None,
    upsample: int = 0,
) -> Optional[dlib.rectangle]:
    """
    Get the bounding box for the biggest face in the image.
    Only one image source should be specified.

    Kwargs:
        data: raw image data.
        image: an existing image array.
        path: path to image file.
        url: URL pointing to image file.
        upsample: number of times to upsample the image.
    Returns:
        bounding box around the biggest face, or None if no faces were found.
    Raises:
        FileNotFoundError: when the path does not point to a file.
        HTTPError: when the HTTP request fails.
        InvalidImage: when the input image is invalid.
        TooLarge: when the input image is too large.
        ValueError: when you fail to specify an image source.
    """
    to_search = _load_image(data=data, image=image, path=path, url=url)
    if _is_too_large(to_search):
        raise TooLarge(to_search)
    return _get_bounding_box(to_search, upsample)


def extract_biggest_face(
    *,
    data: Optional[bytes] = None,
    image: Optional[np.ndarray] = None,
    path: Optional[str] = None,
    url: Optional[str] = None,
    upsample: int = 0,
) -> Optional[np.ndarray]:
    """
    Find and crop out the biggest face in the image.
    Only one image source should be specified.

    Kwargs:
        data: raw image data.
        image: an existing image array.
        path: path to image file.
        url: URL pointing to image file.
        upsample: number of times to upsample the image.
    Returns:
        cropped image containing the biggest face, or None if no faces were found.
    Raises:
        FileNotFoundError: when the path does not point to a file.
        HTTPError: when the HTTP request fails.
        InvalidImage: when the input image is invalid.
        TooLarge: when the input image is too large.
        ValueError: when you fail to specify an image source.
    """
    to_search = _load_image(data=data, image=image, path=path, url=url)
    if _is_too_large(to_search):
        raise TooLarge(to_search)
    box = _get_bounding_box(to_search, upsample)
    if not box:
        return None
    return _crop(to_search, box)


def _load_image(
    *,
    data: Optional[bytes] = None,
    image: Optional[np.ndarray] = None,
    path: Optional[str] = None,
    url: Optional[str] = None,
) -> np.ndarray:
    """
    Load an image.
    Only one image source should be specified.

    Kwargs:
        data: raw image data.
        image: an existing image array.
        path: path to image file.
        url: URL pointing to image file.
        upsample: number of times to upsample the image.
    Returns:
        loaded image.
    Raises:
        FileNotFoundError: when the path does not point to a file.
        HTTPError: when the HTTP request fails.
        InvalidImage: when the input image is invalid.
        ValueError: when you fail to specify an image source.
    """
    if data is not None:
        return _load_from_data(data)
    elif image is not None:
        return image
    elif path is not None:
        return _load_from_path(path)
    elif url is not None:
        return _load_from_url(url)
    else:
        raise ValueError("Must supply one of data, image, path, or url")


def _load_from_data(data: bytes) -> np.ndarray:
    """
    Load an image from raw data.

    Args:
        data: raw image data.
    Returns:
        the loaded image.
    Raises:
        InvalidImage: when the image is of an invalid format.
    """
    # TODO: Load image from memory (https://github.com/davisking/dlib/issues/818).
    _, path = mkstemp()
    with open(path, "wb") as f:
        f.write(data)
    try:
        return _load_from_path(path)
    finally:
        os.remove(path)


def _load_from_path(path: str) -> np.ndarray:
    """
    Load an image from raw data.

    Args:
        data: raw image data.
    Returns:
        the loaded image.
    Raises:
        FileNotFoundError: when the path does not point to a file.
        InvalidImage: when the image is of an invalid format.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    try:
        return dlib.load_rgb_image(path)
    except RuntimeError as e:
        raise InvalidImage(e)


def _load_from_url(url: str) -> np.ndarray:
    """
    Load an image from a URL.

    Args:
        url: URL pointing to image file.
    Returns:
        the loaded image.
    Raises:
        InvalidImage:when the image is of an invalid format.
        HTTPError: when the HTTP request fails.
    """
    resp = requests.get(url)
    resp.raise_for_status()
    return _load_from_data(resp.content)


def _is_too_large(image: np.ndarray) -> bool:
    """
    Determine whether or not an input image is too large to process.
    "Too large" is defined here as: "having more pixels than a 1024x1024 image".
    TODO: Is it more appropriate to define it as "either dimension exceeding 1024"?

    Args:
        image: the input image.
    Returns:
        whether or not the input image is too large.
    """
    pixels: int = image.shape[0] * image.shape[1]
    return pixels > _MAX_PIXELS


def _get_bounding_box(image: np.ndarray, upsample: int) -> Optional[dlib.rectangle]:
    """
    Find the biggest face in the image.

    Args:
        image: input image.
        upsample: number of times to upsample the image.
    Returns:
        bounding box around the biggest face, or None if no faces were found.
    """
    rects = _FACE_DETECTOR(image, upsample)
    if not rects:  # No detections.
        return None
    return max(rects, key=lambda r: r.area())


def _crop(image: np.ndarray, rect: dlib.rectangle) -> np.ndarray:
    """
    Crop an image to the given rectangle.
    The input is not modified.

    Args:
        image: input image to be cropped.
        rect: rectangle to extract.
    Returns:
        the cropped image.
    """
    top = rect.top()
    bottom = rect.bottom()
    left = rect.left()
    right = rect.right()
    return image[top:bottom, left:right]
