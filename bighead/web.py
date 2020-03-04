import logging

from typing import Dict, Optional, Tuple

import dlib

from flask import Flask, request
from requests import codes
from werkzeug.exceptions import InternalServerError, MethodNotAllowed, NotFound

from .face_detection import InvalidImage, TooLarge, find_biggest_face

app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)
JSON = Tuple[Dict[str, object], int]


@app.errorhandler(NotFound)
def not_found(e: NotFound) -> JSON:
    return {"error": "Not found"}, codes.not_found


@app.errorhandler(MethodNotAllowed)
def method_not_allowed(e: MethodNotAllowed) -> JSON:
    return {"error": "Method not allowed"}, codes.method_not_allowed


@app.errorhandler(InternalServerError)
def internal_server_error(e: InternalServerError) -> JSON:
    # Flask already displays the error, we don't need to do anything.
    return {"error": "Internal server error"}, codes.server_error


@app.route("/detect_largest_face", methods=["POST"])
def detect_largest_face() -> JSON:
    """
    Handles POST requests to /detect_largest_face.

    Query parameters:
        upsample (int = 0): number of times to upsample the image.
    Request body:
        raw image data.
    Returns:
        the JSON HTTP response:
        - face detection: {"box": {"left": 1, "right": 2, "top": 3, "bottom": 4}}, 200
        - no detection: {}, 200
        - error: {"error": "message"}, 4xx
    """
    try:
        data, upsample = _preprocess_request()
    except ValueError as e:
        return _error(e.args[0], codes.bad)
    else:
        app.logger.info(f"data length: {len(data)}, upsample: {upsample}")
        return _detect(data, upsample)


def _preprocess_request() -> Tuple[bytes, int]:
    """
    Validates and parses the request.

    Returns:
        the raw image data and the upsample count in a tuple.
    Raises:
        ValueError: when the request is invalid.
    """
    if not request.data:
        app.logger.warning("Request data is missing")
        raise ValueError("No data found, make sure to set Content-Type")
    upsample_arg = request.args.get("upsample", "0")
    try:
        upsample = int(upsample_arg)
    except ValueError:
        app.logger.warning("Invalid upsample argument given: {upsample_arg}")
        raise ValueError("Invalid numeric value for upsample argument")
    if upsample < 0:
        app.logger.warning("Negative number provided for upsample: {upsample}")
        raise ValueError("Value for upsample argument must be nonnegative")
    else:
        return request.data, upsample


def _error(message: str, status: int) -> JSON:
    """
    Returns an HTTP response containing an error.

    Args:
        message: the error message.
        status: the 4xx HTTP status code.
    Returns:
        the JSON HTTP response.
    """
    return {"error": message}, status


def _success(rect: Optional[dlib.rectangle]) -> JSON:
    """
    Return an HTTP response for a successful detection request.

    Args:
        rect: bounding box, if one was found.
    Returns:
        the JSON HTTP response.
    """
    if not rect:
        return {}, codes.ok
    box = {
        "left": rect.left(),
        "right": rect.right(),
        "top": rect.top(),
        "bottom": rect.bottom(),
    }
    return {"box": box}, codes.ok


def _detect(data: bytes, upsample: int) -> JSON:
    """
    Attempts largest face detection with the given data.

    Args:
        data: raw image data.
        upsample: number of times to upsample the image.
    Returns:
        the HTTP response, whose status code varies with the detection outcome.
    """
    try:
        box = find_biggest_face(data=data, upsample=upsample)
    except InvalidImage as e:
        app.logger.warning(f"Invalid image: {e}")
        return _error("The image is invalid", codes.unprocessable)
    except TooLarge as e:
        app.logger.warning(f"Image is too large: {e.dims}")
        return _error(e.msg, codes.request_entity_too_large)
    if box is not None:
        app.logger.info("Found a face")
    else:
        app.logger.info("Did not find a face")
    return _success(box)
