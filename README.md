# bighead

A small library and web server for face detection.

## Installation

You'll need: Python >= 3.6 and CMake.
You can install with [Poetry](https://python-poetry.org) via `poetry install; poetry shell`.
Or, you can install with `pip` via `pip install .`.

## Testing

Run `test.sh`.
It performs unit tests, style checking, linting, and type checking.
The code is tested on Ubuntu.
On MacOS, I have run run into some libjpeg issues along these lines:

```
RuntimeError: jpeg_loader: error while loading image: Wrong JPEG library version: library is 90, caller expects 80
```

If you really want reproducibility, run the tests in Docker:

```sh
docker run python:3.8 --mount type=bind,source=$(pwd),target=/bighead --rm -t sh -c '
  apt-get update
  apt-get -y install cmake
  pip install poetry
  cd /bighead
  poetry install
  poetry run ./test.sh'
```

## Usage (Library)

The user-facing API consists of `bighead.find_biggest_face` for finding bounding boxes, and `bighead.extract_biggest_face` for cropping out faces.
Their parameters are identical.
Here's a quick example:

```py
>>> import bighead as bg, dlib, numpy as np
>>> box: dlib.rectangle = bg.find_biggest_face(path="/my/image.png")  # From a file
>>> face: np.ndarray = bg.extract_biggest_face(data=b"...")           # From raw data
>>> bg.find_biggest_face(image=np.random.randn(256, 256, 3))          # From an image array
>>> bg.find_biggest_face(url="https://example.com/image.png")         # From a URL
>>> bg.find_biggest_face(path="small.png", upsample=1)                # With upsampling
>>> help(bg.find_biggest_face); help(bg.extract_biggest_face)         # For comprehensive documentation
```

## Usage (Web Server)

Run the web server like so:

```sh
poetry shell  # Only if you installed with Poetry
export FLASK_APP=bighead.web
flask run
```

And send requests like this:

```sh
curl 'localhost:5000/detect_largest_face?upsample=1' \
       --data-binary '@image.png' \
       -H 'Content-Type: image/png'
```

For details on the API requests, see the docstring:

```py
>>> from bighead.web import detect_largest_face
>>> help(detect_largest_face)
```
