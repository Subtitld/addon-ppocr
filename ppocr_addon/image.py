"""A subtitle's picture, as the host hands it: lines of ink, ready for the model.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

The host sends each subtitle as an 8-bit grayscale PGM (P5): its text in
dark ink on white, the outline and the background white, cropped to the
text. A subtitle of two lines is cut into two at the rows without ink, which
is exact on subtitle pictures; PP-OCRv5's recognition models read one line
at a time. A run of inked rows much shorter than the others (an accent over
a capital, a line of dots) is not a line: it joins the line next to it.
"""

from __future__ import annotations

import numpy as np

INK = 128            # darker than this is ink
MODEL_HEIGHT = 48    # what PP-OCRv5's recognition models take
_MIN_WIDTH = 16


class ImageError(ValueError):
    """A file that is not the PGM the protocol describes."""


def read_pgm(path) -> np.ndarray:
    """An 8-bit binary PGM's pixels, uint8, rows by columns."""
    with open(path, 'rb') as f:
        data = f.read()
    fields, pos = [], 0
    while len(fields) < 4:
        while pos < len(data) and data[pos:pos + 1].isspace():
            pos += 1
        if data[pos:pos + 1] == b'#':
            pos = data.find(b'\n', pos)
            if pos < 0:
                raise ImageError(f'{path}: truncated PGM header')
            continue
        end = pos
        while end < len(data) and not data[end:end + 1].isspace():
            end += 1
        if end == pos:
            raise ImageError(f'{path}: truncated PGM header')
        fields.append(data[pos:end])
        pos = end
    magic, width, height, maxval = fields[0], *(int(v) for v in fields[1:])
    if magic != b'P5' or maxval != 255 or width <= 0 or height <= 0:
        raise ImageError(f'{path}: not an 8-bit binary PGM (P5, maxval 255)')
    pixels = data[pos + 1:pos + 1 + width * height]
    if len(pixels) != width * height:
        raise ImageError(f'{path}: {len(pixels)} bytes of pixels, {width * height} expected')
    return np.frombuffer(pixels, np.uint8).reshape(height, width)


def _runs(flags) -> list[list[int]]:
    """[start, end) of each run of True."""
    runs, start = [], None
    for i, flag in enumerate(list(flags) + [False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append([start, i])
            start = None
    return runs


def lines(gray: np.ndarray) -> list[np.ndarray]:
    """The picture's lines of text, top to bottom, each cropped to its ink
    with a margin of white."""
    ink = gray < INK
    spans = _runs(ink.any(axis=1))
    if not spans:
        return []
    tallest = max(end - start for start, end in spans)
    merged: list[list[int]] = []
    for span in spans:
        if merged and (span[1] - span[0] < 0.4 * tallest or merged[-1][1] - merged[-1][0] < 0.4 * tallest):
            # The short one belongs to its neighbour: below an accent, above a descender's dot.
            merged[-1][1] = span[1]
        else:
            merged.append(span)
    if len(merged) > 1 and merged[-1][1] - merged[-1][0] < 0.4 * tallest:
        merged[-2][1] = merged.pop()[1]
    out = []
    for top, bottom in merged:
        cols = np.flatnonzero(ink[top:bottom].any(axis=0))
        pad = max(4, (bottom - top) // 4)
        crop = gray[top:bottom, cols[0]:cols[-1] + 1]
        out.append(np.pad(crop, pad, constant_values=255))
    return out


def resize_height(gray: np.ndarray, height: int = MODEL_HEIGHT) -> np.ndarray:
    """`gray` scaled to `height` rows, its width in proportion, float32.

    Bilinear when it grows; when it shrinks, each output pixel averages the
    input pixels it covers, so thin strokes do not drop out."""
    h, w = gray.shape
    width = max(_MIN_WIDTH, round(w * height / h))
    src = gray.astype(np.float32)
    if h > height or w > width:
        src = _area(src, height, width)
        if src.shape == (height, width):
            return src
    return _bilinear(src, height, width)


def _area(src, height, width):
    """Shrink by averaging, along each axis that shrinks."""
    h, w = src.shape
    if h > height:
        edges = np.linspace(0, h, height + 1)
        src = _average_rows(src, edges)
    if w > width:
        edges = np.linspace(0, w, width + 1)
        src = _average_rows(src.T, edges).T
    return src


def _average_rows(src, edges):
    """Rows of `src` averaged between consecutive fractional `edges`."""
    cumulative = np.concatenate([np.zeros((1, src.shape[1]), np.float64), np.cumsum(src, axis=0, dtype=np.float64)])
    def at(y):
        lower = np.floor(y).astype(int)
        frac = (y - lower)[:, None]
        upper = np.minimum(lower + 1, src.shape[0])
        return cumulative[lower] + frac * (cumulative[upper] - cumulative[lower])
    sums = at(edges[1:]) - at(edges[:-1])
    return (sums / np.diff(edges)[:, None]).astype(np.float32)


def _bilinear(src, height, width):
    h, w = src.shape
    ys = np.clip((np.arange(height) + 0.5) * h / height - 0.5, 0, h - 1)
    xs = np.clip((np.arange(width) + 0.5) * w / width - 0.5, 0, w - 1)
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    y1, x1 = np.minimum(y0 + 1, h - 1), np.minimum(x0 + 1, w - 1)
    fy, fx = (ys - y0)[:, None], (xs - x0)[None, :]
    top = src[y0][:, x0] * (1 - fx) + src[y0][:, x1] * fx
    bottom = src[y1][:, x0] * (1 - fx) + src[y1][:, x1] * fx
    return (top * (1 - fy) + bottom * fy).astype(np.float32)


def model_input(line: np.ndarray) -> np.ndarray:
    """A line as the model takes it: 1×3×48×W, scaled to [-1, 1]."""
    x = resize_height(line) / 127.5 - 1.0
    return np.broadcast_to(x, (1, 3) + x.shape).astype(np.float32)
