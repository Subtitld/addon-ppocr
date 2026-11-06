"""PP-OCRv5's mobile recognition models on ONNX Runtime.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

Each model reads one line of text: a 48-pixel-high strip in, one score per
dictionary character (plus the CTC blank first and a space last) for every
4 columns out. Greedy CTC decoding gives the text. No detection model is
needed: subtitle pictures are already cut to their text, and
image.lines() splits them into lines.
"""

from __future__ import annotations

import os
import re
import sys
import threading
from pathlib import Path

import numpy as np

from ppocr_addon import image
from ppocr_addon.scripts import SCRIPTS


class EngineError(RuntimeError):
    pass


def models_dir() -> Path:
    """Where the models are: $PPOCR_ADDON_MODELS, else next to the package
    (in the frozen bundle, `_internal/ppocr_addon/models`)."""
    override = os.environ.get('PPOCR_ADDON_MODELS', '').strip()
    if override:
        return Path(os.path.expanduser(override))
    bundled = getattr(sys, '_MEIPASS', None)
    if bundled:
        return Path(bundled) / 'ppocr_addon' / 'models'
    return Path(__file__).resolve().parent / 'models'


class Model:
    """One script's recognition model."""

    def __init__(self, script: str, threads: int = 0):
        import onnxruntime as ort

        folder = models_dir() / script
        graph, dictionary = folder / 'rec.onnx', folder / 'dict.txt'
        for path in (graph, dictionary):
            if not path.is_file():
                raise EngineError(f'{path} is missing. Reinstall the add-on.')
        chars = dictionary.read_text(encoding='utf-8').split('\n')
        if chars and chars[-1] == '':
            chars.pop()
        self.script = script
        self.chars = [''] + chars + [' ']
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads or default_threads()
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        self.session = ort.InferenceSession(str(graph), options, providers=['CPUExecutionProvider'])
        self.input = self.session.get_inputs()[0].name
        classes = self.session.get_outputs()[0].shape[-1]
        if isinstance(classes, int) and classes != len(self.chars):
            raise EngineError(f'{script}: the model has {classes} classes, its dictionary {len(self.chars)}')
        self._lock = threading.Lock()

    def read_line(self, line: np.ndarray) -> str:
        with self._lock:
            scores = self.session.run(None, {self.input: image.model_input(line)})[0][0]
        best = scores.argmax(axis=1)
        out, previous = [], 0
        for index in best:
            if index != previous and index != 0 and index < len(self.chars):
                out.append(self.chars[index])
            previous = index
        return ''.join(out).strip()

    def read(self, gray: np.ndarray, language: str = '') -> str:
        """A subtitle picture's text, its lines joined by new lines."""
        texts = [self.read_line(line) for line in image.lines(gray)]
        return tidy('\n'.join(text for text in texts if text), language)


def default_threads() -> int:
    """Half the logical processors, 1 to 4: a line takes milliseconds, more
    threads only wait on each other."""
    return max(1, min(4, (os.cpu_count() or 2) // 2))


_LONE_L = re.compile(r"(?<![\w'’])[l|](?=['’](m|ll|d|ve|s|re)\b|\s|$|[,.!?])")
_PIPE = re.compile(r'(?<=\w)\|(?=\w)|(?<![\w])\|(?=\w)')
# Chinese and Japanese write their punctuation full width; the model often
# gives the ASCII mark after a character of theirs.
_FULL_WIDTH = {'?': '？', '!': '！', ',': '，', ':': '：', ';': '；'}
_CJK_PUNCTUATION = re.compile(r'(?<=[぀-ヿ㐀-鿿豈-﫿])[?!,:;]')


def tidy(text: str, language: str = '') -> str:
    """What the models get wrong on subtitle pictures, put right.

    English: a lone `l` or `|` is the pronoun I ("l'm", "l think"), and a
    `|` touching a word is an I. Other Latin scripts: that `|` is an l.
    Chinese and Japanese: ASCII ? ! , : ; after their characters become
    full width. Spaces doubled by the decoder go."""
    primary = str(language or '').lower().replace('_', '-').split('-')[0]
    if primary == 'en':
        text = _LONE_L.sub('I', text)
        text = _PIPE.sub('I', text)
    elif primary in ('zh', 'cmn', 'yue', 'ja'):
        text = _CJK_PUNCTUATION.sub(lambda m: _FULL_WIDTH[m.group(0)], text)
    elif primary != 'ko':
        text = _PIPE.sub('l', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return '\n'.join(line.strip() for line in text.split('\n'))


_models: dict[tuple[str, int], Model] = {}
_models_lock = threading.Lock()


def model(script: str, threads: int = 0) -> Model:
    """The script's model, loaded once."""
    if script not in SCRIPTS:
        raise EngineError(f'no model for {script!r}')
    with _models_lock:
        key = (script, threads)
        if key not in _models:
            _models[key] = Model(script, threads)
        return _models[key]
