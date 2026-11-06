"""The parts that need no model: languages, pictures, lines, clean-up.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ppocr_addon import image  # noqa: E402
from ppocr_addon.engine import tidy  # noqa: E402
from ppocr_addon.scripts import LANGUAGES, SCRIPTS, script_for  # noqa: E402


@pytest.mark.parametrize('language, script', [
    ('en-us', 'en'), ('pt-br', 'latin'), ('fr_FR', 'latin'), ('cmn-hans-cn', 'ch'), ('yue-hant-hk', 'ch'),
    ('ja-jp', 'ch'), ('ko-kr', 'korean'), ('ru-ru', 'eslav'), ('uk-ua', 'eslav'), ('bg-bg', 'cyrillic'),
    ('sr-rs', 'cyrillic'), ('sr-latn-rs', 'latin'), ('uz-cyrl', 'cyrillic'), ('el-gr', 'el'),
    ('fa-ir', 'arabic'), ('ur-pk', 'arabic'), ('hi-in', 'devanagari'), ('th-th', 'th'), ('ta-lk', 'ta'),
    ('te-in', 'te'), ('', 'latin'), ('und', 'latin'),
])
def test_script_for(language, script):
    assert script_for(language) == script


@pytest.mark.parametrize('language', ['vi-vn', 'ro-ro', 'lv-lv', 'he-il', 'ka-ge', 'bn-bd', 'km-kh'])
def test_languages_without_a_model(language):
    assert script_for(language) is None


def test_the_option_wins():
    assert script_for('vi-vn', 'latin') == 'latin'
    assert script_for('en', 'auto') == 'en'


def test_every_language_has_one_script():
    assert len(LANGUAGES) == len(set(LANGUAGES))
    assert all(script_for(lang) in SCRIPTS for lang in LANGUAGES)


def _write_pgm(path, gray):
    h, w = gray.shape
    path.write_bytes(b'P5\n# a comment\n%d %d\n255\n' % (w, h) + gray.astype(np.uint8).tobytes())


def test_read_pgm(tmp_path):
    gray = np.arange(12, dtype=np.uint8).reshape(3, 4) * 20
    _write_pgm(tmp_path / 'a.pgm', gray)
    assert np.array_equal(image.read_pgm(tmp_path / 'a.pgm'), gray)


@pytest.mark.parametrize('data', [b'P6\n2 2\n255\n' + bytes(12), b'P5\n2 2\n65535\n' + bytes(8),
                                  b'P5\n4 4\n255\n' + bytes(3), b'P5\n4'])
def test_read_pgm_refuses(tmp_path, data):
    (tmp_path / 'bad.pgm').write_bytes(data)
    with pytest.raises(image.ImageError):
        image.read_pgm(tmp_path / 'bad.pgm')


def test_lines_split_at_blank_rows_and_keep_accents():
    gray = np.full((100, 60), 255, np.uint8)
    gray[5:8, 10:14] = 0        # an accent over the first line, a blank row between
    gray[9:30, 5:50] = 0        # first line
    gray[50:72, 8:40] = 0       # second line
    found = image.lines(gray)
    assert len(found) == 2
    pad = max(4, (30 - 5) // 4)
    assert found[0].shape == (30 - 5 + 2 * pad, 45 + 2 * pad)


def test_blank_picture_has_no_lines():
    assert image.lines(np.full((10, 10), 255, np.uint8)) == []


@pytest.mark.parametrize('shape', [(24, 300), (48, 100), (120, 900)])
def test_resize_height(shape):
    gray = np.full(shape, 255, np.uint8)
    gray[:, shape[1] // 2:] = 0
    out = image.resize_height(gray)
    assert out.shape[0] == 48 and abs(out.shape[1] - round(shape[1] * 48 / shape[0])) <= 1
    assert out.min() >= 0 and out.max() <= 255
    assert out[:, :out.shape[1] // 3].mean() > 250 and out[:, -out.shape[1] // 3:].mean() < 5


@pytest.mark.parametrize('text, language, expected', [
    ("l'm fine", 'en-us', "I'm fine"),
    ('Then l think so.', 'en', 'Then I think so.'),
    ('| guess', 'en', 'I guess'),
    ("l'été", 'fr', "l'été"),
    ('He|lo', 'de', 'Hello'),
    ('你好?', 'cmn-hans-cn', '你好？'),
    ('Hello?', 'ja', 'Hello?'),
    ('two  spaces \nnext', 'pt', 'two spaces\nnext'),
])
def test_tidy(text, language, expected):
    assert tidy(text, language) == expected
