"""Every model reads, and the sample subtitles come back word for word.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

The pictures in tests/data are rendered by tools/render_samples.py the way
discs draw subtitles (Blu-ray: antialiased 1080p; DVD: 24 px, four colours),
then turned into the dark-on-white PGM the host sends.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ppocr_addon import engine, image  # noqa: E402
from ppocr_addon.scripts import SCRIPTS, script_for  # noqa: E402

DATA = REPO_ROOT / 'tests' / 'data'
EXPECTED = json.loads((DATA / 'expected.json').read_text(encoding='utf-8'))
SELF_TEST = REPO_ROOT / 'ppocr_addon' / 'selftest.pgm'


@pytest.mark.parametrize('name', sorted(EXPECTED))
def test_sample(name):
    sample = EXPECTED[name]
    model = engine.model(script_for(sample['language']))
    assert model.read(image.read_pgm(DATA / f'{name}.pgm'), sample['language']) == sample['text']


@pytest.mark.parametrize('script', sorted(SCRIPTS))
def test_every_model_loads_and_reads_latin_letters(script):
    # Every script's model also knows the Latin alphabet and digits.
    text = engine.model(script).read(image.read_pgm(SELF_TEST), 'en')
    assert 'reading' in text and 'pictures' in text


def test_missing_model(tmp_path, monkeypatch):
    monkeypatch.setenv('PPOCR_ADDON_MODELS', str(tmp_path))
    with pytest.raises(engine.EngineError, match='missing'):
        engine.Model('latin')
