"""The manifest says what the code does.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import json
import runpy
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ppocr_addon import ADDON_ID, __version__  # noqa: E402
from ppocr_addon.scripts import LANGUAGES, SCRIPTS  # noqa: E402

MANIFEST = json.loads((REPO_ROOT / 'manifest.json').read_text(encoding='utf-8'))


def test_identity_and_version():
    pyproject = tomllib.loads((REPO_ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
    assert MANIFEST['id'] == ADDON_ID == 'ppocr'
    assert MANIFEST['version'] == __version__ == pyproject['project']['version']
    assert MANIFEST['tasks'] == ['ocr.recognize']
    assert MANIFEST['executable'] == 'bin/ppocr-addon'
    assert MANIFEST['compute'] == 'cpu'


def test_languages_and_models():
    assert MANIFEST['languages'] == list(LANGUAGES)
    assert [m['id'] for m in MANIFEST['models']] == list(SCRIPTS)
    assert all(m['bundled'] for m in MANIFEST['models'])
    assert [m['id'] for m in MANIFEST['models'] if m['default']] == ['latin']
    assert set(runpy.run_path(str(REPO_ROOT / 'tools' / 'fetch_models.py'))['PINS']) == set(SCRIPTS)


def test_config_schema():
    fields = {f['key']: f for f in MANIFEST['config_schema']['fields']}
    assert [o['value'] for o in fields['script']['options']] == ['auto', *SCRIPTS]
    assert fields['script']['default'] == 'auto'
    assert fields['threads']['type'] == 'int' and fields['threads']['default'] == 0
