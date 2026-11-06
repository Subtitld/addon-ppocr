"""The add-on as Subtitld runs it: a process spoken to over stdio.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

By default the process is the source tree (`python -m ppocr_addon`) with the
models from tools/fetch_models.py. With PPOCR_TEST_BINARY set it is that
executable instead, which is how the release workflow runs these tests: on the
frozen binary unpacked from the release zip the way Subtitld installs it,
with nothing but its own bundle.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ppocr_addon import __version__  # noqa: E402
from ppocr_addon.scripts import LANGUAGES  # noqa: E402

DATA = REPO_ROOT / 'tests' / 'data'
EXPECTED = json.loads((DATA / 'expected.json').read_text(encoding='utf-8'))
BINARY = os.environ.get('PPOCR_TEST_BINARY')


def _start() -> subprocess.Popen:
    env = dict(os.environ)
    if BINARY:
        env.pop('PPOCR_ADDON_MODELS', None)
        env.pop('PYTHONPATH', None)
        command = [BINARY]
    else:
        env['PYTHONPATH'] = str(REPO_ROOT)
        command = [sys.executable, '-m', 'ppocr_addon']
    return subprocess.Popen(command, cwd=REPO_ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, env=env)


def _send(proc, frame):
    proc.stdin.write((json.dumps(frame) + '\n').encode('utf-8'))
    proc.stdin.flush()


def _recv(proc):
    line = proc.stdout.readline()
    assert line, 'the add-on closed its output: ' + proc.stderr.read().decode('utf-8', 'replace')
    return json.loads(line)


def _until_terminal(proc, req_id):
    frames = []
    while True:
        frame = _recv(proc)
        assert frame.get('id') == req_id
        frames.append(frame)
        if frame['type'] in ('result', 'error'):
            return frames


@pytest.fixture
def addon():
    proc = _start()
    hello = _recv(proc)
    _send(proc, {'type': 'ready', 'host': 'tests', 'host_version': '0'})
    yield proc, hello
    try:
        _send(proc, {'type': 'shutdown'})
        assert proc.wait(timeout=15) == 0
    finally:
        if proc.poll() is None:
            proc.kill()


def test_hello(addon):
    _proc, hello = addon
    assert hello['type'] == 'hello' and hello['addon'] == 'ppocr' and hello['version'] == __version__
    capability = hello['capabilities'][0]
    assert capability['task'] == 'ocr.recognize' and capability['languages'] == list(LANGUAGES)


def test_recognize(addon):
    proc, _hello = addon
    names = ['en_two_lines', 'en_dvd']
    _send(proc, {'id': 'r1', 'type': 'ocr.recognize', 'params': {
        'images': [{'id': name, 'path': str(DATA / f'{name}.pgm')} for name in names], 'language': 'en-us'}})
    frames = _until_terminal(proc, 'r1')
    partials = [f['data'] for f in frames if f['type'] == 'partial']
    assert [p['id'] for p in partials] == names
    result = frames[-1]
    assert result['type'] == 'result', result
    assert result['data']['script'] == 'en'
    assert result['data']['texts'] == [{'id': name, 'text': EXPECTED[name]['text']} for name in names]
    assert [f for f in frames if f['type'] == 'progress'][-1]['value'] == 1.0


def test_script_option(addon):
    proc, _hello = addon
    _send(proc, {'id': 'r2', 'type': 'ocr.recognize', 'params': {
        'images': [{'id': 1, 'path': str(DATA / 'de_pgs.pgm')}], 'language': 'xx', 'options': {'script': 'latin'}}})
    result = _until_terminal(proc, 'r2')[-1]
    assert result['data']['texts'] == [{'id': '1', 'text': EXPECTED['de_pgs']['text']}]


@pytest.mark.parametrize('params, code', [
    ({'images': 'nope'}, 'bad_params'),
    ({'images': [{'id': 'a'}]}, 'bad_params'),
    ({'images': [{'id': 'a', 'path': '/no/such.pgm'}], 'language': 'en'}, 'bad_params'),
    ({'images': [], 'language': 'vi-vn'}, 'unsupported_language'),
    ({'images': [], 'options': {'script': 'klingon'}}, 'bad_params'),
])
def test_errors(addon, params, code):
    proc, _hello = addon
    _send(proc, {'id': 'e', 'type': 'ocr.recognize', 'params': params})
    frame = _until_terminal(proc, 'e')[-1]
    assert frame['type'] == 'error' and frame['code'] == code, frame


def test_unknown_task_is_answered(addon):
    proc, _hello = addon
    _send(proc, {'id': 'u', 'type': 'tts.synthesize', 'params': {}})
    frame = _recv(proc)
    assert frame == {'id': 'u', 'type': 'error', 'code': 'bad_params',
                     'message': "unsupported frame type 'tts.synthesize'", 'retryable': False}


def test_self_test():
    command = [BINARY] if BINARY else [sys.executable, '-m', 'ppocr_addon']
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    run = subprocess.run(command + ['--self-test'], capture_output=True, text=True, env=env, timeout=120)
    assert run.returncode == 0, run.stdout + run.stderr
    assert json.loads(run.stdout)['ok'] is True
