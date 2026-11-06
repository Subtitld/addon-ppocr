"""PP-OCRv5 add-on for Subtitld — JSON-line protocol entry point.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License"); you may not
use this file except in compliance with the License. You may obtain a copy of
the License at http://www.apache.org/licenses/LICENSE-2.0. Unless required by
applicable law or agreed to in writing, software distributed under the License
is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.

What this add-on is
-------------------
Offline OCR for picture subtitles (DVD's VobSub, Blu-ray's PGS) with
PaddleOCR's PP-OCRv5 mobile recognition models on ONNX Runtime. Serves
`ocr.recognize`. All twelve models are in the archive; nothing downloads and
nothing is sent anywhere.

Subtitld decodes the subtitles itself, with their times and positions, and
hands over one picture per subtitle: an 8-bit grayscale PGM, the text in dark
ink on white (see ppocr_addon/image.py).

Wire format
-----------
One JSON object per line on stdin/stdout. Stderr is free-form logging, which
the host prefixes with `[addon:ppocr]` and surfaces.

    request:  {"id", "type": "ocr.recognize",
               "params": {"images": [{"id", "path"}], "language", "options"}}
    partial:  {"id": <image id>, "text"}            one per picture, in order
    progress: {"value": 0..1, "message"}
    result:   {"texts": [{"id", "text"}], "script", "model"}

A subtitle of several lines comes back with its lines joined by "\\n".

Options
-------
`script`, or $PPOCR_SCRIPT: the model to read with ('auto', the default, picks
it from the language; or one of ppocr_addon.scripts.SCRIPTS).
`threads`, or $PPOCR_THREADS: ONNX Runtime threads; 0 picks half the
processors, at most 4.

Limits
------
Horizontal text only. A cancel takes effect between pictures.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import sys
import threading
import time
import traceback
from pathlib import Path

from ppocr_addon import ADDON_ID, PROTOCOL_VERSION, __version__, engine, image
from ppocr_addon.engine import EngineError
from ppocr_addon.scripts import LANGUAGES, SCRIPTS, script_for

ADDON_VERSION = __version__
TASK = 'ocr.recognize'
SELF_TEST_TEXT = "I'm reading Subtitld's pictures."

ERR_MODEL_MISSING = 'model_missing'
ERR_UNSUPPORTED_LANGUAGE = 'unsupported_language'
ERR_BAD_PARAMS = 'bad_params'
ERR_INTERNAL = 'internal'
ERR_CANCELLED = 'cancelled'
ERR_OOM = 'oom'

log = logging.getLogger('ppocr_addon')


# ---------------------------------------------------------------------------
# stdio ownership + protocol frames
# ---------------------------------------------------------------------------
_PROTO_IN = None   # binary reader for host frames
_PROTO_OUT = None  # binary writer for our frames
_proto_lock = threading.Lock()


def _claim_protocol_stdio() -> None:
    """Give the protocol exclusive use of the real stdin/stdout.

    fd-level, not just `sys.stdout = sys.stderr`: the engine is native code and
    may write to fd 1 directly. Falls back to the Python-level swap if the fd
    dance fails.
    """
    global _PROTO_IN, _PROTO_OUT
    try:
        sys.stdout.flush()
        in_fd = os.dup(0)
        out_fd = os.dup(1)
        devnull = os.open(os.devnull, os.O_RDONLY)
        try:
            os.dup2(devnull, 0)
        finally:
            os.close(devnull)
        os.dup2(2, 1)
        _PROTO_IN = os.fdopen(in_fd, 'rb')
        _PROTO_OUT = os.fdopen(out_fd, 'wb')
    except (OSError, ValueError, AttributeError):
        _PROTO_IN = sys.stdin.buffer
        _PROTO_OUT = sys.stdout.buffer
    sys.stdin = open(os.devnull, encoding='utf-8')  # stray input() -> EOFError
    sys.stdout = sys.stderr                        # stray print() -> the log


def _emit(frame: dict) -> None:
    data = (json.dumps(frame, ensure_ascii=False, separators=(',', ':')) + '\n').encode('utf-8')
    out = _PROTO_OUT if _PROTO_OUT is not None else sys.stdout.buffer
    with _proto_lock:
        try:
            out.write(data)
            out.flush()
        except (BrokenPipeError, OSError, ValueError):
            pass  # host went away; the frame loop exits on the next empty read


def _emit_progress(req_id: str, value: float, message: str = '') -> None:
    _emit({'id': req_id, 'type': 'progress',
           'value': max(0.0, min(1.0, float(value))), 'message': message})


def _emit_result(req_id: str, data: dict) -> None:
    _emit({'id': req_id, 'type': 'result', 'data': data})


def _emit_error(req_id: str, code: str, message: str, retryable: bool = False) -> None:
    _emit({'id': req_id, 'type': 'error', 'code': code,
           'message': message, 'retryable': retryable})


class _AddonError(Exception):
    """An expected failure that already knows its protocol code."""

    def __init__(self, code: str, message: str, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


class _Cancelled(Exception):
    """The host cancelled this request."""


# ---------------------------------------------------------------------------
# Options and environment
# ---------------------------------------------------------------------------
def _env(*names: str) -> str:
    """First non-empty value among `names`, stripped. '' when none are set."""
    for name in names:
        value = (os.environ.get(name) or '').strip()
        if value:
            return value
    return ''


def _resolve_script(language, options: dict) -> str:
    """The model for the request: the `script` option, else $PPOCR_SCRIPT,
    else the language's own. A language no model reads is an error."""
    override = str(options.get('script') or _env('PPOCR_SCRIPT') or 'auto').strip().lower()
    if override not in ('', 'auto') and override not in SCRIPTS:
        raise _AddonError(ERR_BAD_PARAMS, f'unknown script {override!r}; one of auto, {", ".join(SCRIPTS)}')
    script = script_for(language, override)
    if script is None:
        raise _AddonError(ERR_UNSUPPORTED_LANGUAGE,
                          f'PP-OCRv5 has no model for {language!r}. Pick the script it is written in under '
                          f'Configure, or another OCR engine.')
    return script


def _resolve_threads(options: dict) -> int:
    raw = options.get('threads')
    if raw in (None, ''):
        raw = _env('PPOCR_THREADS') or 0
    try:
        return max(0, min(64, int(raw)))
    except (TypeError, ValueError):
        return 0


# ---------------------------------------------------------------------------
# The request
# ---------------------------------------------------------------------------
class _Request:
    def __init__(self, req_id: str):
        self.req_id = req_id
        self.cancel = threading.Event()
        self.thread: threading.Thread | None = None


_active: dict[str, _Request] = {}
_active_lock = threading.Lock()


def _images(params: dict) -> list[tuple[str, str]]:
    raw = params.get('images')
    if not isinstance(raw, list):
        raise _AddonError(ERR_BAD_PARAMS, 'images must be a list of {id, path}')
    out = []
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get('path'), str) or not item['path']:
            raise _AddonError(ERR_BAD_PARAMS, 'each image needs a path')
        out.append((str(item.get('id', len(out))), item['path']))
    return out


def _handle_recognize(state: _Request, params: dict) -> None:
    req_id = state.req_id
    options = params.get('options')
    if not isinstance(options, dict):
        options = {}
    outcome: tuple = ('error', ERR_INTERNAL, 'no outcome', False)
    try:
        images = _images(params)
        language = str(params.get('language') or '')
        script = _resolve_script(language, options)
        threads = _resolve_threads(options)
        log.info('recognize %s: %d pictures, language=%s script=%s', req_id, len(images), language or '-', script)
        _emit_progress(req_id, 0.0, 'Loading the model...')
        started = time.monotonic()
        try:
            reader = engine.model(script, threads)
        except EngineError as exc:
            raise _AddonError(ERR_MODEL_MISSING, str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 — ONNX Runtime refusing the file
            raise _AddonError(ERR_INTERNAL, f'The {script} model failed to load: {exc}') from exc
        texts = []
        for done, (image_id, path) in enumerate(images, 1):
            if state.cancel.is_set():
                raise _Cancelled()
            try:
                text = reader.read(image.read_pgm(path), language)
            except (OSError, image.ImageError) as exc:
                raise _AddonError(ERR_BAD_PARAMS, f'picture {image_id}: {exc}') from exc
            texts.append({'id': image_id, 'text': text})
            _emit({'id': req_id, 'type': 'partial', 'data': {'id': image_id, 'text': text}})
            _emit_progress(req_id, done / len(images), f'Reading subtitle {done} of {len(images)}')
        log.info('recognize %s: %d pictures in %.2fs', req_id, len(images), time.monotonic() - started)
        outcome = ('result', {'texts': texts, 'script': script, 'model': f'{script}_PP-OCRv5_mobile_rec'})
    except _Cancelled:
        log.info('recognize %s: cancelled', req_id)
        outcome = ('error', ERR_CANCELLED, 'cancelled', False)
    except _AddonError as exc:
        log.warning('recognize %s failed: [%s] %s', req_id, exc.code, exc.message)
        outcome = ('error', exc.code, exc.message, exc.retryable)
    except MemoryError:
        outcome = ('error', ERR_OOM, 'Out of memory while reading the subtitles.', False)
    except Exception as exc:  # noqa: BLE001 — every path must end in one terminal frame
        log.exception('recognize %s failed', req_id)
        outcome = ('error', ERR_INTERNAL, f'{type(exc).__name__}: {exc}', False)

    if outcome[0] == 'result':
        _emit_result(req_id, outcome[1])
    else:
        _emit_error(req_id, *outcome[1:])


def _spawn(req_id: str, target, *args) -> None:
    state = _Request(req_id)
    with _active_lock:
        _active[req_id] = state

    def _run():
        try:
            target(state, *args)
        except BaseException:  # noqa: BLE001
            log.exception('worker crashed')
            tail = traceback.format_exc().strip().splitlines()[-1]
            _emit_error(req_id, ERR_INTERNAL, f'worker crashed: {tail}')
        finally:
            with _active_lock:
                _active.pop(req_id, None)

    thread = threading.Thread(target=_run, name=f'ppocr-{req_id}', daemon=True)
    state.thread = thread
    thread.start()


# ---------------------------------------------------------------------------
# Handshake + frame loop
# ---------------------------------------------------------------------------
def _send_hello() -> None:
    _emit({
        'type': 'hello',
        'protocol': PROTOCOL_VERSION,
        'addon': ADDON_ID,
        'version': ADDON_VERSION,
        'capabilities': [{
            'task': TASK,
            'languages': list(LANGUAGES),
            'scripts': list(SCRIPTS),
            'image_format': 'pgm',
        }],
    })


def _read_frame() -> dict | None:
    source = _PROTO_IN if _PROTO_IN is not None else sys.stdin.buffer
    line = source.readline()
    if not line:
        return None  # stdin closed: the host went away
    text = line.decode('utf-8', errors='replace').strip().lstrip('﻿')
    if not text:
        return {}
    try:
        frame = json.loads(text)
    except ValueError as exc:
        log.warning('malformed frame %r: %s', line[:200], exc)
        return {}
    return frame if isinstance(frame, dict) else {}


def _main_loop() -> int:
    while True:
        frame = _read_frame()
        if frame is None:
            log.info('stdin closed; exiting')
            return 0
        if not frame:
            continue
        ftype = frame.get('type')

        if ftype == 'shutdown':
            log.info('shutdown requested')
            return 0

        if ftype == 'ready':
            log.info('handshake complete; host=%s host_version=%s',
                     frame.get('host'), frame.get('host_version'))
            continue

        if ftype == 'cancel':
            target = frame.get('target')
            with _active_lock:
                state = _active.get(target) if isinstance(target, str) else None
            if state is not None:
                log.info('cancel %s', target)
                state.cancel.set()
            continue

        if ftype == TASK:
            req_id = frame.get('id')
            params = frame.get('params')
            if not isinstance(req_id, str) or not req_id:
                log.warning('%s without id', TASK)
                continue
            if params is None:
                params = {}
            if not isinstance(params, dict):
                _emit_error(req_id, ERR_BAD_PARAMS, 'params must be an object')
                continue
            _spawn(req_id, _handle_recognize, params)
            continue

        # Unknown task: still answer, so the host's future resolves instead of
        # hanging until its timeout.
        req_id = frame.get('id')
        if isinstance(req_id, str) and req_id:
            _emit_error(req_id, ERR_BAD_PARAMS, f'unsupported frame type {ftype!r}')


def _stop_workers(budget_sec: float = 3.5) -> bool:
    """Cancel and join every worker. True when none is left running.

    The host allows 5 s between `shutdown` and SIGTERM, so stay under it.
    """
    with _active_lock:
        states = list(_active.values())
    for state in states:
        state.cancel.set()
    deadline = time.monotonic() + budget_sec
    for state in states:
        if state.thread is not None:
            state.thread.join(timeout=max(0.0, deadline - time.monotonic()))
    return not any(s.thread is not None and s.thread.is_alive() for s in states)


def _self_test() -> int:
    """Check what the frozen bundle needs: every model loads and reads the
    picture in the package, and the result is printed as JSON."""
    gray = image.read_pgm(Path(engine.__file__).resolve().parent / 'selftest.pgm')
    report, ok = {'version': ADDON_VERSION, 'models': {}}, True
    for script in SCRIPTS:
        try:
            text = engine.model(script).read(gray, 'en' if script == 'en' else '')
        except Exception as exc:  # noqa: BLE001
            text, ok = f'FAILED: {exc}', False
        report['models'][script] = text
    english = report['models'].get('en', '')
    ok = ok and english == SELF_TEST_TEXT
    report['ok'] = ok
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0 if ok else 1


def main() -> int:
    logging.basicConfig(
        level=(_env('PPOCR_ADDON_LOG_LEVEL') or 'INFO').upper(),
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
        stream=sys.stderr,
    )
    if '--self-test' in sys.argv[1:]:
        return _self_test()
    _claim_protocol_stdio()
    log.info('ppocr-addon %s on %s %s, python %s',
             ADDON_VERSION, platform.system(), platform.machine(), platform.python_version())
    _send_hello()
    rc = 1
    try:
        rc = _main_loop()
    except KeyboardInterrupt:
        rc = 130
    except Exception:  # noqa: BLE001
        log.exception('main loop crashed')
        rc = 1
    finally:
        stopped = _stop_workers()
    if not stopped:
        # A worker is still inside ONNX Runtime. Normal teardown would free its
        # session under it, so leave without teardown.
        log.warning('a recognition is still running; exiting without cleanup')
        try:
            sys.stderr.flush()
        finally:
            os._exit(rc)
    return rc


if __name__ == '__main__':
    sys.exit(main())
