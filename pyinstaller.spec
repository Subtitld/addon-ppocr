# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Subtitld PP-OCRv5 add-on.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

Produces `dist/ppocr-addon/` (launcher + `_internal/`): ONNX Runtime, numpy,
the add-on, its self-test picture and the twelve recognition models. The
release step copies the folder to `bin/` inside the archive, next to
manifest.json.

Run from the add-on root:

    python tools/fetch_models.py
    pyinstaller pyinstaller.spec --noconfirm
    dist/ppocr-addon/ppocr-addon --self-test
"""

# ruff: noqa: F821  # PyInstaller injects Analysis/PYZ/EXE/COLLECT at runtime.

from __future__ import annotations

import os
import runpy
import subprocess
import sys
from pathlib import Path

SPEC_ROOT = Path(SPECPATH).resolve()
PACKAGE = SPEC_ROOT / 'ppocr_addon'

# Fail the build, loudly, if a model is missing. Otherwise the frozen add-on
# would answer the handshake and then fail the first request in that script.
_SCRIPTS = runpy.run_path(str(PACKAGE / 'scripts.py'))['SCRIPTS']
_FILES = [PACKAGE / 'selftest.pgm'] + [PACKAGE / 'models' / s / f for s in _SCRIPTS for f in ('rec.onnx', 'dict.txt')]
_MISSING = [str(p) for p in _FILES if not p.is_file()]
if _MISSING:
    raise SystemExit(f'FATAL: missing {", ".join(_MISSING)}. Run tools/fetch_models.py first.')

datas = [(str(PACKAGE / 'selftest.pgm'), 'ppocr_addon')]
datas += [(str(PACKAGE / 'models' / s), f'ppocr_addon/models/{s}') for s in _SCRIPTS]


# ---------------------------------------------------------------------------
# libexpat sanity check (from addon-whispercpp).
#
# PyInstaller bundles whatever expat it resolved on the build host, for
# CPython's pyexpat. If that host has an OLD expat shadowing the distro one
# (for example a hand-built 2.5.0 in /usr/local/lib), the frozen binary dies
# before reaching Python:
#     Failed to load libpython3.x.so: undefined symbol: XML_SetReparseDeferralEnabled
# So verify the bundled copy exports the symbol libpython needs, and substitute
# a known-good system copy if it does not.
# ---------------------------------------------------------------------------
_EXPAT_SYMBOL = 'XML_SetReparseDeferralEnabled'


def _exports_symbol(path: str, symbol: str) -> bool:
    try:
        out = subprocess.run(['nm', '-D', '--defined-only', path],
                             capture_output=True, text=True, timeout=30)
        return symbol in out.stdout
    except Exception:
        return True  # no nm (macOS/Windows): assume fine rather than break the build


def _fix_libexpat(entries):
    if not sys.platform.startswith('linux'):
        return entries
    out = []
    for entry in entries:
        name = entry[0] if entry else ''
        src = entry[1] if len(entry) > 1 else ''
        if not os.path.basename(name).startswith('libexpat.so') or _exports_symbol(src, _EXPAT_SYMBOL):
            out.append(entry)
            continue
        for cand in ('/usr/lib/x86_64-linux-gnu/libexpat.so.1', '/usr/lib64/libexpat.so.1',
                     '/lib/x86_64-linux-gnu/libexpat.so.1'):
            if os.path.exists(cand) and _exports_symbol(cand, _EXPAT_SYMBOL):
                print(f'WARNING: {src} lacks {_EXPAT_SYMBOL}; substituting {cand}')
                out.append((name, cand) + tuple(entry[2:]))
                break
        else:
            print(f'ERROR: bundled libexpat lacks {_EXPAT_SYMBOL}; the frozen binary will not start.')
            out.append(entry)
    return out


a = Analysis(
    [str(PACKAGE / '__main__.py')],
    pathex=[str(SPEC_ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # ONNX Runtime's optional parts: training, the converters and tools,
        # and what only they import. Inference needs none of it.
        'onnxruntime.training', 'onnxruntime.tools', 'onnxruntime.transformers',
        'onnxruntime.quantization', 'onnx', 'sympy', 'coloredlogs', 'humanfriendly',
        'torch', 'scipy', 'matplotlib', 'pandas', 'IPython',
        'PySide6', 'PyQt6', 'PyQt5', 'tkinter', 'Tkinter', '_tkinter',
        'pytest', '_pytest', 'pluggy', 'iniconfig', 'setuptools', 'pkg_resources',
        'unittest', 'pydoc_data',
        # Standard modules the add-on never calls, reached only through
        # optional imports. Each would bring a native library, and with it a
        # licence to ship. Nothing in the add-on uses the network.
        'ssl', '_ssl', 'hashlib', '_hashlib', 'urllib.request', 'http.client',
        'bz2', '_bz2', 'lzma', '_lzma', 'compression', '_zstd',
        'decimal', '_decimal', '_pydecimal',
        # GPL (readline) and ncurses: for interactive terminals, which the
        # add-on never has.
        'readline', 'curses', '_curses', '_curses_panel',
    ],
    noarchive=False,
)

a.binaries = _fix_libexpat(a.binaries)
a.binaries = [entry for entry in a.binaries if not os.path.basename(entry[0]).startswith('libmvec.so')]

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ppocr-addon',
    debug=False,
    bootloader_ignore_signals=False,
    # strip=False: stripping the native libraries saves little and breaks
    # macOS code signatures.
    strip=False,
    upx=False,
    console=True,  # stdio is the protocol channel
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='ppocr-addon',
)
