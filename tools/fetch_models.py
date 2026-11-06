#!/usr/bin/env python3
"""Download PP-OCRv5's twelve mobile recognition models into ppocr_addon/models/, checked by sha256.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

PaddlePaddle publishes the models in Paddle's format (Apache-2.0); RapidAI
converts them to ONNX and publishes them on ModelScope with the dictionaries
(RapidAI/RapidOCR, tag v3.10.0). Each file is pinned below; a file that does
not match fails the build. Files already in place and matching are kept.

    python tools/fetch_models.py
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'ppocr_addon' / 'models'
BASE = 'https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/'

# script -> (dictionary's name upstream, rec.onnx sha256, dict.txt sha256)
PINS = {
    'en': ('ppocrv5_en_dict', 'c3461add59bb4323ecba96a492ab75e06dda42467c9e3d0c18db5d1d21924be8',
           'e025a66d31f327ba0c232e03f407ae8d105e1e709e7ccb3f408aa778c24e70d6'),
    'latin': ('ppocrv5_latin_dict', 'b20bd37c168a570f583afbc8cd7925603890efbcdc000a59e22c269d160b5f5a',
              '3c0a8a79b612653c25f765271714f71281e4e955962c153e272b7b8c1d2b13ff'),
    'ch': ('ppocrv5_dict', '5825fc7ebf84ae7a412be049820b4d86d77620f204a041697b0494669b1742c5',
           'd1979e9f794c464c0d2e0b70a7fe14dd978e9dc644c0e71f14158cdf8342af1b'),
    'korean': ('ppocrv5_korean_dict', 'cd6e2ea50f6943ca7271eb8c56a877a5a90720b7047fe9c41a2e541a25773c9b',
               'a88071c68c01707489baa79ebe0405b7beb5cca229f4fc94cc3ef992328802d7'),
    'eslav': ('ppocrv5_eslav_dict', '08705d6721849b1347d26187f15a5e362c431963a2a62bfff4feac578c489aab',
              '3e95f1581557162870cacdba5af91a4c6be2890710d395b0c3c7578e7ee5e6eb'),
    'cyrillic': ('ppocrv5_cyrillic_dict', '90f761b4bfcce0c8c561c0cb5c887b0971d3ec01c32164bdf7374a35b0982711',
                 'db40aa52ceb112055be80c694afdf655d5d2c4f7873704524cc16a447ca913ba'),
    'el': ('ppocrv5_el_dict', 'b4368bccd557123c702b7549fee6cd1e94b581337d1c9b65310f109131542b7f',
           '31defc62c0c3ad3674a82da6192226a2ba98ef4ff014a7045cb88d59f9c3de31'),
    'arabic': ('ppocrv5_arabic_dict', 'c1192e632d0baa9146ae5b756a0e635e3dc63c1733737ebfd1629e87144e9295',
               '7f92f7dbb9b75a4787a83bfb4f6d14a8ab515525130c9d40a9036f61cf6999e9'),
    'devanagari': ('ppocrv5_devanagari_dict', 'd6f0a906580e3fa6b324a318718f1f31f268b6ea8ef985f91c2012a37f52c91e',
                   '09c7440bfc5477e5c41052304b6b185aff8c4a5e8b2b4c23c1c706f6fe1ee9fc'),
    'th': ('ppocrv5_th_dict', 'de541dd83161c241ff426f7ecfd602a0ba77d686cf3ab9a6c255ea82fd08006e',
           '57f5406f94bb6688fb7077f7be65f08bbd71cecf48c01ea26c522cb5c4836b7a'),
    'ta': ('ppocrv5_ta_dict', 'a42448808b7dea87597336f12438935f40353f1949e8360acd9e06b4da21bfe1',
           '85b541352ae18dc6ba6d47152d8bf8adff6b0266e605d2eef2990c1bf466117b'),
    'te': ('ppocrv5_te_dict', 'a3690451b50028a09a3316a1274f7c05728151ea3f8fd392696397a7fefcbd92',
           '42f83f5d3fdb50778e4fa5b66c58d99a59ab7792151c5e74f34b8ffd7b61c9d6'),
}


def files():
    """(url, destination, sha256) of every file."""
    for script, (dict_name, rec_sha, dict_sha) in PINS.items():
        folder = f'{script}_PP-OCRv5_rec_mobile'
        yield (f'{BASE}onnx/PP-OCRv5/rec/{folder}.onnx', DEST / script / 'rec.onnx', rec_sha)
        yield (f'{BASE}paddle/PP-OCRv5/rec/{folder}/{dict_name}.txt', DEST / script / 'dict.txt', dict_sha)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    for url, dest, pin in files():
        if dest.is_file() and sha256(dest) == pin:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_suffix(dest.suffix + '.part')
        request = urllib.request.Request(url, headers={'User-Agent': 'subtitld-addon-ppocr'})
        with urllib.request.urlopen(request, timeout=120) as response, open(part, 'wb') as f:
            while block := response.read(1 << 20):
                f.write(block)
        got = sha256(part)
        if got != pin:
            part.unlink()
            print(f'FATAL: {url} has sha256 {got}, pinned {pin}', file=sys.stderr)
            return 1
        part.replace(dest)
        print(f'{dest.relative_to(ROOT)}: {dest.stat().st_size:,} bytes')
    print(f'{len(PINS)} models in {DEST.relative_to(ROOT)}, all match their pins')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
