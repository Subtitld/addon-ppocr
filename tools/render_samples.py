#!/usr/bin/env python3
"""Render the test pictures: subtitles as a disc draws them, then as the host
hands them over (dark ink on white, PGM).

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

Needs Pillow and the Liberation and Noto CJK fonts; the pictures are
committed, so only changing them needs this.

    python tools/render_samples.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
LIB = '/usr/share/fonts/truetype/liberation/'
CJK = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'

SELF_TEST = "I'm reading Subtitld's pictures."
SAMPLES = [
    # (name, language, text, style, italic)
    ('en_two_lines', 'en', "I told you not to come back here.\nWhy didn't you listen?", 'pgs', False),
    ('en_dvd', 'en', "- Are you okay?\n- I'm fine, thanks.", 'dvd', False),
    ('pt_accents', 'pt-br', 'É difícil explicar\no que aconteceu.', 'dvd', True),
    ('fr_dvd', 'fr-fr', "L'été dernier, à Noël,\nj'étais très ému.", 'dvd', False),
    ('de_pgs', 'de-de', 'Das Mädchen hat große Füße.', 'pgs', True),
    ('ru_pgs', 'ru-ru', 'Это уже слишком поздно,\nнам надо идти.', 'pgs', False),
    ('zh_pgs', 'cmn-hans-cn', '我们必须马上离开这里。\n快点！', 'pgs', False),
    ('ja_dvd', 'ja-jp', 'どこへ行くの？', 'dvd', False),
]


def font(language, size, italic):
    if language.split('-')[0] in ('cmn', 'zh', 'ja'):
        return ImageFont.truetype(CJK, size, index=0)
    return ImageFont.truetype(LIB + ('LiberationSans-BoldItalic.ttf' if italic else 'LiberationSans-Bold.ttf'), size)


def render(text, language, style, italic):
    """The subtitle's RGBA picture: Blu-ray style (antialiased) or DVD style (four colours)."""
    size, stroke = (54, 3) if style == 'pgs' else (24, 2)
    face = font(language, size, italic)
    probe = ImageDraw.Draw(Image.new('RGBA', (1, 1)))
    box = [int(round(v)) for v in probe.multiline_textbbox((0, 0), text, font=face, stroke_width=stroke,
                                                            align='center', spacing=size // 4)]
    picture = Image.new('RGBA', (box[2] - box[0] + 8, box[3] - box[1] + 8), (0, 0, 0, 0))
    ImageDraw.Draw(picture).multiline_text((4 - box[0], 4 - box[1]), text, font=face, fill=(235, 235, 235, 255),
                                           stroke_width=stroke, stroke_fill=(16, 16, 16, 255), align='center',
                                           spacing=size // 4)
    if style == 'dvd':
        pixels = np.array(picture)
        lum = pixels[..., :3].mean(-1)
        out = np.zeros_like(pixels)
        on = pixels[..., 3] > 100
        out[on & (lum > 170)] = (235, 235, 235, 255)
        out[on & (lum <= 170) & (lum > 80)] = (128, 128, 128, 255)
        out[on & (lum <= 80)] = (16, 16, 16, 255)
        picture = Image.fromarray(out)
    return picture


def ink(picture):
    """As the host hands it over: over black, then inverted."""
    black = Image.new('RGBA', picture.size, (0, 0, 0, 255))
    gray = np.array(Image.alpha_composite(black, picture).convert('L'))
    return 255 - gray


def write_pgm(path, gray):
    h, w = gray.shape
    path.write_bytes(b'P5\n%d %d\n255\n' % (w, h) + gray.astype(np.uint8).tobytes())


def main():
    write_pgm(ROOT / 'ppocr_addon' / 'selftest.pgm', ink(render(SELF_TEST, 'en', 'pgs', False)))
    data = ROOT / 'tests' / 'data'
    data.mkdir(parents=True, exist_ok=True)
    expected = {}
    for name, language, text, style, italic in SAMPLES:
        write_pgm(data / f'{name}.pgm', ink(render(text, language, style, italic)))
        expected[name] = {'language': language, 'text': text}
    (data / 'expected.json').write_text(json.dumps(expected, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(f'{len(SAMPLES) + 1} pictures written')


if __name__ == '__main__':
    main()
