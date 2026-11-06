# PP-OCRv5 add-on for Subtitld

Offline **OCR of picture subtitles** for [Subtitld](https://github.com/Subtitld/subtitld),
with PaddleOCR's [PP-OCRv5](https://github.com/PaddlePaddle/PaddleOCR) mobile
text recognition models.

DVD subtitles (VobSub: `.idx` + `.sub`) and Blu-ray subtitles (PGS: `.sup`),
as files or as tracks inside an MKV, MP4 or TS, are drawn as pictures, not
written as text. With this add-on installed, Subtitld opens them, reads every
subtitle as text, and keeps their times and their place on the screen; you
edit them and save them as SRT or any other format.

- Serves the `ocr.recognize` task over Subtitld's add-on protocol. Subtitld
  decodes the subtitles itself and sends one picture per subtitle; the add-on
  sends back its text, lines and all.
- **CPU only, no torch, no cloud account, nothing downloads.** ONNX Runtime
  runs the models; numpy reads the pictures. About 25 ms a line, so a
  feature film's subtitles take a minute or two.
- All twelve models are in the archive (112 MB).

## Languages

The model is picked from the subtitles' language, which Subtitld asks for
when it opens the file (the file's own, when it names one).

| Model | Languages |
|-------|-----------|
| English | English |
| Latin script | French, German, Spanish, Portuguese, Italian, Dutch, Catalan, Galician, Basque, Polish, Czech, Slovak, Slovenian, Croatian, Bosnian, Hungarian, Turkish, Finnish, Swedish, Danish, Norwegian, Icelandic, Estonian, Lithuanian, Albanian, Irish, Indonesian, Malay, Swahili, Tagalog/Filipino, Afrikaans, Latin, Occitan, Uzbek, Javanese, Sundanese, Zulu |
| Chinese and Japanese | Chinese (Simplified and Traditional, Mandarin and Cantonese), Japanese |
| Korean | Korean |
| East Slavic | Russian, Ukrainian, Belarusian |
| Cyrillic | Bulgarian, Serbian, Macedonian, Kazakh, Kyrgyz, Mongolian, Tatar, Tajik, Bashkir, Chuvash |
| Greek | Greek |
| Arabic script | Arabic, Persian, Urdu, Pashto, Kurdish, Uyghur, Sindhi |
| Devanagari | Hindi, Marathi, Nepali, Sanskrit, Maithili, Bhojpuri, Newari |
| Thai, Tamil, Telugu | one each |

A language is listed only when every letter of its alphabet is in the model's
dictionary: Romanian (ș, ț), Vietnamese (ơ, ư), Latvian (ā, ē), Welsh (ŵ, ŷ),
Maltese, Māori, Yakut, Azerbaijani and Esperanto are not, and Hebrew, Georgian, Armenian,
Bengali, Khmer, Lao and Burmese have no model. Serbian is read in Cyrillic;
choose the Latin script model under *Configure* for Serbian in Latin letters
(a `sr-latn` language code picks it too).

After reading, the add-on puts right what the models get wrong on subtitles:
in English a lone `l` or `|` is the pronoun I (`l'm` → `I'm`); in Chinese and
Japanese, ASCII `?`, `!`, `,` after their characters become full width.

## Options

Set these in **Settings ▸ Add-ons ▸ PP-OCRv5 ▸ Configure**.

| Option | Default | Env var | Notes |
|--------|---------|---------|-------|
| Script | `auto` | `PPOCR_SCRIPT` | The model to read with; `auto` picks it from the language. |
| CPU threads | `0` | `PPOCR_THREADS` | `0` = half the logical processors, at most 4. |

Each request takes its option from the request's own `options`, then the
environment variable (Subtitld sets it from the Configure dialog), then the
default. `PPOCR_ADDON_LOG_LEVEL=DEBUG` logs more; `PPOCR_ADDON_MODELS` points
a source run at another models folder.

## The protocol

```
request:  {"id", "type": "ocr.recognize",
           "params": {"images": [{"id", "path"}], "language": "pt-br", "options": {}}}
partial:  {"id": <image id>, "text"}            one per picture, in order
progress: {"value": 0..1, "message"}
result:   {"texts": [{"id", "text"}], "script", "model"}
```

Each picture is an 8-bit grayscale PGM (P5): the text in dark ink on white,
cropped to it. A subtitle of several lines is cut into lines at the rows
without ink, and comes back with its lines joined by `\n`.

## Requirements

- 64-bit Linux, macOS on Apple silicon, or Windows.
- About 260 MB of disk and 512 MB of free memory.

## Development

```bash
python3.12 -m venv .venv
.venv/bin/pip install -c constraints.txt -e '.[dev]'
.venv/bin/python tools/fetch_models.py   # 112 MB, each file checked by sha256
.venv/bin/pytest
```

The test pictures in `tests/data/` and `ppocr_addon/selftest.pgm` are made by
`tools/render_samples.py` (needs Pillow and the Liberation and Noto CJK fonts).

The release workflow (`.github/workflows/release.yml`) freezes the add-on
with PyInstaller on Linux, macOS and Windows, installs each archive the way
Subtitld does, and runs the self-test and `tests/test_protocol.py` against it.

## Where the models come from

PaddlePaddle publishes PP-OCRv5 in Paddle's format (Apache-2.0); RapidAI's
RapidOCR project converts the models to ONNX and publishes them, with their
dictionaries, on [ModelScope](https://www.modelscope.cn/models/RapidAI/RapidOCR)
(tag v3.10.0). `tools/fetch_models.py` takes them from there, unchanged, and
pins each file by sha256.

## Licence

The add-on is Apache-2.0 ([LICENSE](LICENSE)); so are the models (see
[NOTICE](NOTICE)). Each release archive lists everything frozen into it in
`THIRD_PARTY_LICENSES/`. The Linux archive carries libquadmath (LGPL-2.1,
with numpy); its source is attached to each release.
