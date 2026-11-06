"""Which PP-OCRv5 recognition model reads which language.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

PP-OCRv5 has one mobile recognition model per script. The host names a
subtitle's language with a BCP-47-ish tag (`pt-br`, `cmn-hans-cn`,
`sr-latn`); its primary subtag picks the model. A language is listed only
when every letter of its alphabet is in that model's dictionary
(Romanian's ș/ț, Vietnamese's ơ/ư, Latvian's ā/ē, Welsh's ŵ/ŷ are not, in
any of them).
"""

from __future__ import annotations

# id -> (what the Configure dialog calls it, the languages it reads)
SCRIPTS: dict[str, tuple[str, tuple[str, ...]]] = {
    'en': ('English', ('en',)),
    'latin': ('Latin script', (
        'fr', 'de', 'es', 'pt', 'it', 'nl', 'ca', 'gl', 'eu', 'pl', 'cs', 'sk', 'sl', 'hr', 'bs', 'hu',
        'tr', 'fi', 'sv', 'da', 'no', 'nb', 'nn', 'is', 'et', 'lt', 'sq', 'ga', 'id', 'ms', 'sw', 'tl',
        'fil', 'af', 'la', 'oc', 'uz', 'jv', 'su', 'zu')),
    'ch': ('Chinese and Japanese', ('zh', 'cmn', 'yue', 'ja')),
    'korean': ('Korean', ('ko',)),
    'eslav': ('Russian, Ukrainian, Belarusian', ('ru', 'uk', 'be')),
    'cyrillic': ('Other Cyrillic script', ('bg', 'sr', 'mk', 'kk', 'ky', 'mn', 'tt', 'tg', 'ba', 'cv')),
    'el': ('Greek', ('el',)),
    'arabic': ('Arabic script', ('ar', 'fa', 'ur', 'ps', 'ku', 'ug', 'sd')),
    'devanagari': ('Devanagari', ('hi', 'mr', 'ne', 'sa', 'mai', 'bho', 'new')),
    'th': ('Thai', ('th',)),
    'ta': ('Tamil', ('ta',)),
    'te': ('Telugu', ('te',)),
}

# Every language with a model, in the manifest's order.
LANGUAGES: tuple[str, ...] = tuple(lang for _label, langs in SCRIPTS.values() for lang in langs)

_BY_LANGUAGE = {lang: script for script, (_label, langs) in SCRIPTS.items() for lang in langs}
# A script subtag the language's default model does not read.
_BY_SCRIPT_SUBTAG = {'latn': 'latin', 'cyrl': 'cyrillic', 'arab': 'arabic', 'hans': 'ch', 'hant': 'ch',
                     'jpan': 'ch', 'kore': 'korean', 'deva': 'devanagari', 'grek': 'el'}


def script_for(language: str, override: str = '') -> str | None:
    """The model that reads `language`: `override` when it names one
    ('auto' and '' do not), else the language's own; None when no model
    reads it. `sr-latn` reads with the Latin model, `uz-cyrl` with the
    Cyrillic one."""
    if override in SCRIPTS:
        return override
    parts = str(language or '').strip().lower().replace('_', '-').split('-')
    primary = parts[0]
    if primary in ('', 'auto', 'und', 'mul'):
        return 'latin'
    for subtag in parts[1:]:
        if subtag in _BY_SCRIPT_SUBTAG and primary not in ('zh', 'cmn', 'yue', 'ja'):
            return _BY_SCRIPT_SUBTAG[subtag]
    return _BY_LANGUAGE.get(primary)
