"""The ETCBC morphological encoding of Syriac (see Syriac_paradigm.pdf, section 1).

An encoded graphical word consists of analytical words separated by '-'.
Each analytical word has the morpheme slots, in this fixed order:

    pfm  !..!   preformative
    pfx  @..@   passive prefix
    vbs  ]..]   verbal stem morpheme
    lex         lexeme (paradigmatic form, possibly with '=' homograph marks)
    vbe  [..    verbal ending
    nme  /..    nominal ending
    emf  ~..    emphatic marker
    vpm  :..    vowel pattern mark (lowercase letters)

Within a morph, '(X' marks a paradigmatic letter that is not realised and
'&X' a realised letter that is not paradigmatic.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, fields

SLOTS = ('pfm', 'pfx', 'vbs', 'lex', 'vbe', 'nme', 'emf', 'vpm')

# Diacritics in the surface text that are not part of the consonantal skeleton.
SURFACE_DIACRITICS = '^#"'

_MARKERS = set('!@][/~-=')
_VPM_RE = re.compile(r':[a-z]+')
# Characters that may occur inside a morph: consonants, '(' and '&' letter
# operations, and '=' homograph marks.
_M = r'[A-Z<>(&=]'
_ANALYTICAL_RE = re.compile(
    rf'^(?:!(?P<pfm>{_M}*)!)?'
    rf'(?:@(?P<pfx>{_M}*)@)?'
    rf'(?:\](?P<vbs>{_M}*)\])?'
    rf'(?P<lex>{_M}+)'
    rf'(?:\[(?P<vbe>{_M}*))?'
    rf'(?:/(?P<nme>{_M}*))?'
    rf'(?:~(?P<emf>{_M}*))?'
    r'(?::(?P<vpm>[a-z]+))?$'
)


@dataclass(frozen=True)
class AnalyticalWord:
    """One analytical word. Slot values are None when the morpheme is absent;
    an empty string means the morpheme is present with an empty realisation
    and paradigmatic form, e.g. the bare verbal ending '['."""
    pfm: str | None = None
    pfx: str | None = None
    vbs: str | None = None
    lex: str = ''
    vbe: str | None = None
    nme: str | None = None
    emf: str | None = None
    vpm: str | None = None

    def items(self):
        return [(f.name, getattr(self, f.name)) for f in fields(self)]


def strip_surface(word: str) -> str:
    """Remove diacritics from a surface word, leaving the consonantal skeleton."""
    return ''.join(c for c in word if c not in SURFACE_DIACRITICS)


def realise(encoding: str) -> str:
    """Return the surface consonants that an encoded word realises.

    Drops the letter after each '(', keeps the letter after each '&', and
    strips all morpheme markers and vowel pattern marks.
    """
    s = _VPM_RE.sub('', encoding)
    out = []
    i = 0
    while i < len(s):
        c = s[i]
        if c == '(':
            i += 2
        elif c == '&':
            if i + 1 < len(s):
                out.append(s[i + 1])
            i += 2
        else:
            if c not in _MARKERS:
                out.append(c)
            i += 1
    return ''.join(out)


def is_consistent(surface: str, encoding: str) -> bool:
    """True if the encoding realises exactly the consonants of the surface word."""
    return realise(encoding) == strip_surface(surface)


def parse_analytical(s: str) -> AnalyticalWord:
    m = _ANALYTICAL_RE.match(s)
    if m is None:
        raise ValueError(f'cannot parse analytical word {s!r}')
    return AnalyticalWord(**m.groupdict(default=None) | {'lex': m.group('lex')})


def split_analytical(encoding: str) -> list[str]:
    """Split a graphical word into analytical words on '-'.

    A '-' preceded by '(' or '&' is a letter operation, not a boundary; that
    does not occur in the corpus, but guard against it anyway.
    """
    parts, cur, i = [], [], 0
    while i < len(encoding):
        c = encoding[i]
        if c in '(&' and i + 1 < len(encoding):
            cur.append(encoding[i:i + 2])
            i += 2
            continue
        if c == '-':
            parts.append(''.join(cur))
            cur = []
        else:
            cur.append(c)
        i += 1
    parts.append(''.join(cur))
    return parts


def parse_word(encoding: str) -> list[AnalyticalWord]:
    return [parse_analytical(p) for p in split_analytical(encoding)]


def surface_segmentation(encoding: str) -> str:
    """The encoding with paradigmatic information removed: unrealised letters,
    homograph marks and vowel pattern marks are dropped, and '&' letters are
    kept as plain letters. What remains is the realised word with morpheme
    boundaries and types, i.e. the segmentation of the surface form."""
    s = _VPM_RE.sub('', encoding)
    out, i = [], 0
    while i < len(s):
        c = s[i]
        if c == '(':
            i += 2
        elif c == '&':
            i += 1
        elif c == '=':
            i += 1
        else:
            out.append(c)
            i += 1
    return ''.join(out)


def lexemes(encoding: str) -> tuple[str, ...]:
    """The paradigmatic lexemes of all analytical words, homograph marks included."""
    return tuple(_plain(a.lex) for a in parse_word(encoding))


def _plain(morph: str) -> str:
    """Paradigmatic form of a morph: drop '&X' (realised-only letters) and
    keep the letter of '(X' (paradigmatic but unrealised)."""
    out, i = [], 0
    while i < len(morph):
        c = morph[i]
        if c == '&':
            i += 2
        elif c == '(':
            if i + 1 < len(morph):
                out.append(morph[i + 1])
            i += 2
        else:
            out.append(c)
            i += 1
    return ''.join(out)


def paradigmatic_items(encoding: str) -> list[tuple[int, str, str]]:
    """(analytical word index, slot, paradigmatic value) for every morpheme present."""
    items = []
    for idx, aw in enumerate(parse_word(encoding)):
        for slot, value in aw.items():
            if value is None:
                continue
            if slot == 'lex' or slot == 'vpm':
                items.append((idx, slot, value if slot == 'vpm' else _plain(value)))
            else:
                items.append((idx, slot, _plain(value)))
    return items
