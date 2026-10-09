"""Conversion between ETCBC encodings and the per-character labels of
chenzhan4321/etcbc_models_comparison ("structure-first" representation).

Steps, as in upstream data_processing_tools/decompose:
  1. reduce: drop the opening marker of the doubled prefix markers
     (!X! -> X!, @X@ -> X@, ]X] -> X]);
  2. split each reduced word into its surface letters and the n+1 marker
     "patterns" around them ('(' takes the next letter into the pattern, ':'
     takes up to two lowercase letters);
  3. a 7-word line ' w1 w2 ...' gets one label per character: the space
     before a word carries the word's leading pattern, every letter carries
     the pattern that follows it.

The inverse of step 1 is not in the upstream repository, so it is
implemented here: within an analytical word the prefix morphs come first and
in fixed order (pfm, pfx, vbs), and their contents cannot contain markers, so
the reduced form is unambiguous.
"""
from __future__ import annotations

import re
from collections import Counter

from ..encoding import strip_surface

LETTERS = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ<>')
_M = r'[A-Z<>(&=]'
_REDUCED_PREFIX_RE = re.compile(rf'^(?:({_M}*)!)?(?:({_M}*)@)?(?:({_M}*)\])?(.*)$', re.S)


def reduce(encoding: str) -> str:
    s = re.sub(r'!([^!]*?)!', r'\1!', encoding)
    s = re.sub(r'@([^@]*?)@', r'\1@', s)
    return re.sub(r'\]([^\]]*?)\]', r'\1]', s)


def _split_analytical(s: str) -> list[str]:
    parts, cur, i = [], [], 0
    while i < len(s):
        if s[i] in '(&' and i + 1 < len(s):
            cur.append(s[i:i + 2])
            i += 2
            continue
        if s[i] == '-':
            parts.append(''.join(cur))
            cur = []
        else:
            cur.append(s[i])
        i += 1
    parts.append(''.join(cur))
    return parts


def restore(reduced: str) -> str:
    """Inverse of `reduce` for one graphical word."""
    out = []
    for part in _split_analytical(reduced):
        m = _REDUCED_PREFIX_RE.match(part)
        pfm, pfx, vbs, rest = m.groups()
        s = ''
        if pfm is not None:
            s += f'!{pfm}!'
        if pfx is not None:
            s += f'@{pfx}@'
        if vbs is not None:
            s += f']{vbs}]'
        out.append(s + rest)
    return '-'.join(out)


def word_patterns(reduced: str) -> tuple[list[str], list[str]]:
    """Surface letters and the len(letters)+1 patterns around them."""
    letters, patterns, cur, i = [], [], '', 0
    while i < len(reduced):
        c = reduced[i]
        if c in LETTERS:
            patterns.append(cur)
            letters.append(c)
            cur = ''
            i += 1
        elif c == '(':
            cur += c
            i += 1
            if i < len(reduced) and reduced[i] in LETTERS:
                cur += reduced[i]
                i += 1
        elif c == ':':
            cur += c
            i += 1
            n = 0
            while i < len(reduced) and reduced[i].islower() and n < 2:
                cur += reduced[i]
                i += 1
                n += 1
        else:
            cur += c
            i += 1
    patterns.append(cur)
    return letters, patterns


def word_from_patterns(letters: str, patterns: list[str]) -> str:
    """Rebuild the full encoding from surface letters and their patterns."""
    s = patterns[0] + ''.join(c + p for c, p in zip(letters, patterns[1:]))
    return restore(s)


def line_input(surface_words: list[str]) -> str:
    return ''.join(' ' + strip_surface(w) for w in surface_words)


def line_patterns(surface_words: list[str], analyses: list[str]) -> list[str]:
    """One pattern per character of line_input(surface_words)."""
    out = []
    for s, a in zip(surface_words, analyses):
        letters, patterns = word_patterns(reduce(a))
        if ''.join(letters) != strip_surface(s):
            raise ValueError(f'letters of {a!r} do not match surface {s!r}')
        out.extend(patterns)
    return out


def analyses_from_line(surface_words: list[str], patterns: list[str]) -> list[str]:
    """Inverse of line_patterns; missing patterns (short prediction) count as empty."""
    out, pos = [], 0
    for s in surface_words:
        bare = strip_surface(s)
        pats = patterns[pos:pos + len(bare) + 1]
        pats += [''] * (len(bare) + 1 - len(pats))
        out.append(word_from_patterns(bare, pats))
        pos += len(bare) + 1
    return out


class PatternInventory:
    """Pattern <-> integer label, built from training data only.

    Label 0 is the empty pattern; the rest are ordered by descending frequency
    (ties alphabetically). Patterns not seen in training map to 0 when encoding
    gold labels for dev/test files; they never occur in predictions.
    """

    def __init__(self, counts: Counter):
        rest = sorted((p for p in counts if p != ''), key=lambda p: (-counts[p], p))
        self.patterns = [''] + rest
        self.index = {p: i for i, p in enumerate(self.patterns)}
        self.counts = counts

    @classmethod
    def from_lines(cls, lines: list[tuple[list[str], list[str]]]) -> 'PatternInventory':
        counts = Counter()
        for surf, anal in lines:
            counts.update(line_patterns(surf, anal))
        return cls(counts)

    def encode(self, patterns: list[str]) -> list[int]:
        return [self.index.get(p, 0) for p in patterns]

    def decode(self, labels: list[int]) -> list[str]:
        return [self.patterns[k] if 0 <= k < len(self.patterns) else '' for k in labels]
