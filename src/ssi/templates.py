"""Decomposition of encoded words into reusable templates.

A graphical word is split into three parts around one *core* analytical word
(the one with the longest lexeme; see `core_index`):

    proclitics   the analytical words before the core, e.g. 'W-D'
    frame        the core with its lexeme replaced by a lexeme pattern,
                 e.g. '!N!{*(J}[WN' for '!N!BN(J[WN'
    enclitics    the analytical words after the core, e.g. 'HWN'

The lexeme pattern abstracts away the lexeme's untouched leading letters
('*', one or more letters) and keeps the rest of the morph from its first
letter operation on, with each untouched letter written as '.':
'(>T(J&>' stays '(>.(J&>', 'KT&WB' becomes '*&W.', 'MLK' becomes '*'.
So a frame applies to every lexeme that ends in the same letters and has
those letter operations, e.g. all third-yud verbs for '{*(J&>}['. An
unrealised letter that repeats the preceding letter of the lexeme is written
'(%', so that one pattern covers all geminate verbs (Syriac_paradigm.pdf
section 2.3: '!!K&WP(P[' has the pattern '*&W.(%').

A frame can be inverted: given the realised core, it yields the lexeme morph,
from which the full encoding follows (see `Frame.analyses`).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .encoding import _plain, parse_word, realise, split_analytical

_TOKEN_RE = re.compile(r'\([A-Z<>]|&[A-Z<>]|[A-Z<>]')
_AFFIX_PREFIX_RE = re.compile(r'^(?:![^!]*!)?(?:@[^@]*@)?(?:\][^\]]*\])?')


@dataclass(frozen=True)
class Decomposition:
    proclitics: str
    frame: str
    enclitics: str
    lexeme: str  # paradigmatic lexeme of the core, homograph marks included


def lexeme_pattern(morph: str) -> str:
    toks = _TOKEN_RE.findall(morph.rstrip('='))
    first_op = next((i for i, t in enumerate(toks) if len(t) == 2), None)
    if first_op is None:
        return '*'
    out, prev = [], None  # prev: the preceding paradigmatic letter
    for i, t in enumerate(toks):
        if i >= first_op:
            if len(t) == 1:
                out.append('.')
            elif t[0] == '(' and t[1] == prev:
                out.append('(%')
            else:
                out.append(t)
        if t[0] != '&':
            prev = t[-1]
    return ('*' if first_op else '') + ''.join(out)


def core_index(encoding: str) -> int:
    aws = parse_word(encoding)

    def key(i):
        aw = aws[i]
        has_affix = any(v is not None for slot, v in aw.items() if slot != 'lex')
        return len(_plain(aw.lex).rstrip('=')), has_affix, i
    return max(range(len(aws)), key=key)


def decompose(encoding: str) -> Decomposition:
    parts = split_analytical(encoding)
    k = core_index(encoding)
    core = parts[k]
    lex = parse_word(core)[0].lex
    pre = _AFFIX_PREFIX_RE.match(core).group(0)
    post = core[len(pre) + len(lex):]
    frame = f'{pre}{{{lexeme_pattern(lex)}}}{post}'
    return Decomposition('-'.join(parts[:k]), frame, '-'.join(parts[k + 1:]), _plain(lex))


def frame_class(frame: str) -> str:
    """'verb' (has a verbal ending), 'nominal' (nominal ending only) or 'other'."""
    post = frame[frame.index('}') + 1:]
    if '[' in post:
        return 'verb'
    return 'nominal' if '/' in post else 'other'


class LexemePattern:
    """A compiled lexeme pattern: inverts a realised stem into a lexeme morph."""

    def __init__(self, pattern: str):
        self.text = pattern
        self._tokens, k = [], 0  # '*', '.', '(X' or '&X'
        while k < len(pattern):
            n = 2 if pattern[k] in '(&' else 1
            self._tokens.append(pattern[k:k + n])
            k += n
        parts = []
        for t in self._tokens:
            if t == '*':
                parts.append('([A-Z<>]+)')
            elif t == '.':
                parts.append('([A-Z<>])')
            elif t[0] == '&':
                parts.append(re.escape(t[1]))
            # '(X' and '(%' realise nothing
        self._regex = re.compile(''.join(parts) + r'\Z')

    def invert(self, stem: str) -> str | None:
        """Lexeme morph (without homograph marks) realised as `stem`, or None."""
        m = self._regex.match(stem)
        if m is None:
            return None
        groups = iter(m.groups())
        out, prev = [], None
        for t in self._tokens:
            if t in ('*', '.'):
                t = next(groups)
            elif t == '(%':
                if prev is None:
                    return None
                t = '(' + prev
            out.append(t)
            if t[0] != '&':
                prev = t[-1]
        return ''.join(out)

    def __repr__(self):
        return f'LexemePattern({self.text!r})'


def split_frame(frame: str) -> tuple[str, str, str]:
    """(affixes before the lexeme, lexeme pattern, affixes after it)."""
    i, j = frame.index('{'), frame.index('}')
    return frame[:i], frame[i + 1:j], frame[j + 1:]


class Frame:
    """A compiled frame: matches a realised core and rebuilds the lexeme morph."""

    def __init__(self, frame: str):
        self.text = frame
        self.pre, pattern, self.post = split_frame(frame)
        self.pattern = LexemePattern(pattern)
        self.pre_real, self.post_real = realise(self.pre), realise(self.post)
        self.cls = frame_class(frame)

    def invert(self, stem: str) -> str | None:
        return self.pattern.invert(stem)

    def fill(self, lexeme_morph: str) -> str:
        return f'{self.pre}{lexeme_morph}{self.post}'

    def __repr__(self):
        return f'Frame({self.text!r})'
