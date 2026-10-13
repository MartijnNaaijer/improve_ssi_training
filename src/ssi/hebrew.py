"""The Hebrew parallel text (BHSA) as evidence for the ranker (Phase 3).

For the 15 biblical books the Syriac verse has a Hebrew counterpart with the
same reference (checked: the same verse shares 3-4 times as many lexemes as
the next one). The Hebrew is encoded in the same ETCBC style, except that
pronominal suffixes are attached with '+' instead of '-', and compound names
are joined with '_'.

Evidence for a candidate analysis of a Syriac word, computed from its core
lexeme (the analytical word with the longest lexeme) and the Hebrew words of
the parallel verse:

    heb_has    the verse has a Hebrew parallel
    heb_t_max  log of the best translation probability t(lexeme | Hebrew
               lexeme) over the verse (IBM Model 1, Syriac given Hebrew,
               fitted on gold training verses)
    heb_t_al   log of the same probability for the aligned Hebrew word
    heb_cog    cognate similarity with the aligned Hebrew lexeme
    heb_dpos   distance in relative verse position to the aligned word
    heb_known  the lexeme is in the translation table

The aligned Hebrew word is the one that maximises the translation or cognate
score times a position prior. Its full analysis (with proclitics, so that a
wayyiqtol stays recognisable) is passed to the ranker as a string.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from functools import lru_cache

from .corpus import RAW_DIR, Verse
from .encoding import lexemes
from .templates import core_index

BOOKS = {'Gn': 'Gen', 'Ex': 'Exod', 'Lv': 'Lev', 'Nm': 'Num', 'Dt': 'Deut', 'Jos': 'Jos', 'Jd': 'Jud',
         'Ps': 'Psal', '1Rg': '1Reg', '2Rg': '2Reg', 'Jr': 'Jer', 'Ez': 'Ezech', 'Jon': 'Jona', 'Sa': 'Sach',
         'Dn': 'Dan'}

HEB_FEATURES = ('heb_has', 'heb_t_max', 'heb_t_al', 'heb_cog', 'heb_dpos', 'heb_known')
FLOOR = 1e-6
# Feature values when there is no Hebrew (also used for Hebrew dropout in training).
HEB_MISSING = (0.0, math.log(FLOOR), math.log(FLOOR), 0.0, 1.0, 0.0)
NULL = '<null>'


class HebrewWord:
    __slots__ = ('analysis', 'lexemes')

    def __init__(self, analysis: str):
        self.analysis = analysis
        self.lexemes = hebrew_lexemes(analysis)


def hebrew_lexemes(analysis: str) -> tuple[str, ...]:
    norm = analysis.replace('+', '-').replace('_', '')
    try:
        return lexemes(norm)
    except ValueError:
        return tuple(x for x in re.split(r'[-\[/~!@\]:]', norm) if x)


def load_hebrew(path=RAW_DIR / 't-out') -> dict[tuple[str, str, str], list[HebrewWord]]:
    """(Syriac book code, chapter, verse) -> Hebrew words of the parallel verse."""
    to_syr = {h: s for s, h in BOOKS.items()}
    out = {}
    with open(path, encoding='utf8') as f:
        for line in f:
            book, ch, v, text = line.rstrip('\n').split('\t')
            if book in to_syr:
                out[to_syr[book], ch, v] = [HebrewWord(w) for w in text.split()]
    return out


def verse_key(v: Verse) -> tuple[str, str, str]:
    return v.book, v.chapter, v.verse.split('#')[0]


def core_lexeme(analysis: str) -> str:
    try:
        return lexemes(analysis)[core_index(analysis)]
    except ValueError:
        return ''


# -- cognates -------------------------------------------------------------------
# Regular Hebrew -> Syriac consonant correspondences (substitution cost 0.25).
_CORRESPOND = {('C', 'T'), ('Z', 'D'), ('Y', 'V'), ('Y', '<'), ('F', 'S'), ('F', 'C'), ('H', '>'), ('C', 'S')}


@lru_cache(maxsize=2_000_000)
def cognate(syr: str, heb: str) -> float:
    """1 - weighted edit distance / longer length, homograph marks ignored."""
    a, b = syr.replace('=', ''), heb.replace('=', '')
    if not a or not b:
        return 0.0
    prev = [j * 1.0 for j in range(len(b) + 1)]
    for i in range(1, len(a) + 1):
        cur = [i * 1.0] + [0.0] * len(b)
        for j in range(1, len(b) + 1):
            x, y = a[i - 1], b[j - 1]
            sub = 0.0 if x == y else 0.25 if (y, x) in _CORRESPOND else 1.0
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + sub)
        prev = cur
    return max(0.0, 1 - prev[-1] / max(len(a), len(b)))


# -- IBM Model 1 --------------------------------------------------------------
class IBM1:
    """t(syriac lexeme | hebrew lexeme), with a NULL Hebrew word."""

    def fit(self, pairs: list[tuple[list[str], list[str]]], iterations: int = 8) -> 'IBM1':
        syr_vocab = {s for ss, _ in pairs for s in ss}
        uniform = 1.0 / max(len(syr_vocab), 1)
        t = defaultdict(lambda: uniform)
        for _ in range(iterations):
            count = defaultdict(float)
            total = defaultdict(float)
            for ss, hh in pairs:
                hh = hh + [NULL]
                for s in ss:
                    ws = [t[s, h] for h in hh]
                    z = sum(ws)
                    for h, w in zip(hh, ws):
                        c = w / z
                        count[s, h] += c
                        total[h] += c
            t = defaultdict(float, {(s, h): c / total[h] for (s, h), c in count.items() if c / total[h] > 1e-4})
        self.t = dict(t)
        self.syr_vocab = syr_vocab
        return self


def lexeme_pairs(verses: list[Verse], hebrew) -> list[tuple[list[str], list[str]]]:
    pairs = []
    for v in verses:
        heb = hebrew.get(verse_key(v))
        if heb:
            pairs.append(([x for a in v.analysis for x in lexemes(a)], [x for w in heb for x in w.lexemes]))
    return pairs


class HebrewEvidence:
    def __init__(self, hebrew, ibm: IBM1, position_weight: float = 2.0):
        self.hebrew, self.t, self.known, self.pw = hebrew, ibm.t, ibm.syr_vocab, position_weight

    @classmethod
    def fit(cls, train: list[Verse], hebrew) -> 'HebrewEvidence':
        return cls(hebrew, IBM1().fit(lexeme_pairs(train, hebrew)))

    def verse(self, v: Verse, cands: list[list[str]]) -> tuple[list[list[list[float]]], list[list[str]]]:
        """Per word, per candidate: the HEB_FEATURES values and the aligned Hebrew analysis ('' if none)."""
        heb = self.hebrew.get(verse_key(v))
        if not heb:
            return [[list(HEB_MISSING) for _ in cs] for cs in cands], [['' for _ in cs] for cs in cands]
        n, m = len(cands), len(heb)
        cache = {}
        feats, strings = [], []
        for i, cs in enumerate(cands):
            pos = (i + 0.5) / n
            fw, sw = [], []
            for a in cs:
                lex = core_lexeme(a)
                if lex not in cache:
                    ts = [max((self.t.get((lex, h), 0.0) for h in w.lexemes), default=0.0) for w in heb]
                    cs_ = [max((cognate(lex, h) for h in w.lexemes), default=0.0) for w in heb]
                    cache[lex] = (ts, cs_)
                ts, cs_ = cache[lex]
                best_j, best = 0, -1.0
                for j in range(m):
                    evidence = max(ts[j], 0.5 * cs_[j] if cs_[j] >= 0.6 else 0.0)
                    score = evidence * math.exp(-self.pw * abs(pos - (j + 0.5) / m))
                    if score > best:
                        best_j, best = j, score
                dpos = abs(pos - (best_j + 0.5) / m)
                fw.append([1.0, math.log(max(ts) + FLOOR), math.log(ts[best_j] + FLOOR), cs_[best_j], dpos,
                           float(lex in self.known)])
                sw.append(heb[best_j].analysis)
            feats.append(fw)
            strings.append(sw)
        return feats, strings


def add_hebrew(examples, verses: list[Verse], evidence_of) -> None:
    """Append the HEB_FEATURES columns to every candidate's features and set
    `ex.heb` (aligned Hebrew analysis per candidate). `evidence_of(verse)` gives
    the HebrewEvidence to use for that verse (cross-fitted for training verses)."""
    import numpy as np
    for ex, v in zip(examples, verses):
        feats, strings = evidence_of(v).verse(v, ex.cands)
        ex.feats = [np.hstack([f, np.array(h, dtype=np.float32).reshape(len(f), len(HEB_FEATURES))])
                    for f, h in zip(ex.feats, feats)]
        ex.heb = strings
