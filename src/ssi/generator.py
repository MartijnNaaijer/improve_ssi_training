"""Candidate generator for unseen words: training templates + a generative score.

Every training analysis is decomposed (ssi.templates) into proclitics, a core
frame, enclitics and a lexeme; the frame is split further into its affixes and
its lexeme pattern. For a surface word, the generator tries every split of the
consonantal skeleton into proclitics + affixes + stem + affixes + enclitics,
and inverts each lexeme pattern of the frame's class on the stem. Every
candidate therefore realises the surface word exactly.

Each candidate gets a set of features (see `featurise`), most of them parts
of a generative score

    log P(frame) + log P(proclitics | class) + log P(enclitics | class)
      + log P(lexeme | class) + log P(diacritics | frame)

where class is the frame's class (verb / nominal / other) and P(frame)
interpolates the frame's own count with P(affixes) * P(lexeme pattern | class),
so affixes and lexeme patterns that were only seen apart can still combine.
Known lexemes get their relative frequency within the class; the remaining
mass, estimated from lexemes seen once, goes to new lexemes and is spread with
a character trigram model. Counts are over distinct (surface, analysis) pairs
by default, since unseen words behave like rare types rather than like
frequent tokens. Further features: how usual the stem formation is for a known
verb, homograph marks, and the analyses of training words with the same
consonantal skeleton but other diacritics.

`candidates` ranks by a weighted sum of the features. The default weights give
the plain generative score; ssi.rerank learns better ones.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

from .corpus import Verse
from .encoding import SURFACE_DIACRITICS, _plain, realise, strip_surface
from .templates import LexemePattern, decompose, frame_class, split_frame


def diacritic_signature(surface: str) -> str:
    return ''.join(sorted(set(c for c in surface if c in SURFACE_DIACRITICS)))


def _homograph_marks(lexeme: str) -> int:
    return len(lexeme) - len(lexeme.rstrip('='))


# Weights that reproduce the plain generative score; known analyses of the
# same skeleton get a strong preference.
DEFAULT_WEIGHTS = {'generative': 1.0, 'skel_in': 10.0, 'skel_lp': 1.0}


def verb_binyan(pre: str, post: str) -> tuple[str, bool, bool]:
    """Stem formation of a verbal frame: (verbal stem morpheme, passive prefix, D pattern)."""
    vbs = re.search(r'\]([^\]]*)\]', pre)
    vpm = re.search(r':([a-z]+)$', post)
    return (_plain(vbs.group(1)) if vbs else '', '@' in pre, bool(vpm and 'd' in vpm.group(1)))


# Syriac_paradigm.pdf section 2.6, Table 20: the taw of the passive prefix
# changes places with a following sibilant (and becomes D after Z, V after Y),
# as in '@>(T&C@(C&TKX[' and the eshtafal '@>(T&C@](C&T]<BD['.
_METATHESIS = {'C': 'T', 'S': 'T', 'Z': 'D', 'Y': 'V'}
_PASSIVE_T = re.compile(r'@(\(?>)T@(\]C\]|[CSZY])')


def unmetathesised(bare: str) -> list[str]:
    """Surface forms with one sibilant + dental pair put back in prefix order."""
    out = []
    for x, t in _METATHESIS.items():
        start = bare.find(x + t)
        while start != -1:
            out.append(bare[:start] + 'T' + x + bare[start + 2:])
            start = bare.find(x + t, start + 1)
    return out


def metathesise(analysis: str) -> str | None:
    """The metathesised encoding of an analysis of an unmetathesised form."""
    m = _PASSIVE_T.search(analysis)
    if m is None:
        return None
    alaf, nxt = m.groups()
    if nxt == ']C]':
        new = f'@{alaf}(T&C@](C&T]'
    else:
        new = f'@{alaf}(T&{nxt}@({nxt}&{_METATHESIS[nxt]}'
    return analysis[:m.start()] + new + analysis[m.end():]


class CharTrigram:
    """Add-k smoothed character trigram model over strings, with an end symbol."""

    def __init__(self, strings, k: float = 0.1):
        self.k = k
        self.tri, self.bi = Counter(), Counter()
        alphabet = {'$'}
        for s in strings:
            padded = '^^' + s + '$'
            alphabet.update(s)
            for i in range(2, len(padded)):
                self.tri[padded[i - 2:i + 1]] += 1
                self.bi[padded[i - 2:i]] += 1
        self.v = len(alphabet)

    def logp(self, s: str) -> float:
        padded = '^^' + s + '$'
        return sum(math.log((self.tri[padded[i - 2:i + 1]] + self.k)
                            / (self.bi[padded[i - 2:i]] + self.k * self.v))
                   for i in range(2, len(padded)))


class CandidateGenerator:
    """
    counts       'type' (distinct surface/analysis pairs) or 'token'
    beta         weight of the factored estimate P(affixes) P(pattern | class)
                 against the frame's own count
    sig_weight   pseudo-count of the class-level diacritics prior per frame
    min_unknown  floor on the probability mass for new lexemes per class
    """

    def __init__(self, counts: str = 'type', beta: float = 100.0, sig_weight: float = 5.0,
                 min_unknown: float = 0.01, weights: dict[str, float] | None = None):
        self.counts, self.beta, self.sig_weight = counts, beta, sig_weight
        self.min_unknown = min_unknown
        self.weights = dict(weights or DEFAULT_WEIGHTS)

    # -- training -----------------------------------------------------------
    def fit(self, train: list[Verse]) -> 'CandidateGenerator':
        pairs = Counter((s, a) for v in train for s, a in v.words())
        weight = (lambda c: 1) if self.counts == 'type' else (lambda c: c)

        frames, affixes, patterns = Counter(), Counter(), defaultdict(Counter)
        pro, enc, lex = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
        sig, sig_cls = defaultdict(Counter), defaultdict(Counter)
        skel, lexeme_binyan = defaultdict(Counter), defaultdict(Counter)
        for (s, a), c in pairs.items():
            d, w = decompose(a), weight(c)
            cls = frame_class(d.frame)
            pre, pat, post = split_frame(d.frame)
            frames[d.frame] += w
            affixes[pre, post, cls] += w
            patterns[cls][pat] += w
            pro[cls][d.proclitics] += w
            enc[cls][d.enclitics] += w
            lex[cls][d.lexeme] += w
            g = diacritic_signature(s)
            sig[d.frame][g] += w
            sig_cls[cls][g] += w
            skel[strip_surface(s)][a] += c
            if cls == 'verb':
                lexeme_binyan[d.lexeme][verb_binyan(pre, post)] += w

        self.n_frames = sum(frames.values())
        self.frame_counts = frames
        n_aff = sum(affixes.values())
        self.affix_lp = {k: math.log(c / n_aff) for k, c in affixes.items()}
        self.pattern_lp = {cls: self._dist(c) for cls, c in patterns.items()}
        self.patterns = {cls: [LexemePattern(p) for p in c] for cls, c in patterns.items()}
        self.affix_index = defaultdict(list)  # (pre_real, post_real) -> [(pre, post, cls)]
        for pre, post, cls in affixes:
            self.affix_index[realise(pre), realise(post)].append((pre, post, cls))
        self.pre_lens = sorted({len(k[0]) for k in self.affix_index})
        self.post_lens = sorted({len(k[1]) for k in self.affix_index})

        self.pro_lp = {cls: self._dist(c) for cls, c in pro.items()}
        self.enc_lp = {cls: self._dist(c) for cls, c in enc.items()}
        self.pro_by_real, self.enc_by_real = defaultdict(set), defaultdict(set)
        for c in pro.values():
            for chain in c:
                self.pro_by_real[realise(chain)].add(chain)
        for c in enc.values():
            for chain in c:
                self.enc_by_real[realise(chain)].add(chain)

        # Lexicon per class, with Good-Turing style mass for new lexemes.
        self.lex_lp, self.unk_lp, self.char_model = {}, {}, {}
        self.homographs = defaultdict(set)  # plain lexeme -> forms with homograph marks
        for cls, c in lex.items():
            n = sum(c.values())
            unk = min(max(sum(1 for v in c.values() if v == 1) / n, self.min_unknown), 0.5)
            self.lex_lp[cls] = {l: math.log((1 - unk) * v / n) for l, v in c.items()}
            self.unk_lp[cls] = math.log(unk)
            self.char_model[cls] = CharTrigram([l.rstrip('=') for l in c])
            for l in c:
                self.homographs[l.rstrip('=')].add(l)

        sig_space = {g for c in sig_cls.values() for g in c}
        self.n_sigs = len(sig_space)
        self.sig_cls_p = {cls: {g: (c[g] + 1) / (sum(c.values()) + self.n_sigs) for g in sig_space}
                          for cls, c in sig_cls.items()}
        self.sig_counts = sig
        binyans = Counter()
        for c in lexeme_binyan.values():
            binyans.update(c)
        self.binyan_p = {b: n / sum(binyans.values()) for b, n in binyans.items()}
        self.lexeme_binyan = lexeme_binyan
        self.skeleton_p = {s: {a: n / sum(c.values()) for a, n in c.items()} for s, c in skel.items()}
        return self

    @staticmethod
    def _dist(counter: Counter) -> dict[str, float]:
        n = sum(counter.values())
        return {k: math.log(v / n) for k, v in counter.items()}

    def _frame_lp(self, pre: str, pat: str, post: str, cls: str) -> float:
        factored = math.exp(self.affix_lp[pre, post, cls] + self.pattern_lp[cls][pat])
        own = self.frame_counts.get(f'{pre}{{{pat}}}{post}', 0)
        return math.log((own + self.beta * factored) / (self.n_frames + self.beta))

    def _sig_lp(self, frame: str, cls: str, g: str) -> float:
        prior = self.sig_cls_p[cls].get(g, 1 / (self.n_sigs + 1))
        c = self.sig_counts.get(frame)
        n, k = (sum(c.values()), c[g]) if c else (0, 0)
        return math.log((k + self.sig_weight * prior) / (n + self.sig_weight))

    def _lexeme_options(self, cls: str, plain: str):
        """(homograph marks, is new, log P(lexeme | class)) for each reading of
        `plain`: the known lexemes of the class, and a new lexeme."""
        known = self.lex_lp.get(cls, {})
        forms = self.homographs.get(plain, ())
        out = [(_homograph_marks(l), False, known[l]) for l in forms if l in known]
        used = {_homograph_marks(l) for l in forms}
        new = 0 if 0 not in used else min(set(range(len(used) + 1)) - used)
        out.append((new, True, self.unk_lp[cls] + self.char_model[cls].logp(plain)))
        return out

    def _binyan_lp(self, lexeme: str, binyan: tuple) -> float:
        """log P(stem formation | verb lexeme) / P(stem formation | verbs)."""
        prior = self.binyan_p.get(binyan, 1e-4)
        c = self.lexeme_binyan.get(lexeme)
        if not c:
            return 0.0
        n = sum(c.values())
        return math.log((c[binyan] + self.sig_weight * prior) / (n + self.sig_weight) / prior)

    # -- generation ---------------------------------------------------------
    def candidates(self, surface: str, k: int | None = None) -> list[tuple[str, float]]:
        """Candidate analyses with their log posterior under `self.weights`, best first."""
        feats = self.featurise(surface)
        if not feats:
            return []
        w = self.weights
        scores = {a: sum(w.get(name, 0.0) * v for name, v in f.items()) for a, f in feats.items()}
        z = _logsumexp(scores.values())
        ranked = sorted(((a, s - z) for a, s in scores.items()), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:k] if k else ranked

    def featurise(self, surface: str) -> dict[str, dict[str, float]]:
        """Every candidate analysis of `surface` with its feature values.

        When an analysis arises from several decompositions, the features of
        the one with the highest generative score are kept."""
        bare, g = strip_surface(surface), diacritic_signature(surface)
        feats = self._featurise_bare(bare, g)
        for variant in unmetathesised(bare):
            for a, f in self._featurise_bare(variant, g).items():
                a2 = metathesise(a)
                if a2 and a2 not in feats and realise(a2) == bare:
                    feats[a2] = dict(f, metathesis=1.0)
        same_skeleton = self.skeleton_p.get(bare)
        if same_skeleton:
            for a, f in feats.items():
                f['skel_other'] = 1.0
            for a, p in same_skeleton.items():
                f = feats.setdefault(a, {'generative': -100.0})
                f['skel_other'] = 0.0
                f['skel_in'] = 1.0
                f['skel_lp'] = math.log(p)
        return feats

    def _featurise_bare(self, bare: str, g: str) -> dict[str, dict[str, float]]:
        best: dict[str, tuple[float, dict]] = {}
        n = len(bare)
        for i in range(n + 1):
            pros = self.pro_by_real.get(bare[:i])
            if not pros:
                continue
            for j in range(n - i + 1):
                encs = self.enc_by_real.get(bare[n - j:] if j else '')
                if not encs:
                    continue
                for pre, pat, post, cls, morph in self._match_core(bare[i:n - j]):
                    pro_lp, enc_lp = self.pro_lp.get(cls, {}), self.enc_lp.get(cls, {})
                    pe = [(p, e, pro_lp[p], enc_lp[e]) for p in pros if p in pro_lp
                          for e in encs if e in enc_lp]
                    if not pe:
                        continue
                    frame = f'{pre}{{{pat}}}{post}'
                    f_frame = self._frame_lp(pre, pat, post, cls)
                    f_sig = self._sig_lp(frame, cls, g)
                    frame_seen = float(frame in self.frame_counts)
                    binyan = verb_binyan(pre, post) if cls == 'verb' else None
                    plain = _plain(morph)
                    for marks, is_new, lex_lp in self._lexeme_options(cls, plain):
                        lexeme = plain + '=' * marks
                        core = f'{pre}{morph}{"=" * marks}{post}'
                        bin_lp = self._binyan_lp(lexeme, binyan) if binyan and not is_new else 0.0
                        for p, e, lp_p, lp_e in pe:
                            gen = f_frame + f_sig + lp_p + lp_e + lex_lp
                            analysis = '-'.join(x for x in (p, core, e) if x)
                            if analysis in best and best[analysis][0] >= gen:
                                continue
                            best[analysis] = (gen, {
                                'generative': gen, 'frame': f_frame, 'sig': f_sig, 'pro': lp_p,
                                'enc': lp_e, 'lex_known': 0.0 if is_new else lex_lp,
                                'lex_new': lex_lp if is_new else 0.0, 'is_new': float(is_new),
                                'marks': float(marks), 'binyan': bin_lp, 'frame_seen': frame_seen,
                                'n_words': float(analysis.count('-') + 1),
                                'lex_len': float(len(plain)),
                            })
        return {a: f for a, (_, f) in best.items()}

    def _match_core(self, core: str):
        m = len(core)
        for a in self.pre_lens:
            if a > m:
                break
            for b in self.post_lens:
                if a + b > m:
                    break
                affs = self.affix_index.get((core[:a], core[m - b:] if b else ''))
                if not affs:
                    continue
                stem = core[a:m - b]
                for pre, post, cls in affs:
                    for pattern in self.patterns[cls]:
                        morph = pattern.invert(stem)
                        if morph is not None:
                            yield pre, pattern.text, post, cls, morph


def _logsumexp(values) -> float:
    values = list(values)
    hi = max(values)
    return hi + math.log(sum(math.exp(v - hi) for v in values))
