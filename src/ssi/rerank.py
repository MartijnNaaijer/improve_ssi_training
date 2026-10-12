"""Learned weights for the candidate generator's features, and the generator system.

The weights are trained on *pseudo-unseen* words: the training verses are
split into folds, a generator is fitted on all folds but one, and the words of
the held-out fold whose surface form the other folds lack serve as examples.
That mimics the situation of a truly unseen word at test time. The model is a
softmax over the candidates of a word (a conditional logit), fitted with Adam
and a small L2 penalty.

`GeneratorSystem` answers words seen in training by lookup and all other
words with the generator's best candidate.
"""
from __future__ import annotations

import random
from concurrent.futures import ProcessPoolExecutor

from .corpus import Verse
from .generator import DEFAULT_WEIGHTS, CandidateGenerator
from .lookup import LookupBaseline

FEATURES = ('generative', 'frame', 'sig', 'pro', 'enc', 'lex_known', 'lex_new', 'is_new', 'marks',
            'binyan', 'frame_seen', 'n_words', 'lex_len', 'skel_in', 'skel_lp', 'skel_other', 'metathesis')


def _fold_examples(args):
    rest, held, max_candidates, gen_kwargs = args
    gen = CandidateGenerator(**gen_kwargs).fit(rest)
    seen = {s for v in rest for s in v.surface}
    examples, missed = [], 0
    for v in held:
        for s, a in v.words():
            if s in seen:
                continue
            feats = gen.featurise(s)
            if a not in feats:
                missed += 1
                continue
            ranked = sorted(feats, key=lambda x: (-feats[x].get('generative', -100.0), x))
            keep = ranked[:max_candidates]
            keep += [x for x in ranked[max_candidates:] if feats[x].get('skel_in') or x == a]
            examples.append(([[feats[x].get(f, 0.0) for f in FEATURES] for x in keep], keep.index(a)))
    return examples, missed


def cross_fit_examples(train: list[Verse], folds: int = 5, max_candidates: int = 100, seed: int = 42,
                       processes: int | None = None, gen_kwargs: dict | None = None):
    """(feature rows per candidate, index of the gold candidate) for pseudo-unseen words."""
    order = list(range(len(train)))
    random.Random(seed).shuffle(order)
    fold_of = {i: k % folds for k, i in enumerate(order)}
    jobs = []
    for k in range(folds):
        rest = [v for i, v in enumerate(train) if fold_of[i] != k]
        held = [v for i, v in enumerate(train) if fold_of[i] == k]
        jobs.append((rest, held, max_candidates, gen_kwargs or {}))
    examples, missed = [], 0
    with ProcessPoolExecutor(max_workers=processes or folds) as pool:
        for ex, m in pool.map(_fold_examples, jobs):
            examples += ex
            missed += m
    return examples, missed


def learn_weights(examples, l2: float = 1e-3, steps: int = 600, lr: float = 0.05, seed: int = 42) -> dict[str, float]:
    import numpy as np
    n, c_max, f = len(examples), max(len(rows) for rows, _ in examples), len(FEATURES)
    x = np.zeros((n, c_max, f), dtype=np.float32)
    mask = np.zeros((n, c_max), dtype=bool)
    gold = np.zeros(n, dtype=np.int64)
    for i, (rows, g) in enumerate(examples):
        x[i, :len(rows)] = rows
        mask[i, :len(rows)] = True
        gold[i] = g
    # Standardise features over all real candidates so one learning rate fits all.
    mu = x[mask].mean(0)
    sd = x[mask].std(0) + 1e-6
    xs = np.where(mask[..., None], (x - mu) / sd, 0.0).astype(np.float32)

    w = np.zeros(f, dtype=np.float64)
    w[FEATURES.index('generative')] = sd[FEATURES.index('generative')]  # start at the generative score
    m, v = np.zeros_like(w), np.zeros_like(w)
    rows = np.arange(n)
    for t in range(1, steps + 1):
        s = xs @ w
        s = np.where(mask, s, -np.inf)
        s -= s.max(1, keepdims=True)
        p = np.exp(s)
        p /= p.sum(1, keepdims=True)
        expected = np.einsum('nc,ncf->nf', p, xs)
        grad = -(xs[rows, gold] - expected).mean(0) + l2 * w
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad ** 2
        w -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    # Back to raw feature scale; the constant from the mean shift cancels in the softmax.
    raw = w / sd
    return {name: float(raw[i]) for i, name in enumerate(FEATURES)}


class GeneratorSystem:
    name = 'generator'

    def __init__(self, rerank: bool = True, folds: int = 5, verbose: bool = True, **gen_kwargs):
        self.rerank, self.folds, self.verbose, self.gen_kwargs = rerank, folds, verbose, gen_kwargs

    def fit(self, train: list[Verse]) -> 'GeneratorSystem':
        self.lookup = LookupBaseline().fit(train)
        self.generator = CandidateGenerator(**self.gen_kwargs).fit(train)
        if self.rerank:
            examples, missed = cross_fit_examples(train, folds=self.folds, gen_kwargs=self.gen_kwargs)
            self.generator.weights = learn_weights(examples)
            if self.verbose:
                print(f'[generator] {len(examples)} pseudo-unseen examples ({missed} without gold candidate)')
                print('[generator] weights:', {k: round(v, 3) for k, v in self.generator.weights.items()})
        else:
            self.generator.weights = dict(DEFAULT_WEIGHTS)
        return self

    def predict_word(self, surface: str) -> str:
        if surface in self.lookup.exact:
            return self.lookup.exact[surface]
        best = self.generator.candidates(surface, 1)
        return best[0][0] if best else self.lookup.predict_word(surface)

    def predict(self, verses: list[Verse]) -> dict[str, list[str]]:
        return {v.vid: [self.predict_word(s) for s in v.surface] for v in verses}
