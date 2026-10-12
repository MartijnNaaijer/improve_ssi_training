"""Candidate lists with features for every word, for the contextual ranker.

The candidates of a surface form are the analyses it has in the training data
(all of them, most frequent first) followed by the generator's best other
candidates under the Phase 1 weights, up to `max_candidates` in total. Each
candidate carries the generator features (ssi.rerank.FEATURES), lookup
features, and its Phase 1 log posterior.

For the ranker's own training data the training verses are cross-fitted:
the candidates of a word in fold k come from lookup and generator fitted on
the other folds, so seen, ambiguous and unseen words occur in about the same
proportions as at test time.
"""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

from .corpus import Verse
from .generator import CandidateGenerator
from .rerank import FEATURES as GEN_FEATURES

LOOKUP_FEATURES = ('lk_in', 'lk_lp', 'lk_count', 'lk_n', 'seen', 'p1_lp')
FEATURES = GEN_FEATURES + LOOKUP_FEATURES

# {surface: [(analysis, [feature values in FEATURES order]), ...]}
Table = dict[str, list[tuple[str, list[float]]]]


class CandidateBuilder:
    def __init__(self, weights: dict[str, float], max_candidates: int = 24, gen_kwargs: dict | None = None):
        self.weights, self.max_candidates, self.gen_kwargs = weights, max_candidates, gen_kwargs or {}

    def fit(self, train: list[Verse]) -> 'CandidateBuilder':
        self.generator = CandidateGenerator(**self.gen_kwargs, weights=self.weights).fit(train)
        self.exact = defaultdict(Counter)
        for v in train:
            for s, a in v.words():
                self.exact[s][a] += 1
        return self

    def build(self, surface: str) -> list[tuple[str, list[float]]]:
        feats = self.generator.featurise(surface)
        w = self.weights
        scores = {a: sum(w.get(k, 0.0) * x for k, x in f.items()) for a, f in feats.items()}
        z = _logsumexp(scores.values()) if scores else 0.0
        seen = self.exact.get(surface)
        n_seen = sum(seen.values()) if seen else 0

        order = [a for a, _ in seen.most_common()] if seen else []
        order = order[:self.max_candidates]
        for a in sorted(scores, key=lambda x: (-scores[x], x)):
            if len(order) >= self.max_candidates:
                break
            if a not in order:
                order.append(a)

        out = []
        for a in order:
            f = dict(feats.get(a, {'generative': -100.0}))
            c = seen[a] if seen else 0
            f.update(lk_in=float(c > 0), lk_lp=math.log(c / n_seen) if c else 0.0,
                     lk_count=math.log1p(c), lk_n=float(len(seen) if seen else 0), seen=float(n_seen > 0),
                     p1_lp=scores[a] - z if a in scores else -50.0)
            out.append((a, [f.get(k, 0.0) for k in FEATURES]))
        return out


def _build_chunk(args) -> Table:
    train, surfaces, weights, max_candidates, gen_kwargs = args
    builder = CandidateBuilder(weights, max_candidates, gen_kwargs).fit(train)
    return {s: builder.build(s) for s in surfaces}


def build_table(train: list[Verse], surfaces, weights: dict[str, float], max_candidates: int = 24,
                gen_kwargs: dict | None = None, processes: int = 16) -> Table:
    """Candidates for each surface form, from lookup and generator fitted on `train`."""
    surfaces = sorted(set(surfaces))
    chunks = [surfaces[i::processes] for i in range(processes)]
    table: Table = {}
    with ProcessPoolExecutor(max_workers=processes) as pool:
        for part in pool.map(_build_chunk, [(train, c, weights, max_candidates, gen_kwargs) for c in chunks]):
            table.update(part)
    return table


def cross_fit_tables(train: list[Verse], weights: dict[str, float], folds: int = 5, seed: int = 42,
                     max_candidates: int = 24, gen_kwargs: dict | None = None,
                     processes: int = 16) -> tuple[dict[str, int], list[Table]]:
    """Fold of every training verse id, and the candidate table of each fold."""
    order = list(range(len(train)))
    random.Random(seed).shuffle(order)
    fold_of = {train[i].vid: k % folds for k, i in enumerate(order)}
    jobs = []
    per = max(1, processes // folds)
    for k in range(folds):
        rest = [v for v in train if fold_of[v.vid] != k]
        surfaces = sorted({s for v in train if fold_of[v.vid] == k for s in v.surface})
        jobs += [(k, (rest, surfaces[i::per], weights, max_candidates, gen_kwargs)) for i in range(per)]
    tables: list[Table] = [{} for _ in range(folds)]
    with ProcessPoolExecutor(max_workers=processes) as pool:
        for (k, _), part in zip(jobs, pool.map(_build_chunk, [j for _, j in jobs])):
            tables[k].update(part)
    return fold_of, tables


def _logsumexp(values) -> float:
    values = list(values)
    hi = max(values)
    return hi + math.log(sum(math.exp(v - hi) for v in values))
