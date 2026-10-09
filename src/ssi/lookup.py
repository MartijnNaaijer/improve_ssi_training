"""Most-frequent-analysis lookup baseline.

Each surface word gets the analysis it most often has in the training data.
Lookup is tried on the exact surface form (diacritics included, since e.g.
seyame is informative), then on the consonantal skeleton. Unseen words get the
skeleton itself as an unanalysed lexeme, which is at least consistent with the
surface form.
"""
from __future__ import annotations

from collections import Counter, defaultdict

from .corpus import Verse
from .encoding import strip_surface


class LookupBaseline:
    name = 'lookup'

    def fit(self, train: list[Verse]) -> 'LookupBaseline':
        exact, skeleton = defaultdict(Counter), defaultdict(Counter)
        for v in train:
            for s, a in v.words():
                exact[s][a] += 1
                skeleton[strip_surface(s)][a] += 1
        # Ties are broken alphabetically so results do not depend on corpus order.
        self.exact = {s: _best(c) for s, c in exact.items()}
        self.skeleton = {s: _best(c) for s, c in skeleton.items()}
        return self

    def predict_word(self, surface: str) -> str:
        if surface in self.exact:
            return self.exact[surface]
        bare = strip_surface(surface)
        return self.skeleton.get(bare, bare)

    def predict(self, verses: list[Verse]) -> dict[str, list[str]]:
        return {v.vid: [self.predict_word(s) for s in v.surface] for v in verses}


def _best(counter: Counter) -> str:
    return min(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0]
