"""Deterministic train/dev/test splits at verse level.

Splits are stored as JSON lists of verse ids in data/splits/, so every system
is trained and evaluated on exactly the same partition.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from .corpus import REPO_ROOT, Verse

SPLITS_DIR = REPO_ROOT / 'data' / 'splits'

DEFAULT_SEED = 42
# Books held out one at a time: a mix of biblical and non-biblical texts of
# reasonable size. Parallel recensions (EpBarA/B, OrManA/B) are avoided,
# because holding out one would leave its twin in the training data.
HELDOUT_BOOKS = ('Sir', 'BLC', 'Eprx', 'ApBar', 'ESNJ', 'Ez')


@dataclass
class Split:
    name: str
    version: str
    seed: int
    train: list[str]
    dev: list[str]
    test: list[str]

    def save(self, directory: Path = SPLITS_DIR) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f'{self.name}.json'
        with open(path, 'w', encoding='utf8') as f:
            json.dump(self.__dict__, f, ensure_ascii=False, indent=0)
        return path

    @classmethod
    def load(cls, name: str, directory: Path = SPLITS_DIR) -> 'Split':
        with open(directory / f'{name}.json', encoding='utf8') as f:
            return cls(**json.load(f))

    def partition(self, verses: list[Verse]) -> tuple[list[Verse], list[Verse], list[Verse]]:
        by_id = {v.vid: v for v in verses}
        missing = [vid for vid in self.train + self.dev + self.test if vid not in by_id]
        if missing:
            raise ValueError(f'split {self.name}: {len(missing)} verse ids not in corpus, e.g. {missing[:3]}')
        return ([by_id[i] for i in self.train], [by_id[i] for i in self.dev],
                [by_id[i] for i in self.test])


def random_split(verses: list[Verse], version: str, seed: int = DEFAULT_SEED,
                 dev_frac: float = 0.1, test_frac: float = 0.1) -> Split:
    ids = [v.vid for v in verses]
    random.Random(seed).shuffle(ids)
    n_test, n_dev = round(test_frac * len(ids)), round(dev_frac * len(ids))
    return Split('random', version, seed, sorted(ids[n_test + n_dev:]),
                 sorted(ids[n_test:n_test + n_dev]), sorted(ids[:n_test]))


def heldout_book_split(verses: list[Verse], book: str, version: str,
                       seed: int = DEFAULT_SEED, dev_frac: float = 0.1) -> Split:
    test = [v.vid for v in verses if v.book == book]
    if not test:
        raise ValueError(f'book {book!r} not in corpus')
    rest = [v.vid for v in verses if v.book != book]
    random.Random(seed).shuffle(rest)
    n_dev = round(dev_frac * len(rest))
    return Split(f'heldout-{book}', version, seed, sorted(rest[n_dev:]),
                 sorted(rest[:n_dev]), sorted(test))


def cross_validation_splits(verses: list[Verse], version: str, folds: int = 5, seed: int = DEFAULT_SEED,
                            dev_frac: float = 0.05) -> list[Split]:
    """Splits cv<folds>-0 .. cv<folds>-<folds-1>: every verse is in exactly one
    test fold, so the folds' test predictions together cover the corpus (used
    to look for gold errors). Each split's dev part (`dev_frac` of all verses)
    comes from the other folds, for checkpoint selection and calibration."""
    ids = [v.vid for v in verses]
    random.Random(seed).shuffle(ids)
    n_dev = round(dev_frac * len(ids))
    out = []
    for k in range(folds):
        test = ids[k::folds]
        rest = [i for j, i in enumerate(ids) if j % folds != k]
        random.Random(seed + k).shuffle(rest)
        out.append(Split(f'cv{folds}-{k}', version, seed, sorted(rest[n_dev:]), sorted(rest[:n_dev]),
                         sorted(test)))
    return out
