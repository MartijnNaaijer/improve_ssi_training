"""Evaluation of predicted analyses against gold, word by word.

The corpus is aligned one-to-one at the level of graphical words, so every
metric is computed per word and micro-averaged. Words are grouped into
categories relative to the training data:

    unseen     surface form does not occur in training
    ambiguous  surface form occurs in training with more than one analysis
    unambig    surface form occurs in training with exactly one analysis
"""
from __future__ import annotations

from collections import Counter, defaultdict

from .corpus import Verse
from .encoding import (SLOTS, is_consistent, lexemes, paradigmatic_items,
                       surface_segmentation)

CATEGORIES = ('all', 'unseen', 'ambiguous', 'unambig')


def training_analyses(train: list[Verse]) -> dict[str, Counter]:
    table = defaultdict(Counter)
    for v in train:
        for s, a in v.words():
            table[s][a] += 1
    return table


def categorise(surface: str, table: dict[str, Counter]) -> str:
    if surface not in table:
        return 'unseen'
    return 'ambiguous' if len(table[surface]) > 1 else 'unambig'


def edit_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


class _Tally:
    def __init__(self):
        self.n = 0
        self.word = self.lexeme = self.segmentation = self.consistent = 0
        self.edits = self.ref_chars = 0
        self.slot_tp, self.slot_pred, self.slot_gold = Counter(), Counter(), Counter()

    def add(self, surface: str, gold: str, pred: str):
        self.n += 1
        self.word += pred == gold
        self.consistent += is_consistent(surface, pred)
        self.edits += edit_distance(pred, gold)
        self.ref_chars += len(gold)
        try:
            self.lexeme += lexemes(pred) == lexemes(gold)
            self.segmentation += surface_segmentation(pred) == surface_segmentation(gold)
            p_items = Counter(paradigmatic_items(pred))
        except ValueError:  # unparseable prediction: counts as wrong everywhere
            p_items = Counter()
        g_items = Counter(paradigmatic_items(gold))
        for (_, slot, _), k in (p_items & g_items).items():
            self.slot_tp[slot] += k
        for (_, slot, _), k in p_items.items():
            self.slot_pred[slot] += k
        for (_, slot, _), k in g_items.items():
            self.slot_gold[slot] += k

    def summary(self) -> dict:
        if self.n == 0:
            return {'n': 0}
        slots = {}
        for s in SLOTS:
            p = self.slot_tp[s] / self.slot_pred[s] if self.slot_pred[s] else 0.0
            r = self.slot_tp[s] / self.slot_gold[s] if self.slot_gold[s] else 0.0
            f = 2 * p * r / (p + r) if p + r else 0.0
            slots[s] = {'p': p, 'r': r, 'f1': f, 'support': self.slot_gold[s]}
        return {
            'n': self.n,
            'word_acc': self.word / self.n,
            'lexeme_acc': self.lexeme / self.n,
            'segmentation_acc': self.segmentation / self.n,
            'consistency': self.consistent / self.n,
            'cer': self.edits / self.ref_chars,
            'slots': slots,
        }


def evaluate(test: list[Verse], predictions: dict[str, list[str]],
             train: list[Verse]) -> dict:
    """Score predictions (verse id -> list of analyses) on the test verses."""
    table = training_analyses(train)
    tallies = {c: _Tally() for c in CATEGORIES}
    for v in test:
        pred = predictions[v.vid]
        if len(pred) != len(v.analysis):
            raise ValueError(f'{v.vid}: {len(pred)} predictions for {len(v.analysis)} words')
        for s, g, p in zip(v.surface, v.analysis, pred):
            tallies['all'].add(s, g, p)
            tallies[categorise(s, table)].add(s, g, p)
    result = {c: t.summary() for c, t in tallies.items()}
    for c in CATEGORIES[1:]:
        result[c]['share'] = result[c]['n'] / result['all']['n'] if result['all']['n'] else 0.0
    return result
