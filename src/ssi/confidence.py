"""Calibrated confidence for the ranker's choices (Phase 4).

The ranker gives a softmax over each word's candidates. Its top probability
is already a good confidence score, but it is somewhat overconfident. Two
calibrators are fitted on dev:

    temperature  p_i ** (1/T), renormalised; T minimises the NLL of the gold
                 candidate
    logistic     logistic regression of "top candidate is correct" on the
                 softmax shape (top log probability, margin, entropy, number
                 of candidates) and the surface category

plus metrics to compare them (ECE, Brier score, AUROC for telling right from
wrong) and the triage curve: if annotators check the least confident share
of words, how many errors do they catch, and how accurate is the rest?
"""
from __future__ import annotations

import math

import numpy as np

CONF_FEATURES = ('log_p1', 'margin', 'entropy', 'log_n', 'unseen', 'ambiguous', 'no_cands')


def word_features(probs: np.ndarray, category: str) -> list[float]:
    if len(probs) == 0:
        return [math.log(1e-6), 0.0, 0.0, 0.0, float(category == 'unseen'), float(category == 'ambiguous'), 1.0]
    p = np.sort(probs)[::-1]
    p1 = float(p[0])
    p2 = float(p[1]) if len(p) > 1 else 0.0
    ent = float(-(probs * np.log(np.clip(probs, 1e-12, 1))).sum())
    return [math.log(max(p1, 1e-6)), p1 - p2, ent, math.log(len(p)),
            float(category == 'unseen'), float(category == 'ambiguous'), 0.0]


def scale(probs: np.ndarray, t: float) -> np.ndarray:
    if len(probs) == 0:
        return probs
    logp = np.log(np.clip(probs, 1e-12, 1)) / t
    logp -= logp.max()
    e = np.exp(logp)
    return e / e.sum()


def fit_temperature(probs: list[np.ndarray], gold: list[int]) -> float:
    """Temperature minimising the NLL of the gold candidate (words without it are skipped)."""
    pairs = [(np.log(np.clip(p, 1e-12, 1)), g) for p, g in zip(probs, gold) if g >= 0 and len(p) > 1]

    def nll(t):
        total = 0.0
        for lp, g in pairs:
            z = lp / t
            hi = z.max()
            total += hi + math.log(np.exp(z - hi).sum()) - z[g]
        return total / max(len(pairs), 1)

    grid = np.exp(np.linspace(math.log(0.5), math.log(5.0), 61))
    best = min(grid, key=nll)
    fine = np.exp(np.linspace(math.log(best) - 0.04, math.log(best) + 0.04, 21))
    return float(min(fine, key=nll))


class LogisticConfidence:
    """P(top candidate correct | word features), L2-regularised logistic regression."""

    def __init__(self, l2: float = 1e-3, steps: int = 50):
        self.l2, self.steps = l2, steps

    def fit(self, x: np.ndarray, y: np.ndarray) -> 'LogisticConfidence':
        self.mu, self.sd = x.mean(0), x.std(0) + 1e-6
        z = np.hstack([(x - self.mu) / self.sd, np.ones((len(x), 1))])
        w = np.zeros(z.shape[1])
        reg = self.l2 * len(x) * np.eye(z.shape[1])
        reg[-1, -1] = 0.0
        for _ in range(self.steps):  # Newton's method
            p = 1 / (1 + np.exp(-z @ w))
            grad = z.T @ (p - y) + reg @ w
            hess = (z * (p * (1 - p))[:, None]).T @ z + reg
            step = np.linalg.solve(hess, grad)
            w -= step
            if np.abs(step).max() < 1e-8:
                break
        self.w = w
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = np.hstack([(x - self.mu) / self.sd, np.ones((len(x), 1))])
        return 1 / (1 + np.exp(-z @ self.w))


# -- metrics ------------------------------------------------------------------
def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error with equal-width bins."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.sum() * abs(conf[m].mean() - correct[m].mean())
    return float(total / len(conf))


def brier(conf: np.ndarray, correct: np.ndarray) -> float:
    return float(((conf - correct) ** 2).mean())


def auroc(conf: np.ndarray, correct: np.ndarray) -> float:
    """Probability that a random correct word gets higher confidence than a random wrong one."""
    order = np.argsort(conf, kind='stable')
    ranks = np.empty(len(conf))
    ranks[order] = np.arange(1, len(conf) + 1)
    # Average ranks over ties.
    sorted_conf = conf[order]
    i = 0
    while i < len(conf):
        j = i
        while j + 1 < len(conf) and sorted_conf[j + 1] == sorted_conf[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2
        i = j + 1
    pos = correct.astype(bool)
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def triage_curve(conf: np.ndarray, correct: np.ndarray,
                 budgets=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)) -> list[dict]:
    """For each share of words checked (least confident first): errors caught and
    the accuracy of the unchecked words."""
    order = np.argsort(conf, kind='stable')
    wrong = ~correct.astype(bool)
    out = []
    for b in budgets:
        k = int(round(b * len(conf)))
        rest = order[k:]
        out.append({'checked': b, 'errors_caught': float(wrong[order[:k]].sum() / max(wrong.sum(), 1)),
                    'unchecked_acc': float(1 - wrong[rest].mean()) if len(rest) else 1.0})
    return out


def budget_for_accuracy(conf: np.ndarray, correct: np.ndarray, target: float) -> float:
    """Smallest share of words to check (least confident first) so that the
    unchecked words reach `target` accuracy."""
    order = np.argsort(conf, kind='stable')
    wrong = (~correct.astype(bool))[order]
    # Errors left among words k.. for every k.
    left = np.concatenate([np.cumsum(wrong[::-1])[::-1], [0]])
    n = len(conf)
    for k in range(n + 1):
        if n - k == 0 or 1 - left[k] / (n - k) >= target:
            return k / n
    return 1.0
