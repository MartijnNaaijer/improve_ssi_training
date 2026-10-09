"""Fixed-length word windows for systems that work on n-word input sequences.

At prediction time every test word must get exactly one analysis. The test
words of a book are covered by non-overlapping windows over the book's full
surface text (in corpus order). Words outside the test set may appear in a
window as context: their surface form is input, never their analysis.
"""
from __future__ import annotations

from dataclasses import dataclass

from .corpus import Verse


@dataclass
class Window:
    words: list[str]                      # surface words of the window
    targets: list[tuple[str, int, int]]   # (verse id, word index in verse, offset in window)
    ids: list[tuple[str, int]]            # (verse id, word index) of every word in the window


def book_streams(verses: list[Verse]) -> dict[str, list[tuple[str, int, str]]]:
    """Per book, the sequence of (verse id, word index, surface word) in corpus order."""
    streams: dict[str, list[tuple[str, int, str]]] = {}
    for v in verses:
        stream = streams.setdefault(v.book, [])
        for i, s in enumerate(v.surface):
            stream.append((v.vid, i, s))
    return streams


def cover_with_windows(corpus: list[Verse], targets: list[Verse], length: int) -> list[Window]:
    """Non-overlapping windows of `length` words that cover every word of `targets`.

    A window starts at the first uncovered target word and is shifted left if it
    would run past the end of the book. Books shorter than `length` words give
    one shorter window.
    """
    target_ids = {v.vid for v in targets}
    windows = []
    for stream in book_streams(corpus).values():
        n = len(stream)
        covered_until = 0
        for pos, (vid, _, _) in enumerate(stream):
            if vid not in target_ids or pos < covered_until:
                continue
            start = max(0, min(pos, n - length))
            end = min(start + length, n)
            words = [w for _, _, w in stream[start:end]]
            tgts = [(stream[p][0], stream[p][1], p - start)
                    for p in range(max(start, covered_until), end) if stream[p][0] in target_ids]
            ids = [(stream[p][0], stream[p][1]) for p in range(start, end)]
            windows.append(Window(words, tgts, ids))
            covered_until = end
    return windows


def assemble(targets: list[Verse], windows: list[Window],
             window_predictions: list[list[str]], missing: str = '') -> dict[str, list[str]]:
    """Turn per-window predicted word lists into per-verse analyses.

    A window prediction with too few words leaves the missing positions as
    `missing`, which counts as wrong.
    """
    out = {v.vid: [missing] * len(v.surface) for v in targets}
    for win, pred in zip(windows, window_predictions):
        for vid, idx, offset in win.targets:
            out[vid][idx] = pred[offset] if offset < len(pred) else missing
    return out

