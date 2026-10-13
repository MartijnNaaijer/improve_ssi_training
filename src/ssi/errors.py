"""Classify how two analyses of the same surface word differ."""
from __future__ import annotations

from .encoding import paradigmatic_items, surface_segmentation

STEM_SLOTS = {'pfm', 'pfx', 'vbs'}
ENDING_SLOTS = {'vbe', 'nme', 'emf'}


def _boundaries(encoding: str):
    """Letter positions with a morpheme boundary before them, and the marker
    string at every position (leading and trailing included)."""
    seg = surface_segmentation(encoding)
    marks, cur = [], ''
    for c in seg:
        if c.isupper() or c in '<>':
            marks.append(cur)
            cur = ''
        else:
            cur += c
    marks.append(cur)
    return {k for k, m in enumerate(marks[1:-1], 1) if m}, marks


def marker_diff(gold: str, pred: str) -> str:
    """The marker strings that differ, as 'gold>pred' per position."""
    g, p = _boundaries(gold)[1], _boundaries(pred)[1]
    return ' '.join(f'{a or "_"}>{b or "_"}' for a, b in zip(g, p) if a != b)


def error_type(gold: str, pred: str) -> str:
    """The coarsest level at which prediction and gold differ."""
    if surface_segmentation(gold) != surface_segmentation(pred):
        if gold.count('-') != pred.count('-'):
            return 'segm: clitics'
        gp, pp = _boundaries(gold), _boundaries(pred)
        if gp[0] != pp[0]:
            return 'segm: boundary position'
        return 'segm: same boundaries, other markers'
    gi, pi = paradigmatic_items(gold), paradigmatic_items(pred)
    glex = [v for _, s, v in gi if s == 'lex']
    plex = [v for _, s, v in pi if s == 'lex']
    if glex != plex:
        if [x.replace('=', '') for x in glex] == [x.replace('=', '') for x in plex]:
            return 'lexeme: homograph index'
        return 'lexeme: other lexeme'
    gs, ps = set(gi), set(pi)
    slots = {s for _, s, _ in gs ^ ps}
    if not slots:
        return 'letter operations only'
    if slots == {'vpm'}:
        return 'vowel pattern mark'
    if slots <= STEM_SLOTS | {'vpm'}:
        return 'stem formation'
    if slots <= ENDING_SLOTS:
        return 'endings (vbe/nme/emf)'
    return 'mixed morphemes'
