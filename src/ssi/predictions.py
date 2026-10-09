"""Reading and writing saved predictions.

Format: a JSON metadata line starting with '#', then one line per verse:
verse id, a tab, and the predicted analyses separated by single spaces.
An empty analysis (missing prediction) is written as '<none>'.
"""
from __future__ import annotations

import json
from pathlib import Path

NONE = '<none>'


def write_predictions(path: Path, predictions: dict[str, list[str]], meta: dict | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf8', newline='\n') as f:
        f.write('#' + json.dumps(meta or {}) + '\n')
        for vid, analyses in predictions.items():
            f.write(vid + '\t' + ' '.join(a if a else NONE for a in analyses) + '\n')


def read_predictions(path: Path) -> tuple[dict[str, list[str]], dict]:
    preds, meta = {}, {}
    with open(path, encoding='utf8') as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('#'):
                meta = json.loads(line[1:])
                continue
            vid, _, text = line.partition('\t')
            preds[vid] = ['' if a == NONE else a for a in text.split(' ')] if text else []
    return preds, meta
