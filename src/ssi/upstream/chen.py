"""Adapter that trains and runs chenzhan4321/etcbc_models_comparison models on our splits.

Upstream `train.py` is run unchanged as a subprocess. This module writes its
data directory (train/val/test .in/.out) from our split and turns its
per-character label predictions back into encodings.

Data, following upstream: every line is a 7-word window with a leading space;
train and val lines are rolling windows (stride 1) over the train and dev
verses of each book; test lines are non-overlapping windows covering the test
verses (see ssi.windows). The label inventory is built from the training
lines only (upstream builds it from all files, including test).

Presets are the configurations of the paper's archived runs (see
third_party/README.md for where they come from).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from ..corpus import REPO_ROOT, Verse
from ..encoding import strip_surface
from ..windows import assemble, cover_with_windows
from .chen_labels import PatternInventory, analyses_from_line, line_input, line_patterns

UPSTREAM = REPO_ROOT / 'third_party' / 'etcbc_models_comparison'
LINE_WORDS = 7

PRESETS = {
    # outputs/encoder_only_100ep_bs128_lr3e-4_d512_l4_h16_d0.25_5seeds (seed 42)
    'chen-transformer': ['--model_type', 'transformer', '--d_model', '512', '--num_layers', '4',
                         '--num_heads', '16', '--dropout', '0.25', '--learning_rate', '3e-4',
                         '--batch_size', '128', '--num_epochs', '100', '--weight_decay', '0.03',
                         '--max_length', '64', '--patience', '50'],
    # README headline run: mdlm d768 l10 h6, 3 diffusion steps
    'chen-mdlm': ['--model_type', 'mdlm', '--d_model', '768', '--num_layers', '10',
                  '--num_heads', '6', '--dropout', '0.23', '--learning_rate', '5e-5',
                  '--batch_size', '16', '--num_epochs', '250', '--weight_decay', '0.01',
                  '--num_timesteps', '3', '--max_length', '64', '--patience', '50'],
}


def rolling_lines(verses: list[Verse]) -> list[tuple[list[str], list[str]]]:
    by_book: dict[str, list[Verse]] = {}
    for v in verses:
        by_book.setdefault(v.book, []).append(v)
    lines = []
    for bv in by_book.values():
        surf = [s for v in bv for s in v.surface]
        anal = [a for v in bv for a in v.analysis]
        for i in range(max(0, len(surf) - LINE_WORDS) + 1):
            if surf[i:i + LINE_WORDS]:
                lines.append((surf[i:i + LINE_WORDS], anal[i:i + LINE_WORDS]))
    return lines


def write_data_dir(directory: Path, corpus: list[Verse], train: list[Verse], dev: list[Verse],
                   targets: list[Verse]) -> tuple[PatternInventory, list]:
    """Write train/val/test files. Returns the inventory and the test windows."""
    directory.mkdir(parents=True, exist_ok=True)
    train_lines, val_lines = rolling_lines(train), rolling_lines(dev)
    inventory = PatternInventory.from_lines(train_lines)

    gold = {v.vid: v for v in corpus}
    windows = cover_with_windows(corpus, targets, LINE_WORDS)
    # Upstream needs aligned gold labels for the test file; they only define
    # which positions are predicted and are never used for scoring here.
    test_lines = [(w.words, [gold[vid].analysis[i] for vid, i in w.ids]) for w in windows]

    for name, lines in (('train', train_lines), ('val', val_lines), ('test', test_lines)):
        with open(directory / f'{name}.in', 'w', encoding='utf8', newline='\n') as fi, \
             open(directory / f'{name}.out', 'w', encoding='utf8', newline='\n') as fo:
            for surf, anal in lines:
                labels = inventory.encode(line_patterns(surf, anal))
                fi.write(line_input(surf) + '\n')
                fo.write(' '.join(map(str, labels)) + '\n')
    with open(directory / 'patterns.csv', 'w', encoding='utf8', newline='\n') as f:
        f.write('label,pattern,count\n')
        for i, p in enumerate(inventory.patterns):
            f.write(f'{i},"{p}",{inventory.counts[p]}\n')
    (directory / 'inventory.json').write_text(json.dumps(inventory.patterns), encoding='utf8')
    print(f'[chen] {len(train_lines)} train / {len(val_lines)} val / {len(test_lines)} test lines, '
          f'{len(inventory.patterns)} labels', flush=True)
    return inventory, windows


def train_and_predict(preset: str, run_dir: Path, corpus: list[Verse], train: list[Verse],
                      dev: list[Verse], targets: list[Verse], epochs: int | None = None,
                      seed: int = 42) -> dict[str, list[str]]:
    data_dir = run_dir / 'data'
    inventory, windows = write_data_dir(data_dir, corpus, train, dev, targets)

    args = list(PRESETS[preset])
    if epochs is not None:
        args[args.index('--num_epochs') + 1] = str(epochs)
    cmd = [sys.executable, '-u', str(UPSTREAM / 'train.py'), '--data_dir', str(data_dir),
           '--seed', str(seed), '--force_device', 'cuda'] + args
    print('[chen] running:', ' '.join(cmd), flush=True)
    # Upstream prints emoji; force UTF-8 so redirected output works on Windows.
    env = os.environ | {'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8'}
    subprocess.run(cmd, cwd=run_dir, check=True, env=env)

    pred_files = sorted(run_dir.glob('outputs/*_train_*/results/*_predictions_*.out'))
    if not pred_files:
        raise FileNotFoundError(f'no predictions file under {run_dir / "outputs"}')
    return read_upstream_predictions(pred_files[-1], inventory, windows, targets)


def read_upstream_predictions(path: Path, inventory: PatternInventory, windows,
                              targets: list[Verse]) -> dict[str, list[str]]:
    lines = path.read_text(encoding='utf8').splitlines()
    if len(lines) != len(windows):
        raise ValueError(f'{path}: {len(lines)} prediction lines for {len(windows)} test lines; '
                         'upstream dropped lines, alignment is lost')
    window_preds = []
    for w, line in zip(windows, lines):
        labels = [int(x) for x in line.split()]
        window_preds.append(analyses_from_line(w.words, inventory.decode(labels)))
    return assemble(targets, windows, window_preds)


def load_inventory(run_dir: Path) -> PatternInventory:
    from collections import Counter
    patterns = json.loads((run_dir / 'data' / 'inventory.json').read_text(encoding='utf8'))
    inv = PatternInventory(Counter())
    inv.patterns = patterns
    inv.index = {p: i for i, p in enumerate(patterns)}
    return inv
