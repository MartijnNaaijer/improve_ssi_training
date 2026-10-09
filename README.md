# Syriac morphological parsing (ETCBC encoding)

Work towards a parser for Classical Syriac that improves on
[ETCBC/ssi_morphology](https://github.com/ETCBC/ssi_morphology) and
[chenzhan4321/etcbc_models_comparison](https://github.com/chenzhan4321/etcbc_models_comparison),
using a hybrid approach: generate candidate analyses (lookup, rule-based
generator, neural n-best), filter them with hard constraints, and rank them in
context. See `CLAUDE.md` for the encoding and project conventions.

## Setup

```
pip install -e .[dev]
python -m pytest
```

## Usage

```
python scripts/make_splits.py                 # writes data/splits/*.json
python scripts/evaluate.py --system lookup    # writes results/<split>/lookup.json and results/summary.md
```

## Status

- Phase 0 (test bench): corpus loader, encoding parser and realisation check,
  splits, metrics, lookup baseline. Results: `results/summary.md`.
