"""Build and cache the candidate tables for the contextual ranker.

    python scripts/prepare_candidates.py --split random

Learns the Phase 1 feature weights on the training part, builds cross-fitted
candidate tables for the training verses and tables for dev and test (from
lookup and generator fitted on all training verses), and writes them to
models/ranker/<split>/candidates.pkl. Prints the oracle recall (gold among the
candidates) per category.
"""
import argparse
import json
import pickle
import time

from ssi.candidates import FEATURES, build_table, cross_fit_tables
from ssi.corpus import REPO_ROOT, load_corpus
from ssi.metrics import categorise, training_analyses
from ssi.rerank import cross_fit_examples, learn_weights
from ssi.splits import Split


def oracle(verses, table, train):
    seen = training_analyses(train)
    hit, n = {}, {}
    for v in verses:
        for s, a in v.words():
            c = categorise(s, seen)
            n[c] = n.get(c, 0) + 1
            hit[c] = hit.get(c, 0) + any(x == a for x, _ in table.get(s, ()))
    return {c: round(hit[c] / n[c], 4) for c in n} | {'all': round(sum(hit.values()) / sum(n.values()), 4)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--split', required=True)
    ap.add_argument('--max-candidates', type=int, default=24)
    ap.add_argument('--folds', type=int, default=5)
    args = ap.parse_args()

    split = Split.load(args.split)
    train, dev, test = split.partition(load_corpus(split.version))
    out_dir = REPO_ROOT / 'models' / 'ranker' / split.name
    out_dir.mkdir(parents=True, exist_ok=True)

    t = time.time()
    examples, missed = cross_fit_examples(train, folds=args.folds)
    weights = learn_weights(examples)
    (out_dir / 'phase1_weights.json').write_text(json.dumps(weights, indent=1), encoding='utf8')
    print(f'phase 1 weights from {len(examples)} examples in {time.time() - t:.0f}s', flush=True)

    t = time.time()
    fold_of, train_tables = cross_fit_tables(train, weights, folds=args.folds, max_candidates=args.max_candidates)
    print(f'cross-fitted train tables in {time.time() - t:.0f}s', flush=True)
    t = time.time()
    eval_table = build_table(train, {s for v in dev + test for s in v.surface}, weights,
                             max_candidates=args.max_candidates)
    print(f'dev/test table in {time.time() - t:.0f}s', flush=True)

    with open(out_dir / 'candidates.pkl', 'wb') as f:
        pickle.dump({'features': FEATURES, 'fold_of': fold_of, 'train_tables': train_tables,
                     'eval_table': eval_table, 'max_candidates': args.max_candidates}, f)
    for name, part in (('dev', dev), ('test', test)):
        print(f'oracle recall {name}:', oracle(part, eval_table, train))
    # Cross-fitted train recall, against the other folds' training data.
    hits = n = 0
    for v in train:
        table = train_tables[fold_of[v.vid]]
        for s, a in v.words():
            n += 1
            hits += any(x == a for x, _ in table[s])
    print(f'oracle recall train (cross-fitted): {hits / n:.4f}')


if __name__ == '__main__':
    main()
