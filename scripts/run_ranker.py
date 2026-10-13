"""Train the contextual ranker on one split and write its predictions.

    .venv/Scripts/python.exe scripts/run_ranker.py --split random
    .venv/Scripts/python.exe scripts/evaluate.py --system ranker --from-predictions --splits random [--part dev]

Needs models/ranker/<split>/candidates.pkl from scripts/prepare_candidates.py.
Writes results/<split>/<name>.pred.tsv and <name>-dev.pred.tsv.

--hebrew adds the Hebrew parallel text as evidence (ssi.hebrew; use e.g.
--name ranker-heb). The translation table for training verses is fitted on the
other folds of the candidate cross-fitting, for dev and test on all of train.
"""
import argparse
import pickle
import time
from concurrent.futures import ProcessPoolExecutor

from ssi.corpus import REPO_ROOT, load_corpus
from ssi.hebrew import HEB_MISSING, HebrewEvidence, add_hebrew, load_hebrew
from ssi.predictions import write_predictions
from ssi.ranker import ContextRanker, make_examples
from ssi.splits import Split


def _fit_evidence(args):
    train, hebrew = args
    return HebrewEvidence.fit(train, hebrew)


def hebrew_evidence(train, fold_of):
    """Evidence per fold (for training verses) and on all of train (for dev and test)."""
    hebrew = load_hebrew()
    folds = sorted(set(fold_of.values()))
    jobs = [[v for v in train if fold_of[v.vid] != k] for k in folds] + [train]
    with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
        fitted = list(pool.map(_fit_evidence, [(j, hebrew) for j in jobs]))
    return dict(zip(folds, fitted[:-1])), fitted[-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--split', required=True)
    ap.add_argument('--name', default='ranker', help='system name for the prediction files')
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--d', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--hebrew', action='store_true', help='use the Hebrew parallel text')
    ap.add_argument('--heb-dropout', type=float, default=0.3)
    args = ap.parse_args()

    split = Split.load(args.split)
    train, dev, test = split.partition(load_corpus(split.version))
    model_dir = REPO_ROOT / 'models' / 'ranker' / split.name
    with open(model_dir / 'candidates.pkl', 'rb') as f:
        data = pickle.load(f)
    n_feat = len(data['features'])
    fold_of, train_tables, eval_table = data['fold_of'], data['train_tables'], data['eval_table']

    train_ex = make_examples(train, lambda v: train_tables[fold_of[v.vid]], n_feat)
    dev_ex = make_examples(dev, lambda v: eval_table, n_feat)
    test_ex = make_examples(test, lambda v: eval_table, n_feat)
    if args.hebrew:
        t = time.time()
        per_fold, full = hebrew_evidence(train, fold_of)
        add_hebrew(train_ex, train, lambda v: per_fold[fold_of[v.vid]])
        add_hebrew(dev_ex, dev, lambda v: full)
        add_hebrew(test_ex, test, lambda v: full)
        print(f'hebrew evidence in {time.time() - t:.0f}s', flush=True)

    t = time.time()
    ranker = ContextRanker(d=args.d, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                           dropout=args.dropout, seed=args.seed, heb_dropout=args.heb_dropout,
                           heb_missing=HEB_MISSING if args.hebrew else None,
                           log=lambda m: print(m, flush=True)).fit(train_ex, dev_ex)
    print(f'trained in {time.time() - t:.0f}s', flush=True)
    ranker.save(model_dir / f'{args.name}.pt')

    meta = {'system': args.name, 'split': split.name, 'version': split.version} | vars(args)
    for suffix, examples in (('-dev', dev_ex), ('', test_ex)):
        path = REPO_ROOT / 'results' / split.name / f'{args.name}{suffix}.pred.tsv'
        write_predictions(path, ranker.predict(examples), meta=meta)
        print('->', path)


if __name__ == '__main__':
    main()
