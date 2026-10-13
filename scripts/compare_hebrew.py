"""Compare two systems' word accuracy on verses with and without a Hebrew parallel.

    PYTHONPATH=src python scripts/compare_hebrew.py --split random --part dev --systems ranker ranker-heb

Also breaks the difference down by error type (ssi.errors) and surface category.
"""
import argparse
from collections import Counter

from ssi.corpus import REPO_ROOT, load_corpus
from ssi.errors import error_type
from ssi.hebrew import load_hebrew, verse_key
from ssi.metrics import categorise, training_analyses
from ssi.predictions import read_predictions
from ssi.splits import Split


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--split', default='random')
    ap.add_argument('--part', default='dev', choices=('dev', 'test'))
    ap.add_argument('--systems', nargs=2, default=['ranker', 'ranker-heb'])
    args = ap.parse_args()

    split = Split.load(args.split)
    train, dev, test = split.partition(load_corpus(split.version))
    verses = dev if args.part == 'dev' else test
    suffix = '-dev' if args.part == 'dev' else ''
    preds = [read_predictions(REPO_ROOT / 'results' / split.name / f'{s}{suffix}.pred.tsv')[0]
             for s in args.systems]
    hebrew = load_hebrew()
    seen = training_analyses(train)

    n, ok = Counter(), [Counter(), Counter()]
    errs = [Counter(), Counter()]
    for v in verses:
        group = 'with Hebrew' if verse_key(v) in hebrew else 'without Hebrew'
        for i, (s, g) in enumerate(v.words()):
            c = categorise(s, seen)
            for key in (group, f'{group}, {c}'):
                n[key] += 1
            for k, p in enumerate(preds):
                if p[v.vid][i] == g:
                    ok[k][group] += 1
                    ok[k][f'{group}, {c}'] += 1
                elif group == 'with Hebrew':
                    errs[k][error_type(g, p[v.vid][i])] += 1

    a, b = args.systems
    print(f'{split.name} {args.part}: {a} vs {b}')
    for key in sorted(n, key=lambda x: (x.split(',')[0], ',' in x, x)):
        x, y = ok[0][key] / n[key], ok[1][key] / n[key]
        print(f'  {key:32s} {n[key]:6d} words   {x:.4f} -> {y:.4f}  ({y - x:+.4f})')
    print('\nerrors on verses with Hebrew, by type')
    for t in sorted(set(errs[0]) | set(errs[1]), key=lambda t: -errs[0][t]):
        print(f'  {t:40s} {errs[0][t]:5d} -> {errs[1][t]:5d}')


if __name__ == '__main__':
    main()
