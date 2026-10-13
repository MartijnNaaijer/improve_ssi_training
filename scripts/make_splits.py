"""Create the standard splits in data/splits/.

    python scripts/make_splits.py [--version s12] [--cv 5]

--cv N writes only the N cross-validation splits cvN-0 .. cvN-(N-1).
"""
import argparse

from ssi.corpus import load_corpus
from ssi.splits import HELDOUT_BOOKS, cross_validation_splits, heldout_book_split, random_split


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--version', default='s12')
    ap.add_argument('--cv', type=int, default=0, help='write N cross-validation splits instead')
    args = ap.parse_args()

    verses = load_corpus(args.version)
    if args.cv:
        splits = cross_validation_splits(verses, args.version, folds=args.cv)
    else:
        splits = [random_split(verses, args.version)]
        splits += [heldout_book_split(verses, b, args.version) for b in HELDOUT_BOOKS]
    for s in splits:
        path = s.save()
        print(f'{s.name:16} train {len(s.train):6} dev {len(s.dev):5} test {len(s.test):5}  -> {path.name}')


if __name__ == '__main__':
    main()
