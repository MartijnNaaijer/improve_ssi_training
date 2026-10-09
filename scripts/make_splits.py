"""Create the standard splits in data/splits/.

    python scripts/make_splits.py [--version s12]
"""
import argparse

from ssi.corpus import load_corpus
from ssi.splits import HELDOUT_BOOKS, heldout_book_split, random_split


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--version', default='s12')
    args = ap.parse_args()

    verses = load_corpus(args.version)
    splits = [random_split(verses, args.version)]
    splits += [heldout_book_split(verses, b, args.version) for b in HELDOUT_BOOKS]
    for s in splits:
        path = s.save()
        print(f'{s.name:16} train {len(s.train):6} dev {len(s.dev):5} test {len(s.test):5}  -> {path.name}')


if __name__ == '__main__':
    main()
