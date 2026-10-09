import pytest

from ssi.corpus import Verse, load_corpus
from ssi.lookup import LookupBaseline
from ssi.metrics import edit_distance, evaluate
from ssi.splits import heldout_book_split, random_split


def V(vid, book, pairs):
    return Verse(vid, book, '1', vid, tuple(p[0] for p in pairs), tuple(p[1] for p in pairs))


TRAIN = [
    V('1', 'A', [('MLK>', 'MLK/~>'), ('WMLK>', 'W-MLK/~>')]),
    V('2', 'A', [('MLK>', 'MLK/~>'), ('VB>', 'VB/>')]),
    V('3', 'A', [('VB>', 'VB/~>'), ('VB>', 'VB/~>')]),
]
TEST = [V('4', 'B', [('MLK>', 'MLK/~>'), ('VB>', 'VB/>'), ('BJT>', 'BJT/~>')])]


def test_lookup_and_categories():
    model = LookupBaseline().fit(TRAIN)
    pred = model.predict(TEST)
    assert pred['4'] == ['MLK/~>', 'VB/~>', 'BJT>']
    m = evaluate(TEST, pred, TRAIN)
    assert m['all']['n'] == 3
    assert m['all']['word_acc'] == pytest.approx(1 / 3)
    assert m['unambig']['word_acc'] == 1.0
    assert m['ambiguous']['word_acc'] == 0.0
    assert m['unseen']['n'] == 1
    assert m['all']['consistency'] == 1.0


def test_lookup_backs_off_to_skeleton():
    model = LookupBaseline().fit(TRAIN)
    assert model.predict_word('ML"K>') == 'MLK/~>'


def test_edit_distance():
    assert edit_distance('', 'abc') == 3
    assert edit_distance('kitten', 'sitting') == 3
    assert edit_distance('MLK/~>', 'MLK/~>') == 0


def test_splits_are_disjoint_and_complete():
    verses = TRAIN + TEST
    for s in (random_split(verses, 'x', dev_frac=0.25, test_frac=0.25),
              heldout_book_split(verses, 'B', 'x', dev_frac=0.34)):
        ids = s.train + s.dev + s.test
        assert len(ids) == len(set(ids)) == len(verses)
    assert heldout_book_split(verses, 'B', 'x').test == ['4']


def test_real_corpus_is_consistent():
    verses = load_corpus('s12')
    assert len(verses) == 18146
    assert len({v.vid for v in verses}) == len(verses)
