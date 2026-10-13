import math

import numpy as np
import pytest

from ssi.corpus import RAW_DIR, Verse
from ssi.hebrew import (HEB_FEATURES, HEB_MISSING, IBM1, HebrewEvidence, HebrewWord, add_hebrew, cognate,
                        core_lexeme, hebrew_lexemes, lexeme_pairs, load_hebrew)
from ssi.ranker import VerseExample


def verse(ref, analyses):
    book, ch, v = ref
    return Verse(f'{book} {ch}:{v}', book, ch, v, tuple(a.replace('-', '') for a in analyses), tuple(analyses))


def test_hebrew_lexemes_normalises_suffixes_and_compounds():
    assert hebrew_lexemes('W+!J!>MR[+HW') == ('W', '>MR', 'HW')
    assert hebrew_lexemes('BJT_>L/') == ('BJT>L',)


def test_core_lexeme():
    assert core_lexeme('W-MLK/~>') == 'MLK'
    assert core_lexeme('L-H=') == 'H='
    assert core_lexeme('???') == ''


def test_cognate():
    assert cognate('MLK', 'MLK') == 1.0
    assert cognate('MN=', 'MN') == 1.0                   # homograph marks ignored
    assert cognate('TWR', 'CWR') == pytest.approx(1 - 0.25 / 3)  # Hebrew C ~ Syriac T
    assert cognate('DHB', 'ZHB') == pytest.approx(1 - 0.25 / 3)
    assert cognate('KTB', '>MR') == 0.0
    assert cognate('', 'X') == 0.0


def test_ibm1_learns_translations():
    pairs = [(['>ZL', 'MLK'], ['HLK', 'MLK']), (['>ZL', 'BJT'], ['HLK', 'BJT']), (['JHB', 'MLK'], ['NTN', 'MLK']),
             (['JHB', 'BJT'], ['NTN', 'BJT'])] * 3
    t = IBM1().fit(pairs).t
    assert t[('>ZL', 'HLK')] > t.get(('>ZL', 'MLK'), 0.0)
    assert t[('JHB', 'NTN')] > t.get(('JHB', 'BJT'), 0.0)


HEB = {('Gn', '1', '1'): [HebrewWord('W-HLK['), HebrewWord('H-MLK/')],
       ('Gn', '1', '2'): [HebrewWord('W-NTN['), HebrewWord('H-BJT/')]}
TRAIN = [verse(('Gn', '1', '1'), ['W->ZL[', 'MLK/~>']), verse(('Gn', '1', '2'), ['W-JHB[', 'BJT/~>'])] * 3


def test_lexeme_pairs_skips_verses_without_hebrew():
    pairs = lexeme_pairs(TRAIN + [verse(('Sir', '1', '1'), ['MLK/~>'])], HEB)
    assert len(pairs) == len(TRAIN)
    assert pairs[0] == (['W', '>ZL', 'MLK'], ['W', 'HLK', 'H', 'MLK'])


def test_evidence_aligns_and_prefers_translations():
    ev = HebrewEvidence.fit(TRAIN, HEB)
    v = verse(('Gn', '1', '1'), ['W->ZL[', 'MLK/~>'])
    feats, strings = ev.verse(v, [['W->ZL[', 'W-JHB['], ['MLK/~>']])
    assert [len(f) for f in feats] == [2, 1] and all(len(x) == len(HEB_FEATURES) for f in feats for x in f)
    good, bad = feats[0]
    assert good[0] == 1.0 and good[1] > bad[1]           # heb_has, heb_t_max
    assert strings[0][0] == 'W-HLK['
    assert strings[1][0] == 'H-MLK/' and feats[1][0][3] == 1.0  # cognate MLK ~ MLK


def test_evidence_without_parallel():
    ev = HebrewEvidence.fit(TRAIN, HEB)
    feats, strings = ev.verse(verse(('Sir', '1', '1'), ['MLK/~>']), [['MLK/~>', 'MLK/']])
    assert feats == [[list(HEB_MISSING)] * 2] and strings == [['', '']]
    assert HEB_MISSING[1] == math.log(1e-6)


def test_add_hebrew_appends_columns():
    ev = HebrewEvidence.fit(TRAIN, HEB)
    v = TRAIN[0]
    cands = [['W->ZL[', 'W-JHB['], ['MLK/~>']]
    ex = VerseExample(vid=v.vid, words=list(v.surface), cands=cands,
                      feats=[np.zeros((len(c), 3), dtype=np.float32) for c in cands], gold=[0, 0])
    add_hebrew([ex], [v], lambda _: ev)
    assert [f.shape for f in ex.feats] == [(2, 3 + len(HEB_FEATURES)), (1, 3 + len(HEB_FEATURES))]
    assert ex.heb == [['W-HLK[', 'W-HLK['], ['H-MLK/']]


@pytest.mark.skipif(not (RAW_DIR / 't-out').exists(), reason='BHSA data not present')
def test_load_hebrew():
    heb = load_hebrew()
    assert heb[('Gn', '1', '1')][1].analysis == 'BR>['
    assert {b for b, _, _ in heb} <= {'Gn', 'Ex', 'Lv', 'Nm', 'Dt', 'Jos', 'Jd', 'Ps', '1Rg', '2Rg', 'Jr', 'Ez',
                                      'Jon', 'Sa', 'Dn'}
