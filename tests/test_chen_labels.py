from collections import Counter

import pytest

from ssi.upstream.chen_labels import (PatternInventory, analyses_from_line, line_input,
                                      line_patterns, reduce, restore, word_patterns)

WORDS = ['!>!(>T(J&>[', 'W-!>!]C]ML(J&>[', 'ML(>/J-KJ', '!M!](>]CKX[/J(N-(XNN',
         'L-!M!BRK[/WT:d-H', '!!KT&WB[', '@>T@QBR[', '!M!@(>(T&S@(S&TKL=[/J(N:d-(XNN', 'MLK/~>']


@pytest.mark.parametrize('enc', WORDS)
def test_reduce_restore_roundtrip(enc):
    assert restore(reduce(enc)) == enc


def test_reduce_matches_upstream_examples():
    # from the archived reduced/original pairs in the upstream repository
    assert reduce('!>!CR(J&>[') == '>!CR(J&>['
    assert reduce('L-!M!QVL[/') == 'L-M!QVL[/'
    assert reduce('!M!](>]XJ(J[/W:d') == 'M!(>]XJ(J[/W:d'


def test_word_patterns():
    letters, pats = word_patterns(reduce('!>!(>T(J&>['))
    assert ''.join(letters) == '>T>'
    assert pats == ['', '!(>', '(J&', '[']


def test_line_alignment_and_inverse():
    surf = ['>T>', 'W>CML>', 'M"LJKJ']
    anal = ['!>!(>T(J&>[', 'W-!>!]C]ML(J&>[', 'ML(>/J-KJ']
    pats = line_patterns(surf, anal)
    assert line_input(surf) == ' >T> W>CML> MLJKJ'
    assert len(pats) == len(line_input(surf))
    assert analyses_from_line(surf, pats) == anal


def test_short_prediction_fills_empty_patterns():
    assert analyses_from_line(['MLK>'], ['', '', '', '/~']) == ['MLK/~>']


def test_inventory_maps_unseen_to_zero():
    inv = PatternInventory(Counter({'': 10, '/': 5, '-': 7}))
    assert inv.patterns == ['', '-', '/']
    assert inv.encode(['/', '[:d']) == [2, 0]
    assert inv.decode([1, 99]) == ['-', '']
