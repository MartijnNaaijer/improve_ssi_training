import pytest

from ssi.encoding import (AnalyticalWord, is_consistent, lexemes, paradigmatic_items,
                          parse_analytical, parse_word, realise, surface_segmentation)

# (surface, encoding) pairs from Syriac_paradigm.pdf and the corpus.
EXAMPLES = [
    ('>T>', '!>!(>T(J&>['),                  # PDF 1.1.3
    ('W>CML>', 'W-!>!]C]ML(J&>['),
    ('MLJKJ', 'ML(>/J-KJ'),
    ('MCKXJNN', '!M!](>]CKX[/J(N-(XNN'),     # contraction with pronoun
    ('LMBRKWTH', 'L-!M!BRK[/WT:d-H'),        # vowel pattern mark before '-'
    ('MLK>', 'MLK/~>'),
    ('KTWB', '!!KT&WB['),                    # empty preformative, '&' letter
    ('>TQBR', '@>T@QBR['),
    ('MCBX>', '!M!CBX[/~>:dp'),
    ('B<JT', 'B<J[T'),
]


@pytest.mark.parametrize('surface,encoding', EXAMPLES)
def test_realise_matches_surface(surface, encoding):
    assert realise(encoding) == surface
    assert is_consistent(surface, encoding)


def test_surface_diacritics_ignored():
    assert is_consistent('M"N>', 'MN/~>')
    assert not is_consistent('MN>', 'MLK/~>')


def test_parse_slots():
    aw = parse_analytical('!M!](>]CKX[/J(N')
    assert aw == AnalyticalWord(pfm='M', vbs='(>', lex='CKX', vbe='', nme='J(N')
    aw = parse_analytical('!M!CBX[/~>:dp')
    assert (aw.pfm, aw.lex, aw.vbe, aw.nme, aw.emf, aw.vpm) == ('M', 'CBX', '', '', '>', 'dp')
    assert parse_analytical('@>T@QBR[').pfx == '>T'
    assert parse_analytical('!!KT&WB[').pfm == ''


def test_parse_word_splits_analytical_words():
    words = parse_word('L-!M!BRK[/WT:d-H')
    assert [w.lex for w in words] == ['L', 'BRK', 'H']
    assert words[1].vpm == 'd'


def test_parse_rejects_garbage():
    with pytest.raises(ValueError):
        parse_analytical('[[/!')


def test_lexemes_are_paradigmatic():
    assert lexemes('!>!(>T(J&>[') == ('>TJ',)
    assert lexemes('W-!>!]C]ML(J&>[') == ('W', 'MLJ')
    assert lexemes('ML(>/J-KJ') == ('ML>', 'KJ')
    assert lexemes('!!KT&WB[') == ('KTB',)


def test_surface_segmentation_drops_paradigmatic_info():
    assert surface_segmentation('!>!(>T(J&>[') == '!>!T>['
    assert surface_segmentation('HW(J[T==') == 'HW[T'
    assert surface_segmentation('L-!M!BRK[/WT:d-H') == 'L-!M!BRK[/WT-H'


def test_paradigmatic_items():
    items = paradigmatic_items('W-MLK/~>')
    assert (0, 'lex', 'W') in items
    assert (1, 'lex', 'MLK') in items
    assert (1, 'nme', '') in items
    assert (1, 'emf', '>') in items
