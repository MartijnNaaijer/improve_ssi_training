import pytest

from ssi.corpus import Verse, load_corpus
from ssi.encoding import is_consistent
from ssi.generator import CandidateGenerator, verb_binyan
from ssi.templates import Frame, decompose, lexeme_pattern


def V(vid, pairs):
    return Verse(vid, 'A', '1', vid, tuple(p[0] for p in pairs), tuple(p[1] for p in pairs))


@pytest.mark.parametrize('morph, pattern', [
    ('MLK', '*'),
    ('MLK=', '*'),
    ('(>T(J&>', '(>.(J&>'),
    ('KT&WB', '*&W.'),
    ('ML(>', '*(>'),
])
def test_lexeme_pattern(morph, pattern):
    assert lexeme_pattern(morph) == pattern


def test_decompose_picks_longest_lexeme_as_core():
    d = decompose('W-!N!BN(J[WN-HWN')
    assert (d.proclitics, d.frame, d.enclitics, d.lexeme) == ('W', '!N!{*(J}[WN', 'HWN', 'BNJ')
    d = decompose('W-L-MLK=/~>')
    assert (d.proclitics, d.frame, d.enclitics, d.lexeme) == ('W-L', '{*}/~>', '', 'MLK=')


def test_frame_inverts_lexeme_pattern():
    f = Frame('!!{*&W.}[')
    assert (f.pre_real, f.post_real) == ('', '')
    assert f.invert('KTWB') == 'KT&WB'
    assert f.invert('KTB') is None
    assert Frame('{*(J&>}[').invert('BN>') == 'BN(J&>'


def test_verb_binyan():
    assert verb_binyan('!N!](>]', '[WN') == ('>', False, False)
    assert verb_binyan('@>T@', '[:d') == ('', True, True)


TRAIN = [
    V('1', [('KTB', 'KTB['), ('MLK>', 'MLK/~>'), ('WQBR', 'W-QBR[')]),
    V('2', [('NKTBWN', '!N!KTB[WN'), ('MLKJN', 'MLK/JN'), ('BJT>', 'BJT/~>')]),
]


def test_generator_combines_seen_parts():
    gen = CandidateGenerator().fit(TRAIN)
    cands = dict(gen.candidates('WNQBRWN'))
    assert 'W-!N!QBR[WN' in cands
    assert all(is_consistent('WNQBRWN', a) for a in cands)
    assert gen.candidates('BJTJN')[0][0] == 'BJT/JN'


def test_generator_covers_corpus_words():
    """Every word of a verse is among the candidates of a generator fitted on the others."""
    verses = load_corpus('s12')[:400]
    gen = CandidateGenerator().fit(verses[:-1])
    for s, a in verses[-1].words():
        assert a in dict(gen.candidates(s)), (s, a)


def test_geminate_pattern_generalises():
    assert lexeme_pattern('K&WP(P') == '*&W.(%'
    assert Frame('!!{*&W.(%}[').invert('MWK') == 'M&WK(K'


@pytest.mark.parametrize('plain, metathesised, surface', [
    ('@>T@CKX[', '@>(T&C@(C&TKX[', '>CTKX'),
    ('!!@>T@ZHR[:d', '!!@>(T&Z@(Z&DHR[:d', '>ZDHR'),
    ('@>T@]C]<BD[', '@>(T&C@](C&T]<BD[', '>CT<BD'),
])
def test_metathesis_rule(plain, metathesised, surface):
    from ssi.encoding import realise
    from ssi.generator import metathesise, unmetathesised
    assert metathesise(plain) == metathesised
    assert realise(metathesised) == surface
    assert realise(plain) in unmetathesised(surface)


def test_generator_applies_metathesis():
    train = [V('1', [('>TQBR', '@>T@QBR['), ('CKX', 'CKX[')])]
    gen = CandidateGenerator().fit(train)
    assert '@>(T&C@(C&TKX[' in dict(gen.candidates('>CTKX'))
