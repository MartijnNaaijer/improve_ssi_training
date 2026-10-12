import pytest

from ssi.candidates import FEATURES, CandidateBuilder
from ssi.corpus import Verse

torch = pytest.importorskip('torch')
from ssi.ranker import ContextRanker, make_examples  # noqa: E402


def V(vid, pairs):
    return Verse(vid, 'A', '1', vid, tuple(p[0] for p in pairs), tuple(p[1] for p in pairs))


TRAIN = [
    V('1', [('KTB', 'KTB['), ('MLK>', 'MLK/~>'), ('WQBR', 'W-QBR[')]),
    V('2', [('NKTBWN', '!N!KTB[WN'), ('MLKJN', 'MLK/JN'), ('BJT>', 'BJT/~>')]),
]


def test_builder_puts_lookup_first():
    builder = CandidateBuilder({'generative': 1.0}).fit(TRAIN)
    cands = builder.build('MLK>')
    assert cands[0][0] == 'MLK/~>'
    assert all(len(f) == len(FEATURES) for _, f in cands)
    assert cands[0][1][FEATURES.index('lk_in')] == 1.0
    assert 'BJT/JN' in [a for a, _ in builder.build('BJTJN')]


def test_ranker_trains_and_predicts():
    builder = CandidateBuilder({'generative': 1.0}).fit(TRAIN)
    table = {s: builder.build(s) for v in TRAIN for s in v.surface}
    ex = make_examples(TRAIN, lambda v: table, len(FEATURES))
    ranker = ContextRanker(d=16, epochs=2, batch_size=2, device='cpu', log=lambda m: None).fit(ex, ex)
    preds = ranker.predict(ex)
    assert set(preds) == {'1', '2'}
    assert all(len(preds[v.vid]) == len(v.surface) for v in TRAIN)
