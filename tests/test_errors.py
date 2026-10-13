import pytest

from ssi.errors import error_type


@pytest.mark.parametrize('gold, pred, kind', [
    ('W->MR[', 'W-!!>MR[', 'segm: same boundaries, other markers'),
    ('L-H=', 'L-H', 'lexeme: homograph index'),
    ('QRB[', 'QRB[:d', 'vowel pattern mark'),
    ('>JK-(HN>', '>JKN>', 'segm: clitics'),
    ('MLK/~>', 'MLK/~>', 'letter operations only'),
])
def test_error_type(gold, pred, kind):
    assert error_type(gold, pred) == kind
