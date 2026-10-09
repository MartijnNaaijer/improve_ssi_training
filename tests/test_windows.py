from ssi.corpus import Verse
from ssi.predictions import read_predictions, write_predictions
from ssi.upstream.ssi_seq2seq import _blocks
from ssi.windows import assemble, cover_with_windows


def V(vid, book, n):
    words = tuple(f'{vid}w{i}' for i in range(n))
    return Verse(vid, book, '1', vid, words, tuple(w.upper() for w in words))


CORPUS = [V('a', 'X', 4), V('b', 'X', 3), V('c', 'X', 5), V('d', 'Y', 2)]


def test_every_target_word_covered_exactly_once():
    targets = [CORPUS[1], CORPUS[2], CORPUS[3]]
    windows = cover_with_windows(CORPUS, targets, 4)
    seen = [(vid, i) for w in windows for vid, i, _ in w.targets]
    expected = [(v.vid, i) for v in targets for i in range(len(v.surface))]
    assert sorted(seen) == sorted(expected)
    for w in windows:
        assert len(w.words) <= 4
        for vid, i, off in w.targets:
            assert w.words[off] == next(v for v in CORPUS if v.vid == vid).surface[i]


def test_short_book_gives_short_window_and_end_is_shifted_left():
    windows = cover_with_windows(CORPUS, [CORPUS[2], CORPUS[3]], 4)
    assert windows[-1].words == ['dw0', 'dw1']
    # verse c (stream positions 7..11 of book X, 12 words): the last window is shifted left
    assert all(len(w.words) == 4 for w in windows[:-1])


def test_assemble_marks_missing_positions():
    targets = [CORPUS[3]]
    windows = cover_with_windows(CORPUS, targets, 4)
    preds = assemble(targets, windows, [['DW0']])
    assert preds == {'d': ['DW0', '']}


def test_predictions_roundtrip(tmp_path):
    p = tmp_path / 'x.pred.tsv'
    write_predictions(p, {'Gn 1:1': ['A/', '', 'B['], 'Gn 1:2': []}, {'system': 's'})
    preds, meta = read_predictions(p)
    assert preds == {'Gn 1:1': ['A/', '', 'B['], 'Gn 1:2': []}
    assert meta == {'system': 's'}


def test_blocks_reproduce_upstream_grouping():
    # 9 one-word verses, seq_len 2 -> groups of 2 verses; the short last group
    # replaces the previous one, doubled (upstream behaviour).
    verses = [V(f'v{i}', 'X', 1) for i in range(9)]
    blocks = _blocks(verses, 2)
    firsts = [b[0][0] for b in blocks]
    assert firsts == ['v0w0 v1w0', 'v2w0 v3w0', 'v4w0 v5w0', 'v8w0 v8w0']
