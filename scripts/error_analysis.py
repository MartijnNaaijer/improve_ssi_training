"""Break down the errors of a system's predictions.

    PYTHONPATH=src python scripts/error_analysis.py --system ranker --split random --part dev

For every wrong word it records the surface category (unseen / ambiguous /
unambig), whether the gold analysis was among the ranker's candidates and
among the surface form's training analyses, and what kind of difference
separates prediction from gold. Prints tables and writes all errors to
results/<split>/<system>-<part>.errors.tsv.
"""
import argparse
import pickle
import re
from collections import Counter

from ssi.corpus import REPO_ROOT, load_corpus
from ssi.encoding import paradigmatic_items, parse_word, surface_segmentation
from ssi.metrics import categorise, training_analyses
from ssi.predictions import read_predictions
from ssi.splits import Split

STEM_SLOTS = {'pfm', 'pfx', 'vbs'}
ENDING_SLOTS = {'vbe', 'nme', 'emf'}


def _boundaries(encoding: str):
    """Letter positions with a morpheme boundary before them, and the marker
    string at every position (leading and trailing included)."""
    seg = surface_segmentation(encoding)
    marks, cur = [], ''
    for c in seg:
        if c.isupper() or c in '<>':
            marks.append(cur)
            cur = ''
        else:
            cur += c
    marks.append(cur)
    return {k for k, m in enumerate(marks[1:-1], 1) if m}, marks


def marker_diff(gold: str, pred: str) -> str:
    """The marker strings that differ, as 'gold>pred' per position."""
    g, p = _boundaries(gold)[1], _boundaries(pred)[1]
    return ' '.join(f'{a or "_"}>{b or "_"}' for a, b in zip(g, p) if a != b)


def error_type(gold: str, pred: str) -> str:
    """The coarsest level at which prediction and gold differ."""
    if surface_segmentation(gold) != surface_segmentation(pred):
        if gold.count('-') != pred.count('-'):
            return 'segm: clitics'
        gp, pp = _boundaries(gold), _boundaries(pred)
        if gp[0] != pp[0]:
            return 'segm: boundary position'
        return 'segm: same boundaries, other markers'
    gi, pi = paradigmatic_items(gold), paradigmatic_items(pred)
    glex = [v for _, s, v in gi if s == 'lex']
    plex = [v for _, s, v in pi if s == 'lex']
    if glex != plex:
        if [x.replace('=', '') for x in glex] == [x.replace('=', '') for x in plex]:
            return 'lexeme: homograph index'
        return 'lexeme: other lexeme'
    gs, ps = set(gi), set(pi)
    slots = {s for _, s, _ in gs ^ ps}
    if not slots:
        return 'letter operations only'
    if slots == {'vpm'}:
        return 'vowel pattern mark'
    if slots <= STEM_SLOTS | {'vpm'}:
        return 'stem formation'
    if slots <= ENDING_SLOTS:
        return 'endings (vbe/nme/emf)'
    return 'mixed morphemes'


def table(title, counter, total, top=None):
    print(f'\n{title}')
    for k, n in counter.most_common(top):
        print(f'  {n:6d}  {n / total:6.1%}  {k}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--system', default='ranker')
    ap.add_argument('--split', default='random')
    ap.add_argument('--part', default='dev', choices=('dev', 'test'))
    args = ap.parse_args()

    split = Split.load(args.split)
    train, dev, test = split.partition(load_corpus(split.version))
    verses = dev if args.part == 'dev' else test
    suffix = '-dev' if args.part == 'dev' else ''
    preds, _ = read_predictions(REPO_ROOT / 'results' / split.name / f'{args.system}{suffix}.pred.tsv')
    seen = training_analyses(train)
    cand_path = REPO_ROOT / 'models' / 'ranker' / split.name / 'candidates.pkl'
    cands = pickle.load(open(cand_path, 'rb'))['eval_table'] if cand_path.exists() else {}

    n_words, by_cat, err_cat = 0, Counter(), Counter()
    types, types_cat, status, pairs, lex_pairs = Counter(), Counter(), Counter(), Counter(), Counter()
    majority, mdiff = Counter(), Counter()
    rows = []
    for v in verses:
        for i, (s, g) in enumerate(v.words()):
            c = categorise(s, seen)
            n_words += 1
            by_cat[c] += 1
            p = preds[v.vid][i]
            if p == g:
                continue
            err_cat[c] += 1
            t = error_type(g, p)
            types[t] += 1
            types_cat[c, t] += 1
            names = [a for a, _ in cands.get(s, [])]
            in_cands = g in names
            in_train = g in seen.get(s, {})
            st = ('gold not in candidates' if not in_cands else
                  'gold seen with this form' if in_train else 'gold only from generator')
            status[c, st] += 1
            if t == 'segm: same boundaries, other markers':
                mdiff[marker_diff(g, p)] += 1
            if c == 'ambiguous':
                top = seen[s].most_common(1)[0][0]
                majority['pred is majority analysis' if p == top else
                         'gold is majority analysis' if g == top else 'neither is majority'] += 1
            pairs[s, g, p] += 1
            if t.startswith('lexeme'):
                lex_pairs[' '.join(x for _, sl, x in paradigmatic_items(g) if sl == 'lex'),
                          ' '.join(x for _, sl, x in paradigmatic_items(p) if sl == 'lex')] += 1
            ctx = ' '.join(v.surface[max(0, i - 3):i]) + ' [' + s + '] ' + ' '.join(v.surface[i + 1:i + 4])
            rows.append((v.vid, s, c, t, st, g, p, ctx))

    n_err = sum(err_cat.values())
    print(f'{args.system} {split.name} {args.part}: {n_words} words, {n_err} errors '
          f'(word acc {1 - n_err / n_words:.4f})')
    print('\nerrors by category  (errors, share of all errors, error rate in category)')
    for c in ('unseen', 'ambiguous', 'unambig'):
        print(f'  {c:10s} {err_cat[c]:6d}  {err_cat[c] / n_err:6.1%}  {err_cat[c] / max(by_cat[c], 1):6.1%}'
              f'  of {by_cat[c]}')
    table('error type', types, n_err)
    for c in ('unseen', 'ambiguous', 'unambig'):
        table(f'error type, {c}', Counter({t: n for (cc, t), n in types_cat.items() if cc == c}),
              max(err_cat[c], 1))
    table('candidate status', Counter({f'{c:10s} {st}': n for (c, st), n in status.items()}), n_err)
    table('ambiguous errors vs majority training analysis', majority, max(err_cat['ambiguous'], 1))
    table('segm errors with the same boundaries: differing markers (gold>pred)', mdiff,
          max(types['segm: same boundaries, other markers'], 1), top=20)
    table('most frequent lexeme confusions (gold -> pred)',
          Counter({f'{g} -> {p}': n for (g, p), n in lex_pairs.items()}), n_err, top=25)
    table('most frequent word errors (surface: gold -> pred)',
          Counter({f'{s}: {g} -> {p}': n for (s, g, p), n in pairs.items()}), n_err, top=30)

    out = REPO_ROOT / 'results' / split.name / f'{args.system}-{args.part}.errors.tsv'
    with open(out, 'w', encoding='utf8', newline='') as f:
        f.write('vid\tsurface\tcategory\ttype\tcandidates\tgold\tpred\tcontext\n')
        for r in rows:
            f.write('\t'.join(r) + '\n')
    print('\n->', out)


if __name__ == '__main__':
    main()
