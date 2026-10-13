"""Find words that the gold analyses differently in identical contexts.

    PYTHONPATH=src python scripts/find_inconsistencies.py [--version s12] [--window 2]

Groups all tokens of the corpus by their surface form plus `window` surface
words on either side. A group with more than one analysis is either a genuine
ambiguity that the context does not resolve or an annotation inconsistency;
the minority analyses are the ones to check. No model is involved, so this
covers the whole corpus. Writes results/consistency/<version>-w<window>.tsv,
most frequent groups first.
"""
import argparse
from collections import Counter, defaultdict

from ssi.corpus import REPO_ROOT, load_corpus
from ssi.errors import error_type


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--version', default='s12')
    ap.add_argument('--window', type=int, default=2)
    args = ap.parse_args()

    w = args.window
    groups = defaultdict(lambda: defaultdict(list))
    n_tokens = 0
    for v in load_corpus(args.version):
        padded = ('<s>',) * w + v.surface + ('</s>',) * w
        for i, (s, a) in enumerate(v.words()):
            key = padded[i:i + w] + (s,) + padded[i + w + 1:i + 2 * w + 1]
            groups[key][a].append(f'{v.vid}#{i + 1}')
            n_tokens += 1

    rows, types = [], Counter()
    n_groups = n_minority = 0
    for key, analyses in groups.items():
        if len(analyses) < 2:
            continue
        n_groups += 1
        ranked = sorted(analyses.items(), key=lambda x: (-len(x[1]), x[0]))
        major = ranked[0][0]
        for a, where in ranked[1:]:
            n_minority += len(where)
            types[error_type(major, a)] += len(where)
        total = sum(len(x) for x in analyses.values())
        context = ' '.join(key[:w]) + ' [' + key[w] + '] ' + ' '.join(key[w + 1:])
        rows.append((total, key[w], context, ranked))

    rows.sort(key=lambda r: (-r[0], r[1]))
    out = REPO_ROOT / 'results' / 'consistency' / f'{args.version}-w{w}.tsv'
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf8', newline='\n') as f:
        f.write('n\tsurface\tcontext\tmajority\tmajority_n\tminority\tminority_n\ttype\tminority_where\n')
        for total, s, context, ranked in rows:
            major, major_where = ranked[0]
            for a, where in ranked[1:]:
                f.write(f'{total}\t{s}\t{context}\t{major}\t{len(major_where)}\t{a}\t{len(where)}\t'
                        f'{error_type(major, a)}\t{" ".join(where)}\n')

    print(f'{args.version}, window {w}: {n_tokens} tokens; {n_groups} contexts with more than one analysis, '
          f'{n_minority} minority tokens ({n_minority / n_tokens:.2%})')
    for t, n in types.most_common():
        print(f'  {n:6d}  {n / max(n_minority, 1):6.1%}  {t}')
    print('->', out)


if __name__ == '__main__':
    main()
