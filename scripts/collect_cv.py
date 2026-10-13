"""Combine the cross-validation folds into corpus-wide results.

    PYTHONPATH=src python scripts/collect_cv.py [--folds 5]

Every verse is in the test part of exactly one fold (see
ssi.splits.cross_validation_splits), so the folds' test outputs together
cover the corpus. Writes to results/cv<folds>/:

    suspects.tsv   suspected gold errors over the whole corpus (from each
                   fold's triage suspects, test part only), the least
                   probable gold first
    words.tsv      every word with prediction and calibrated confidence
    summary.json   word accuracy per fold and pooled
"""
import argparse
import csv
import json

from ssi.corpus import REPO_ROOT


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--folds', type=int, default=5)
    args = ap.parse_args()

    out_dir = REPO_ROOT / 'results' / f'cv{args.folds}'
    out_dir.mkdir(parents=True, exist_ok=True)
    suspects, words, summary = [], [], {'folds': {}}
    header_s = header_w = None
    for k in range(args.folds):
        tri = REPO_ROOT / 'results' / f'cv{args.folds}-{k}' / 'triage'
        if not (tri / 'ranker-test.tsv').exists():
            print(f'fold {k}: no triage output, skipped')
            continue
        with open(tri / 'ranker-suspects.tsv', encoding='utf8', newline='') as f:
            rows = list(csv.reader(f, delimiter='\t'))
        header_s = rows[0]
        suspects += [r for r in rows[1:] if r[0] == 'test']
        with open(tri / 'ranker-test.tsv', encoding='utf8', newline='') as f:
            rows = list(csv.reader(f, delimiter='\t'))
        header_w = rows[0]
        fold_words = rows[1:]
        words += fold_words
        n_ok = sum(r[-1] == '1' for r in fold_words)
        summary['folds'][k] = {'words': len(fold_words), 'word_acc': n_ok / len(fold_words)}

    p = header_s.index('p_gold')
    c = header_s.index('confidence')
    suspects.sort(key=lambda r: (float(r[p]), -float(r[c])))
    with open(out_dir / 'suspects.tsv', 'w', encoding='utf8', newline='\n') as f:
        f.write('\t'.join(header_s[1:]) + '\n')
        for r in suspects:
            f.write('\t'.join(r[1:]) + '\n')
    with open(out_dir / 'words.tsv', 'w', encoding='utf8', newline='\n') as f:
        f.write('\t'.join(header_w) + '\n')
        for r in words:
            f.write('\t'.join(r) + '\n')
    n = sum(v['words'] for v in summary['folds'].values())
    summary['words'] = n
    summary['word_acc'] = sum(v['words'] * v['word_acc'] for v in summary['folds'].values()) / max(n, 1)
    summary['suspects'] = len(suspects)
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=1), encoding='utf8')
    print(f"{n} words in {len(summary['folds'])} folds, pooled word acc {summary['word_acc']:.4f}, "
          f'{len(suspects)} suspected gold errors -> {out_dir}')


if __name__ == '__main__':
    main()
