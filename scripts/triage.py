"""Calibrated confidence, annotator triage and suspected gold errors for the ranker.

    .venv/Scripts/python.exe scripts/triage.py --split random

Needs models/ranker/<split>/{candidates.pkl,ranker.pt}. Fits the calibrators
on dev, compares them on dev (5-fold cross-validation) and on test, and
writes to results/<split>/triage/:

    ranker.json            calibration metrics and triage curves
    ranker-<part>.tsv      per word: prediction, calibrated confidence, the
                           best alternatives, and the gold (for evaluation)
    ranker-suspects.tsv    dev and test words where a confident prediction
                           disagrees with the gold: candidates for re-annotation

For held-out books it also reports how much refitting the temperature on the
first 5% / 10% of the book's verses improves calibration on the rest.
"""
import argparse
import json
import pickle

import numpy as np

from ssi.confidence import (CONF_FEATURES, LogisticConfidence, auroc, brier, budget_for_accuracy, ece,
                            fit_temperature, scale, triage_curve, word_features)
from ssi.corpus import REPO_ROOT, load_corpus
from ssi.metrics import categorise, training_analyses
from ssi.ranker import ContextRanker, make_examples
from ssi.splits import Split


def word_records(verses, examples, probs, seen):
    rows = []
    for v, ex, pv in zip(verses, examples, probs):
        for i, (s, g) in enumerate(zip(ex.words, v.analysis)):
            p = pv[i]
            k = int(p.argmax()) if len(p) else -1
            rows.append({'vid': v.vid, 'i': i, 'surface': s, 'gold': g, 'cands': ex.cands[i], 'probs': p,
                         'gold_idx': ex.gold[i], 'pred': ex.cands[i][k] if k >= 0 else '',
                         'correct': k >= 0 and ex.cands[i][k] == g, 'category': categorise(s, seen),
                         'context': ' '.join(ex.words[max(0, i - 4):i]) + ' [' + s + '] '
                                    + ' '.join(ex.words[i + 1:i + 5])})
    return rows


def metrics(conf, correct):
    return {'ece': ece(conf, correct), 'brier': brier(conf, correct), 'auroc': auroc(conf, correct),
            'mean_conf': float(conf.mean()), 'acc': float(correct.mean())}


def confidences(method, rows, fit_rows):
    """Confidence of the top candidate for `rows`, calibrated on `fit_rows`."""
    if method == 'raw':
        return np.array([r['probs'].max() if len(r['probs']) else 0.0 for r in rows])
    if method == 'temperature':
        t = fit_temperature([r['probs'] for r in fit_rows], [r['gold_idx'] for r in fit_rows])
        return np.array([scale(r['probs'], t).max() if len(r['probs']) else 0.0 for r in rows])
    x_fit = np.array([word_features(r['probs'], r['category']) for r in fit_rows])
    y_fit = np.array([r['correct'] for r in fit_rows], dtype=float)
    model = LogisticConfidence().fit(x_fit, y_fit)
    return model.predict(np.array([word_features(r['probs'], r['category']) for r in rows]))


def cross_validated(method, rows, folds=5, seed=42):
    """Out-of-fold confidences on dev; folds are whole verses."""
    vids = sorted({r['vid'] for r in rows})
    rng = np.random.default_rng(seed)
    fold = dict(zip(vids, rng.permutation(len(vids)) % folds))
    conf = np.zeros(len(rows))
    for k in range(folds):
        test_idx = [j for j, r in enumerate(rows) if fold[r['vid']] == k]
        fit_rows = [r for r in rows if fold[r['vid']] != k]
        conf[test_idx] = confidences(method, [rows[j] for j in test_idx], fit_rows)
    return conf


def write_words(path, rows, conf, top=3):
    with open(path, 'w', encoding='utf8', newline='\n') as f:
        f.write('vid\tword\tsurface\tprediction\tconfidence\talternatives\tgold\tcorrect\n')
        for r, c in zip(rows, conf):
            order = np.argsort(-r['probs'])[1:top + 1] if len(r['probs']) else []
            alts = ' '.join(f"{r['cands'][j]}:{r['probs'][j]:.3f}" for j in order)
            f.write(f"{r['vid']}\t{r['i'] + 1}\t{r['surface']}\t{r['pred']}\t{c:.4f}\t{alts}\t{r['gold']}\t"
                    f"{int(r['correct'])}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--split', required=True)
    ap.add_argument('--name', default='ranker')
    ap.add_argument('--suspect-threshold', type=float, default=0.9,
                    help='minimum calibrated confidence of a prediction that disagrees with the gold')
    args = ap.parse_args()

    split = Split.load(args.split)
    train, dev, test = split.partition(load_corpus(split.version))
    model_dir = REPO_ROOT / 'models' / 'ranker' / split.name
    with open(model_dir / 'candidates.pkl', 'rb') as f:
        data = pickle.load(f)
    n_feat = len(data['features'])
    train_ex = make_examples(train, lambda v: data['train_tables'][data['fold_of'][v.vid]], n_feat)
    dev_ex = make_examples(dev, lambda v: data['eval_table'], n_feat)
    test_ex = make_examples(test, lambda v: data['eval_table'], n_feat)
    ranker = ContextRanker.load(model_dir / f'{args.name}.pt', train=train_ex)

    seen = training_analyses(train)
    dev_rows = word_records(dev, dev_ex, ranker.probabilities(dev_ex), seen)
    test_rows = word_records(test, test_ex, ranker.probabilities(test_ex), seen)
    y_dev = np.array([r['correct'] for r in dev_rows])
    y_test = np.array([r['correct'] for r in test_rows])

    report = {'split': split.name, 'version': split.version, 'system': args.name,
              'features': list(CONF_FEATURES), 'methods': {}}
    for method in ('raw', 'temperature', 'logistic'):
        cv = cross_validated(method, dev_rows) if method != 'raw' else confidences('raw', dev_rows, dev_rows)
        conf_test = confidences(method, test_rows, dev_rows)
        report['methods'][method] = {'dev_cv': metrics(cv, y_dev), 'test': metrics(conf_test, y_test)}
    best = min(('temperature', 'logistic'), key=lambda m: report['methods'][m]['dev_cv']['ece'])
    report['chosen'] = best
    if best == 'temperature':
        report['temperature'] = fit_temperature([r['probs'] for r in dev_rows], [r['gold_idx'] for r in dev_rows])

    conf_dev = cross_validated(best, dev_rows)
    conf_test = confidences(best, test_rows, dev_rows)
    report['triage'] = {}
    for part, conf, y in (('dev_cv', conf_dev, y_dev), ('test', conf_test, y_test)):
        report['triage'][part] = {
            'curve': triage_curve(conf, y),
            'check_share_for_unchecked_acc': {str(t): budget_for_accuracy(conf, y, t) for t in (0.98, 0.99, 0.995)},
        }
    by_cat = {}
    for c in ('unseen', 'ambiguous', 'unambig'):
        m = np.array([r['category'] == c for r in test_rows])
        by_cat[c] = metrics(conf_test[m], y_test[m])
    report['test_by_category'] = by_cat

    # A new book: how much does refitting the temperature on its first verses
    # (as if an annotator had corrected them) help on the rest of the book?
    if split.name.startswith('heldout'):
        t_dev = fit_temperature([r['probs'] for r in dev_rows], [r['gold_idx'] for r in dev_rows])
        report['book_recalibration'] = []
        for share in (0.05, 0.10):
            first = {v.vid for v in test[:int(share * len(test))]}
            cal = [r for r in test_rows if r['vid'] in first]
            rest = [r for r in test_rows if r['vid'] not in first]
            y = np.array([r['correct'] for r in rest])
            t_book = fit_temperature([r['probs'] for r in cal], [r['gold_idx'] for r in cal])
            entry = {'first_verses_share': share, 'calibration_words': len(cal), 'dev_t': t_dev, 'book_t': t_book}
            for label, t in (('dev_t', t_dev), ('book_t', t_book)):
                conf = np.array([scale(r['probs'], t).max() if len(r['probs']) else 0.0 for r in rest])
                entry[f'rest_ece_{label}'] = ece(conf, y)
            report['book_recalibration'].append(entry)

    out_dir = REPO_ROOT / 'results' / split.name / 'triage'
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f'{args.name}.json').write_text(json.dumps(report, indent=1), encoding='utf8')
    write_words(out_dir / f'{args.name}-dev.tsv', dev_rows, conf_dev)
    write_words(out_dir / f'{args.name}-test.tsv', test_rows, conf_test)

    # Suspected gold errors: confident predictions that disagree with the gold,
    # the least probable gold first.
    suspects = []
    for part, rows, conf in (('dev', dev_rows, conf_dev), ('test', test_rows, conf_test)):
        for r, c in zip(rows, conf):
            if not r['correct'] and c >= args.suspect_threshold:
                p_gold = float(r['probs'][r['gold_idx']]) if r['gold_idx'] >= 0 else 0.0
                suspects.append((p_gold, -c, part, r))
    suspects.sort(key=lambda x: (x[0], x[1]))
    with open(out_dir / f'{args.name}-suspects.tsv', 'w', encoding='utf8', newline='\n') as f:
        f.write('part\tvid\tword\tsurface\tgold\tp_gold\tprediction\tconfidence\tcategory\tcontext\n')
        for p_gold, neg_c, part, r in suspects:
            f.write(f"{part}\t{r['vid']}\t{r['i'] + 1}\t{r['surface']}\t{r['gold']}\t{p_gold:.4f}\t{r['pred']}\t"
                    f"{-neg_c:.4f}\t{r['category']}\t{r['context']}\n")

    print(f'{split.name}: chosen calibrator {best}')
    for m, v in report['methods'].items():
        d, t = v['dev_cv'], v['test']
        print(f"  {m:12s} dev ECE {d['ece']:.4f} AUROC {d['auroc']:.4f} | test ECE {t['ece']:.4f} "
              f"Brier {t['brier']:.4f} AUROC {t['auroc']:.4f} mean conf {t['mean_conf']:.4f} acc {t['acc']:.4f}")
    for row in report['triage']['test']['curve']:
        print(f"  test: check {row['checked']:.0%} -> catch {row['errors_caught']:.1%}, "
              f"unchecked acc {row['unchecked_acc']:.4f}")
    print('  test: share to check for unchecked acc', report['triage']['test']['check_share_for_unchecked_acc'])
    for e in report.get('book_recalibration', []):
        print(f"  temperature refitted on the first {e['first_verses_share']:.0%} of the book: rest ECE "
              f"{e['rest_ece_dev_t']:.4f} -> {e['rest_ece_book_t']:.4f}")
    print(f'  {len(suspects)} suspected gold errors -> {out_dir}')


if __name__ == '__main__':
    main()
