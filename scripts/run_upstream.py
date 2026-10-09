"""Train an upstream system on a split and save its test (or dev) predictions.

    python scripts/run_upstream.py --system ssi-seq2seq --split random [--epochs 30] [--part test]
    python scripts/run_upstream.py --system chen-transformer --split random [--epochs N]

Systems: ssi-seq2seq (ETCBC/ssi_morphology), chen-transformer and chen-mdlm
(chenzhan4321/etcbc_models_comparison, archived paper configurations).

The model goes to models/<system>/<split>/ (git-ignored). Predictions go to
results/<split>/<system>.pred.tsv; score them with
    python scripts/evaluate.py --system <system> --from-predictions
"""
import argparse
import time

from ssi.corpus import REPO_ROOT, load_corpus
from ssi.predictions import write_predictions
from ssi.splits import Split


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--system', choices=['ssi-seq2seq', 'chen-transformer', 'chen-mdlm'], required=True)
    ap.add_argument('--split', required=True)
    ap.add_argument('--part', choices=['dev', 'test'], default='test')
    ap.add_argument('--epochs', type=int, help='default: 30 for ssi-seq2seq, the preset for chen-*')
    ap.add_argument('--reuse-model', action='store_true', help='load a saved model instead of training')
    args = ap.parse_args()

    split = Split.load(args.split)
    corpus = load_corpus(split.version)
    train, dev, test = split.partition(corpus)
    targets = dev if args.part == 'dev' else test
    model_dir = REPO_ROOT / 'models' / args.system / args.split

    t0 = time.time()
    if args.system.startswith('chen-'):
        from ssi.upstream import chen
        preds = chen.train_and_predict(args.system, model_dir, corpus, train, dev, targets, args.epochs)
        print(f'trained and predicted in {time.time() - t0:.0f}s', flush=True)
        save(args, preds)
        return

    from ssi.upstream.ssi_seq2seq import SsiSeq2Seq
    args.epochs = args.epochs or 30
    if args.reuse_model:
        system = SsiSeq2Seq.load(model_dir)
    else:
        system = SsiSeq2Seq(epochs=args.epochs, log_dir=str(model_dir / 'tb'))
        system.fit(train, dev, all_inputs=corpus)
        system.save(model_dir)
        print(f'trained in {time.time() - t0:.0f}s', flush=True)

    t1 = time.time()
    preds = system.predict(corpus, targets)
    print(f'predicted in {time.time() - t1:.0f}s', flush=True)
    save(args, preds)


def save(args, preds):
    suffix = '' if args.part == 'test' else '-dev'
    path = REPO_ROOT / 'results' / args.split / f'{args.system}{suffix}.pred.tsv'
    write_predictions(path, preds, meta={'system': args.system, 'split': args.split,
                                         'part': args.part, 'epochs': args.epochs})
    print(f'-> {path}', flush=True)


if __name__ == '__main__':
    main()
