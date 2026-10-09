"""Adapter that trains and runs the ETCBC/ssi_morphology transformer on our splits.

The model, training loop and beam search are imported unchanged from
third_party/ssi_morphology/src. Only the data preparation is replaced, so the
model sees our train/dev partition instead of its own internal random split.
The preparation mirrors upstream `DataReader`: verses are grouped per book in
blocks of `seq_len` verses, rolling windows of `seq_len` words are made within
each block, and the order of blocks is shuffled once (upstream gets that from
`train_test_split`). As upstream, there is no per-epoch shuffling and the model
after the last epoch is used.

Upstream defaults reproduced here: seq_len 7, lr 1e-4, 3+3 layers, emb 512,
8 heads, ffn 512, dropout 0.1, batch 128, Adam(0.9, 0.98, eps 1e-9), beam 3,
beam alpha 0.75. Upstream README trains 30 epochs.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from ..corpus import REPO_ROOT, Verse
from ..windows import assemble, cover_with_windows

UPSTREAM_SRC = REPO_ROOT / 'third_party' / 'ssi_morphology' / 'src'


def _import_upstream():
    if not UPSTREAM_SRC.is_dir():
        raise FileNotFoundError(f'{UPSTREAM_SRC} not found; see third_party/README.md')
    if str(UPSTREAM_SRC) not in sys.path:
        sys.path.insert(0, str(UPSTREAM_SRC))
    import torch  # noqa: F401  (fail early if the ml extra is missing)
    import config, data, evaluate_transformer, transformer_train_fns  # noqa: E401
    transformer_train_fns.accuracy = _fast_accuracy
    return config, data, evaluate_transformer, transformer_train_fns


def _fast_accuracy(target, labels):
    """Drop-in for upstream `transformer_train_fns.accuracy` with identical output.

    Upstream iterates over CUDA tensors element by element (one GPU sync per
    element, ~5 s per batch); this copies to the CPU once. The value is only
    logged and does not influence training.
    """
    import torch
    from Levenshtein import distance
    o = ord(' ')
    gold = labels[1:, :].cpu().tolist()
    pred = torch.max(target, -1)[1].cpu().tolist()
    wrong = sum(distance(''.join(chr(o + i) for i in g), ''.join(chr(o + i) for i in p))
                for g, p in zip(gold, pred))
    return 1 - wrong / (len(gold) * len(gold[0]))


def _blocks(verses: list[Verse], seq_len: int) -> list[tuple[list[str], list[str]]]:
    """Upstream DataReader.group_verses + make_rolling_window_strings, per block."""
    by_book: dict[str, list[Verse]] = {}
    for v in verses:
        by_book.setdefault(v.book, []).append(v)
    blocks = []
    for book_verses in by_book.values():
        if len(book_verses) <= seq_len:
            groups = [book_verses]
        else:
            groups = [book_verses[i:i + seq_len] for i in range(0, len(book_verses), seq_len)]
            if len(groups[-1]) < seq_len:
                # Upstream intends to merge a short last group into the previous
                # one, but `groups[-1] += groups.pop(-1)` replaces the previous
                # (full) group with the short group doubled. Reproduced as is.
                groups[-1] += groups.pop(-1)
        for g in groups:
            surf = [s for v in g for s in v.surface]
            anal = [a for v in g for a in v.analysis]
            wins = [(' '.join(surf[i:i + seq_len]), ' '.join(anal[i:i + seq_len]))
                    for i in range(len(surf) - seq_len + 1)]
            if wins:
                blocks.append(wins)
    return blocks


class SsiSeq2Seq:
    name = 'ssi-seq2seq'

    def __init__(self, epochs: int = 30, seq_len: int = 7, lr: float = 1e-4, emb: int = 512,
                 nhead: int = 8, enc_layers: int = 3, dec_layers: int = 3, ffn: int = 512,
                 dropout: float = 0.1, batch_size: int = 128, beam_size: int = 3,
                 beam_alpha: float = 0.75, seed: int = 42, log_dir: str | None = None):
        self.epochs, self.seq_len, self.lr = epochs, seq_len, lr
        self.emb, self.nhead, self.enc_layers, self.dec_layers = emb, nhead, enc_layers, dec_layers
        self.ffn, self.dropout, self.batch_size = ffn, dropout, batch_size
        self.beam_size, self.beam_alpha, self.seed = beam_size, beam_alpha, seed
        self.log_dir = log_dir
        self.model = None

    # -- vocabularies -------------------------------------------------------
    def _vocab(self, strings, reduce=None):
        d = {'PAD': 0, 'SOS': 1, 'EOS': 2}
        for s in strings:
            for c in (reduce(s) if reduce else s):
                d.setdefault(c, len(d))
        return d

    def fit(self, train: list[Verse], dev: list[Verse], all_inputs: list[Verse] | None = None) -> 'SsiSeq2Seq':
        """Train on `train`, report validation loss on `dev` each epoch.

        `all_inputs` (surface text only) extends the input character vocabulary
        so that unseen characters at prediction time do not crash; upstream
        builds it from all data as well.
        """
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader
        config, data, _, tf = _import_upstream()

        rng = random.Random(self.seed)
        train_blocks, dev_blocks = _blocks(train, self.seq_len), _blocks(dev, self.seq_len)
        rng.shuffle(train_blocks)
        rng.shuffle(dev_blocks)
        Xtr = [x for b in train_blocks for x, _ in b]
        ytr = [y for b in train_blocks for _, y in b]
        Xdv = [x for b in dev_blocks for x, _ in b]
        ydv = [y for b in dev_blocks for _, y in b]

        input_texts = Xtr + Xdv + [' '.join(v.surface) for v in (all_inputs or [])]
        self.in_vocab = self._vocab(input_texts)
        self.out_vocab = self._vocab(ytr + ydv, reduce=data.mc_reduce)
        self.out_inv = {i: c for c, i in self.out_vocab.items()}
        src_vocab, tgt_vocab = len(self.in_vocab) + 2, len(self.out_vocab) + 2  # as upstream

        torch.manual_seed(self.seed)
        train_set = data.HebrewWords(Xtr, ytr, self.in_vocab, self.out_vocab)
        dev_set = data.HebrewWords(Xdv, ydv, self.in_vocab, self.out_vocab)
        train_dl = DataLoader(train_set, batch_size=self.batch_size, collate_fn=data.collate_transformer_fn)
        dev_dl = DataLoader(dev_set, batch_size=self.batch_size, collate_fn=data.collate_transformer_fn)

        model = tf.initialize_transformer_model(self.enc_layers, self.dec_layers, self.emb, self.nhead,
                                                src_vocab, tgt_vocab, self.ffn, self.dropout)
        loss_fn = nn.CrossEntropyLoss(ignore_index=config.PAD_IDX)
        optimizer = optim.Adam(model.parameters(), lr=self.lr, betas=(0.9, 0.98), eps=1e-9)
        print(f'[ssi-seq2seq] {len(Xtr)} train windows, {len(Xdv)} dev windows, '
              f'vocab in {len(self.in_vocab)} out {len(self.out_vocab)}', flush=True)
        self.model = tf.train_transformer(model, loss_fn, optimizer, train_dl, dev_dl, self.epochs,
                                          config.PAD_IDX, self.seed, self.lr,
                                          self.log_dir or str(REPO_ROOT / 'models' / 'tb'),
                                          self.batch_size, self.in_vocab, self.out_vocab)
        return self

    def predict(self, corpus: list[Verse], targets: list[Verse], log_every: int = 500) -> dict[str, list[str]]:
        config, data, ev, _ = _import_upstream()
        windows = cover_with_windows(corpus, targets, self.seq_len)
        preds = []
        for k, w in enumerate(windows):
            text = ' '.join(w.words)
            enc = data.encode_string(text, self.in_vocab, add_sos=False, add_eos=True)
            out = ev.translate(self.model, enc, self.out_inv, self.out_vocab, self.beam_size, self.beam_alpha)
            preds.append(out.split())
            if log_every and (k + 1) % log_every == 0:
                print(f'[ssi-seq2seq] predicted {k + 1}/{len(windows)} windows', flush=True)
        return assemble(targets, windows, preds)

    # -- persistence --------------------------------------------------------
    def save(self, directory: Path):
        import torch
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), directory / 'model.pt')
        cfg = {k: v for k, v in self.__dict__.items() if k not in ('model', 'out_inv')}
        (directory / 'config.json').write_text(json.dumps(cfg, indent=1), encoding='utf8')

    @classmethod
    def load(cls, directory: Path) -> 'SsiSeq2Seq':
        import torch
        _, _, _, tf = _import_upstream()
        cfg = json.loads((directory / 'config.json').read_text(encoding='utf8'))
        in_vocab, out_vocab = cfg.pop('in_vocab'), cfg.pop('out_vocab')
        self = cls(**cfg)
        self.in_vocab, self.out_vocab = in_vocab, out_vocab
        self.out_inv = {i: c for c, i in out_vocab.items()}
        model = tf.initialize_transformer_model(self.enc_layers, self.dec_layers, self.emb, self.nhead,
                                                len(in_vocab) + 2, len(out_vocab) + 2, self.ffn, self.dropout)
        model.load_state_dict(torch.load(directory / 'model.pt', map_location=model.generator.weight.device))
        model.eval()
        self.model = model
        return self
