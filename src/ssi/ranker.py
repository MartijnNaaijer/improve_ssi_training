"""Contextual neural ranker over candidate analyses (Phase 2).

For every word the candidates come from ssi.candidates. The model scores each
candidate against the word in its verse:

    context    characters of each surface word (diacritics included) -> BiGRU
               -> word vector; word vectors of the whole verse -> BiLSTM
    candidate  characters of the candidate analysis -> BiGRU -> vector
    features   the candidate's generator / lookup / Phase 1 features -> MLP
    score      MLP over [context, candidate, context * candidate, features]

and is trained with a softmax over the candidates of each word (words whose
gold analysis is not among the candidates are left out of the loss). The
checkpoint with the best dev word accuracy is kept.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .corpus import Verse
from .encoding import strip_surface


@dataclass
class VerseExample:
    vid: str
    words: list[str]
    cands: list[list[str]]
    feats: list[np.ndarray]  # per word: [n candidates, n features]
    gold: list[int]          # index of the gold candidate, -1 if absent or unknown


def make_examples(verses: list[Verse], table_of, n_features: int, with_gold: bool = True) -> list[VerseExample]:
    """`table_of(verse)` gives the candidate table to use for that verse."""
    out = []
    for v in verses:
        table = table_of(v)
        cands, feats, gold = [], [], []
        for s, a in zip(v.surface, v.analysis):
            entries = table.get(s, [])
            names = [x for x, _ in entries]
            cands.append(names)
            feats.append(np.array([f for _, f in entries], dtype=np.float32).reshape(len(entries), n_features))
            gold.append(names.index(a) if with_gold and a in names else -1)
        out.append(VerseExample(v.vid, list(v.surface), cands, feats, gold))
    return out


class Vocab:
    def __init__(self, strings):
        chars = sorted({c for s in strings for c in s})
        self.index = {c: i + 2 for i, c in enumerate(chars)}  # 0 pad, 1 unknown

    def __len__(self):
        return len(self.index) + 2

    def encode(self, s: str) -> list[int]:
        return [self.index.get(c, 1) for c in s] or [1]


def _encode_strings(strings: list[str], vocab: Vocab, device) -> tuple[torch.Tensor, torch.Tensor]:
    ids = [vocab.encode(s) for s in strings]
    lengths = torch.tensor([len(x) for x in ids])
    out = torch.zeros(len(ids), int(lengths.max()), dtype=torch.long)
    for i, x in enumerate(ids):
        out[i, :len(x)] = torch.tensor(x)
    return out.to(device), lengths


class CharEncoder(nn.Module):
    def __init__(self, n_chars: int, char_dim: int, hidden: int):
        super().__init__()
        self.emb = nn.Embedding(n_chars, char_dim, padding_idx=0)
        self.rnn = nn.GRU(char_dim, hidden, batch_first=True, bidirectional=True)

    def forward(self, ids, lengths):
        packed = pack_padded_sequence(self.emb(ids), lengths, batch_first=True, enforce_sorted=False)
        _, h = self.rnn(packed)  # [2, N, hidden]
        return torch.cat([h[0], h[1]], dim=-1)


class RankerModel(nn.Module):
    def __init__(self, n_in_chars: int, n_out_chars: int, n_features: int, d: int = 256,
                 char_dim: int = 64, char_hidden: int = 128, ctx_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.word_enc = CharEncoder(n_in_chars, char_dim, char_hidden)
        self.ctx = nn.LSTM(2 * char_hidden, d // 2, num_layers=ctx_layers, batch_first=True,
                           bidirectional=True, dropout=dropout if ctx_layers > 1 else 0.0)
        self.cand_enc = CharEncoder(n_out_chars, char_dim, char_hidden)
        self.cand_proj = nn.Linear(2 * char_hidden, d)
        self.word_proj = nn.Linear(2 * char_hidden, d)
        self.feat = nn.Sequential(nn.Linear(n_features, d), nn.ReLU())
        self.drop = nn.Dropout(dropout)
        self.scorer = nn.Sequential(nn.Linear(5 * d, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, 1))

    def forward(self, word_ids, word_lens, verse_lens, cand_ids, cand_lens, cand_index, feats, cand_mask):
        b, w = cand_index.shape[:2]
        words = self.word_enc(word_ids, word_lens)                       # [B*W, 2h]
        words = words.view(b, w, -1)
        packed = pack_padded_sequence(self.drop(words), verse_lens, batch_first=True, enforce_sorted=False)
        ctx, _ = self.ctx(packed)
        ctx, _ = pad_packed_sequence(ctx, batch_first=True, total_length=w)  # [B, W, d]
        ctx = torch.cat([ctx, self.word_proj(words)], dim=-1)             # [B, W, 2d]
        cand = self.cand_proj(self.cand_enc(cand_ids, cand_lens))         # [U, d]
        c = cand[cand_index]                                              # [B, W, K, d]
        k = c.shape[2]
        h = ctx.unsqueeze(2).expand(-1, -1, k, -1)
        d = c.shape[-1]
        x = torch.cat([h, c, h[..., :d] * c, self.feat(feats)], dim=-1)
        scores = self.scorer(self.drop(x)).squeeze(-1)
        return scores.masked_fill(~cand_mask, -1e4)


class ContextRanker:
    def __init__(self, d: int = 256, epochs: int = 12, batch_size: int = 32, lr: float = 1e-3,
                 dropout: float = 0.2, seed: int = 42, device: str | None = None, log=print):
        self.d, self.epochs, self.batch_size, self.lr, self.dropout = d, epochs, batch_size, lr, dropout
        self.seed, self.log = seed, log
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))

    # -- batching -----------------------------------------------------------
    def _batch(self, examples: list[VerseExample]):
        words = [w for ex in examples for w in ex.words]
        b, w = len(examples), max(len(ex.words) for ex in examples)
        k = max([len(c) for ex in examples for c in ex.cands] + [1])
        # Word grid padded to [B, W]: padding words get a one-character dummy.
        grid = []
        for ex in examples:
            grid += ex.words + ['_'] * (w - len(ex.words))
        word_ids, word_lens = _encode_strings(grid, self.in_vocab, self.device)
        verse_lens = torch.tensor([len(ex.words) for ex in examples])

        uniq, pos = ['_'], {'_': 0}
        cand_index = torch.zeros(b, w, k, dtype=torch.long)
        cand_mask = torch.zeros(b, w, k, dtype=torch.bool)
        feats = torch.zeros(b, w, k, self.n_features)
        gold = torch.full((b, w), -1, dtype=torch.long)
        for i, ex in enumerate(examples):
            for j, (cs, f, g) in enumerate(zip(ex.cands, ex.feats, ex.gold)):
                for m, a in enumerate(cs):
                    if a not in pos:
                        pos[a] = len(uniq)
                        uniq.append(a)
                    cand_index[i, j, m] = pos[a]
                cand_mask[i, j, :len(cs)] = True
                if len(cs):
                    feats[i, j, :len(cs)] = torch.from_numpy(np.clip((f - self.mu) / self.sd, -5, 5))
                gold[i, j] = g
        cand_ids, cand_lens = _encode_strings(uniq, self.out_vocab, self.device)
        dev = self.device
        return (word_ids, word_lens, verse_lens, cand_ids, cand_lens, cand_index.to(dev), feats.to(dev),
                cand_mask.to(dev)), gold.to(dev)

    # -- set-up, saving and loading --------------------------------------------
    def _setup(self, train: list[VerseExample]):
        """Vocabularies, feature scaling and a fresh model, all from the training examples."""
        self.in_vocab = Vocab(w for ex in train for w in ex.words)
        self.out_vocab = Vocab(a for ex in train for cs in ex.cands for a in cs)
        all_f = np.concatenate([f for ex in train for f in ex.feats if len(f)])
        self.n_features = all_f.shape[1]
        self.mu, self.sd = all_f.mean(0), all_f.std(0) + 1e-6
        self._new_model()

    def _new_model(self):
        self.model = RankerModel(len(self.in_vocab), len(self.out_vocab), self.n_features, d=self.d,
                                 dropout=self.dropout).to(self.device)

    def save(self, path):
        torch.save({'state': self.model.state_dict(), 'in_vocab': self.in_vocab.index,
                    'out_vocab': self.out_vocab.index, 'mu': self.mu, 'sd': self.sd,
                    'config': {'d': self.d, 'dropout': self.dropout}}, path)

    @classmethod
    def load(cls, path, train: list[VerseExample] | None = None, device: str | None = None,
             **config) -> 'ContextRanker':
        """Load a saved ranker. Files that hold only a state dict (written before
        save() existed) also need the training examples to rebuild vocabularies
        and feature scaling, and `config` (d, dropout) if not the defaults."""
        data = torch.load(path, map_location='cpu', weights_only=False)
        if 'state' not in data:
            if train is None:
                raise ValueError(f'{path} holds only a state dict; pass the training examples')
            ranker = cls(device=device, **config)
            ranker._setup(train)
            ranker.model.load_state_dict(data)
            return ranker
        ranker = cls(device=device, **data['config'])
        ranker.in_vocab, ranker.out_vocab = Vocab(()), Vocab(())
        ranker.in_vocab.index, ranker.out_vocab.index = data['in_vocab'], data['out_vocab']
        ranker.mu, ranker.sd = data['mu'], data['sd']
        ranker.n_features = len(ranker.mu)
        ranker._new_model()
        ranker.model.load_state_dict(data['state'])
        return ranker

    # -- training -----------------------------------------------------------
    def fit(self, train: list[VerseExample], dev: list[VerseExample] | None = None) -> 'ContextRanker':
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        self._setup(train)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        steps = self.epochs * math.ceil(len(train) / self.batch_size)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=steps, pct_start=0.1)
        loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        n_params = sum(p.numel() for p in self.model.parameters())
        self.log(f'[ranker] {len(train)} train verses, {n_params:,} parameters, device {self.device}')

        best, best_state = -1.0, None
        order = list(range(len(train)))
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            random.shuffle(order)
            t, total, n = time.time(), 0.0, 0
            for i in range(0, len(order), self.batch_size):
                inputs, gold = self._batch([train[j] for j in order[i:i + self.batch_size]])
                scores = self.model(*inputs)
                loss = loss_fn(scores.flatten(0, 1), gold.flatten())
                if not torch.isfinite(loss):
                    continue
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                sched.step()
                total += loss.item()
                n += 1
            msg = f'[ranker] epoch {epoch}: loss {total / max(n, 1):.4f} ({time.time() - t:.0f}s)'
            if dev is not None:
                acc = self.accuracy(dev)
                msg += f', dev acc (words with candidates) {acc:.4f}'
                if acc > best:
                    best = acc
                    best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            self.log(msg)
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.log(f'[ranker] kept best dev checkpoint ({best:.4f})')
        return self

    # -- prediction ---------------------------------------------------------
    @torch.no_grad()
    def probabilities(self, examples: list[VerseExample]) -> list[list[np.ndarray]]:
        """Per verse, per word: the softmax over that word's candidates (empty if it has none)."""
        self.model.eval()
        out = []
        for i in range(0, len(examples), 64):
            batch = examples[i:i + 64]
            inputs, _ = self._batch(batch)
            probs = torch.softmax(self.model(*inputs).float(), -1).cpu().numpy()
            for j, ex in enumerate(batch):
                out.append([probs[j, m, :len(cs)] for m, cs in enumerate(ex.cands)])
        return out

    def choices(self, examples: list[VerseExample]) -> list[list[int]]:
        return [[int(p.argmax()) if len(p) else -1 for p in verse] for verse in self.probabilities(examples)]

    def accuracy(self, examples: list[VerseExample]) -> float:
        hit = n = 0
        for ex, ch in zip(examples, self.choices(examples)):
            for g, c, cs in zip(ex.gold, ch, ex.cands):
                if cs:
                    n += 1
                    hit += g >= 0 and g == c
        return hit / max(n, 1)

    def predict(self, examples: list[VerseExample]) -> dict[str, list[str]]:
        preds = {}
        for ex, ch in zip(examples, self.choices(examples)):
            preds[ex.vid] = [cs[c] if c >= 0 else strip_surface(w) for w, cs, c in zip(ex.words, ex.cands, ch)]
        return preds
