# CLAUDE.md

## Project goal

Build a morphological parser for Classical Syriac (ETCBC encoding) that outperforms:

- **ETCBC/ssi_morphology** (https://github.com/ETCBC/ssi_morphology): transformer seq2seq over 7-word windows (Naaijer et al. 2023, ALP workshop).
- **chenzhan4321/etcbc_models_comparison** (https://github.com/chenzhan4321/etcbc_models_comparison): "structure-first" per-character marker classification (330-label inventory) + masked diffusion LM; CER 3.38% on the S2 test set.

Approach: **hybrid generate → filter → rank**:
1. Candidates from (a) training-data lookup, (b) a rule-based generator built from `Syriac_paradigm.pdf` + lexicon, (c) neural n-best.
2. Hard filters: surface realisation check, word grammar, lexicon membership.
3. Contextual neural ranker (verse-level context), with calibrated confidence for annotator triage.

## Layout

- `Syriac_paradigm.pdf`: Sikkel (2023), *The ETCBC Paradigm for the Encoding of Syriac*. The authoritative spec for the encoding, paradigms, and special cases. Red text in it marks unresolved questions; don't treat that text as settled policy. **Local only: it is git-ignored and must never be committed or pushed** (it was removed from the history on 2026-10-09).
- `data/raw/`: corpus files copied from ssi_morphology (provenance in `data/README.md`). `data/splits/`: verse-id lists per split (committed).
- `src/ssi/`: `corpus.py` (loading), `encoding.py` (parse, realise, segmentation), `splits.py`, `metrics.py`, `lookup.py` (baseline).
- `scripts/make_splits.py`, `scripts/evaluate.py --system <name>`: results go to `results/<split>/<system>.json`; `results/summary.md` is regenerated, never hand-edited.
- New systems: implement `fit(train) -> self` and `predict(verses) -> {vid: [analysis, ...]}`, then register them in `SYSTEMS` in `scripts/evaluate.py`.
- Tests: `python -m pytest`. Scripts need `PYTHONPATH=src` or `pip install -e .`.
- Upstream systems: clones in `third_party/` (git-ignored; pinned commits, recreate steps and faithfulness notes in `third_party/README.md`). Adapters in `src/ssi/upstream/`: `ssi_seq2seq.py` (imports upstream modules), and `chen.py` + `chen_labels.py` (runs upstream `train.py` as a subprocess on a data dir we write).
  - Train and predict: `.venv/Scripts/python.exe scripts/run_upstream.py --system {ssi-seq2seq,chen-transformer,chen-mdlm} --split <name>`. This writes `results/<split>/<system>.pred.tsv`, with models and run dirs under `models/` (git-ignored).
  - Score: `scripts/evaluate.py --system <name> --from-predictions`.
  - Systems that work on 7-word windows predict each test word once, from non-overlapping windows over the book's full surface text (`ssi.windows`). Context words outside the test set are input only.
  - Run upstream code with `PYTHONUTF8=1` when output is redirected: it prints emoji, which crashes the Windows cp1252 console encoding.

## Data

Tab-separated files `book<TAB>chapter<TAB>verse<TAB>text`, one verse per line; `*-in` holds surface text and `*-out` holds the morphological encoding. The two files are aligned word by word (same number of space-separated words).

- `s2`, `s3`, …, `s12` are **cumulative versions** of the same corpus. `s12` (18,146 verses, ~255k words) is the latest and contains all S2 verses; 1,530 of 5,588 S2 verses were re-analysed after S2.
- **Use `s12` as gold.** Report results on S2 only for comparison with published numbers, and flag them as such.
- `t-in_con` / `t-in_voc` / `t-out`: Hebrew Bible (BHSA), usable for auxiliary training.
- Input text may contain `^`, `#`, `"` (diacritics such as seyame). They are not part of the consonantal skeleton, so strip them for the realisation check, but keep them as model input: they are informative.
- Nine references occur twice in `s12` (a verse split over two lines). Verse ids get a `#2` suffix for the second line.

## Encoding cheat sheet (see PDF §1)

Morpheme order within an analytical word: `pfm pfx vbs LEXEME vbe nme emf vpm`.

| Marker | Type | Example |
|---|---|---|
| `!…!` | preformative (pfm) | `!N!CXN[` |
| `@…@` | passive prefix (pfx) | `@>T@QBR[` |
| `]…]` | verbal stem morpheme (vbs) | `]>]MLK=[` |
| `[` | verbal ending (vbe) | `CM<[TWN` |
| `/` | nominal ending (nme) | `GBR/JN` |
| `~` | emphatic marker (emf) | `MLK/~>` |
| `:x` | vowel pattern mark (`d` = D-stem, `p` = passive) | `PQD[:d` |
| `-` | analytical word boundary (proclitics, pronominal suffixes) | `W-MLK/~>` |
| `(X` | paradigmatic letter X **not realised** | `ML(>/J` |
| `&X` | letter X realised but **not paradigmatic** | `KT&WB[` |
| `=` | homograph index on lexemes/morphemes (rarer = more `=`) | `HWN=` |

**Realisation invariant:** deleting the character after each `(`, keeping the character after each `&`, and stripping all markers (`! @ ] [ / ~ - =` and `:[a-z]+`) reproduces the surface word. It holds for 100% of `s12` tokens. Every predicted analysis must pass this check.

ssi_morphology's `mc_reduce` / `mc_expand` drop and restore the leading marker of the doubled prefix markers and the `:` of vowel pattern marks. Keep that in mind when comparing outputs.

## Evaluation conventions

- Primary metric: **exact word accuracy** on the full encoding. Also report lexeme accuracy, segmentation accuracy, and F1 per morpheme type.
- Always break results down into **seen / unseen / ambiguous** surface forms.
- Splits (`data/splits/`): `random` (80/10/10 by verse) plus `heldout-{Sir,BLC,Eprx,ApBar,ESNJ,Ez}`. Never split at window level: overlapping windows leak. Tune on `dev`; touch `test` only for reported numbers.
- Baseline to beat (`lookup`, see `results/summary.md`): 86.2% word accuracy on `random`, 68–83% on held-out books. Unseen words are 11–23% of tokens and almost all wrong.
- For comparison with Chen et al., also report character-level CER on the restored encoding.

## Environment

- Virtual env: `.venv/` in the repo root (git-ignored), Python 3.13.
- Installed: torch 2.11.0+cu128 (CUDA 12.8 build), numpy 2.5, pytest, and this package in editable mode.
- GPU: NVIDIA RTX A4000 Laptop, 8 GB VRAM; driver 596.71 (supports up to CUDA 13.2). Size batches and models for 8 GB: use mixed precision and gradient accumulation for larger models.
- Use the venv interpreter directly; there's no need to activate it:
  - Git Bash: `.venv/Scripts/python.exe -m pytest`, `.venv/Scripts/python.exe scripts/evaluate.py --system lookup`
  - PowerShell: `.venv\Scripts\python.exe ...`
- The system `python` has no torch; scripts that don't need torch also run with `PYTHONPATH=src python ...`.
- To recreate the env:
  ```
  python -m venv .venv
  .venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
  .venv/Scripts/python.exe -m pip install -e ".[dev,ml]"
  ```
  Install torch from the CUDA index *first*: plain `pip install torch` gets the CPU-only build. The download is about 2.75 GB.

## Conventions

- Python. Keep data loading, candidate generation, filtering, and ranking as separate modules so each can be evaluated on its own.
- Fix random seeds; record the data version (`s12` etc.) and split in every results file.
- Platform is Windows (PowerShell primary, Git Bash available).
