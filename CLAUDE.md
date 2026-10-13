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
- Candidate generation (Phase 1):
  - `templates.py`: splits an analysis into proclitics + frame (affixes around a lexeme pattern) + enclitics + lexeme, and inverts lexeme patterns.
  - `generator.py`: `CandidateGenerator`. `featurise(surface)` returns every analysis that realises the word, with features; `candidates(surface)` ranks them with `weights`. It also holds the PDF rules: geminates via the `(%` pattern token (§2.3) and T-stem metathesis (§2.6).
  - `rerank.py`: learns the feature weights on pseudo-unseen words from cross-fitted folds of train. `GeneratorSystem` (systems `generator` and `generator-plain`) uses lookup for words seen in training and the generator for all others.
- Contextual ranker (Phase 2):
  - `candidates.py`: up to 24 candidates per word: all training analyses of the surface form, then generator candidates ranked by the Phase 1 weights. Each carries the generator features, lookup features, and the Phase 1 log posterior. The ranker's training verses are cross-fitted in 5 folds, so their candidates come from the other folds.
  - `ranker.py`: `ContextRanker`. A char BiGRU encodes each word, a BiLSTM runs over the verse, a char BiGRU encodes each candidate, and an MLP scores each candidate against its context. It trains with a softmax over each word's candidates and keeps the best dev checkpoint.
  - Workflow, per split:
    1. `PYTHONPATH=src python scripts/prepare_candidates.py --split <name>` (CPU, 16 processes; writes `models/ranker/<split>/candidates.pkl`).
    2. `.venv/Scripts/python.exe scripts/run_ranker.py --split <name>` (GPU, about 1 h).
    3. `scripts/evaluate.py --system ranker --from-predictions --splits <name>`, adding `--part dev` for the dev predictions.
  - `scripts/queue_ranker.sh` chains these steps for the held-out splits. It checks run_ranker's exit code, which can be non-zero even after a successful run, so check `results/` and score by hand if needed.
- Hebrew parallel text (Phase 3):
  - `hebrew.py`: for the 15 biblical books (`BOOKS` maps Syriac to BHSA book codes), the Syriac verse and the BHSA verse with the same reference are parallel. The Hebrew Dan holds only its Hebrew verses. In total, 12,106 of the 18,146 Syriac verses have a parallel.
    - BHSA `t-out` uses `+` before pronominal suffixes and `_` in compound names; `hebrew_lexemes` normalises both.
    - `IBM1` fits t(Syriac lexeme | Hebrew lexeme) on gold training verses. `HebrewEvidence` aligns each candidate's core lexeme to one Hebrew word, maximising translation or cognate score (`cognate`: weighted edit distance with regular consonant correspondences) times a position prior.
    - Each candidate gets the `HEB_FEATURES` columns and the aligned Hebrew analysis as a string. Verses without a parallel get `HEB_MISSING`.
  - `ranker.py`: if examples carry `heb`, a char BiGRU encodes the aligned Hebrew analysis and feeds it to the scorer. In training, each verse's Hebrew is hidden with probability `heb_dropout` (0.3), so the model still works on non-biblical books.
  - `run_ranker.py --split <name> --hebrew --name ranker-heb`. The translation tables are cross-fitted per candidate fold, like the candidates. Building the evidence for training takes about 6 min on the CPU, and training takes about 1.5 h.
  - `scripts/compare_hebrew.py --split <name> --part dev --systems ranker ranker-heb`: accuracy on verses with and without Hebrew, by category and error type.
  - Random test results: 93.4% → 94.8% word accuracy, CER 2.35% → 1.86%. Verses with a parallel go from 94.1% to 96.0% (unseen +5.6, ambiguous +3.6); verses without one stay the same. Vowel pattern marks don't improve.
  - Held-out Ezekiel test (a biblical book unseen in training): 91.8% → 94.0%, CER 2.91% → 2.12%, unseen words 75.9% → 82.0%.
  - Not yet checked: whether `--hebrew` costs anything on the non-biblical held-out books (Sir, ESNJ).
- Error analysis and triage (Phase 4):
  - `errors.py`: `error_type(gold, pred)` classifies how two analyses differ (segmentation, homograph index, vowel pattern mark, endings, lexeme, stem formation). `scripts/error_analysis.py` uses it to break down a system's errors and writes `results/<split>/<system>-<part>.errors.tsv`.
  - `confidence.py`: calibrators (temperature scaling, logistic) and metrics (ECE, Brier, AUROC, triage curve).
  - `scripts/triage.py --split <name>` writes to `results/<split>/triage/` (subfolder so `summary.md` ignores it):
    - the calibration metrics;
    - a per-word confidence and alternatives table;
    - `ranker-suspects.tsv`: confident predictions that disagree with the gold.
  - Temperature scaling fitted on dev wins. New books stay overconfident (ESNJ test ECE 4%); refitting on the first 5% of the book's verses halves that.
  - `scripts/find_inconsistencies.py --window 2`: identical surface contexts with different gold analyses across the whole corpus (`results/consistency/`).
  - `ContextRanker.save` / `.load`. `.pt` files written before save() existed hold only a state dict; `load` rebuilds them from the training examples.
- `scripts/make_splits.py`, `scripts/evaluate.py --system <name>`: results go to `results/<split>/<system>.json`; `results/summary.md` is regenerated, never hand-edited.
- New systems: implement `fit(train) -> self` and `predict(verses) -> {vid: [analysis, ...]}`, then register them in `SYSTEMS` in `scripts/evaluate.py`.
- Tests: `python -m pytest`. Scripts need `PYTHONPATH=src` or `pip install -e .`.
- Upstream systems: clones in `third_party/` (git-ignored; pinned commits, recreate steps and faithfulness notes in `third_party/README.md`). Adapters in `src/ssi/upstream/`: `ssi_seq2seq.py` (imports upstream modules), and `chen.py` + `chen_labels.py` (runs upstream `train.py` as a subprocess on a data dir we write).
  - Train and predict: `.venv/Scripts/python.exe scripts/run_upstream.py --system {ssi-seq2seq,chen-transformer,chen-bilstm-crf,chen-mdlm} --split <name>`. This writes `results/<split>/<system>.pred.tsv`, with models and run dirs under `models/` (git-ignored).
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
