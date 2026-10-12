# Upstream systems

Clones of the systems we compare against. The clones themselves are
git-ignored; this file pins the commits. Our adapters live in `src/ssi/upstream/`.

| Directory | Repository | Commit |
|---|---|---|
| `ssi_morphology/` | https://github.com/ETCBC/ssi_morphology | `d05e2da61a8fba96ab7a03dc42c8208982c4a0e8` |
| `etcbc_models_comparison/` | https://github.com/chenzhan4321/etcbc_models_comparison | `7de4b7b6bd8a6bbb6e0a05f2a54ba6a8c4d54d13` |

## Recreate

Run from this directory in Git Bash. `MSYS_NO_PATHCONV=1` stops Git Bash from
rewriting the leading-slash sparse patterns into Windows paths.

```
git clone --depth 1 -c core.longpaths=true https://github.com/ETCBC/ssi_morphology.git

# The full repo is ~11 GB of prediction archives; fetch only the code.
git clone --depth 1 --filter=blob:none --no-checkout -c core.longpaths=true \
    https://github.com/chenzhan4321/etcbc_models_comparison.git
cd etcbc_models_comparison
git sparse-checkout init --no-cone
MSYS_NO_PATHCONV=1 git sparse-checkout set '/*.py' '/*.md' '/requirements.txt' '/models/' \
    '/src/' '/data_processing_tools/' '/data/README.md' '/data/*/patterns.csv'
git checkout
```

Extra packages their code imports (installed in `.venv`):

```
.venv/Scripts/python.exe -m pip install python-Levenshtein tensorboard scikit-learn requests pyyaml tqdm matplotlib seaborn
```

## Notes on faithfulness

- **ssi_morphology** (`ssi.upstream.ssi_seq2seq`): model, training loop and
  beam search are imported unchanged; only the data preparation is ours. Two
  quirks are reproduced on purpose:
  - there is no shuffling between epochs and no checkpoint selection (the model
    after the last epoch is used);
  - in `DataReader.group_verses`, `grouped_verses[-1] += grouped_verses.pop(-1)`
    replaces the second-to-last block of 7 verses with the short last block
    written out twice.

  One function is replaced. Upstream `transformer_train_fns.accuracy` (only
  logged, it does not affect training) walks over CUDA tensors element by
  element, which takes about 5 s per batch, or 37 min per epoch on s12. The
  adapter swaps in `_fast_accuracy`, which gives identical values and is about
  800x faster.
- **etcbc_models_comparison** (`ssi.upstream.chen`): `train.py` runs unchanged
  as a subprocess.
  - **Label inventory:** we build it from the training lines only. Upstream's
    `02_generate_patterns_csv_from_reduced.py` reads every `*.out.reduced`
    file, including test.
  - **Restoring markers:** upstream does not include the script that turns the
    reduced markers (`M!`) back into full ones (`!M!`; `restore_to_original.py`
    is imported by `src/eval_cer.py` but missing from the repo). Our inverse is
    in `ssi.upstream.chen_labels.restore`, and round-trips all 255,339 words of
    s12.
  - **Presets:** these are the archived paper configurations. The encoder-only
    preset comes from
    `outputs/encoder_only_100ep_bs128_lr3e-4_d512_l4_h16_d0.25_5seeds/`; the
    README's `python train.py --model_type transformer` uses different defaults.
    The MDLM preset is the README headline command. The archived runs used an
    earlier version of `train.py`.
  - **BiLSTM-CRF preset:** from the hyperparameters in
    `outputs/models_1_revision/bilstm_crf/s4_on_s2_seed49/lstm_results_*.json`,
    which match the current `train.py`. It needs `pip install pytorch-crf`.
  - **Windows crash on exit:** the lstm/CRF run crashes natively (exit code
    0xC0000409) when Python shuts down, after every output has been written.
    The adapter accepts a non-zero exit code only when the run wrote a new
    predictions file whose line count matches the test windows.
