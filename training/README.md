# Training and benchmark scripts (Phase 3)

Offline Python for fine-tuning and benchmarking the speaker model. The plan is
[docs/superpowers/plans/2026-10-28-phase3-fine-tuning.md](../docs/superpowers/plans/2026-10-28-phase3-fine-tuning.md).
Nothing here is part of the Gradle build. All data lives under the gitignored `data/` and
results under `reports/`.

Needs Python 3.10+ with numpy, and ffmpeg on the PATH. Run everything from the repository root
with `python -I`. `check_checkpoint.py` also needs the packages in `requirements.txt`, which
belong in a virtual environment (`.venv/`, gitignored).

## Starting checkpoint (Task 1)

The checkpoint is WeSpeaker `voxceleb_resnet34_LM` from Hugging Face,
`Wespeaker/wespeaker-voxceleb-resnet34-LM` at revision `f0c48c298fd835726c27956a5d617bad7115627e`
(CC BY 4.0):

| File | SHA-256 |
|---|---|
| `avg_model` (PyTorch state dict) | `9872b375f2c6a3851ca471cbbf59e06efd23a627d78bf5872e1f0269fd298449` |
| `config.yaml` | `3cf7d3243464cd939083e29d2be65c2abcdd954c1a64559bad73b74ffdb0db3e` |

```bash
python -I training/check_checkpoint.py data/wespeaker/resnet34_LM/avg_model models/speaker_embedding.onnx
```

**It is the network the app ships.** After folding each BatchNorm into its convolution, as the
exporter does, all 36 convolutions and the embedding layer match `models/speaker_embedding.onnx`.
The largest relative difference is 1.7e-7, which is float32 rounding. The mean vector is zero.
The ONNX file that WeSpeaker publishes beside the checkpoint has the same weights, and the same
graph node for node. sherpa-onnx's copy only adds metadata. Its shape matches WeSpeaker's
`export_onnx.py`, with `feats` and `embs` at opset 14. Identical weights in an identical graph
give identical embeddings, so this replaces the cosine check on clips. The check reads weights
only and runs no WeSpeaker code. WeSpeaker itself first runs on Kaggle.

The checkpoint's `config.yaml` is already the large-margin fine-tuning recipe:

- **Schedule:** learning rate 1e-4 decaying to 2.5e-5, margin 0.5, 5 epochs, 600-frame chunks.
- **Speaker count:** the head has 17,982 classes, which is VoxCeleb2's 5,994 speakers × 3. Speed
  perturbation counts each speed as a new speaker. Task 4's head grows the same way when speed
  perturbation is on.

## Training data (Task 2)

```bash
python -I training/prepare_samromur.py data/samromur --fetch    # 6.6 GB at a pinned revision, hashes checked
python -I training/make_trials.py data/samromur holdout          # trials for model selection
```

The FLACs are already 16 kHz mono 16-bit, and WeSpeaker reads FLAC, so the lists point at them
directly instead of writing 15 GB of WAV.

| | Speakers (sessions) | Utterances | Hours |
|---|---:|---:|---:|
| Corpus train split | 2,517 | 134,394 | 127.4 |
| **Training** (sessions with 2+ utterances, holdout removed) | 2,327 | 130,465 | 123.6 |
| **Holdout** for selection: 19 per gender × age band (4–8, 9–11, 12–14, 15–17), 5–100 utterances each | 152 | 3,891 | 3.8 |

- **Almost no young children.** Ages 4–5 are 11 sessions and 0.1 h in all. Ages 4–6 together are
  under 1 h. Most of the audio is ages 9–13. Our target children are 1–5.
- **Unbalanced per speaker.** The median is 18 utterances per session and the largest is 2,399.
- **Gender** (sessions): 1,357 female, 1,097 male, 63 unknown or other. Speakers with unknown
  gender or age stay in training but are never held out.
- **Session IDs are not children.** Train and test share no session ID. Train and dev share one,
  010363, as the dataset card says. Nothing in the metadata links two sessions of one child, and
  linking them by voice would mean identifying speakers, which the terms forbid. So a repeat
  child stays two classes, which is a known bias towards a pessimistic EER.
- **Holdout trials:** 2,910 target pairs, plus 8,730 hard and 8,730 random non-target pairs.

## Fine-tuning on Kaggle (Task 4)

`kaggle_finetune.ipynb` runs the whole job on a Kaggle GPU. Its first cell says how to import and
commit it. In outline:

1. **Environment.** Clones this repository and WeSpeaker at pinned commits. Builds a Python 3.11
   venv from `requirements-kaggle.txt` with uv.
2. **Data.** Fetches the checkpoint and Samrómur and checks their hashes.
3. **Checks.** Confirms that `model_init` loads every backbone tensor, and that the GPU scorer
   reproduces the app's 11.26% on Samrómur test.
4. **Training.** Trains on `conf/samromur_resnet34_ft.yaml`.
5. **Selection.** Scores every epoch, and the average of the last three, on the 152-speaker
   holdout. Picks the lowest hard EER.
6. **Export.** Exports the pick to ONNX and verifies the tensor contract, plus parity with PyTorch.

Training stops itself at a time budget, and a second commit with `RESUME_FROM` continues from the
last epoch. A resumed run restarts SGD's momentum, which WeSpeaker does not checkpoint.

**The recipe** is WeSpeaker's own for moving a trained model to a new corpus
(`examples/cnceleb/v3_finetune`), not the large-margin settings the checkpoint was last trained
with. The new head starts from random weights, and a learning rate of 1e-4 with margin 0.5 from
the first step gives it little chance to learn. So the run uses:

- learning rate 5e-3 decaying to 2.5e-5 after a 2-epoch warm-up from zero;
- margin 0 rising to 0.2 over epochs 3–5;
- speed perturbation, so 3 × 2,327 classes;
- 2 s chunks and 15 epochs.

The config's header lists every difference from WeSpeaker's file.

**Pinned environment.** WeSpeaker's training imports every frontend it ships, so s3prl, whisper,
transformers and peft must install even though this recipe uses none of them. Two of their
source packages (whisper, and visdom through torchnet) need setuptools older than 81, hence
`build-constraints-kaggle.txt`. The full set resolves under uv on Python 3.11. That is checked
here with CPU wheels; the CUDA build has not run yet.

**The app's front end is WeSpeaker's.** `embed_onnx.py` computes WeSpeaker's inference features
in Python: Kaldi fbank, Hamming window, int16 scale, no dither, mean normalization. On all 1,714
Samrómur test files, with the shipped model, its embeddings match the Kotlin harness's at cosine
1.000000. Both give 11.26% EER. FLAC and the converted WAV give identical embeddings. So
selecting on Python-side embeddings selects for what the app will compute, and the parity half of
Task 5 holds for any model trained on these features.

```bash
python -I training/embed_onnx.py models/speaker_embedding.onnx data/samromur/test.list reports/samromur-baseline/test-embeddings-python.csv
python -I training/compare_embeddings.py reports/samromur-baseline/test-embeddings.csv reports/samromur-baseline/test-embeddings-python.csv
```

## Samrómur Children benchmark

Samrómur Children is CC BY 4.0. Its one condition: do not attempt to identify any speaker.

```bash
# dev.tar.gz, test.tar.gz and metadata_{dev,test}.tsv from
# https://huggingface.co/datasets/language-and-voice-lab/samromur_children (corpus/), unpacked so
# that the audio sits in data/samromur/wav/{dev,test}/<speaker>/<audio_id>.flac
python -I training/prepare_samromur_eval.py data/samromur test      # FLAC → 16 kHz WAV + test.list
python -I training/make_trials.py data/samromur test                # trials_test.txt
./gradlew :harness:embedFiles --args="--list data/samromur/test.list --models models --out reports/samromur-baseline/test-embeddings.csv"
python -I training/score_trials.py reports/samromur-baseline/test-embeddings.csv data/samromur/trials_test.txt
```

`embedFiles` embeds through the app's own Kotlin front end and ONNX model, so these numbers
describe what ships. Pass `--model other.onnx` to score a candidate beside the current model.

## Baselines (Phase 3, Task 0) — current model

Model: WeSpeaker `voxceleb_resnet34_LM`, adult-trained, as in `models/PROVENANCE.txt`.

**Samrómur Children**, utterance against utterance. "Hard" non-targets are the same gender and at
most a year apart in age; "random" are any other child.

| Split | Speakers | Target trials | EER (hard) | EER (random) | minDCF 0.01 (hard) |
|---|---:|---:|---:|---:|---:|
| dev | 34 | 446 | 8.97% | 8.33% | 0.666 |
| test | 625 (ages 6–16) | 1,989 | 11.26% | 9.05% | 0.876 |

Dev is lopsided (one speaker holds 1,000 of its 1,489 utterances), so Task 4 selects on speakers
held out from the training split instead.

**CHILDES**, through the product harness: speaker episodes, default engine settings. FA is false
alarms per hour.

| Corpus | Ages | Episodes scored | Best detection | FA/h there | AUC child vs parents | AUC child vs sibling |
|---|---|---:|---:|---:|---:|---:|
| MacWhinney, Ross 4;00–4;05 | 4 | 367 | 5.7% | 1.1 | 0.57 | 0.53 |
| MacWhinney, Ross 4;00–6;11 | 4–6 | 1,997 | 2.9% | 7.8 | — | — |
| Gleason, 22 families | 2–5 | 557 (7 families skipped) | 8.1% | 0.35 | 0.89 | — |
| BermanLong (Hebrew), 4 children | 1–3 | 1,524 | 7.0% | 6.8 | 0.78 | 0.51 against other children |

Every row is FALL_BACK. The rows fail for different reasons:

- **MacWhinney: the embedding.** The VAD hears Ross, but his speech scores closest to his own
  centroid only 27% of the time, and against his brother it is at chance. This is what
  fine-tuning targets.
- **Gleason: the VAD and the gates, not the embedding.** When the child's speech is scored, it
  separates from the parents well (AUC 0.89). But Silero rates only 13% of the children's episode
  audio as speech, against 36% of the parents'. So 7 of 22 children could not be enrolled at all,
  about 60% of episodes produced no observation, and the 0.35 absolute gate removed 37.5% of the
  child's remaining positive evidence. Fine-tuning the speaker model cannot fix that.
- **BermanLong (Hebrew): both failures at once.** The child separates from the parents
  (AUC 0.78) but not from other children: playmates and visitors, unenrolled, score at chance
  (0.51), and half of their speech produces positive evidence for the target, which drives most
  of the false alarms. As in Gleason, 55% of episodes yield no observation, and 70% of the rest
  yield exactly one, which cannot reach the default threshold.
- **Samrómur** (clean read speech, ages 6–16) is far easier than either. Gains there will
  overstate gains in a home.

## VAD and gate calibration (CHILDES)

Whether a looser VAD recovers the toddler speech Silero misses. Silero's per-frame probabilities
are recorded once, then gate settings are replayed against them:

```bash
./gradlew :harness:vadProbes --args="--manifest data/childes-gleason --models models --out reports/vad-probes/gleason"
python -I tools/experiments/vad_sweep.py reports/vad-probes/gleason          # segments only, seconds per setting
./gradlew :harness:run --args="--manifest data/childes-gleason --models models --out reports/vad-cal/gleason-vad25 \
  --vad-probes reports/vad-probes/gleason --vad 0.25,0.17,16,400 --min-embed 400 --min-target-similarity 0.35,0.3,0.25,0.2"
python -I tools/experiments/compare_runs.py data/childes-gleason reports/vad-probes/gleason RUN_DIR [RUN_DIR ...]
```

`compare_runs.py` scores every run on the households all of them enrolled, since a looser VAD
also enrols harder children. Replaying the probabilities reproduces a Silero run exactly (all
5,842 Gleason observations identical).

Chosen on Gleason, then checked unchanged on the other two. Detection is the best reachable at
≤ 2 false alarms per hour (the ITERATE bar) across gates 0.2–0.35 and fire thresholds 2–10.

| Corpus | Default (0.5/0.35, hang 8, embed 800 ms) | VAD 0.25/0.17, hang 16, embed 400 ms |
|---|---:|---:|
| Gleason (tuning), 15 households | 9.9% | 21.9% |
| BermanLong (Hebrew), 4 children | 0.9% | 0.5% |
| MacWhinney slice | 5.7% | 6.8% |

- **The VAD loss is mostly recoverable.** On Gleason, the share of episodes yielding any
  observation rises from 41% to 74%, and 20 of 22 households enrol instead of 15. The 800 ms
  embedding floor alone accounts for about half of that gain.
- **The embedding is now the limit everywhere.** Only 42% of Gleason episodes produce any
  evidence for the child, and most new false alarms are mothers' short, high-pitched fragments
  scored as the child. In BermanLong, unenrolled children absorb the whole false-alarm budget at
  either setting.
- **Lowering the 0.35 absolute gate does not help.** It buys a point or two on Gleason and costs
  it back on BermanLong, where it is the only defence against other children.
- **Not measurable here:** each CHILDES file is one speaker's speech, so neither the battery cost
  of a looser VAD on room noise and TV nor the effect of a 512 ms hangover joining two speakers'
  turns shows up. Those need real-room audio before any default changes.
