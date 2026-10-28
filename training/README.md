# Training and benchmark scripts (Phase 3)

Offline Python for fine-tuning and benchmarking the speaker model. The plan is
[docs/superpowers/plans/2026-10-28-phase3-fine-tuning.md](../docs/superpowers/plans/2026-10-28-phase3-fine-tuning.md).
Nothing here is part of the Gradle build. All data lives under the gitignored `data/` and
results under `reports/`.

Needs Python 3.10+ with numpy, and ffmpeg on the PATH. Run everything from the repository root
with `python -I`.

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
